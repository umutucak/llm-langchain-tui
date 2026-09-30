"""For all the MCP client interfaces."""
import httpx

from collections.abc import Awaitable, Callable

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools.base import BaseTool

from llmtui.config import OBSIDIAN_API_KEY, OBSIDIAN_CERT_PATH


def _obsidian_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """The httpx client the MCP session talks over, trusting Obsidian's cert.

    The Local REST API plugin serves https on a certificate it signed itself,
    so the default trust store rejects it and the handshake dies before the API
    key is ever sent -- the failure reads as an auth problem but is not one.
    Naming the plugin's own certificate as the trust root fixes it while
    leaving verification on, which verify=False would not: that would accept
    any certificate at all, on every request this client makes.

    The signature is the McpHttpClientFactory protocol, and the defaults
    mirror mcp's own create_mcp_http_client so only the trust root differs.
    """

    kwargs = {"follow_redirects": True, "verify": OBSIDIAN_CERT_PATH}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


# one entry per server, keyed by the name that shows up in the status bar
MCP_SERVERS: dict = {
    "obsidian": {
        "transport": "http",
        "url": "https://127.0.0.1:27124/mcp/",
        "headers": {
            "Authorization": f"Bearer {OBSIDIAN_API_KEY}"
        },
        "httpx_client_factory": _obsidian_http_client
    }
}

# ditched a bunch of the tools from
# https://coddingtonbear.github.io/obsidian-local-rest-api/#/paths/mcp/post
# i dont have use for them
WHITELISTED_TOOLS: set[str] = {
    "vault_list",
    "vault_read",
    "vault_write",
    "vault_append",
    "vault_patch",
    "vault_delete",
    "vault_move",
    "vault_copy",
    "vault_get_document_map",
    "search_simple",
    "open_file"
}


# the tools that can change what a note says. every one of these is held at the
# approval gate in build_agent -- moving, copying and opening a note are writes
# too, but none of them can rewrite the text of one
WRITE_TOOLS: set[str] = {
    "vault_write",
    "vault_append",
    "vault_patch",
    "vault_delete",
}


def tool_text(content) -> str:
    """The text a tool returned, out of whichever shape it came back in.

    Local tools return a string. MCP tools return a list of content blocks, and
    str() on that gives a python repr -- newlines escaped, so the whole result
    lands on one line and wrapping it is what locks the screen up.
    """

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def vault_reader(tools: list[BaseTool]) -> Callable[..., Awaitable[str]] | None:
    """An await-able read of one note, or None if the server never offered one.

    Handed to the preview builder so it can show what a write would change.
    Returns the note's text, and raises whatever the server raised -- a missing
    file and a missing heading both arrive that way, and the caller decides
    which of those is worth showing as an error.
    """

    read = next((tool for tool in tools if tool.name == "vault_read"), None)
    if read is None:
        return None

    async def read_note(path: str, **target) -> str:
        args = {"path": path, **{k: v for k, v in target.items() if v is not None}}
        return tool_text(await read.ainvoke(args))

    return read_note


def _reason(exc: BaseException, dropped: bool = False) -> str:
    """The shortest honest account of why a server did not answer.

    anyio wraps a transport failure in an ExceptionGroup several layers deep,
    and the group's own str() is "unhandled errors in a TaskGroup". Walk down
    to the first real exception and describe that instead.

    dropped tells apart a server that was never there from one that went away
    mid-session: the first needs a restart to pick up, the second recovers on
    its own, and the two deserve different words on screen.
    """

    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return f"refused the api key ({code})"
        return f"answered {code}"

    # the ssl error arrives as a ConnectError too, so it has to be checked first
    if "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return "certificate not trusted"
    if isinstance(exc, httpx.ConnectError):
        return "lost connection" if dropped else "not running"

    return f"{type(exc).__name__}: {str(exc)[:60]}"


class MCPStatus:
    """Which servers are reachable: None if they are, or why they are not.

    Shared and mutated, not snapshotted. The loader fills it in, the
    interceptor below keeps it current, and the status bar reads it at render
    time -- so nothing has to notify anything.
    """

    def __init__(self, servers: dict[str, str | None] | None = None) -> None:
        self.servers: dict[str, str | None] = dict(servers or {})

    def mark_up(self, server: str) -> None:
        self.servers[server] = None

    def mark_down(self, server: str, reason: str) -> None:
        self.servers[server] = reason

    @property
    def live(self) -> list[str]:
        return [name for name, failed in self.servers.items() if failed is None]


def _watch_servers(status: MCPStatus):
    """Interceptor that keeps status honest from what calls actually do.

    Only transport failures raise in here. An isError result from the server is
    converted into one a layer further out, after this chain has returned -- so
    in here a return means the server answered, whatever it said, and a raise
    means it could not be reached at all.
    """

    async def watch(request, handler):
        try:
            result = await handler(request)
        except Exception as exc:
            status.mark_down(request.server_name, _reason(exc, dropped=True))
            # the middleware still needs this to tell the model what happened
            raise
        status.mark_up(request.server_name)
        return result

    return watch


async def get_mcp_tools(status: MCPStatus) -> list[BaseTool]:
    """Get every server's tools and update the given MCPStatus according to uptime.
    """

    # tool interceptor will catch the tool returns to see if the mcp server
    # connection is still alive
    client = MultiServerMCPClient(
        MCP_SERVERS, tool_interceptors=[_watch_servers(status)]
    )
    tools: list[BaseTool] = []

    for name in MCP_SERVERS:
        try:
            loaded = await client.get_tools(server_name=name)
            tools.extend(t for t in loaded if t.name in WHITELISTED_TOOLS)
            status.mark_up(name)
        except Exception as exc:
            status.mark_down(name, _reason(exc))

    return tools
