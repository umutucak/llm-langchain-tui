"""For all the MCP client interfaces."""
import httpx

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
            tools.extend(await client.get_tools(server_name=name))
            status.mark_up(name)
        except Exception as exc:
            status.mark_down(name, _reason(exc))

    return tools
