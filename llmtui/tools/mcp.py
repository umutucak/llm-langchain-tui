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


async def get_mcp_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "obsidian": {
                "transport": "http",
                "url": "https://127.0.0.1:27124/mcp/",
                "headers": {
                    "Authorization": f"Bearer {OBSIDIAN_API_KEY}"
                },
                "httpx_client_factory": _obsidian_http_client
            }
        }
    )
    return await client.get_tools()
