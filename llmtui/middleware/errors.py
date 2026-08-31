"""Per-tool handlers for ToolErrorMiddleware, one function per kind of failure.

A handler never works out which tool it is speaking for. route_tool_error
decides that before calling it.
"""

from collections.abc import Callable

from langchain.agents.middleware import ToolCallRequest


def route_tool_error(
    mcp_tool_names: set[str]
) -> Callable[[Exception, ToolCallRequest], str | None]:
    """Build the on_error that picks a handler per tool.

    The agent can have only one ToolErrorMiddleware, so we need a on_error
    trigger that can facilitate multiple errors.

    Returns a closure because on_error is always called with just
    (exc, request), and the MCP names are only known once the servers answer.
    """

    def on_tool_error(exc: Exception, request: ToolCallRequest) -> str | None:
        if request.tool_call["name"] in mcp_tool_names:
            return on_mcp_error(exc, request)
        return on_search_error(exc, request)

    return on_tool_error


def on_mcp_error(exc: Exception, request: ToolCallRequest) -> str | None:
    """Hand a failed MCP call back to the model instead of killing the run.

    Scoped to whichever MCP tools loaded at startup, so this speaks for all of
    them and names none of them -- the tool name in the message is the only
    identifying detail, and it comes from the call itself.

    MCP tools open a fresh session per call, so a server only has to be up at
    the moment it is used -- which also means it can go away between two calls
    in one turn. langchain_mcp_adapters converts the errors a server reports
    about itself and explicitly does not convert transport failures, which is
    precisely the case where the server has since been shut down. Those would
    otherwise raise and take the whole run down.
    """

    return (
        f"`{request.tool_call['name']}` could not reach the server providing "
        f"it: {type(exc).__name__}: {str(exc)[:200]}\n"
        f"The tool is unavailable for the rest of this turn, so do not retry "
        f"it. Say plainly that it could not be reached, and fall back to your "
        f"other tools only if they can actually answer the question."
    )


def on_search_error(exc: Exception, request: ToolCallRequest) -> str | None:
    """Hand a failed search back to the model instead of killing the run.

    Scoped to search_books by the middleware, so anything arriving here is a
    search failure and there is no exception type to discriminate on. The
    class name goes into the message so a genuine bug is still visible in the
    [TOOL] line rather than being silently swallowed.
    """

    return (
        f"`{request.tool_call['name']}` failed to run: "
        f"{type(exc).__name__}: {str(exc)[:200]}\n"
        f"This is a failure of the search itself, NOT a result. It does not "
        f"mean the library lacks this topic, so do not treat it as 'nothing "
        f"found' and do not answer from general knowledge. Try the search once "
        f"more; if it fails again, tell the user the document library is "
        f"currently unavailable."
    )
