"""Per-tool handlers for ToolErrorMiddleware, one function per tool.

Each is scoped to its own tool by the `tools=[...]` argument where the
middleware is constructed, so a handler never has to work out which tool it
is speaking for.
"""

from langchain.agents.middleware import ToolCallRequest


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
