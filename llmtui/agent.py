"""The agent itself: model, tools, middleware stack, checkpointer."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ToolCallLimitMiddleware, ToolErrorMiddleware,
)

from langchain_core.tools.base import BaseTool

from langchain_ollama.chat_models import ChatOllama

from langgraph.graph.state import CompiledStateGraph

from llmtui.config import (
    CONTEXT_SIZE,
    IS_REASONING,
    MAX_TOOL_CALLS,
    MODEL,
    REPETITION_PENALTY,
    SYSTEM_PROMPT_PATH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from llmtui.middleware import repair_tool_calls, route_tool_error
from llmtui.tools import TOOLS
from llmtui.tools.mcp import WRITE_TOOLS

with open(SYSTEM_PROMPT_PATH, 'r') as f:
    SYSTEM_PROMPT: str = f.read()


def build_model() -> ChatOllama:
    # values from .env, which are the recommended values from
    # https://huggingface.co/Qwen/Qwen3.8-27B
    # ctx extended according to my 3090 with q8 kv caching
    return ChatOllama(
        model=MODEL,
        reasoning=IS_REASONING,
        num_ctx=CONTEXT_SIZE,
        temperature=TEMPERATURE,
        validate_model_on_init=True,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPETITION_PENALTY
    )


def build_agent(
    model: ChatOllama, checkpointer, mcp_tools: list[BaseTool]
) -> CompiledStateGraph:
    """Assemble the graph out of parts. The MCP tools are fetched by the caller.
    """

    mcp_names = {tool.name for tool in mcp_tools}

    # only the write tools that actually loaded, so an obsidian that is down
    # cannot leave the gate holding names no tool answers to
    gated = WRITE_TOOLS & mcp_names

    # goes in front, which by the rule below means it runs last, and that is the
    # point: the writes it holds up are the repaired ones, and the ones the call
    # limit already let through. asking about a call another middleware is about
    # to rewrite or drop would be asking about nothing
    approval = []
    if gated:
        interrupt_on = {}
        for name in gated:
            interrupt_on[name] = {"allowed_decisions": ["approve", "reject"]}
        approval.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model=model,
        tools=TOOLS+mcp_tools,
        # after_model hooks run in reverse list order: repair_tool_calls first,
        # then the call limit, then the approval gate last -- so the gate only
        # ever sees repaired calls the limit already let through. the tool-error
        # wrapper runs around the tool node, after all of them
        middleware=approval + [
            ToolCallLimitMiddleware(
                tool_name="search_books",
                run_limit=MAX_TOOL_CALLS
            ),
            # one middleware for all types tool, so we need a on_error trigger
            # that can facilitate for all types of errors
            ToolErrorMiddleware(
                on_error=route_tool_error(mcp_names),
                tools=["search_books", *mcp_names]
            ),
            # after_model custom middleware to fix malformed tool calls
            # https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks
            # i am so proud of this one
            repair_tool_calls
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
