"""The agent itself: model, tools, middleware stack, checkpointer."""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolErrorMiddleware

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
from llmtui.middleware import on_search_error, repair_tool_calls
from llmtui.tools import TOOLS

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


def build_agent(model: ChatOllama, checkpointer) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=TOOLS,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_books",
                run_limit=MAX_TOOL_CALLS
            ),
            ToolErrorMiddleware(
                on_error=on_search_error,
                tools=["search_books"]
            ),
            # after_model custom middleware to fix malformed tool calls
            # https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks
            # i am so proud of this one
            repair_tool_calls
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
