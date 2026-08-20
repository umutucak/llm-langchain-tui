"""Letting the model title its own conversation once the first turn is done."""

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage

from llmtui.config import SELF_NAME_PROMPT_PATH
from llmtui.sessions import is_session_named, set_session_title


def strip_tool_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Used to extract only the input and output messages. Strips reasoning and tool blocks.

    Args:
    full context: list[BaseMessage]

    Return:
    only output and input messages: list[BaseMessage]
    """

    cleaned = []
    for m in messages:
        # ignore tool outputs
        if isinstance(m, ToolMessage):
            continue
        # rid AI output from its reasoning and tool call blocks
        if isinstance(m, AIMessage):
            # holy list comp that extracts only the text output from the AIMessage
            text: str = "\n".join([block.get("text") for block in m.content if block.get("type") == "text"])
            if not text.strip():
                continue
            cleaned.append(AIMessage(content=text))
        # user input can be processed directly
        else:
            cleaned.append(m.content)

    return cleaned


def name_session_if_unnamed(agent, model, sqlite_connection, agent_thread_config) -> None:
    """Ask the model for a title, but only the first time a thread finishes a turn."""

    # self-name the session if not already named
    if not is_session_named(sqlite_connection, agent_thread_config["configurable"]["thread_id"]):
        print("Watch for slight freeze, naming conversation...")
        self_naming_prompt = strip_tool_context(agent.get_state(agent_thread_config).values["messages"])
        with open(SELF_NAME_PROMPT_PATH, 'r') as f:
            p = f.read()
        self_naming_prompt.append(HumanMessage(p))
        set_session_title(
            sqlite_connection,
            agent_thread_config["configurable"]["thread_id"],
            model.invoke(self_naming_prompt).content
        )
