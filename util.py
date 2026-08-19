import os

from dotenv import load_dotenv

from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages.base import BaseMessage

from langgraph.stream.run_stream import GraphRunStream


load_dotenv()
MAX_TOOL_CALLS: int = 3
DISPLAY_THINKING: bool = bool(int(os.getenv("DISPLAY_THINKING")))


def stream_output(stream:GraphRunStream):
    # guide i read to write this abomination:
    # https://docs.langchain.com/oss/python/deepagents/event-streaming#stream-messages
    
    # message type = ChatModelStream
    for message in stream.messages:
        thinking_gate = False
        for event in message:
            # check if event is streamed data, and check if streamed delta has data
            delta = event.get("delta")
            if event.get("event") != "content-block-delta" or not isinstance(delta, dict):
                continue
            # reasoning stream
            if delta.get("type") == "reasoning-delta" and DISPLAY_THINKING:
                if not thinking_gate:
                    print("<THINKING>", end="", flush=True)
                    thinking_gate = True
                print(delta.get("reasoning", ""), end="", flush=True)
            # output stream
            elif delta.get("type") == "text-delta":
                if thinking_gate:
                    print("</THINKING>\n", flush=True)
                    thinking_gate = False
                print(delta.get("text", ""), end="", flush=True)
        # a tool-calling turn ends on the tool call, with no text delta to
        # close the block, so it has to be closed out here instead
        if thinking_gate:
            print("</THINKING>\n", flush=True)
    print("\n")
    # check if tool were used, if so, print what tool and its result
    print_tool_use(stream.output)



def print_tool_use(result) -> None:
    # naive tool use search for printing
    _start = len(result["messages"])-2
    _end = max(0, _start-MAX_TOOL_CALLS+1)
    for r in result["messages"][_start:_end:-1]:
        if isinstance(r, ToolMessage):
            print(f"[TOOL] {r.name}: {r.content}")


def is_session_named(sqlite_connection, thread_id: str) -> bool:
    row = sqlite_connection.execute(
        "SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return row is not None


def delete_session(sqlite_connection, thread_id: str) -> None:
    sqlite_connection.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    sqlite_connection.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    sqlite_connection.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    sqlite_connection.commit()


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


def set_session_title(sqlite_connection, thread_id:str, title:str) -> None:
    sqlite_connection.execute("""
        INSERT INTO sessions (thread_id, title, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at    
        """,
        (thread_id, title)
    )
    sqlite_connection.commit()

    return


def list_sessions(sqlite_connection):
    return sqlite_connection.execute(
        "SELECT thread_id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
