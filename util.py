import os

from dotenv import load_dotenv

from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages.base import BaseMessage


load_dotenv()
MAX_TOOL_CALLS: int = 3
DISPLAY_THINKING: bool = bool(int(os.getenv("DISPLAY_THINKING")))


def stream_output(stream):
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
    """Strip full context from tool calls.

    Used for cleaning up the first exchange for feeding it
    back into the model to self-name the session.
    """

    cleaned = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage) and m.tool_calls:
            cleaned.append(AIMessage(content=m.content))
        else:
            cleaned.append(m)
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
