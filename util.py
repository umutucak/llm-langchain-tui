from dotenv import load_dotenv

from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages.base import BaseMessage


load_dotenv()
MAX_TOOL_CALLS: int = 3


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
