"""The sessions table: the human-readable names sitting beside LangGraph's threads.

LangGraph's checkpointer owns `checkpoints` and `writes` and knows nothing about
titles, so a session's name lives in a table of our own keyed by the same
thread_id. Deleting a session has to clear all three.
"""

import sqlite3

from llmtui.config import SQLITE_DB_PATH


def connect() -> sqlite3.Connection:
    """Open the checkpoint database and make sure our own table is there."""

    sqlite_connection = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    sqlite_connection.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    sqlite_connection.commit()

    return sqlite_connection


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
