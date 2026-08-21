"""Entry point for the TUI: `python app.py`."""

import asyncio
import warnings

from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llmtui.agent import build_agent, build_model
from llmtui.config import SQLITE_DB_PATH
from llmtui.sessions import connect
from llmtui.tui import LlmTui

# announces that its beta, so we shush it
warnings.filterwarnings(
    "ignore",
    message=".*v3 streaming protocol on Pregel is experimental.*"
)


async def main_async() -> None:
    sqlite_connection = connect()

    model = build_model()

    # the graph is driven with astream_events, and the plain SqliteSaver raises
    # NotImplementedError on every async checkpoint call, so the saver has to be
    # the aiosqlite one. same file and the same schema, so inspect_context.py
    # still reads these threads with the sync saver from its own process
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as memory:
        await memory.setup()

        agent = build_agent(model, memory)

        agent_thread_config = {
            "configurable": {
                "thread_id": str(uuid7())
            }
        }

        await LlmTui(agent, model, sqlite_connection, agent_thread_config).run_async()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
