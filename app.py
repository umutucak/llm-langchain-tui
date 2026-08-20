"""Entry point for the TUI: `python app.py`."""

import warnings

from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.sqlite import SqliteSaver

from llmtui.agent import build_agent, build_model
from llmtui.cli import repl
from llmtui.sessions import connect

# announces that its beta, so we shush it
warnings.filterwarnings(
    "ignore",
    message=".*v3 streaming protocol on Pregel is experimental.*"
)


def main() -> None:
    sqlite_connection = connect()

    memory: SqliteSaver = SqliteSaver(sqlite_connection)
    memory.setup()

    model = build_model()
    agent = build_agent(model, memory)

    agent_thread_config = {
        "configurable": {
            "thread_id": str(uuid7())
        }
    }

    repl.run(agent, model, sqlite_connection, agent_thread_config)


if __name__ == "__main__":
    main()
