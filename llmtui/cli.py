"""Entry point for the packaged `gm` binary (`gm` bare opens the TUI)."""

import argparse
import asyncio
import warnings

from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llmtui.agent import build_agent, build_model
from llmtui.config import LIBRARY_DIR, SQLITE_DB_PATH
from llmtui.models_config import flatten, load_models
from llmtui.sessions import connect
from llmtui.tools.mcp import MCPStatus, get_mcp_tools, vault_reader
from llmtui.tui import LlmTui

# announces that its beta, so we shush it
warnings.filterwarnings(
    "ignore", message=".*v3 streaming protocol on Pregel is experimental.*"
)


async def main_async() -> None:
    # first-boot workspace: prompts once, before anything else is built
    from llmtui.workspace import ensure_workspace

    if await ensure_workspace() is None:
        return

    sqlite_connection = connect()

    # first entry of models.json is the default: file order is preference
    entry = flatten(load_models())[0]
    model = build_model(f"{entry.provider}/{entry.id}")

    # the graph is driven with astream_events, and the plain SqliteSaver raises
    # NotImplementedError on every async checkpoint call, so the saver has to be
    # the aiosqlite one. same file and the same schema, so inspect_context.py
    # still reads these threads with the sync saver from its own process
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as memory:
        await memory.setup()

        # mcp_tools goes into build_agent to give the graph the tools
        # mcp_status goes into the tui to report what (if at all) mcp servers are connected to
        # one live object, not a snapshot: get_mcp_tools fills it in, and an
        # interceptor keeps it current as calls succeed and fail during the run
        mcp_status = MCPStatus()
        mcp_tools = await get_mcp_tools(mcp_status)

        # the tui shows a pending write as a diff, which means reading the note
        # first. None when the vault never answered, and the approval still runs
        note_reader = vault_reader(mcp_tools)

        agent = build_agent(model, memory, mcp_tools)

        agent_thread_config = {"configurable": {"thread_id": str(uuid7())}}

        await LlmTui(
            agent,
            model,
            memory,
            mcp_tools,
            entry,
            sqlite_connection,
            agent_thread_config,
            mcp_status,
            note_reader,
            library_dir=LIBRARY_DIR,
        ).run_async()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gm", description="the gm assistant")
    parser.add_argument(
        "--version", action="store_true", help="print the installed version and exit"
    )
    args = parser.parse_args()
    if args.version:
        from importlib.metadata import version

        print(f"gm {version('llmtui')}")
        return
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
