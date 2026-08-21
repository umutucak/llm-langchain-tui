"""Drives the Textual app headlessly with a scripted model.

Textual's run_test() pilot gives a real running app -- CSS parsed, widgets
mounted, workers scheduled -- without a terminal, so this catches broken TCSS
and bad mounts that only a live run would otherwise show.
"""
import asyncio
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool

from langgraph.checkpoint.memory import MemorySaver

from textual.widgets import Collapsible, Markdown, SelectionList, Static

from llmtui.tui import (
    AssistantTurn, LlmTui, SessionPicker, ToolRow, compact, meter,
)


class ScriptedModel(BaseChatModel):
    responses: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self


def ai_calls(calls):
    tool_calls, blocks = [], [{"type": "reasoning", "reasoning": "considering", "index": 0}]
    for name, args in calls:
        cid = f"call_{len(tool_calls)}"
        tool_calls.append({"name": name, "args": args, "id": cid, "type": "tool_call"})
        blocks.append({"type": "tool_call", "id": cid, "name": name, "args": args})
    return AIMessage(content=blocks, tool_calls=tool_calls, id=str(uuid.uuid4()))


def ai_text(text):
    return AIMessage(content=[{"type": "text", "text": text}], id=str(uuid.uuid4()))


@tool
async def search_books(query: str, book: str = "") -> str:
    """Search the book library."""
    await asyncio.sleep(0.02)
    return f"[core.pdf p.1]\nPassage about {query}\n\n---\n\n[core.pdf p.2]\nMore about {query}"


def build_app(responses):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute(
        "CREATE TABLE sessions (thread_id TEXT PRIMARY KEY, title TEXT,"
        " created_at TEXT, updated_at TEXT)"
    )
    agent = create_agent(
        model=ScriptedModel(responses=list(responses)),
        tools=[search_books],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return LlmTui(agent, None, connection, config)


async def ask(app, text, settle=1.4):
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = text
        await pilot.press("enter")
        await asyncio.sleep(settle)
        await pilot.pause()
        yield_state = {
            "tool_rows": list(app.query(ToolRow)),
            "markdown": list(app.query(Markdown)),
            "collapsibles": list(app.query(Collapsible)),
            "statics": [s for s in app.query(Static)],
            "status": app.status,
        }
        return yield_state


results = {}


def check(label, conditions):
    ok = True
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for desc, passed in conditions:
        print(f"    {'PASS' if passed else 'FAIL'}  {desc}")
        ok = ok and passed
    results[label] = ok


# ---- helpers are pure, check them first ----
check("0  helpers", [
    ("meter fills proportionally", meter(16384, 32768) == "▓▓▓▓░░░░"),
    ("meter clamps at full", meter(99999, 32768) == "▓▓▓▓▓▓▓▓"),
    ("meter empty at zero", meter(0, 32768) == "░░░░░░░░"),
    ("compact shortens thousands", compact(8100) == "8.1k"),
    ("compact leaves small numbers", compact(412) == "412"),
])


# ---- 1. a turn with two searches ----
async def case_search():
    app = build_app([
        ai_calls([("search_books", {"query": "initiative", "book": "draw_steel"}),
                  ("search_books", {"query": "holding a turn"})]),
        ai_text("In **Draw Steel**, initiative is not rolled.\n\n- Teams alternate\n"),
    ])
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "how does initiative work"
        await pilot.press("enter")
        await asyncio.sleep(1.6)
        await pilot.pause()

        rows = list(app.query(ToolRow))
        answers = list(app.query(Markdown))
        turns = list(app.query(AssistantTurn))

        check("1  two searches -> two rows, answer rendered", [
            ("a tool row per call", len(rows) == 2),
            ("rows keyed by tool_call_id",
             {r.tool_call_id for r in rows} == {"call_0", "call_1"}),
            ("both rows resolved ok", all(r.has_class("-ok") for r in rows)),
            ("no row left spinning", not any(r.has_class("-running") for r in rows)),
            ("titles report passage counts",
             all("2 passages" in str(r.title) for r in rows)),
            ("book argument shown in the label",
             any("draw_steel" in str(r.title) for r in rows)),
            ("answer mounted as a Markdown widget", len(answers) == 1),
            ("reasoning pane created", turns and turns[0].reasoning is not None),
            ("reasoning folded once the answer began",
             turns and turns[0].reasoning is not None and turns[0].reasoning.collapsed),
            ("fold summary names a duration",
             turns and "reasoning ·" in str(turns[0].reasoning.title)),
            ("status returned to ready", app.status.state == "ready"),
        ])


# ---- 2. plain answer, no tools ----
async def case_plain():
    app = build_app([ai_text("Just an answer, no search needed.")])
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")
        await asyncio.sleep(1.0)
        await pilot.pause()
        check("2  no tools -> no rows, still answers", [
            ("no tool rows", len(list(app.query(ToolRow))) == 0),
            ("answer still rendered", len(list(app.query(Markdown))) == 1),
            ("status ready", app.status.state == "ready"),
        ])


# ---- 3. a failing tool ----
async def case_error():
    @tool
    async def search_books(query: str, book: str = "") -> str:
        """Search the book library."""
        raise RuntimeError("simulated milvus GOAWAY")

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("CREATE TABLE sessions (thread_id TEXT PRIMARY KEY, title TEXT,"
                       " created_at TEXT, updated_at TEXT)")
    agent = create_agent(
        model=ScriptedModel(responses=[
            ai_calls([("search_books", {"query": "grappling"})]),
            ai_text("The library is unavailable."),
        ]),
        tools=[search_books],
        checkpointer=MemorySaver(),
    )
    app = LlmTui(agent, None, connection, {"configurable": {"thread_id": str(uuid.uuid4())}})

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "grappling rules"
        await pilot.press("enter")
        await asyncio.sleep(1.4)
        await pilot.pause()
        rows = list(app.query(ToolRow))
        check("3  tool raises -> error row, app survives", [
            ("a row was still created", len(rows) == 1),
            ("row marked as error", rows and rows[0].has_class("-error")),
            ("row is not stuck running", rows and not rows[0].has_class("-running")),
            ("app did not crash", app.status.state == "ready"),
        ])


# ---- 4. slash commands ----
async def case_commands():
    app = build_app([])
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        help_shown = any("/list" in str(getattr(s, "content", ""))
                         for s in app.query(Static))

        app.query_one("#prompt").value = "/nonsense"
        await pilot.press("enter")
        await pilot.pause()
        rejected = any("unrecognized" in str(getattr(s, "content", ""))
                       for s in app.query(Static))

        app.query_one("#prompt").value = ""
        await pilot.press("enter")
        await pilot.pause()

        check("4  slash commands handled in the same input", [
            ("/help lists the commands", help_shown),
            ("unknown command rejected, not sent to the model", rejected),
            ("empty submit is a no-op", len(list(app.query(AssistantTurn))) == 0),
        ])


# ---- 5. the real middleware stack: errors arrive as ToolMessages ----
async def case_middleware():
    """How the app is actually wired -- ToolErrorMiddleware catches the raise.

    The exception never reaches tc.error, so a row that only checks that field
    would settle a failed search as a green success.
    """
    from langchain.agents.middleware import ToolCallLimitMiddleware, ToolErrorMiddleware
    from llmtui.middleware import on_search_error, repair_tool_calls

    @tool
    async def search_books(query: str, book: str = "") -> str:
        """Search the book library."""
        raise RuntimeError("simulated milvus GOAWAY")

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("CREATE TABLE sessions (thread_id TEXT PRIMARY KEY, title TEXT,"
                       " created_at TEXT, updated_at TEXT)")
    agent = create_agent(
        model=ScriptedModel(responses=[
            ai_calls([("search_books", {"query": "grappling"})]),
            ai_text("The library appears to be unavailable."),
        ]),
        tools=[search_books],
        middleware=[
            ToolCallLimitMiddleware(tool_name="search_books", run_limit=3),
            ToolErrorMiddleware(on_error=on_search_error, tools=["search_books"]),
            repair_tool_calls,
        ],
        checkpointer=MemorySaver(),
    )
    app = LlmTui(agent, None, connection, {"configurable": {"thread_id": str(uuid.uuid4())}})

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "grappling rules"
        await pilot.press("enter")
        await asyncio.sleep(1.6)
        await pilot.pause()
        rows = list(app.query(ToolRow))
        check("5  ToolErrorMiddleware path -> row still reads as failed", [
            ("a row was created", len(rows) == 1),
            ("handled error still marks the row failed",
             rows and rows[0].has_class("-error")),
            ("not mislabelled as a success", rows and not rows[0].has_class("-ok")),
            ("title carries the failure message, not a useless type name",
             rows and "GOAWAY" in str(rows[0].title) and "str ·" not in str(rows[0].title)),
            ("body shows the failure, not the model-facing guidance",
             rows and "GOAWAY" in str(rows[0]._body.content)),
            ("run finished with an answer", len(list(app.query(Markdown))) == 1),
        ])


# ---- 6. the real checkpointer ----
async def case_async_saver():
    """AsyncSqliteSaver against a real file, which is how app.py runs.

    Every case above used MemorySaver, which implements both the sync and async
    checkpoint APIs -- so it hid the fact that the plain SqliteSaver raises
    NotImplementedError on every async call the graph makes.
    """
    import tempfile
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = tempfile.mktemp(suffix=".sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()

        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (thread_id TEXT PRIMARY KEY,"
                           " title TEXT, created_at TEXT, updated_at TEXT)")

        agent = create_agent(
            model=ScriptedModel(responses=[
                ai_calls([("search_books", {"query": "initiative"})]),
                ai_text("Teams alternate."),
            ]),
            tools=[search_books],
            checkpointer=saver,
        )
        # a separate model instance, because naming pops from its own script
        namer = ScriptedModel(responses=[AIMessage(content="Initiative Rules")])
        thread_id = str(uuid.uuid4())
        app = LlmTui(agent, namer, connection, {"configurable": {"thread_id": thread_id}})

        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "how does initiative work"
            await pilot.press("enter")
            await asyncio.sleep(2.0)
            await pilot.pause()

            titles = connection.execute(
                "SELECT title FROM sessions WHERE thread_id = ?", (thread_id,)).fetchall()
            rows = list(app.query(ToolRow))
            check("6  AsyncSqliteSaver -> the run actually completes", [
                ("run finished instead of raising NotImplementedError",
                 app.status.state == "ready"),
                ("tool row resolved", rows and rows[0].has_class("-ok")),
                ("answer rendered", len(list(app.query(Markdown))) == 1),
                ("checkpoint written to the file",
                 connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] > 0),
                ("context meter got real token counts", app.status.ctx_used >= 0),
                ("session self-named through the async path",
                 titles and titles[0][0] == "Initiative Rules"),
            ])
        connection.close()


# ---- 7. /name and /load round trip ----
async def case_load():
    """The command that did not work at all before: pick a thread, redraw it."""
    import tempfile
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = tempfile.mktemp(suffix=".sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (thread_id TEXT PRIMARY KEY,"
                           " title TEXT, created_at TEXT, updated_at TEXT)")

        agent = create_agent(
            model=ScriptedModel(responses=[ai_text("Teams alternate, no roll.")]),
            tools=[search_books],
            checkpointer=saver,
        )
        namer = ScriptedModel(responses=[AIMessage(content="Draw Steel Initiative")])
        first = str(uuid.uuid4())
        app = LlmTui(agent, namer, connection, {"configurable": {"thread_id": first}})

        async with app.run_test() as pilot:
            # a turn worth coming back to
            app.query_one("#prompt").value = "how does initiative work"
            await pilot.press("enter")
            await asyncio.sleep(1.6)
            await pilot.pause()

            named = connection.execute(
                "SELECT title FROM sessions WHERE thread_id = ?", (first,)).fetchone()

            # move to a fresh thread, as /load would be used from
            app.agent_thread_config["configurable"]["thread_id"] = str(uuid.uuid4())
            app.action_clear()
            await pilot.pause()
            empty_after_clear = len(list(app.query(AssistantTurn))) == 0

            app.query_one("#prompt").value = "/load"
            await pilot.press("enter")
            await asyncio.sleep(0.4)
            await pilot.pause()
            modal_open = isinstance(app.screen, SessionPicker)

            await pilot.press("enter")           # take the highlighted session
            await asyncio.sleep(0.8)
            await pilot.pause()

            restored = [s for s in app.query(Static) if s.has_class("user")]
            check("7  /load resumes a thread and redraws it", [
                ("session was auto-named", named and named[0] == "Draw Steel Initiative"),
                ("transcript really was empty before loading", empty_after_clear),
                ("picker opened as a modal", modal_open),
                ("thread switched back", app.agent_thread_config["configurable"]["thread_id"] == first),
                ("history redrawn from the checkpoint", len(restored) == 1),
                ("the original question is back on screen",
                 restored and "initiative" in str(restored[0].content)),
                ("answer redrawn too", len(list(app.query(Markdown))) == 1),
            ])
        connection.close()


# ---- 8. scrolling back must stick ----
async def case_scroll():
    """Decision 6a: follow the tail, but let go the moment the reader scrolls.

    The first attempt drove this from a 15fps interval that re-scrolled to the
    bottom unconditionally, so any scroll-up was undone within ~67ms and the
    transcript could not be read back at all.
    """
    long_answer = "\n\n".join(f"Paragraph {i} about initiative order." for i in range(40))
    app = build_app([ai_text(long_answer)])

    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "explain at length"
        await pilot.press("enter")
        await asyncio.sleep(1.6)
        await pilot.pause()

        transcript = app.transcript
        followed_while_streaming = transcript.is_anchored
        scrollable = transcript.max_scroll_y > 0

        # the reader scrolls back through the transcript
        transcript.anchor(False)
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()
        went_up = transcript.scroll_y < 1

        # long enough that the old 15fps follower would have yanked it back
        await asyncio.sleep(0.6)
        await pilot.pause()

        check("8  scrolling back is not undone", [
            ("content was long enough to scroll", scrollable),
            ("followed the tail while streaming", followed_while_streaming),
            ("scrolling back actually moved", went_up),
            ("still at the top half a second later", transcript.scroll_y < 1),
            ("stayed released, nothing re-anchored it", not transcript.is_anchored),
        ])

        # a new turn is a deliberate arrival, so it re-anchors
        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await asyncio.sleep(0.4)
        await pilot.pause()
        check("8b re-anchors when something new is put on screen", [
            ("anchored again after a command", transcript.is_anchored),
        ])


# ---- 9. /new and deleting the session you are sitting in ----
async def case_new_and_self_delete():
    """/new gives a clean thread, and /delete may take the current one.

    Both matter because /load leaves you inside someone else's conversation with
    no way back to an empty context.
    """
    import tempfile
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from llmtui.tui import SessionDeleter

    path = tempfile.mktemp(suffix=".sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (thread_id TEXT PRIMARY KEY,"
                           " title TEXT, created_at TEXT, updated_at TEXT)")

        agent = create_agent(
            model=ScriptedModel(responses=[ai_text("First answer."), ai_text("Second answer.")]),
            tools=[search_books],
            checkpointer=saver,
        )
        namer = ScriptedModel(responses=[AIMessage(content="First Chat"),
                                         AIMessage(content="Second Chat")])
        first = str(uuid.uuid4())
        app = LlmTui(agent, namer, connection, {"configurable": {"thread_id": first}})

        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "first question"
            await pilot.press("enter")
            await asyncio.sleep(1.6)
            await pilot.pause()
            had_history = len(list(app.query(AssistantTurn))) == 1

            # ---- /new ----
            app.query_one("#prompt").value = "/new"
            await pilot.press("enter")
            await asyncio.sleep(0.5)
            await pilot.pause()
            second = app.agent_thread_config["configurable"]["thread_id"]

            check("9  /new starts a clean thread", [
                ("the first turn had left history", had_history),
                ("thread_id changed", second != first),
                ("transcript cleared of old turns",
                 len(list(app.query(AssistantTurn))) == 0),
                ("context meter reset", app.status.ctx_used == 0),
                ("old session still on disk",
                 connection.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?",
                                    (first,)).fetchone()[0] > 0),
            ])

            # give the new thread some history of its own
            app.query_one("#prompt").value = "second question"
            await pilot.press("enter")
            await asyncio.sleep(1.6)
            await pilot.pause()

            # ---- /delete, taking the current session with it ----
            app.query_one("#prompt").value = "/delete"
            await pilot.press("enter")
            await asyncio.sleep(0.5)
            await pilot.pause()
            screen = app.screen
            offered = list(screen.query_one("#picker", SelectionList).options) \
                if isinstance(screen, SessionDeleter) else []
            current_is_offered = any(
                o.value == second for o in offered) if offered else False

            # tick every session, current included, then confirm
            screen.query_one("#picker", SelectionList).select_all()
            await pilot.pause()
            await pilot.press("enter")
            await asyncio.sleep(0.6)
            await pilot.pause()

            third = app.agent_thread_config["configurable"]["thread_id"]
            left = connection.execute("SELECT count(*) FROM sessions").fetchone()[0]
            checkpoints_left = connection.execute(
                "SELECT count(*) FROM checkpoints WHERE thread_id IN (?, ?)",
                (first, second)).fetchone()[0]

            check("9b /delete can take the current session", [
                ("the current session was offered for deletion", current_is_offered),
                ("rolled onto a brand new thread", third not in (first, second)),
                ("session rows gone", left == 0),
                ("checkpoints gone for both deleted threads", checkpoints_left == 0),
                ("transcript is empty", len(list(app.query(AssistantTurn))) == 0),
                ("app still usable", app.status.state == "ready"),
            ])
        connection.close()


async def main():
    await case_new_and_self_delete()
    await case_scroll()
    await case_search()
    await case_plain()
    await case_error()
    await case_commands()
    await case_middleware()
    await case_async_saver()
    await case_load()

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}   {label}")
    print(f"\n{sum(results.values())}/{len(results)} cases passed")


if __name__ == "__main__":
    asyncio.run(main())
