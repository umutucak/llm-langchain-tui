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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool

from langgraph.checkpoint.memory import MemorySaver

from textual.widgets import Collapsible, Markdown, SelectionList, Static

from llmtui.middleware import route_tool_error
from llmtui.tools.mcp import MCPStatus
from llmtui.tui import (
    AssistantTurn, LlmTui, PromptArea, SessionPicker, StatusBar, ToolRow, compact, meter,
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
    return LlmTui(agent, None, connection, config, MCPStatus())


async def ask(app, text, settle=1.4):
    async with app.run_test() as pilot:
        app.query_one("#prompt").text = text
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
        ai_calls([("search_books", {"query": "initiative", "book": "core_rulebook"}),
                  ("search_books", {"query": "holding a turn"})]),
        ai_text("In the **core rulebook**, initiative is not rolled.\n\n- Teams alternate\n"),
    ])
    async with app.run_test() as pilot:
        app.query_one("#prompt").text = "how does initiative work"
        await pilot.press("enter")
        await asyncio.sleep(1.6)
        await pilot.pause()

        rows = list(app.query(ToolRow))
        answers = list(app.query(Markdown))
        turns = list(app.query(AssistantTurn))
        panes = list(turns[0].query(".reasoning")) if turns else []

        check("1  two searches -> two rows, answer rendered", [
            ("a tool row per call", len(rows) == 2),
            ("rows keyed by tool_call_id",
             {r.tool_call_id for r in rows} == {"call_0", "call_1"}),
            ("both rows resolved ok", all(r.has_class("-ok") for r in rows)),
            ("no row left spinning", not any(r.has_class("-running") for r in rows)),
            ("titles report passage counts",
             all("2 passages" in str(r.title) for r in rows)),
            ("book argument shown in the label",
             any("core_rulebook" in str(r.title) for r in rows)),
            ("answer mounted as a Markdown widget", len(answers) == 1),
            # query the DOM, not the attribute: .reasoning holds the *current*
            # round's pane, and this run's second round emits no reasoning at all
            ("reasoning pane created", len(panes) == 1),
            ("reasoning folded once the answer began", panes and panes[0].collapsed),
            ("fold summary names a duration",
             panes and "reasoning ·" in str(panes[0].title)),
            ("status returned to ready", app.status.state == "ready"),
        ])


# ---- 2. plain answer, no tools ----
async def case_plain():
    app = build_app([ai_text("Just an answer, no search needed.")])
    async with app.run_test() as pilot:
        app.query_one("#prompt").text = "hello"
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
    app = LlmTui(agent, None, connection,
                  {"configurable": {"thread_id": str(uuid.uuid4())}}, MCPStatus())

    async with app.run_test() as pilot:
        app.query_one("#prompt").text = "grappling rules"
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
        app.query_one("#prompt").text = "/help"
        await pilot.press("enter")
        await pilot.pause()
        help_shown = any("/list" in str(getattr(s, "content", ""))
                         for s in app.query(Static))

        app.query_one("#prompt").text = "/nonsense"
        await pilot.press("enter")
        await pilot.pause()
        rejected = any("unrecognized" in str(getattr(s, "content", ""))
                       for s in app.query(Static))

        app.query_one("#prompt").text = ""
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
    from llmtui.middleware import repair_tool_calls, route_tool_error

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
            # no MCP tools here, so the router falls through to the search
            # handler for everything -- the same stack build_agent assembles
            ToolErrorMiddleware(on_error=route_tool_error(set()),
                                tools=["search_books"]),
            repair_tool_calls,
        ],
        checkpointer=MemorySaver(),
    )
    app = LlmTui(agent, None, connection,
                  {"configurable": {"thread_id": str(uuid.uuid4())}}, MCPStatus())

    async with app.run_test() as pilot:
        app.query_one("#prompt").text = "grappling rules"
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
        app = LlmTui(agent, namer, connection,
                      {"configurable": {"thread_id": thread_id}}, MCPStatus())

        async with app.run_test() as pilot:
            app.query_one("#prompt").text = "how does initiative work"
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
        app = LlmTui(agent, namer, connection,
                      {"configurable": {"thread_id": first}}, MCPStatus())

        async with app.run_test() as pilot:
            # a turn worth coming back to
            app.query_one("#prompt").text = "how does initiative work"
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

            app.query_one("#prompt").text = "/load"
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
        app.query_one("#prompt").text = "explain at length"
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
        app.query_one("#prompt").text = "/help"
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
        app = LlmTui(agent, namer, connection,
                      {"configurable": {"thread_id": first}}, MCPStatus())

        async with app.run_test() as pilot:
            app.query_one("#prompt").text = "first question"
            await pilot.press("enter")
            await asyncio.sleep(1.6)
            await pilot.pause()
            had_history = len(list(app.query(AssistantTurn))) == 1

            # ---- /new ----
            app.query_one("#prompt").text = "/new"
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
            app.query_one("#prompt").text = "second question"
            await pilot.press("enter")
            await asyncio.sleep(1.6)
            await pilot.pause()

            # ---- /delete, taking the current session with it ----
            app.query_one("#prompt").text = "/delete"
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


def ai_round(reasoning, text=None, calls=()):
    """An AIMessage shaped the way qwen actually emits them.

    Taken from a real checkpoint: reasoning, then prose, then a tool call, all
    in one message — which is what broke the single-pane assumption.
    """
    blocks = [{"type": "reasoning", "reasoning": reasoning, "index": 0}]
    if text:
        blocks.append({"type": "text", "text": text})
    tool_calls = []
    for name, args in calls:
        cid = f"call_{len(tool_calls)}"
        tool_calls.append({"name": name, "args": args, "id": cid, "type": "tool_call"})
        blocks.append({"type": "tool_call", "id": cid, "name": name, "args": args})
    return AIMessage(content=blocks, tool_calls=tool_calls, id=str(uuid.uuid4()))


# ---- 10. every model round gets its own reasoning pane ----
async def case_multi_round_reasoning():
    """Two model passes, each with its own thinking.

    Previously the second round's reasoning was appended to the first round's
    pane: collapsed, mounted above the search it was reacting to, and counted
    under a word total computed before it arrived.
    """
    app = build_app([
        ai_round("FIRSTROUND deciding where to look for population figures",
                 "Let me check the books.",
                 [("search_books", {"query": "town population"})]),
        ai_round("SECONDROUND weighing what the passages actually said",
                 "A frontier town of that size runs 400-900 people."),
    ])

    async with app.run_test() as pilot:
        app.query_one("#prompt").text = "how big is a frontier town"
        await pilot.press("enter")
        await asyncio.sleep(2.0)
        await pilot.pause()

        turn = app.query_one(AssistantTurn)
        bodies = [str(s.content) for s in turn.query(".reasoning-body")]
        panes = list(turn.query(".reasoning"))

        # where each thing sits in the turn, top to bottom
        order, tool_at = [], None
        for i, child in enumerate(turn.children):
            if child.has_class("reasoning"):
                order.append(i)
            if isinstance(child, ToolRow):
                tool_at = i

        check("10  each model round gets its own reasoning pane", [
            ("two rounds were counted", turn.rounds == 2),
            ("two reasoning panes exist", len(bodies) == 2),
            ("first pane holds only the first round's thinking",
             bodies and "FIRSTROUND" in bodies[0] and "SECONDROUND" not in bodies[0]),
            ("second round's thinking is visible in its own pane",
             len(bodies) > 1 and "SECONDROUND" in bodies[1]),
            ("the second pane sits after the search it reacted to",
             len(order) == 2 and tool_at is not None and order[0] < tool_at < order[1]),
            ("both panes folded once their round produced text",
             len(panes) == 2 and all(p.collapsed for p in panes)),
            ("each pane's word count covers only its own round",
             len(panes) == 2 and "words" in str(panes[0].title)
             and str(panes[0].title) != str(panes[1].title)),
            ("an answer block per round", len(list(app.query(Markdown))) == 2),
        ])


# ---- 11. the prompt wraps instead of scrolling sideways ----
async def case_prompt_wraps():
    """A long question must stay readable from its first character.

    Input is single-line: past the width of the box it scrolls horizontally and
    the start of what you typed becomes unreachable. TextArea soft-wraps, so the
    box grows downward and scroll_x never leaves zero.
    """
    long_question = (
        "what is an appropriate population for a frontier town that has a "
        "blacksmith, a chapel, two taverns, a small garrison and a weekly "
        "market, given the surrounding farmland can support maybe forty households"
    )
    app = build_app([ai_text("Four to nine hundred.")])

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        min_height = prompt.size.height

        prompt.text = long_question
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        wrapped_rows = prompt.wrapped_document.height
        real_lines = prompt.document.line_count
        grew_to = prompt.size.height
        scrolled_sideways = prompt.scroll_x

        await pilot.press("enter")
        await asyncio.sleep(1.2)
        await pilot.pause()

        check("11  prompt wraps rather than running off to the left", [
            ("the question is longer than the box is wide", len(long_question) > 80),
            ("it is still one logical line", real_lines == 1),
            ("but it is laid out over several rows", wrapped_rows > 1),
            ("the box grew to fit them", grew_to > min_height),
            ("nothing scrolled horizontally, so the start stays visible",
             scrolled_sideways == 0),
            ("enter still sends rather than inserting a newline",
             len(list(app.query(AssistantTurn))) == 1),
            ("the prompt cleared on send", prompt.text == ""),
            ("the question reached the transcript",
             any("frontier town" in str(s.content)
                 for s in app.query(Static) if s.has_class("user"))),
        ])


# ---- 12. the status bar never sits on top of the prompt ----
async def case_footer_stacking():
    """A tall prompt in a short terminal must not push the status bar off screen.

    Prompt and status used to be docked to the bottom edge independently. When
    the prompt grew, the stack ran past the bottom of the terminal: the status
    bar's row no longer existed, so it was drawn over the prompt's last lines
    and typed text vanished under it. Verified failing on the old arrangement
    before this was written -- at 12 rows it put the status bar on row 14.
    """
    app = build_app([ai_text("ok")])

    async with app.run_test(size=(100, 12)) as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        status = app.query_one(StatusBar)
        prompt.focus()
        prompt.text = "w" * 900          # far taller than a 12 row screen allows
        await asyncio.sleep(0.35)
        await pilot.pause()

        screen_height = app.screen.size.height
        prompt_bottom = prompt.region.y + prompt.region.height

        check("12  status bar stacks under the prompt, never over it", [
            ("the prompt really did grow tall", prompt.region.height > 6),
            ("the status bar is on screen at all", status.region.y < screen_height),
            ("it is on the very last row", status.region.y == screen_height - 1),
            ("the prompt stops before the status bar", prompt_bottom <= status.region.y),
            ("the prompt does not run off the bottom", prompt_bottom <= screen_height),
        ])


async def case_mcp_degrades():
    """A server that is not listening costs its own tools and nothing else.

    Obsidian being closed used to take the whole app down on launch, because
    get_mcp_tools raised straight through build_agent into main_async. The
    reasons matter as much as the survival: they are what the status bar shows,
    and they arrive buried several ExceptionGroups deep with the group's own
    str() reading "unhandled errors in a TaskGroup".
    """
    import httpx
    from llmtui.tools import mcp

    def wrap(exc):
        """Two groups deep, the way anyio actually delivers these."""
        return ExceptionGroup("unhandled errors in a TaskGroup",
                              [ExceptionGroup("inner", [exc])])

    response = httpx.Response(401, request=httpx.Request("POST", "https://127.0.0.1/mcp/"))

    reasons = [
        ("connection refused reads as not running",
         mcp._reason(wrap(httpx.ConnectError("All connection attempts failed")))
         == "not running"),
        ("the same failure mid-session reads as a dropped connection",
         mcp._reason(wrap(httpx.ConnectError("All connection attempts failed")),
                     dropped=True) == "lost connection"),
        ("a self-signed cert is named as such, not as a connection failure",
         mcp._reason(wrap(httpx.ConnectError(
             "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
             "self-signed certificate"))) == "certificate not trusted"),
        ("a 401 is reported as the key being refused",
         mcp._reason(wrap(httpx.HTTPStatusError(
             "401", request=response.request, response=response)))
         == "refused the api key (401)"),
    ]

    # port 1 is not going to be listening, so this is a real refused connection
    # through the real client rather than a stubbed one
    original = mcp.MCP_SERVERS
    mcp.MCP_SERVERS = {
        "ghost": {"transport": "http", "url": "http://127.0.0.1:1/mcp/"},
    }
    status = MCPStatus()
    try:
        tools = await mcp.get_mcp_tools(status)
    finally:
        mcp.MCP_SERVERS = original

    check("13  a dead MCP server degrades instead of raising", reasons + [
        ("no tools came back", tools == []),
        ("every configured server is accounted for", set(status.servers) == {"ghost"}),
        ("and it says why", status.servers["ghost"] == "not running"),
        ("none of them count as live", status.live == []),
    ])


async def case_mcp_live_status():
    """A call that fails marks its server down; one that answers marks it up.

    The discriminator is structural, not a guess: only transport failures raise
    inside the interceptor chain. An isError result from the server is turned
    into an exception one layer further out, after the chain has returned, so
    in here it arrives as an ordinary return -- which is right, because the
    server plainly answered.
    """
    import httpx
    from llmtui.tools import mcp

    status = MCPStatus({"obsidian": None})
    watch = mcp._watch_servers(status)
    request = SimpleNamespace(server_name="obsidian")

    async def unreachable(req):
        raise httpx.ConnectError("All connection attempts failed")

    async def server_answered_no(req):
        # what execute_tool returns for isError=True: a result, not a raise
        return SimpleNamespace(isError=True, content=[])

    reraised = False
    try:
        await watch(request, unreachable)
    except httpx.ConnectError:
        reraised = True
    after_drop = status.servers["obsidian"]

    await watch(request, server_answered_no)
    after_server_error = status.servers["obsidian"]

    check("16  call outcomes keep the server verdict current", [
        ("a transport failure marks the server down",
         after_drop == "lost connection"),
        ("and is re-raised so the middleware can still tell the model", reraised),
        ("a server-reported error still counts as reachable",
         after_server_error is None),
        ("so it is live again", status.live == ["obsidian"]),
    ])


async def case_mcp_status_is_live():
    """The bar reflects a mid-run change with nothing wired to notify it.

    StatusBar holds the MCPStatus by reference rather than copying it, and the
    bar is repainted on every state change anyway, so a server going down
    during a turn shows up on the next repaint without a callback.
    """
    status = MCPStatus({"obsidian": None})
    app = build_app([ai_text("ok")])
    app.mcp_status = status

    async with app.run_test(size=(120, 12)) as pilot:
        bar = app.query_one(StatusBar)
        before = bar.render().plain

        # exactly what the interceptor does, from outside the widget
        status.mark_down("obsidian", "lost connection")
        stale = bar.render().plain          # not repainted yet
        app.status.set_state("thinking")    # any state change repaints
        await pilot.pause()
        after = bar.render().plain

        check("17  the bar picks up a mid-run change on the next repaint", [
            ("green before", "mcp 1/1" in before),
            ("the widget was never told anything", "mcp 1/1" in stale),
            ("and it is red after the next repaint", "mcp 0/1" in after),
        ])


async def case_mcp_status_bar():
    """The mcp tally opens into the server names when clicked."""
    app = build_app([ai_text("ok")])
    app.mcp_status = MCPStatus({"obsidian": None, "ghost": "not running"})

    async with app.run_test(size=(120, 12)) as pilot:
        status = app.query_one(StatusBar)
        collapsed = status.render().plain

        # click the segment where it is actually drawn, rather than calling the
        # action directly -- that would leave the markup wiring untested
        await pilot.click(StatusBar, offset=(collapsed.index("mcp") + 1, 0))
        await pilot.pause()
        expanded = status.render().plain
        opened = status.mcp_open

        await pilot.click(StatusBar, offset=(expanded.index("mcp") + 1, 0))
        await pilot.pause()

        check("14  the mcp button opens into the server names", [
            ("collapsed shows how many of how many", "mcp 1/2" in collapsed),
            ("collapsed does not spend the row on names", "obsidian" not in collapsed),
            ("clicking it expands", opened),
            ("the connected server is named", "obsidian" in expanded),
            ("the missing one says why", "ghost (not running)" in expanded),
            ("clicking again collapses", not status.mcp_open),
            ("no markup leaked into the text", "[" not in collapsed),
        ])


async def case_build_agent():
    """build_agent itself has to run, with MCP tools and without them.

    Every other case here builds its agent with create_agent directly, so the
    real builder was never executed by a test -- which is how it shipped with
    two ToolErrorMiddleware in it, a combination create_agent rejects outright
    ("Please remove duplicate middleware instances"). Both arities are checked
    because the empty one is what a closed vault produces.
    """
    from llmtui.agent import build_agent

    @tool
    def vault_read(path: str) -> str:
        """Read a note."""
        return "note"

    built = {}
    for label, mcp_tools in (("without mcp", []), ("with mcp", [vault_read])):
        try:
            built[label] = build_agent(None, MemorySaver(), mcp_tools) is not None
        except Exception as exc:
            built[label] = f"{type(exc).__name__}: {exc}"

    # the routing is what the single middleware bought, so check it still sorts
    handler = route_tool_error({"vault_read"})
    boom = RuntimeError("server went away")

    def said(tool_name):
        return handler(boom, SimpleNamespace(tool_call={"name": tool_name}))

    check("15  build_agent runs, and tool errors reach the right handler", [
        ("builds with no MCP tools", built["without mcp"] is True),
        ("builds with MCP tools", built["with mcp"] is True),
        ("an MCP failure talks about reaching a server",
         "could not reach the server" in said("vault_read")),
        ("a search failure still talks about the library",
         "document library" in said("search_books")),
    ])


async def main():
    await case_build_agent()
    await case_mcp_degrades()
    await case_mcp_live_status()
    await case_mcp_status_is_live()
    await case_mcp_status_bar()
    await case_footer_stacking()
    await case_prompt_wraps()
    await case_multi_round_reasoning()
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
