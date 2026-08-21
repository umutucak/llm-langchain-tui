"""Textual frontend: a streaming transcript, live tool rows, markdown answers.

Replaces the print-based renderer. Nothing below the UI changes -- the agent,
the middleware and the tools are untouched, and this module only consumes what
the run already emits.

The run is driven through astream_events(version="v3"), which hands back an
AsyncGraphRunStream. That splits into named channels, and the two we care about
are consumed in parallel: .messages carries reasoning and answer deltas,
.tool_calls carries one ToolCallStream per call so rows can appear the moment a
search is dispatched rather than after the fact.
"""

import asyncio
import time

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.uuid import uuid7

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Collapsible, Input, Markdown, OptionList, SelectionList, Static,
)

from llmtui import config
from llmtui.naming import name_session_if_unnamed
from llmtui.sessions import delete_session, list_sessions, set_session_title


SPINNER: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# how many blocks wide the context meter is drawn
METER_WIDTH: int = 8


def meter(used: int, total: int) -> str:
    """A little bar showing how full the context window is."""

    if not total:
        return ""
    filled = min(METER_WIDTH, round(METER_WIDTH * used / total))
    return "▓" * filled + "░" * (METER_WIDTH - filled)


def compact(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class ToolRow(Collapsible):
    """One tool call: a status line that expands in place into its passages.

    Keyed by tool_call_id rather than position, because the repair middleware
    rewrites the assistant message and mints fresh ids for recovered calls -- a
    row keyed on order would end up attached to the wrong call.
    """

    def __init__(self, tool_call_id: str, tool_name: str, args: dict) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.args = args or {}
        self._started = time.monotonic()
        self._spin = 0
        self._timer = None
        self._body = Static("running…", classes="passages")
        super().__init__(
            self._body,
            title=self._row_title(),
            collapsed=True,
            collapsed_symbol="▸",
            expanded_symbol="▾",
            classes="toolrow -running",
        )

    def _label(self) -> str:
        """What was actually asked for, short enough to sit on one line."""

        query = self.args.get("query") if isinstance(self.args, dict) else None
        book = self.args.get("book") if isinstance(self.args, dict) else None
        text = f'"{query}"' if query else str(self.args)[:52]
        return f"{text} · {book}" if book else text

    def _row_title(self, mark: str | None = None, meta: str = "") -> str:
        mark = mark or SPINNER[self._spin]
        line = f"{mark} {self.tool_name}  {self._label()}"
        return f"{line}   {meta}" if meta else line

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.08, self._tick)

    def _tick(self) -> None:
        self._spin = (self._spin + 1) % len(SPINNER)
        self.title = self._row_title()

    def resolve(self, output, error, timed: bool = True) -> None:
        """Settle the row once output_deltas has run dry.

        timed=False is for rows rebuilt from a checkpoint, where the duration
        would be the age of the widget rather than of the search.
        """

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        when = f" · {time.monotonic() - self._started:.1f}s" if timed else ""
        self.remove_class("-running")

        # output is a ToolMessage, not the string the tool returned. a failure can
        # arrive two ways: an exception reaches tc.error only when nothing caught
        # it, and ToolErrorMiddleware catches search failures first and hands back
        # a ToolMessage with status="error" instead. the repair middleware's
        # dropped-call notices arrive the same way
        text = str(getattr(output, "content", output) or "")
        failed = error is not None or getattr(output, "status", None) == "error"

        if failed:
            self.add_class("-error")
            # tc.error is a string, not an exception, so there is no type name to
            # show -- the first line of the message is the useful part
            detail = str(error).splitlines()[0][:44] if error is not None else "failed"
            self.title = self._row_title("✕", f"{detail}{when}")
            # an uncaught raise arrives on tc.error with no output at all, while
            # a ToolMessage with no exception is how the repair middleware reports
            # a dropped call. on_search_error's guidance is deliberately not shown
            # here -- it is instruction written for the model, not for the reader
            self._body.update(str(error) if error is not None else text)
            return

        # search_books joins its passages with this separator, so splitting on it
        # gets the count back without the tool having to report one
        passages = text.split("\n\n---\n\n") if text.strip() else []
        self.add_class("-ok")
        plural = "" if len(passages) == 1 else "s"
        self.title = self._row_title("✓", f"{len(passages)} passage{plural}{when}")
        self._body.update(text or "(nothing returned)")


class AssistantTurn(Vertical):
    """One reply: the reasoning, the searches it fired, and the answer."""

    def __init__(self) -> None:
        super().__init__(classes="turn")
        self.reasoning: Collapsible | None = None
        self.reasoning_body: Static | None = None
        self.answer: Markdown | None = None
        self._md = None
        self._reasoning: list[str] = []
        self._reason_started: float | None = None
        self._folded = False

    async def write_reasoning(self, text: str) -> None:
        if self.reasoning is None:
            self.reasoning_body = Static("", classes="reasoning-body")
            self.reasoning = Collapsible(
                self.reasoning_body,
                title="reasoning",
                collapsed=False,
                collapsed_symbol="▸",
                expanded_symbol="▾",
                classes="reasoning",
            )
            self._reason_started = time.monotonic()
            await self.mount(self.reasoning)
        self._reasoning.append(text)
        self.reasoning_body.update("".join(self._reasoning))

    def fold_reasoning(self) -> None:
        """Once the answer starts, reasoning collapses to a one-line summary."""

        if self.reasoning is None or self._folded:
            return
        self._folded = True
        words = len("".join(self._reasoning).split())
        elapsed = time.monotonic() - (self._reason_started or time.monotonic())
        self.reasoning.title = f"reasoning · {elapsed:.1f}s · {words} words"
        self.reasoning.collapsed = True

    async def add_tool(self, tool_call_id: str, tool_name: str, args: dict) -> ToolRow:
        row = ToolRow(tool_call_id, tool_name, args)
        await self.mount(row)
        return row

    async def write_answer(self, text: str) -> None:
        if self.answer is None:
            self.fold_reasoning()
            self.answer = Markdown(classes="answer")
            await self.mount(self.answer)
            # MarkdownStream re-parses only the final block, so this stays cheap
            # however long the answer runs
            self._md = Markdown.get_stream(self.answer)
        await self._md.write(text)

    async def note(self, text: str, classes: str = "note") -> None:
        await self.mount(Static(text, classes=classes))

    async def finish(self) -> None:
        if self._md is not None:
            await self._md.stop()
            self._md = None
        self.fold_reasoning()


class StatusBar(Static):
    """State, context pressure and generation speed."""

    def __init__(self) -> None:
        super().__init__(id="status")
        self.state = "ready"
        self.ctx_used = 0
        self.tps = 0.0
        self.elapsed = 0.0

    def refresh_line(self) -> None:
        parts = [f"[b]{self.state}[/b]"]
        if self.ctx_used:
            parts.append(
                f"ctx {meter(self.ctx_used, config.CONTEXT_SIZE)} "
                f"{compact(self.ctx_used)}/{compact(config.CONTEXT_SIZE)}"
            )
        if self.tps:
            parts.append(f"{self.tps:.0f} tok/s")
        if self.elapsed:
            parts.append(f"{self.elapsed:.1f}s")
        parts += ["esc stop", "^q quit", "/help"]
        self.update("  ·  ".join(parts))

    def set_state(self, state: str) -> None:
        self.state = state
        self.refresh_line()


class NamePrompt(ModalScreen[str | None]):
    """Type a title for the current conversation."""

    BINDINGS = [("escape", "cancel", "cancel")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("name this conversation", classes="modal-title"),
            Input(placeholder="title", id="title"),
            Static("enter save · esc cancel", classes="modal-hint"),
            classes="modal",
        )

    def on_mount(self) -> None:
        self.query_one("#title", Input).focus()

    @on(Input.Submitted, "#title")
    def save(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionPicker(ModalScreen[str | None]):
    """Pick a session to resume. Returns its thread_id."""

    BINDINGS = [("escape", "cancel", "cancel")]

    def __init__(self, sessions: list, current: str) -> None:
        super().__init__()
        self.sessions = sessions
        self.current = current

    def compose(self) -> ComposeResult:
        rows = []
        for thread_id, title, updated_at in self.sessions:
            here = "  ·  current" if thread_id == self.current else ""
            rows.append(f"{title or '(untitled)'}{here}\n  {updated_at}")
        yield Vertical(
            Static("resume a session", classes="modal-title"),
            OptionList(*rows, id="picker"),
            Static("enter resume · esc cancel", classes="modal-hint"),
            classes="modal",
        )

    def on_mount(self) -> None:
        self.query_one("#picker", OptionList).focus()

    @on(OptionList.OptionSelected, "#picker")
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.sessions[event.option_index][0])

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionDeleter(ModalScreen[list[str] | None]):
    """Tick sessions to remove, the current one included.

    Deleting the conversation you are sitting in is allowed -- the app just
    rolls a fresh thread afterwards, the same as /new. It is marked in the list
    so that is a choice rather than an accident.
    """

    BINDINGS = [
        ("escape", "cancel", "cancel"),
        # SelectionList already owns enter, so the screen has to claim it first
        Binding("enter", "confirm", "delete", priority=True),
    ]

    def __init__(self, sessions: list, current: str) -> None:
        super().__init__()
        self.sessions = sessions
        self.current = current

    def compose(self) -> ComposeResult:
        rows = []
        for thread_id, title, updated_at in self.sessions:
            here = "  ·  current" if thread_id == self.current else ""
            rows.append((f"{title or '(untitled)'}{here}   {updated_at}", thread_id))
        yield Vertical(
            Static("delete sessions", classes="modal-title"),
            SelectionList(*rows, id="picker"),
            Static("space tick · enter delete · esc cancel", classes="modal-hint"),
            classes="modal",
        )

    def on_mount(self) -> None:
        self.query_one("#picker", SelectionList).focus()

    def action_confirm(self) -> None:
        self.dismiss(list(self.query_one("#picker", SelectionList).selected))

    def action_cancel(self) -> None:
        self.dismiss(None)


class LlmTui(App):
    """The app itself: input at the bottom, transcript above it."""

    CSS_PATH = "tui.tcss"
    TITLE = "llm-tui"
    BINDINGS = [
        ("escape", "stop", "stop generating"),
        ("ctrl+q", "quit", "quit"),
        ("ctrl+l", "clear", "clear transcript"),
    ]

    def __init__(self, agent, model, sqlite_connection, agent_thread_config) -> None:
        super().__init__()
        self.agent = agent
        self.model = model
        self.sqlite_connection = sqlite_connection
        self.agent_thread_config = agent_thread_config
        self._turn_started = 0.0
        self._out_chars = 0

    # ---------- layout ----------

    def compose(self) -> ComposeResult:
        yield Static(f"llm-tui   [dim]{config.MODEL}[/dim]", id="chrome")
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="ask about the library, or /help", id="prompt")
        yield StatusBar()

    def on_mount(self) -> None:
        self.transcript = self.query_one("#transcript", VerticalScroll)
        self.status = self.query_one(StatusBar)
        self.status.refresh_line()
        self.query_one("#prompt", Input).focus()
        # anchoring is Textual's own: the container follows new content and lets
        # go the moment the reader scrolls, by whatever means. re-anchored below
        # whenever something new is deliberately put on screen
        self.transcript.anchor()

    # ---------- input ----------

    @on(Input.Submitted, "#prompt")
    async def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.run_command(text)
            return

        self.transcript.anchor()
        await self.transcript.mount(Static(text, classes="turn user"))
        self.run_turn(text)

    # ---------- the turn ----------

    @work(exclusive=True, group="turn")
    async def run_turn(self, text: str) -> None:
        turn = AssistantTurn()
        await self.transcript.mount(turn)

        self.status.set_state("thinking")
        self._turn_started = time.monotonic()
        self._out_chars = 0

        try:
            stream = await self.agent.astream_events(
                {"messages": [HumanMessage(text)]},
                config=self.agent_thread_config,
                version="v3",
            )
            await asyncio.gather(
                self._pump_messages(stream, turn),
                self._pump_tools(stream, turn),
            )
        except asyncio.CancelledError:
            await turn.note("⏹ stopped · esc pressed", classes="note stopped")
            raise
        except Exception as exc:
            await turn.note(f"✕ {type(exc).__name__}: {exc}", classes="note failed")
        finally:
            await turn.finish()

        # deliberately outside the finally: both of these await, and awaiting
        # while the worker is being cancelled just raises CancelledError again
        await self._read_usage()
        self.status.set_state("naming")
        try:
            await name_session_if_unnamed(
                self.agent, self.model, self.sqlite_connection, self.agent_thread_config)
        except Exception:
            # a failed title is not worth losing the turn over
            pass
        self.status.set_state("ready")

    async def _pump_messages(self, stream, turn: AssistantTurn) -> None:
        """Reasoning and answer deltas, in the shape the sync renderer used."""

        async for message in stream.messages:
            async for event in message:
                delta = event.get("delta")
                if event.get("event") != "content-block-delta" or not isinstance(delta, dict):
                    continue
                kind = delta.get("type")
                if kind == "reasoning-delta" and config.DISPLAY_THINKING:
                    self.status.set_state("reasoning")
                    await turn.write_reasoning(delta.get("reasoning", ""))
                elif kind == "text-delta":
                    chunk = delta.get("text", "")
                    self._out_chars += len(chunk)
                    self.status.set_state("streaming")
                    await turn.write_answer(chunk)

    async def _pump_tools(self, stream, turn: AssistantTurn) -> None:
        """One row per call, each watched on its own task.

        tc.completed is a plain bool and is False when the stream object first
        arrives, so completion is signalled by output_deltas running dry. Each
        call gets its own task, otherwise the rows resolve one after another and
        the concurrency the ToolNode already has is lost on the way to the screen.
        """

        watching: list[asyncio.Task] = []
        async for tc in stream.tool_calls:
            self.status.set_state("searching")
            row = await turn.add_tool(tc.tool_call_id, tc.tool_name, tc.input)
            watching.append(asyncio.create_task(self._watch_tool(tc, row)))
        if watching:
            await asyncio.gather(*watching)

    async def _watch_tool(self, tc, row: ToolRow) -> None:
        try:
            async for _ in tc.output_deltas:
                pass
        finally:
            row.resolve(tc.output, tc.error)

    async def _read_usage(self) -> None:
        """Token counts straight from the model, rather than guessed from text.

        input_tokens on the last assistant message is exactly what the turn cost
        to send, and eval_count/eval_duration give generation speed excluding
        prompt processing -- which is the number worth showing.
        """

        elapsed = time.monotonic() - self._turn_started
        self.status.elapsed = elapsed
        try:
            # aget_state, not get_state -- AsyncSqliteSaver raises
            # NotImplementedError on every sync accessor
            state = await self.agent.aget_state(self.agent_thread_config)
            messages = state.values["messages"]
        except Exception:
            self.status.refresh_line()
            return

        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            usage = message.usage_metadata or {}
            if usage.get("input_tokens"):
                self.status.ctx_used = usage["input_tokens"] + usage.get("output_tokens", 0)
            meta = message.response_metadata or {}
            count, duration = meta.get("eval_count"), meta.get("eval_duration")
            if count and duration:
                self.status.tps = count / (duration / 1e9)
            elif elapsed > 0 and self._out_chars:
                # no timings came back, so fall back to a rough client-side rate
                self.status.tps = (self._out_chars / 4) / elapsed
            break

        self.status.refresh_line()

    # ---------- commands ----------

    @work(group="command")
    async def run_command(self, text: str) -> None:
        """Slash commands, run in a worker because push_screen_wait needs one."""

        command = text[1:].strip()
        current = self.agent_thread_config["configurable"]["thread_id"]

        if command == "help":
            await self._note(
                "/help    this list\n"
                "/new     start a fresh conversation\n"
                "/list    every conversation so far\n"
                "/name    title this conversation\n"
                "/load    resume a conversation\n"
                "/delete  remove conversations, this one included")

        elif command == "new":
            await self.start_new_thread()
            await self._note("started a new conversation")

        elif command == "list":
            lines = []
            for i, (thread_id, title, updated_at) in enumerate(list_sessions(self.sqlite_connection), start=1):
                mark = "[b]›[/b]" if thread_id == current else " "
                lines.append(f"{mark} {i}. {title or '(untitled)'}  [dim]{updated_at}[/dim]")
            await self._note("\n".join(lines) or "no sessions yet")

        elif command == "name":
            title = await self.push_screen_wait(NamePrompt())
            if title:
                set_session_title(self.sqlite_connection, current, title)
                await self._note(f"named this conversation “{title}”")

        elif command == "load":
            sessions = list_sessions(self.sqlite_connection)
            if not sessions:
                await self._note("no saved sessions yet", failed=True)
                return
            thread_id = await self.push_screen_wait(SessionPicker(sessions, current))
            if thread_id and thread_id != current:
                self.agent_thread_config["configurable"]["thread_id"] = thread_id
                await self.reload_transcript()

        elif command == "delete":
            sessions = list_sessions(self.sqlite_connection)
            if not sessions:
                await self._note("no saved sessions yet", failed=True)
                return
            doomed = await self.push_screen_wait(SessionDeleter(sessions, current))
            for thread_id in doomed or []:
                delete_session(self.sqlite_connection, thread_id)
            if doomed:
                # deleting the conversation you are sitting in leaves the thread
                # pointing at checkpoints that no longer exist, so move to a new
                # one rather than carrying on in a thread that was just erased
                if current in doomed:
                    await self.start_new_thread()
                    await self._note(
                        f"deleted {len(doomed)} session(s), including this one — "
                        f"started a new conversation")
                else:
                    await self._note(f"deleted {len(doomed)} session(s)")

        else:
            await self._note(f"{command} unrecognized. Try /help.", failed=True)

        self.transcript.anchor()

    async def _note(self, text: str, failed: bool = False) -> None:
        classes = "turn note failed" if failed else "turn note"
        await self.transcript.mount(Static(text, classes=classes))
        self.transcript.anchor()

    async def start_new_thread(self) -> None:
        """Point at a fresh thread and drop everything describing the old one.

        Used by /new, and by /delete when the conversation being removed is the
        one on screen -- carrying on in a thread whose checkpoints were just
        erased would leave the status bar quoting a context that no longer exists.
        """

        # a turn still streaming belongs to the thread we are leaving
        for worker in self.workers:
            if worker.is_running and worker.group == "turn":
                worker.cancel()

        self.agent_thread_config["configurable"]["thread_id"] = str(uuid7())
        await self.transcript.remove_children()
        self.status.ctx_used = 0
        self.status.tps = 0.0
        self.status.elapsed = 0.0
        self.status.set_state("ready")
        self.transcript.anchor()

    async def reload_transcript(self) -> None:
        """Redraw the transcript for whichever thread is now current.

        Rebuilt from the checkpoint rather than kept in memory, so a resumed
        session shows the same history the model is about to be sent.
        """

        await self.transcript.remove_children()
        state = await self.agent.aget_state(self.agent_thread_config)
        messages = (state.values or {}).get("messages", []) if state else []

        # tool results are separate messages, so index them by the call they answer
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}

        for message in messages:
            if isinstance(message, HumanMessage):
                await self.transcript.mount(Static(str(message.content), classes="turn user"))
                continue
            if not isinstance(message, AIMessage):
                continue

            turn = AssistantTurn()
            await self.transcript.mount(turn)
            for call in message.tool_calls:
                row = await turn.add_tool(call["id"], call["name"], call.get("args"))
                row.resolve(results.get(call["id"]), None, timed=False)
            answer = "".join(
                block.get("text", "") for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if answer.strip():
                await turn.write_answer(answer)
                await turn.finish()

        self.transcript.anchor()
        await self._read_usage()

    # ---------- actions ----------

    def action_stop(self) -> None:
        """Cancel the worker rather than abort()ing the run.

        Whether abort() leaves an in-flight Milvus query and the checkpoint in a
        consistent state has not been tested, so this stops the display and lets
        the graph finish its step.
        """

        for worker in self.workers:
            if worker.is_running and worker.group == "turn":
                worker.cancel()
        self.status.set_state("stopped")

    def action_clear(self) -> None:
        self.transcript.remove_children()
        self.transcript.anchor()
