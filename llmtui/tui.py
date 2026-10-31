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
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.uuid import uuid7
from langgraph.types import Command
from rich.syntax import Syntax
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Collapsible,
    Input,
    Markdown,
    OptionList,
    SelectionList,
    Static,
    TextArea,
)

from llmtui import citations, config
from llmtui.agent import build_agent, build_model
from llmtui.models_config import load_models
from llmtui.naming import name_session_if_unnamed
from llmtui.sessions import delete_session, list_sessions, set_session_title
from llmtui.tools.mcp import WRITE_TOOLS, MCPStatus, tool_text
from llmtui.writes import WritePreview, preview_write

SPINNER: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# how many blocks wide the context meter is drawn
METER_WIDTH: int = 8
# most a tool row will put on screen. the row is a preview, and an auto-height
# Static holding a whole note re-lays the log out every time it is opened
ROW_PREVIEW_CHARS: int = 4000


def preview(text: str) -> str:
    """Enough of a tool result to read, without the rest of it on screen."""

    if len(text) <= ROW_PREVIEW_CHARS:
        return text
    return f"{text[:ROW_PREVIEW_CHARS]}\n\n… {len(text) - ROW_PREVIEW_CHARS} more characters"


def meter(used: int, total: int) -> str:
    """A little bar showing how full the context window is."""

    if not total:
        return ""
    filled = min(METER_WIDTH, round(METER_WIDTH * used / total))
    return "▓" * filled + "░" * (METER_WIDTH - filled)


def compact(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


# the menu is built from this rather than from the help text, so /help and the
# popup can never drift apart. (name, one-line description)
COMMANDS: list[tuple[str, str]] = [
    ("help", "this list"),
    ("new", "start a fresh conversation"),
    ("list", "every conversation so far"),
    ("name", "title this conversation"),
    ("load", "resume a conversation"),
    ("model", "switch model"),
    ("delete", "remove conversations, this one included"),
]


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
        # a Markdown body, not a Static: search results carry [/path p.N] markers
        # that Static's markup parser would eat as closing tags (crash), and
        # verbatim-looking plain text misses what users actually do -- click the
        # citation in the passage to open the book. Markdown parses it as literal
        # text, so citations normalize() links up and nothing crashes
        self._body = Markdown("running…", classes="passages", open_links=False)
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

    async def resolve(self, output, error, timed: bool = True) -> None:
        """Settle the row once output_deltas has run dry.

        On the live path output is only the tool's content string: the artifact
        and the middleware's error status ride on the ToolMessage, which the
        graph commits at the end of the superstep. finish() applies those once
        it exists; a row settled from a ToolMessage already applies them here.

        timed=False is for rows rebuilt from a checkpoint, where the duration
        would be the age of the widget rather than of the search.
        """

        self._stop_spinner()
        when = f" · {time.monotonic() - self._started:.1f}s" if timed else ""
        self.remove_class("-running")

        # a failure can arrive two ways: an exception reaches tc.error only when
        # nothing caught it, and ToolErrorMiddleware catches search failures
        # first and hands back a ToolMessage with status="error" instead. the
        # repair middleware's dropped-call notices arrive the same way
        text = tool_text(getattr(output, "content", output))
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
            await self._body.update(
                citations.normalize(preview(str(error) if error is not None else text))
            )
            return

        if isinstance(output, ToolMessage):
            await self.finish(output, timed=timed)
            return

        # the stream's string only, so no artifact to classify on yet. show the
        # text now and let finish() correct the title once the round commits
        self.add_class("-ok")
        self.title = self._row_title("✓", f"{compact(len(text))} chars{when}")
        await self._body.update(
            citations.normalize(preview(text)) or "(nothing returned)"
        )

    async def finish(self, output, timed: bool = True) -> None:
        """Classify a settled row from its committed ToolMessage.

        a search that matched nothing is not a failure, so it arrives here
        looking like any other answer. search_books puts its count in the
        artifact, which is the only way to tell the two apart -- a tool that
        reports no count gets its size instead of an invented passage figure
        """

        self._stop_spinner()
        when = f" · {time.monotonic() - self._started:.1f}s" if timed else ""
        self.remove_class("-running")
        text = tool_text(getattr(output, "content", output))
        passages = (getattr(output, "artifact", None) or {}).get("passages")

        if passages == 0:
            self.add_class("-empty")
            # resolve() may have already marked the row ok on the char fallback
            self.remove_class("-ok")
            self.title = self._row_title("∅", f"no passages{when}")
        elif passages is not None:
            self.add_class("-ok")
            self.remove_class("-empty")
            plural = "" if passages == 1 else "s"
            self.title = self._row_title("✓", f"{passages} passage{plural}{when}")
        else:
            self.add_class("-ok")
            self.remove_class("-empty")
            self.title = self._row_title("✓", f"{compact(len(text))} chars{when}")

        await self._body.update(
            citations.normalize(preview(text)) or "(nothing returned)"
        )

    def _stop_spinner(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


class AssistantTurn(Vertical):
    """One reply, in as many rounds as the graph takes to finish it.

    The graph loops model -> tools -> model, and every pass emits its own
    reasoning and its own text. Widgets are therefore per round, not per turn:
    a second round gets a second reasoning pane and a second answer block,
    mounted after the tool rows that came between them. Holding one pane for
    the whole turn merged round two's thinking into round one's, above the
    search it was reacting to, behind a fold with a stale word count.
    """

    def __init__(self) -> None:
        super().__init__(classes="turn")
        self.reasoning: Collapsible | None = None
        self.reasoning_body: Static | None = None
        self.answer: Markdown | None = None
        self._md = None
        # the raw answer text of the current round, for the whole-answer
        # normalize pass in close_round (citations split across appends)
        self._answer_text = ""
        self._reasoning: list[str] = []
        self._reason_started: float | None = None
        self._folded = False
        self.rounds = 0

    async def start_round(self) -> None:
        """Called once per model pass, before any of its deltas arrive."""

        await self.close_round()
        self.reasoning = None
        self.reasoning_body = None
        self.answer = None
        self._reasoning = []
        self._reason_started = None
        self._folded = False
        self.rounds += 1

    async def close_round(self) -> None:
        """Settle whatever the previous pass left open."""

        if self._md is not None:
            await self._md.stop()
            self._md = None
        # a citation split across stream chunks only normalizes whole: re-render
        # the finished answer into a FRESH Markdown (updating the streamed widget
        # in place left links stale)
        if self.answer is not None:
            whole = citations.normalize(self._answer_text)
            if whole != self._answer_text:
                fresh = Markdown(classes="answer", open_links=False)
                parent = self.answer.parent or self
                await parent.mount(
                    fresh, before=self.answer if self.answer.parent else None
                )
                await self.answer.remove()
                self.answer = fresh
                await fresh.update(whole)
            self._answer_text = whole
        self.fold_reasoning()

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
            # open_links=False: link clicks arrive as Markdown.LinkClicked for
            # the app to route (citations open PDFs, web links still open a
            # browser). The flag covers streamed content too, not just the
            # final document.
            self.answer = Markdown(classes="answer", open_links=False)
            self._answer_text = ""
            await self.mount(self.answer)
            # MarkdownStream re-parses only the final block, so this stays cheap
            # however long the answer runs
            self._md = Markdown.get_stream(self.answer)
        # raw text accumulates for close_round: a citation split across chunks
        # only normalizes whole (answer is re-rendered once the round ends)
        self._answer_text += text
        await self._md.write(text)

    async def note(self, text: str, classes: str = "note") -> None:
        await self.mount(Static(text, classes=classes, markup=False))

    async def finish(self) -> None:
        await self.close_round()


class DiffScreen(ModalScreen[None]):
    """The whole diff, for when the card's summary is not enough to judge by.

    Read-only: the decision is still taken on the card underneath. Rich already
    colours a unified diff, so there is nothing to write here but the frame.
    """

    BINDINGS = [("escape,q,enter", "close", "close")]

    def __init__(self, preview: WritePreview) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        body = Static(classes="diff-body")
        if self.preview.diff:
            body.update(
                Syntax(
                    self.preview.diff,
                    "diff",
                    word_wrap=True,
                    theme="ansi_dark",
                    background_color="default",
                )
            )
        else:
            body.update(self.preview.note or "(nothing to show)")
        yield Vertical(
            Static(f"{self.preview.tool} · {self.preview.path}", classes="modal-title"),
            VerticalScroll(body),
            Static("esc close", classes="modal-hint"),
            classes="modal modal-wide",
        )

    def action_close(self) -> None:
        self.dismiss(None)


class ReasonScreen(ModalScreen[str | None]):
    """A line of explanation to send back with a refusal.

    Refusing without one leaves the model guessing why, and guessing usually
    means trying the same write again in a slightly different shape.
    """

    BINDINGS = [("escape", "cancel", "cancel")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("why are you refusing this write?", classes="modal-title"),
            Input(placeholder="the model reads this", id="reason"),
            Static("enter send · esc back", classes="modal-hint"),
            classes="modal",
        )

    def on_mount(self) -> None:
        self.query_one("#reason", Input).focus()

    @on(Input.Submitted, "#reason")
    def submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class WriteCard(Vertical):
    """One pending write, waiting on a person.

    The graph is interrupted while this is on screen, so nothing has touched the
    vault yet and nothing will until decision() resolves. It settles in place
    afterwards rather than disappearing, so the transcript keeps a record of
    what was allowed and what was not.
    """

    BINDINGS = [
        ("a", "approve", "approve"),
        ("r", "reject", "reject"),
        ("R", "reject_with_reason", "reject with a reason"),
        Binding("enter", "show_diff", "diff", priority=True),
        ("escape", "reject", "reject"),
    ]

    can_focus = True

    def __init__(self, preview: WritePreview) -> None:
        super().__init__(classes="writecard -pending")
        self.preview = preview
        self._decided: asyncio.Future = asyncio.get_running_loop().create_future()

    def compose(self) -> ComposeResult:
        yield Static(
            f"✎ {self.preview.tool}  {self.preview.path}", classes="write-head"
        )
        yield Static(self.preview.summary, classes="write-summary")
        if self.preview.note:
            yield Static(self.preview.note, classes="write-note")
        hint = "a approve · r reject · R reject with a reason"
        if self.preview.diff:
            hint = "enter diff · " + hint
        yield Static(hint, classes="write-hint")

    def on_mount(self) -> None:
        self.focus()

    async def decision(self) -> dict:
        """Block the turn here until someone says yes or no."""

        return await self._decided

    def _settle(self, decision: dict, mark: str, style: str) -> None:
        if self._decided.done():
            return
        self.remove_class("-pending")
        self.add_class(style)
        self.query_one(".write-hint", Static).update(mark)
        self._decided.set_result(decision)

    def abandon(self) -> None:
        """Give the card up without deciding it, when the turn is stopped.

        Counts as a refusal, because a write nobody approved is a write that
        does not happen. The graph is still parked on its interrupt; the next
        turn clears that before sending anything.
        """

        self._settle({"type": "reject"}, "⏹ left undecided", "-error")

    def action_approve(self) -> None:
        self._settle({"type": "approve"}, "✓ approved", "-ok")

    def action_reject(self) -> None:
        self._settle({"type": "reject"}, "✕ refused", "-error")

    @work
    async def action_reject_with_reason(self) -> None:
        reason = await self.app.push_screen_wait(ReasonScreen())
        decision = {"type": "reject"}
        if reason:
            decision["message"] = reason
        self._settle(decision, f"✕ refused · {reason or 'no reason given'}", "-error")

    @work
    async def action_show_diff(self) -> None:
        await self.app.push_screen_wait(DiffScreen(self.preview))
        self.focus()


class CommandMenu(OptionList):
    """The floating list of viable commands, shown above the prompt while it
    holds a "/fragment".

    Not a ModalScreen -- the prompt must keep the focus and keep accepting
    keystrokes while the menu is up, the way pi does it. The prompt drives it:
    the menu only paints whichever commands still match and reports the
    highlighted one back when the prompt asks for a completion.
    """

    def __init__(self) -> None:
        super().__init__(id="command-menu", classes="hidden")
        # the entries currently painted, kept so the highlight can be mapped
        # back to a command name. positional indexing of COMMANDS would break
        # the moment the list is filtered -- highlight 0 of "/l" is list, not
        # the first command ever defined
        self._shown: list[tuple[str, str]] = []

    def show(self, matches: list[tuple[str, str]]) -> None:
        """Repaint with the given matches and make the first one the target."""

        self._shown = matches
        self.clear_options().add_options(
            f"/[b]{name}[/b]  [dim]· {about}[/]" for name, about in matches
        )
        self.remove_class("hidden")
        self.highlighted = 0

    def hide(self) -> None:
        self.add_class("hidden")
        self.clear_options()

    @property
    def active(self) -> bool:
        return not self.has_class("hidden") and self.option_count > 0

    def current(self) -> str | None:
        """The command name under the highlight, for completion."""

        if not self.active or self.highlighted is None:
            return None
        return self._shown[self.highlighted][0]


class PromptArea(TextArea):
    """The prompt box: soft-wrapped, and it grows instead of scrolling sideways.

    Input is single line by definition -- a long question runs off to the left
    and the start of it becomes unreachable. TextArea wraps, but it owns enter
    for inserting newlines, and it takes that in _on_key with prevent_default(),
    so a binding never gets a look. Intercepting here is the only way to keep
    enter meaning send.
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        # while the menu is up the prompt lends it the navigation keys, the same
        # bargain a combobox makes. tab completes into the prompt; enter runs
        # the highlighted command outright; escape puts it
        # away without touching the stop-generating binding underneath
        menu = self.app.query_one(CommandMenu)
        if menu.active:
            if event.key == "down":
                event.stop()
                event.prevent_default()
                menu.action_cursor_down()
                return
            if event.key == "up":
                event.stop()
                event.prevent_default()
                menu.action_cursor_up()
                return
            if event.key == "tab":
                event.stop()
                event.prevent_default()
                self.complete(menu.current())
                return
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                # enter does not just fill the prompt in -- it runs the
                # highlighted command, the way submitting a typed "/help"
                # would
                command = menu.current()
                if command is not None:
                    self.post_message(self.Submitted(f"/{command}"))
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                menu.hide()
                return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        # a real newline, where the terminal can tell the two apart at all
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

    def complete(self, command: str | None) -> None:
        """Fill in the highlighted command and hand the prompt back."""

        self.app.query_one(CommandMenu).hide()
        if command is None:
            return
        self.text = f"/{command} "
        # setting text rewinds the cursor to the start; put it back at the end
        self.move_cursor(self.document.end)
        self.focus()


class StatusBar(Static):
    """State, model, context pressure, generation speed and the MCP servers."""

    def __init__(
        self, mcp_status: MCPStatus, context_window: int = 0, model_label: str = ""
    ) -> None:
        super().__init__(id="status")
        self.state = "ready"
        # provider/model shown next to the context usage; swap_model updates it
        self.model_label = model_label
        self.ctx_used = 0
        # filled in only when the provider reports its context length; 0 keeps
        # the meter a plain token count (see refresh_line)
        self.context_window = context_window
        self.tps = 0.0
        self.elapsed = 0.0
        # held by reference and read at render time, never copied. the
        # interceptor in mcp.py writes to this same object as calls fail and
        # recover, so every repaint shows the current verdict for free
        self.mcp = mcp_status
        self.mcp_open = False

    def mcp_segment(self) -> str:
        """The mcp button: a tally that opens into the roll call when clicked.

        How many servers there are is not known here, and the bar is one row,
        so the names cannot all live on it permanently. Collapsed it is a count
        you can read at a glance; clicking swaps in the names, and the ones
        that failed say why. [@click=...] is Textual's own markup, so this
        stays one Static rather than becoming a row of widgets.
        """

        servers = self.mcp.servers
        live = self.mcp.live

        if not self.mcp_open:
            colour = "$success" if len(live) == len(servers) else "$error"
            label = f"mcp [{colour}]{len(live)}/{len(servers)}[/]"
        else:
            label = "mcp " + " ".join(
                f"[$success]{name}[/]"
                if failed is None
                else f"[$error]{name} ({failed})[/]"
                for name, failed in servers.items()
            )

        return f"[@click=app.toggle_mcp()]{label}[/]"

    def refresh_line(self) -> None:
        parts = [f"[b]{self.state}[/b]"]
        if self.model_label:
            parts.append(f"[dim]{self.model_label}[/dim]")
        if self.ctx_used:
            # context_window arrives filled in only when the provider said how
            # big it is (step 5 of SPEC-provider-migration.md). without it the
            # meter degrades to a plain token count -- no invented ceiling
            if self.context_window:
                parts.append(
                    f"ctx {meter(self.ctx_used, self.context_window)} "
                    f"{compact(self.ctx_used)}/{compact(self.context_window)}"
                )
            else:
                parts.append(f"ctx {compact(self.ctx_used)}")
        if self.tps:
            parts.append(f"{self.tps:.0f} tok/s")
        if self.elapsed:
            parts.append(f"{self.elapsed:.1f}s")
        # nothing to say when no servers are configured at all
        if self.mcp.servers:
            parts.append(self.mcp_segment())
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


class ModelPicker(ModalScreen[str | None]):
    """Pick a model. Returns its "provider/model" selector.

    models.json is reloaded here, on every open, so edits to the file show
    up without restarting the app (pi parity). The swap itself is cheap:
    the graph holds no history -- that lives in the checkpointer keyed by
    thread_id -- so a new graph over the same memory resumes the same
    conversation. Reasoning/sampling params travel with the model entry.
    """

    BINDINGS = [("escape", "cancel", "cancel")]

    def __init__(self, entries, current: str) -> None:
        super().__init__()
        self.entries = entries  # (selector, label) rows
        self.current = current

    def compose(self) -> ComposeResult:
        rows = []
        for selector, label in self.entries:
            here = "  ·  current" if selector == self.current else ""
            rows.append(f"{label}{here}")
        yield Vertical(
            Static("switch model", classes="modal-title"),
            OptionList(*rows, id="picker"),
            Static("enter switch · esc cancel", classes="modal-hint"),
            classes="modal",
        )

    def on_mount(self) -> None:
        self.query_one("#picker", OptionList).focus()

    @on(OptionList.OptionSelected, "#picker")
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.entries[event.option_index][0])

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

    def __init__(
        self,
        agent,
        model,
        memory,
        mcp_tools,
        entry,
        sqlite_connection,
        agent_thread_config,
        mcp_status: MCPStatus,
        note_reader=None,
        # where citations look for the cited PDFs (see llmtui/citations.py);
        # a string here, resolved to a Path only at click time
        library_dir: str = "",
    ) -> None:
        super().__init__()
        self.agent = agent
        self.model = model
        # kept for switch_model: a new graph over the same memory and the same
        # tools resumes the same thread with the new model
        self.memory = memory
        self.mcp_tools = mcp_tools
        self.model_entry = entry
        self.model_provider = entry.provider
        self.sqlite_connection = sqlite_connection
        self.agent_thread_config = agent_thread_config
        self.mcp_status = mcp_status
        # reads a note so a pending write can be shown as a diff. None when the
        # vault is not there, and the approval still happens either way
        self.note_reader = note_reader
        self.library_dir = library_dir
        self._turn_started = 0.0
        self._out_chars = 0

    async def context_window(self) -> int:
        """Context length for the meter: the entry's metadata when declared,
        else the provider asked live, else 0 (plain token count).

        The live query is only the fallback for entries without contextWindow
        metadata, and it is best-effort: a server that cannot say leaves the
        meter with no ceiling rather than inventing one.
        """

        if self.model_entry.context_window:
            return self.model_entry.context_window
        providers = load_models()
        provider = providers.get(self.model_entry.provider)
        if provider is None:
            return 0
        from llmtui.models_config import fetch_context_window

        return await fetch_context_window(self.model_entry, provider)

    # ---------- layout ----------

    def compose(self) -> ComposeResult:
        yield Static(f"llm-tui   [dim]{self.model.model_name}[/dim]", id="chrome")
        yield VerticalScroll(id="transcript")
        # the statusbar blocks the last line of the TextArea because both of their
        # CSS are docked to the bottom. we do a hard split here and each are
        # bottoming their own containers. nothing a 30 minute debug sidetracking cant fix :')
        with Vertical(id="footer"):
            # the floating command menu; hidden until the prompt holds a "/..."
            yield CommandMenu()
            yield PromptArea(
                placeholder="ask about the library, or /help",
                soft_wrap=True,
                highlight_cursor_line=False,
                id="prompt",
            )
            yield StatusBar(
                self.mcp_status,
                model_label=f"{self.model_provider}/{self.model_entry.id}",
            )

    async def on_mount(self) -> None:
        self.transcript = self.query_one("#transcript", VerticalScroll)
        self.status = self.query_one(StatusBar)
        # the live context fallback is async, so the statusbar gets its meter
        # filled in here rather than in compose
        self.status.context_window = await self.context_window()
        self.status.refresh_line()
        self.query_one("#prompt", PromptArea).focus()
        # anchoring is Textual's own: the container follows new content and lets
        # go the moment the reader scrolls, by whatever means. re-anchored below
        # whenever something new is deliberately put on screen
        self.transcript.anchor()

    @on(PromptArea.Changed, "#prompt")
    def prompt_changed(self) -> None:
        """Keeps the command menu in step with what is being typed.

        A menu is shown while the prompt holds "/" plus a fragment with no
        space in it -- once the command is being argued with, it is no longer
        being picked. Matches are prefix, so the list thins out as the command
        is spelled: / shows everything, /l leaves list and load, /load leaves
        one.
        """

        text = self.query_one("#prompt", PromptArea).text
        menu = self.query_one(CommandMenu)
        if not text.startswith("/") or " " in text:
            menu.hide()
            return
        fragment = text[1:].lower()
        matches = [c for c in COMMANDS if c[0].startswith(fragment)]
        if matches:
            menu.show(matches)
        else:
            menu.hide()

    # ---------- input ----------

    @on(PromptArea.Submitted)
    async def submitted(self, event: PromptArea.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#prompt", PromptArea).text = ""
        # the menu filters on the prompt's text, which was just emptied
        self.query_one(CommandMenu).hide()
        if not text:
            return
        if text.startswith("/"):
            self.run_command(text)
            return

        self.transcript.anchor()
        await self.transcript.mount(Static(text, classes="turn user", markup=False))
        self.run_turn(text)

    # ---------- links ----------

    @on(Markdown.LinkClicked)
    async def on_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Citations open the PDF, everything else opens as a web link."""

        event.stop()
        citation = citations.parse(event.href)
        if citation is None:
            self.open_url(event.href)
            return
        error = citations.open_pdf(citation, Path(self.library_dir))
        if error:
            self.notify(error, severity="error", timeout=8)

    # ---------- the turn ----------

    @work(exclusive=True, group="turn")
    async def run_turn(self, text: str) -> None:
        turn = AssistantTurn()
        await self.transcript.mount(turn)

        self.status.set_state("thinking")
        self._turn_started = time.monotonic()
        self._out_chars = 0

        try:
            refusals = await self._pending_refusals()
            payload = {"messages": [HumanMessage(text)]}
            if refusals:
                # a parked write from an abandoned turn is refused in the same
                # resume that carries the user's next message, so the model
                # answers once instead of emitting an unseen reply to the
                # rejection first
                payload = Command(
                    resume={"decisions": refusals},
                    update={"messages": [HumanMessage(text)]},
                )
            # a turn is one pass unless a write needs approving. every approval
            # interrupts the graph, so the turn is really: run until it stops,
            # ask, resume, and again until it stops without asking
            while True:
                stream = await self.agent.astream_events(
                    payload,
                    config=self.agent_thread_config,
                    version="v3",
                )
                await asyncio.gather(
                    self._pump_messages(stream, turn),
                    self._pump_tools(stream, turn),
                )
                await self._sync_tool_rows(turn)
                interrupts = await stream.interrupts()
                if not interrupts:
                    break
                decisions = await self._review_writes(interrupts, turn)
                payload = Command(resume={"decisions": decisions})
        except asyncio.CancelledError:
            # a waiting card holds the focus, so a stopped turn has to hand it
            # back or the prompt cannot be typed into again
            for card in turn.query(WriteCard):
                card.abandon()
            self.query_one("#prompt", PromptArea).focus()
            await turn.note("⏹ stopped · esc pressed", classes="note stopped")
            raise
        except Exception as exc:
            await turn.note(await self._explain_failure(exc), classes="note failed")
        finally:
            await turn.finish()

        # deliberately outside the finally: both of these await, and awaiting
        # while the worker is being cancelled just raises CancelledError again
        await self._read_usage()
        self.status.set_state("naming")
        try:
            await name_session_if_unnamed(
                self.agent, self.model, self.sqlite_connection, self.agent_thread_config
            )
        except Exception:
            # a failed title is not worth losing the turn over
            pass
        self.status.set_state("ready")

    async def _explain_failure(self, exc: Exception) -> str:
        """Name the write the model was setting up, when it was setting one up.

        These break while a write is being put together, and the exception is
        the model complaining about its own request -- it says nothing about
        the vault, which is the part the reader actually wants to know about.
        """

        detail = f"✕ {type(exc).__name__}: {exc}"

        try:
            messages = (await self.agent.aget_state(self.agent_thread_config)).values[
                "messages"
            ]
        except Exception:
            return detail

        # only the newest assistant message matters -- an older write in the
        # same thread has already been decided and is not what broke
        newest = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if newest is None:
            return detail

        writes = [c for c in newest.tool_calls if c["name"] in WRITE_TOOLS]
        if not writes:
            return detail

        asked = ", ".join(
            f"{call['name']} on {call['args'].get('path') or '?'}" for call in writes
        )
        return (
            f"{detail}\n"
            f"   this broke while setting up {asked}\n"
            f"   the vault was not touched -- nothing runs before you approve it"
        )

    async def _review_writes(self, interrupts, turn: AssistantTurn) -> list[dict]:
        """One card at a time, and one decision back for each write asked about.

        The middleware raises unless it gets exactly as many decisions as it
        asked for, in the order it asked, so this walks the requests rather than
        collecting whatever the cards happen to produce.
        """

        self.status.set_state("approve?")
        decisions: list[dict] = []

        for pending in interrupts:
            request = getattr(pending, "value", pending) or {}
            for action in request.get("action_requests", []):
                preview = await preview_write(
                    action["name"], action.get("args") or {}, self.note_reader
                )
                self.transcript.anchor()
                card = WriteCard(preview)
                await turn.mount(card)
                decisions.append(await card.decision())

        self.query_one("#prompt", PromptArea).focus()
        self.status.set_state("thinking")
        return decisions

    async def _pending_refusals(self) -> list[dict]:
        """Refusals for anything left hanging from a turn that never finished.

        Stopping with esc, a crash, or loading a thread that was mid-approval
        all leave the graph parked on an interrupt. Sending a new message into
        that would resume it with no decision at all, so the safe reading of an
        abandoned write is that it was not approved.

        Returns the decisions rather than resuming here: the caller folds them
        into the same resume that carries the user's next message, so the model
        answers once instead of emitting a discarded reply to the rejection.
        """

        try:
            state = await self.agent.aget_state(self.agent_thread_config)
        except Exception:
            return []

        pending = getattr(state, "interrupts", None) or []
        wanted = 0
        for item in pending:
            request = getattr(item, "value", item) or {}
            wanted += len(request.get("action_requests", []))
        if not wanted:
            return []

        return [
            {
                "type": "reject",
                "message": "The turn was interrupted before this was approved, "
                "so it was not run. Ask again if it is still wanted.",
            }
            for _ in range(wanted)
        ]

    async def _pump_messages(self, stream, turn: AssistantTurn) -> None:
        """Reasoning and answer deltas, in the shape the sync renderer used."""

        # one item per model pass, so this is where a round begins
        async for message in stream.messages:
            await turn.start_round()
            async for event in message:
                delta = event.get("delta")
                if event.get("event") != "content-block-delta" or not isinstance(
                    delta, dict
                ):
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
            self.status.set_state("tooling")
            row = await turn.add_tool(tc.tool_call_id, tc.tool_name, tc.input)
            watching.append(asyncio.create_task(self._watch_tool(tc, row)))
        if watching:
            await asyncio.gather(*watching)

    async def _watch_tool(self, tc, row: ToolRow) -> None:
        try:
            async for _ in tc.output_deltas:
                pass
        finally:
            await row.resolve(tc.output, tc.error)

    async def _sync_tool_rows(self, turn: AssistantTurn) -> None:
        """Finish the round's rows from the committed ToolMessages.

        tc.output is only the tool's content string; the artifact and the
        middleware's error status live on the ToolMessage, which the graph
        commits at the end of the superstep. Reading state here, after the
        pumps drain, is what makes that commit guaranteed -- a lookup when
        output_deltas runs dry races it (issue #18).
        """

        state = await self.agent.aget_state(self.agent_thread_config)
        by_id = {
            message.tool_call_id: message
            for message in state.values.get("messages", [])
            if isinstance(message, ToolMessage) and message.tool_call_id
        }
        for row in turn.query(ToolRow):
            # a row settled from tc.error already shows the raised message,
            # which the ToolMessage does not carry
            if row.has_class("-error"):
                continue
            message = by_id.get(row.tool_call_id)
            if message is not None:
                await row.finish(message)

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
                self.status.ctx_used = usage["input_tokens"] + usage.get(
                    "output_tokens", 0
                )
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
            width = max(len(name) for name, _ in COMMANDS)
            await self._note(
                "\n".join(f"/{name:<{width}}  {about}" for name, about in COMMANDS)
            )

        elif command == "new":
            await self.start_new_thread()
            await self._note("started a new conversation")

        elif command == "list":
            lines = []
            for i, (thread_id, title, updated_at) in enumerate(
                list_sessions(self.sqlite_connection), start=1
            ):
                mark = "[b]›[/b]" if thread_id == current else " "
                lines.append(
                    f"{mark} {i}. {title or '(untitled)'}  [dim]{updated_at}[/dim]"
                )
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

        elif command == "model":
            # models.json is read now, at open, never at launch: edits to the
            # file land in the very next /model
            try:
                providers = load_models()
            except RuntimeError as exc:
                await self._note(f"✕ models.json: {exc}", failed=True)
                return
            entries = [
                (f"{p.name}/{m.id}", f"{m.id}  [dim]· {p.name}[/]")
                for p in providers.values()
                for m in p.models
            ]
            current = f"{self.model_provider}/{self.model.model_name}"
            selector = await self.push_screen_wait(ModelPicker(entries, current))
            if selector and selector != current:
                try:
                    await self.switch_model(selector)
                except RuntimeError as exc:
                    await self._note(f"✕ {exc}", failed=True)
                else:
                    await self._note(f"model switched to {selector}")

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
                        f"started a new conversation"
                    )
                else:
                    await self._note(f"deleted {len(doomed)} session(s)")

        else:
            await self._note(f"{command} unrecognized. Try /help.", failed=True)

        self.transcript.anchor()

    async def _note(self, text: str, failed: bool = False) -> None:
        classes = "turn note failed" if failed else "turn note"
        await self.transcript.mount(Static(text, classes=classes, markup=False))
        self.transcript.anchor()

    async def switch_model(self, selector: str) -> None:
        """Rebuild the graph over a new model, same memory, same thread.

        The graph itself carries no conversation state -- history lives in the
        checkpointer keyed by thread_id -- so pointing a fresh build_agent at
        the same memory resumes the conversation with the new model. Sampling
        and reasoning settings travel with the model entry in models.json.
        """

        # a turn still streaming belongs to the model being replaced
        for worker in self.workers:
            if worker.is_running and worker.group == "turn":
                worker.cancel()

        model = build_model(selector)
        providers = load_models()
        entry = next(
            m
            for p in providers.values()
            for m in p.models
            if f"{p.name}/{m.id}" == selector
        )
        self.model = model
        self.model_entry = entry
        self.model_provider = selector.split("/", 1)[0]
        self.agent = build_agent(
            model,
            self.memory,
            self.mcp_tools,
        )
        self.status.context_window = await self.context_window()
        self.status.model_label = f"{self.model_provider}/{self.model_entry.id}"
        # refresh the header line
        self.query_one("#chrome", Static).update(
            f"llm-tui   [dim]{model.model_name}[/dim]"
        )

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
                await self.transcript.mount(
                    Static(str(message.content), classes="turn user", markup=False)
                )
                continue
            if not isinstance(message, AIMessage):
                continue

            turn = AssistantTurn()
            await self.transcript.mount(turn)
            for call in message.tool_calls:
                row = await turn.add_tool(call["id"], call["name"], call.get("args"))
                await row.resolve(results.get(call["id"]), None, timed=False)
            answer = "".join(
                block.get("text", "")
                for block in message.content
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

    def action_toggle_mcp(self) -> None:
        """Open and close the mcp tally in the status bar. Fired by the click
        markup in StatusBar.mcp_segment, so there is no key bound to it."""

        self.status.mcp_open = not self.status.mcp_open
        self.status.refresh_line()
