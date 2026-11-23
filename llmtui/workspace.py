"""First-boot workspace prompt (issue #38): asked once, before the chat TUI.

Runs as its own tiny Textual app so the main TUI stays unaware of setup. The
choice is persisted via llmtui.config.save_workspace and the skeleton
(templates/, artifacts/, Memory/) is created on confirm.
"""

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static

DEFAULT_WORKSPACE = "~/Documents/gm-assistant"

CSS = """
Screen { align: center middle; }
#box { width: 70; padding: 1 2; border: round $accent; }
.modal-title { text-style: bold; margin-bottom: 1; }
.modal-hint { color: $text-muted; margin-top: 1; }
"""


class WorkspacePrompt(App[str | None]):
    """Ask where the workspace goes; dismiss with the expanded path or None."""

    CSS = CSS

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("where should the workspace live?", classes="modal-title"),
            # the default is real text, not a placeholder: enter accepts it as
            # is, and the reader can edit it like any path
            Input(value=DEFAULT_WORKSPACE, id="workspace"),
            Static("enter confirm · esc quit", classes="modal-hint"),
            id="box",
        )

    def on_mount(self) -> None:
        self.query_one("#workspace", Input).focus()

    @on(Input.Submitted, "#workspace")
    def confirm(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return
        path = Path(raw).expanduser().resolve()
        self.existing = path.is_dir()
        # skeleton: the two top folders #37 writes into, plus Memory
        for child in ("templates", "artifacts", "Memory"):
            (path / child).mkdir(parents=True, exist_ok=True)
        self.exit(str(path))

    def action_cancel(self) -> None:
        self.exit(None)


async def ensure_workspace() -> str | None:
    """Prompt when the configured workspace is missing; return the final path.

    No workspace configured, or a configured one that no longer exists,
    triggers the prompt. An existing (even empty) directory never re-prompts.
    None means the user quit out of the prompt.
    """

    from llmtui import config

    if config.WORKSPACE_DIR and Path(config.WORKSPACE_DIR).is_dir():
        return config.WORKSPACE_DIR

    path = await WorkspacePrompt().run_async()
    if path is not None:
        config.save_workspace(path)
    return path
