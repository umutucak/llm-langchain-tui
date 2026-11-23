"""First-boot workspace prompt (issue #38): asked once, before the chat TUI.

Runs as its own tiny Textual app so the main TUI stays unaware of setup. The
choice is persisted via llmtui.config.save_workspace and the skeleton
(templates/, artifacts/, Memory/) is created on confirm.
"""

import filecmp
import shutil
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static

DEFAULT_WORKSPACE = "~/Documents/gm-assistant"


def migrate_workspace(old: str, new: str) -> None:
    """Move a workspace's contents to a new directory, without data loss.

    Order matters: copy everything first, verify every file landed byte-for-
    byte identical, and only then remove the old tree. Any failure at any
    stage cancels the migration, deletes what the copy created so the system
    is back to the pre-migration state, and raises with the reason.
    """

    src = Path(old).expanduser().resolve()
    dst = Path(new).expanduser().resolve()
    if src == dst:
        return
    if src in dst.parents or dst in src.parents:
        raise RuntimeError(
            f"{dst} nests inside {src} (or the reverse); pick a sibling directory"
        )
    if not src.is_dir():
        raise RuntimeError(f"{src} does not exist; nothing to migrate")

    # an existing target must be empty: we only ever write into a place that
    # is either brand new or guaranteed to hold nothing of the user's
    pre_existing = dst.exists()
    if pre_existing:
        if any(dst.iterdir()):
            raise RuntimeError(f"{dst} already exists and is not empty")
    else:
        dst.mkdir(parents=True)

    try:
        for item in src.rglob("*"):
            target = dst / item.relative_to(src)
            if item.is_dir():
                target.mkdir(exist_ok=True)
            else:
                shutil.copy2(item, target)

        # verify 1:1: same file set, every file byte-identical
        src_files = {p.relative_to(src) for p in src.rglob("*") if p.is_file()}
        dst_files = {p.relative_to(dst) for p in dst.rglob("*") if p.is_file()}
        if src_files != dst_files:
            raise RuntimeError("copied file set does not match the original")
        for rel in src_files:
            if not filecmp.cmp(src / rel, dst / rel, shallow=False):
                raise RuntimeError(f"{rel} differs after the copy")
    except Exception as error:
        # roll back: everything under dst is ours (it was empty or new)
        for child in dst.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        if not pre_existing:
            dst.rmdir()
        raise RuntimeError(
            f"workspace migration cancelled, nothing changed: {error}"
        ) from error

    shutil.rmtree(src)


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
