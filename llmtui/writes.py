"""What a pending vault write would do, worked out before it is allowed to run.

The model asks to change a note; this reads what is there now, works out what
would be there afterwards, and hands back a diff to approve or refuse.

Nothing here imports the UI or the MCP client. The read is injected, so the
whole module tests against a dict standing in for a vault.
"""

import difflib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

# reading a heading returns exactly what replacing it would consume, so a patch
# can be previewed by reading its own target back first. these are the argument
# names that select it, passed through to vault_read untouched
TARGET_ARGS: tuple[str, ...] = ("targetType", "target", "scope")
# lines of context around each change
DIFF_CONTEXT: int = 3


@dataclass
class WritePreview:
    """One pending write, in the terms a person needs to judge it."""

    tool: str
    path: str
    summary: str
    # the unified diff, or None when this write has no before and after to
    # compare -- the summary carries it alone then, and note says why
    diff: str | None = None
    note: str | None = None


def _diff(path: str, before: str, after: str) -> str | None:
    """A unified diff, or None when the write would not change anything."""

    lines = list(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        f"{path} (now)",
        f"{path} (after)",
        n=DIFF_CONTEXT,
    ))
    if not lines:
        return None
    # a file with no trailing newline would otherwise run its last line into the
    # next hunk header
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


def _counts(diff: str) -> str:
    """How much moves, counted off the diff rather than the two texts.

    The +++/--- header lines start with the same characters as real changes, so
    they have to be skipped or every diff reads as one line more than it is.
    """

    added = sum(1 for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines()
                  if line.startswith("-") and not line.startswith("---"))
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return " ".join(parts) or "no change"


def _patched(before: str, operation: str, content: str) -> str | None:
    """What a patch leaves behind, or None for operations with no text form.

    Mirrors the plugin's own operations. A patch that moves a block or sets a
    frontmatter value has no before-and-after text to lay side by side, so it
    gets a summary instead of a diff rather than a guessed one.
    """

    if operation == "replace":
        return content
    if operation == "append":
        return before + content
    if operation == "prepend":
        return content + before
    if operation == "delete":
        return ""
    return None


async def preview_write(
    tool_name: str, args: dict, read: Callable[..., Awaitable[str]] | None
) -> WritePreview:
    """Work out what one pending write would change.

    Never raises. A vault that cannot be read still has to be approvable, so
    every failure below degrades to a summary of what was asked for -- the
    decision is the point, and the diff is what makes it easier, not possible.
    """

    path = str(args.get("path", "") or "")
    content = str(args.get("content", "") or "")

    if read is None:
        return WritePreview(
            tool_name, path,
            summary=f"{tool_name} on {path}",
            note="obsidian is not connected, so there is nothing to compare against",
        )

    # a heading target reads back the same span the patch will consume. anything
    # else reads the whole note
    target = {key: args[key] for key in TARGET_ARGS if key in args}

    first_failure = ""
    try:
        before = await read(path, **target)
        missing = False
    except Exception as exc:
        # the note or the heading is not there yet, which is a normal thing to
        # ask for -- a write creates it. the text still goes in the note in case
        # the read failed for some other reason
        before, missing = "", True
        first_failure = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__

    if tool_name == "vault_write":
        after = content
    elif tool_name == "vault_append":
        after = before + content
    elif tool_name == "vault_delete":
        after = ""
    elif tool_name == "vault_patch":
        after = _patched(before, str(args.get("operation", "")), content)
        if after is None:
            return WritePreview(
                tool_name, path,
                summary=f"{args.get('operation')} on {path}",
                note=f"a {args.get('operation')} patch has no text to compare, "
                     f"so the arguments are all there is to go on: {args}",
            )
    else:
        return WritePreview(
            tool_name, path,
            summary=f"{tool_name} on {path}",
            note=f"no preview is built for this tool, so the arguments are all "
                 f"there is to go on: {args}",
        )

    diff = _diff(path, before, after)

    if diff is None:
        return WritePreview(
            tool_name, path,
            summary=f"{tool_name} on {path}",
            note="this would leave the note exactly as it is",
        )

    where = f"{path}{_where(target)}"
    if missing:
        return WritePreview(
            tool_name, where,
            summary=f"creates {where} · {_counts(diff)}",
            diff=diff,
            note=f"nothing is there to read yet ({first_failure})",
        )
    return WritePreview(tool_name, where, summary=f"{_counts(diff)} in {where}", diff=diff)


def _where(target: dict) -> str:
    """The heading path a patch aims at, written the way the vault shows it."""

    aim = target.get("target")
    if not aim:
        return ""
    if isinstance(aim, list):
        return " › " + " › ".join(str(part) for part in aim)
    return f" › {aim}"
