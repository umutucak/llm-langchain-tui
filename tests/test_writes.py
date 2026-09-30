"""What the approval card is told a pending write would do."""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmtui.writes import preview_write

results = {}


def check(label, conditions):
    ok = True
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for desc, passed in conditions:
        print(f"    {'PASS' if passed else 'FAIL'}  {desc}")
        ok = ok and passed
    results[label] = ok


NOTE = "# Caley\nshe is kind\nand wise\n"
SECTION = "she is kind\nand wise\n"


async def read(path, **target):
    """A vault of exactly one note, with one readable heading in it."""

    if path != "note.md":
        raise RuntimeError(f"File not found: {path}")
    if target.get("target") == ["Caley"]:
        return SECTION
    if target.get("target"):
        raise RuntimeError("Target not found")
    return NOTE


async def broken_read(path, **target):
    raise RuntimeError("connection lost")


def get(coro):
    return asyncio.run(coro)


# ---- 1. the three content shapes ----
overwrite = get(preview_write(
    "vault_write", {"path": "note.md", "content": "# Caley\nshe is wary\n"}, read))
append = get(preview_write(
    "vault_append", {"path": "note.md", "content": "and wary of Grexes\n"}, read))
delete = get(preview_write("vault_delete", {"path": "note.md"}, read))

check("1  a whole-note write is diffed against the note", [
    ("the replaced lines are counted", overwrite.summary == "+1 -2 in note.md"),
    ("a diff is built", overwrite.diff is not None),
    ("the new line is in it", "+she is wary" in (overwrite.diff or "")),
    ("the old line is marked gone", "-she is kind" in (overwrite.diff or "")),
    ("an append only adds", append.summary == "+1 in note.md"),
    ("nothing is marked removed in an append",
     not any(line.startswith("-") and not line.startswith("---")
             for line in (append.diff or "").splitlines())),
    ("a delete removes the whole note", delete.summary == "-3 in note.md"),
])

# ---- 2. a patch is diffed against its own target, not the whole note ----
patch = get(preview_write("vault_patch", {
    "path": "note.md", "targetType": "heading", "target": ["Caley"],
    "operation": "append", "content": "and wary of Grexes\n",
}, read))
replace = get(preview_write("vault_patch", {
    "path": "note.md", "targetType": "heading", "target": ["Caley"],
    "operation": "replace", "content": "she is wary\n",
}, read))
# the plugin can move blocks and set frontmatter values, and neither has a
# before-and-after text to lay side by side
moved = get(preview_write("vault_patch", {
    "path": "note.md", "targetType": "block", "target": "abc",
    "operation": "move", "destination": {"path": "other.md"},
}, read))

check("2  a patch is previewed against the span it will consume", [
    ("the heading path is named", patch.path == "note.md › Caley"),
    ("only the section is diffed, not the note",
     "# Caley" not in (patch.diff or "")),
    ("appending to a section adds one line", patch.summary.startswith("+1")),
    ("replacing a section counts both sides", replace.summary.startswith("+1 -2")),
    ("an operation with no text form gets no invented diff", moved.diff is None),
    ("and says why", "no text to compare" in (moved.note or "")),
])

# ---- 3. writes that create something ----
fresh = get(preview_write("vault_write", {"path": "new.md", "content": "fresh\n"}, read))
section = get(preview_write("vault_patch", {
    "path": "note.md", "targetType": "heading", "target": ["Grexes"],
    "operation": "replace", "content": "a hunter\n",
}, read))

check("3  a write to something that is not there yet reads as a creation", [
    ("a missing file is a creation, not an error", fresh.summary.startswith("creates")),
    ("it still gets a diff", fresh.diff is not None),
    ("it says nothing was there", "nothing is there to read yet" in (fresh.note or "")),
    ("a missing heading is a creation too", section.summary.startswith("creates")),
])

# ---- 4. the preview never blocks the decision ----
# the approval is the point, the diff only makes it easier. a vault that cannot
# be read has to leave a card that can still be answered
no_server = get(preview_write("vault_write", {"path": "note.md", "content": "x"}, None))
unreadable = get(preview_write("vault_write", {"path": "note.md", "content": "x"}, broken_read))
unchanged = get(preview_write("vault_write", {"path": "note.md", "content": NOTE}, read))
unknown = get(preview_write("vault_move", {"path": "note.md", "destination": "b.md"}, read))

check("4  a preview that cannot be built still describes the write", [
    ("no vault reader at all is survivable", no_server.diff is None),
    ("and says why", "not connected" in (no_server.note or "")),
    ("a read that raises does not raise here",
     unreadable.summary.startswith("creates")),
    ("the failure is quoted", "connection lost" in (unreadable.note or "")),
    ("a write that changes nothing says so",
     "exactly as it is" in (unchanged.note or "")),
    ("an unrecognised tool is described, not guessed at",
     unknown.diff is None and "no preview is built" in (unknown.note or "")),
    ("every path names the tool it came from",
     all(p.tool for p in (no_server, unreadable, unchanged, unknown))),
])

print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
for label, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'}   {label}")
print(f"\n{sum(results.values())}/{len(results)} cases passed")
