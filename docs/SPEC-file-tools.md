# SPEC: file tools and rendered artifacts

Issue #37. Branch: `feat/file-tools`.

The agent gets plain-disk file tools — read, write, patch, render — so it can produce
game artifacts (statblocks, dungeons, encounters) as PDF/PNG files the user actually
opens. No bash tool, and the Obsidian MCP integration goes away: the app becomes a
start-in-cwd agent like claude code or pi, and `templates/` and `artifacts/` folders
simply appear in whatever directory it is launched from.

The motivating flow, from the user's seat:

1. The user discusses an enemy, the agent researches it (search_books against the pdf
   library), stats and abilities get pinned in conversation.
1. The user says "print it". The agent knows the system from context (or asks once, in
   chat), writes the artifact, renders it, and names the PDF/PNG path.
1. The user reads the PDF, gives feedback, the agent patches and re-renders.

The user never sees an HTML file, never hears about templates unless they ask, and never
approves a write twice for the same job.

## Goals

- Creation is frictionless: one instruction produces a rendered file on disk.
- Every disk write passes the existing diff-preview approval gate (`llmtui/writes.py`),
  unchanged in spirit.
- No bash. The agent cannot run commands, only read/write/patch/render.
- The game system is fluid. Nothing in the tool layer knows what "AC" is; the agent
  adapts to homebrew because the structure lives in its prompt, not in a schema.

## Non-goals

- No slot-filling `make_document` tool with a generated schema. The template reuse
  convention (below) buys the consistency; a schema would fight the homebrew
  requirement.
- No auto-render inside `write_file`. A feedback loop of five patches would render five
  times. Rendering is an explicit tool call.
- No template marketplace, no template versioning, no per-system registries beyond what
  directory names already give.
- No memory persistence of the chosen system (conversation-only for now; obsidian-based
  memory may come later).

## Tools

Five tools total, all operating on paths relative to the working directory:

### read_file(path) -> string

Returns the file's text. Refuses paths outside the sandbox (below), refuses anything it
cannot decode as utf-8 with a message that names the encoding problem rather than
leaking a traceback. There is no size limit beyond a sanity cap (say 2 MB) — the agent
reads templates, not corpora.

### write_file(path, content) -> string

Creates or overwrites. Parent directories are created as needed. Held at the approval
gate; the preview shows the unified diff (whole file is new, so the "before" side is
empty — `preview_write` already degrades to that correctly when the read fails).

### patch_file(path, content, operation) -> string

The patch operations the vault tools had, minus the heading-target ones: `replace`
(whole file), `append`, `prepend`, `delete`. Whole-file replace is enough — statblock
HTML is one block, and heading addressing was an Obsidian note concept. Same approval
gate, same `preview_write` diff.

### render_file(path, format) -> string

`path` is an HTML file; `format` is `"pdf"` or `"png"`. Produces `path` with the
extension swapped (`vecna.html` -> `vecna.pdf`), overwriting a previous render of the
same artifact. Returns the rendered path plus a one-line size/page count so the agent
can report it naturally.

- pdf: `xhtml2pdf` (`pisa.CreatePDF`).
- png: render a temp PDF, then `pymupdf` (`page.get_pixmap(dpi=150).save()`). A
  statblock is one page; png always renders page 1.

`render_file` is NOT held at the approval gate — it only ever creates a file derived
from an already-approved one, and gating it would double-prompt every "print it" turn.
It is listed in the error-middleware tool set and the call limit does not count it.

## Sandboxing

Every tool resolves its `path` against the cwd and rejects anything that escapes it:
absolute paths, `..` traversal, and symlinks that resolve outside (via `Path.resolve()`
then `is_relative_to(cwd)`). The rejection message tells the agent to use a relative
path. This is one shared helper in the tools module, not per-tool logic.

Config gains no new keys. The cwd is the sandbox and the artifact root, which is what
makes the app portable to any folder.

## Rendering stack

- `xhtml2pdf` — HTML to PDF, pure python on ReportLab, pip-only. Supports flexbox, L4
  selectors, CSS custom properties, border-radius, page-break control, `@page`. Not a
  browser: the agent writes templates within its CSS subset, and the system prompt says
  so explicitly.
- `pymupdf` — PDF to PNG rasterization. Also the intended dependency for issue #8 (PDF
  attachments), one engine for both.

New deps in `pyproject.toml`: `xhtml2pdf`, `pymupdf`. Both ship wheels for macOS arm64
and linux x86_64.

## Conventions (system prompt, not tool logic)

These live in `prompts/system.txt`; nothing in Python enforces them.

- Templates live in `templates/{system}/statblock.html` etc., artifacts in
  `artifacts/{type}/name.html`, renders beside them. Both under the cwd.
- First artifact in a system: the agent improvises the template and saves it — writing
  the template first, then the artifact using it. The user's feedback on the rendered
  artifact is the approval; structural feedback patches the template, content feedback
  patches the artifact.
- Later artifacts in the same system: read the existing template first, reuse it.
  Consistency without a registry.
- The system is established conversationally. If the conversation has clearly
  established it (e.g. "my 5.5e game with Nimble damage rolls"), proceed; if not, the
  agent states a proposal in chat and asks once. Templates encode the homebrew as-is —
  if the table rolls damage directly, no attack-roll stats appear, because nothing in
  the pipeline validates mechanics.
- Always render after writing an artifact, and report the rendered path to the user,
  never the HTML path.

## What gets removed

The Obsidian MCP integration, entirely:

- `llmtui/tools/mcp.py` — the whole module. `tool_text` moves into the tools package
  (local tools already return strings, so only the TUI's use of it survives, if any).
- `WRITE_TOOLS` / approval gating in `llmtui/agent.py` — re-pointed at the new tool
  names (`write_file`, `patch_file`); the `HumanInTheLoopMiddleware` wiring itself
  stays.
- `MCPStatus`, the status-bar mcp segment, and `action_toggle_mcp` in `llmtui/tui.py`.
- `llmtui/config.py`: `OBSIDIAN_CERT_PATH`, `OBSIDIAN_API_KEY`, the `obsidian` section;
  matching keys out of `config/config.example.json` and the user's local `config.json`.
- `langchain-mcp-adapters` and `httpx` from `pyproject.toml` (httpx only if nothing else
  imports it).
- `resources/obsidian-local-rest-api.pem`.
- `llmtui/writes.py` keeps its role but loses the Obsidian vocabulary: the read is now
  injected as the sandboxed local read, `TARGET_ARGS` (heading targeting) goes away, and
  the tool-name branch narrows to the two write tools.

## Testing

- Sandboxing: traversal, absolute paths, symlink escape — one parametrized test against
  the shared helper.
- `preview_write` against plain files: create, append, prepend, replace, delete;
  missing-file create; the existing `tests/test_writes.py` ports with the injected read
  swapped for a real file read.
- `render_file`: golden smoke test — a tiny known HTML renders to a PDF with nonzero
  size and a PNG with nonzero size (no pixel comparison; the agent's own eyes on real
  artifacts are the layout QA).
- Existing suites (`test_tools.py`, `test_middleware.py`, `test_tui.py`) updated for the
  removed MCP surface.

## Work order

1. Genericize `llmtui/writes.py` + add `read_file` / `write_file` / `patch_file`,
   sandboxed, gated.
1. Add `render_file` + the two dependencies.
1. Conventions into `prompts/system.txt`.
1. Remove the Obsidian MCP surface (module, config, TUI status segment, deps).
1. Port the tests; run `make test`.
