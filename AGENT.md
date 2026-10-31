# AGENT.md — guidance for agents working in this repo

A LangChain/LangGraph practice project: a self-hosted chat agent (Ollama local models)
behind a Textual TUI, with RAG over a PDF library and Obsidian vault tools over MCP.

## Git protocol (binding on agents)

- **Never commit or push autonomously.** Make changes in the working tree, show the
  diff/summary, and wait. The user commits and pushes explicitly — or instructs the
  agent to do so in that message. No push-per-fix chains, no "I'll just push this one."
  This includes PRs and workflow monitoring: hand the user the commands and stop.

## Run it (Python 3.12)

```bash
uv sync                                     # creates .venv with dependencies
uv run python app.py                        # the chat TUI
uv run python scripts/ingest.py <pdf_dir>   # build the vector store
uv run python scripts/inspect_context.py    # dump a thread's context as HTML
```

## CI, formatting, and the pre-commit hook

- All commands go through uv; dev tools (ruff, mdformat) install via
  `uv sync --group dev`.

- `make format` reformats everything. `make lint` and `make test` are what CI runs
  (`.github/workflows/ci.yml` is just `make lint`, then `make test`), so hook, CI, and
  local can't drift: lint is ruff format --check, ruff check, mdformat --check --wrap
  88; test is pytest over `tests/`.

- A checked-in pre-commit hook runs `make lint` on every commit and blocks on failure.
  It needs one-time activation per clone: `git config core.hooksPath hooks`.

- Requires a running model server, described in `config/models.json` (copy
  `config/models.example.json`). Any OpenAI-compatible endpoint works — Ollama,
  llama.cpp server, vLLM, LM Studio, or a remote API (OpenAI-compat dialect only).

- All settings live in `config/config.json` (copy `config/config.example.json`). Every
  key is read, type-checked, and path-resolved against the project root in
  `llmtui/config.py` — a bad file fails at import with the offending key. Both files are
  gitignored (they may hold secrets); model/provider configuration lives in
  `config/models.json` (see `llmtui/models_config.py`), loaded at call time and re-read
  every time `/model` opens. `$NAME` in any string interpolates from the environment.
  Add new settings there, never via raw `os.getenv` elsewhere.

- Prompts are plain files under `prompts/` (`system.txt`, `self_name.txt`), pointed at
  by `config/config.json` paths.

## Architecture

- `app.py` — entry point. Wires model + checkpointer + MCP tools + TUI together.
- `llmtui/config.py` — app-level settings from `config/config.json`: fail-fast JSON
  loader, paths resolved against the project root. Knows nothing about providers.
- `llmtui/models_config.py` — providers and models from `config/models.json` (pi-harness
  shape: `providers` map, per-model `samplingParams`). Loaded at call time, never at
  import, so `/model` re-reads the file live. One dialect (`openai-completions`);
  sampling params are merged into the request body verbatim — core never interprets
  them. Reserved request fields (model, messages, tools, ...) cannot be overridden.
  `fetch_context_window` is the live ollama fallback for the context meter.
- `llmtui/agent.py` — the agent: model construction, tool list, and the LangChain
  `create_agent` middleware stack (approval gate, tool-call limit, tool-error handling,
  malformed-call repair). Ordering of middleware matters; the comments in `build_agent`
  explain why.
- `llmtui/tools/` — built-in tools (`search_books`, RAG over Milvus) and the MCP client
  layer (`mcp.py`: server config, tool whitelist, write-tool classification, live
  `MCPStatus`, self-signed-cert httpx factory).
- `llmtui/middleware/` — custom LangChain middleware: `repair.py` (after_model hook that
  recovers malformed tool calls) and `errors.py` (per-tool error handlers that feed
  failures back to the model instead of killing the run).
- `llmtui/tui.py` — Textual frontend only. Streams via `astream_events(version="v3")`;
  it consumes events and never touches agent/tool logic. `/model` hot-swaps providers: a
  new graph over the same checkpointer memory resumes the same thread — the graph holds
  no history.
- `llmtui/writes.py` — diff previews for pending Obsidian writes (pure logic, view
  injected, no UI or MCP imports).
- `llmtui/sessions.py`, `llmtui/naming.py` — session titles in a `sessions` table beside
  LangGraph's checkpoint tables; model names its own threads.
- `scripts/` — async support tasks run straight from a shell. They insert the project
  root onto `sys.path` themselves before importing `llmtui`.
  - `ingest.py`: PDFs → Milvus live db (hybrid dense+BM25). A preexisting store is moved
    to `db/backups/` (last-1 kept) before rebuilding; `restore.py` swaps a backup back
    into place.
  - `inspect_context.py`: renders a checkpointed thread as a self-contained HTML page in
    `resources/context_inspection/`.

**Boundary rule:** LangChain implementation code (models, tools, middleware, graph
assembly) stays inside `llmtui/agent.py`, `llmtui/tools/`, and `llmtui/middleware/`. The
TUI, writes, sessions, and naming modules consume what the run emits — don't leak
LangChain concerns into them.

## Conventions & gotchas

- Text style (enforced by tooling: `ruff format` line-length 88 for code,
  `mdformat --wrap 88` for .md — one width everywhere; .json is hand-formatted, 4-space
  indent — mdformat's json style is too flat for config files): wrap prose and comments
  at that width, breaking at clause/punctuation boundaries. When an edit touches a
  sentence inside a wrapped paragraph, re-wrap the whole paragraph rather than splicing
  lines — a mid-sentence splice is exactly what mdformat then has to undo.
- The codebase is heavily commented with reasoning about _why_; read and preserve that
  style. Don't strip explanatory comments.
- The checkpointer must be `AsyncSqliteSaver` (the sync saver raises on async calls);
  `sessions.py` opens the same file with WAL for the sync side.
- Config errors are deliberately self-describing — every loader failure names the file,
  provider, and key. Read the message before guessing.
- MCP failures are transport-level vs server-reported — `mcp.py` and
  `middleware/errors.py` distinguish and handle these deliberately.
- Tests live in `tests/`, all of them pytest modules (`make test`). None of them need a
  model server, an ingested store, or `config/models.json`: the TUI suite drives Textual
  headlessly with a scripted model, and the vector store is built lazily, on the first
  search. Run them before handing changes back.
