# llm-tui

Langchain practice: a chat agent behind a Textual TUI, running on any OpenAI-compatible
model endpoint — Ollama locally, or a remote API — on python 3.12.

Dependencies are managed with [uv](https://docs.astral.sh/uv/). After cloning:

```sh
uv sync                    # creates .venv, installs the pinned deps
```

## install as gm

```sh
uv tool install .          # from the repo; puts `gm` on the PATH
gm                         # the chat TUI, from any directory
gm update                  # update to the latest tagged release
gm update --nightly        # update to the latest commit on main
gm --version
```

Updates after that are always `gm update` (or `uv tool upgrade llmtui`); re-install with
`uv tool install .` while developing.

## configuration

Everything lives under one root: `~/.config/llmtui/` (Linux and macOS) or
`%APPDATA%\llmtui\` (Windows):

```text
~/.config/llmtui/
├── config.json          # app settings
├── models.json          # providers + models
├── db/                  # sqlite checkpoints, milvus store, backups
└── last_update_check    # written by gm's daily update check
```

Copy the committed templates there on first run:

```sh
mkdir -p ~/.config/llmtui
cp config/config.example.json ~/.config/llmtui/config.json
cp config/models.example.json ~/.config/llmtui/models.json
```

(Coming from an older checkout: your `config/config.json` and `config/models.json`
migrate by hand — copy them across; db files go into `db/`.)

Paths inside config.json may be relative (resolved against the root) or absolute; the
system prompts are packaged inside the app and need no config entry, but a
`system_prompt` / `self_name_prompt` path in config.json overrides the packaged one. A
bad file fails at import, naming the offending key.

config.json — app settings: sqlite/milvus db paths, milvus index options, the obsidian
api key and cert, agent behaviour (`max_tool_calls`, `display_thinking`), and the
`workspace` path (asked once on first boot; everything the agent produces —
`templates/`, `artifacts/`, `memory/` — lives there, default
`~/Documents/gm-assistant`).

`config/models.json` in the config dir — providers and models, pi-harness style. Any
OpenAI-compatible endpoint (ollama, llama.cpp server, vLLM, LM Studio, remote apis) is
just a JSON entry; no code changes. Sampling params (`temperature`, `top_k`, ...) are
per model and sent through to the server verbatim. `"$ENV_VAR"` in any string reads from
the environment. `/model` in the TUI switches between entries; the file is re-read every
time the picker opens, so edits land without a restart.

## models and providers

ollama locally: pull models through `ollama run` and point a provider entry at
`http://localhost:11434/v1`. the context meter picks up the model's real context length
from ollama automatically; other servers should declare `contextWindow` on the entry.

remote: entries point at whatever speaks the openai chat-completions dialect. reasoning
models get `reasoning_effort` from their `reasoning` field.

## running it

```sh
uv run gm                                  # the chat TUI (dev path; same main())
uv run python app.py                       # equivalent dev shim
uv run python scripts/ingest.py <pdf_dir>   # build the vector store
uv run python scripts/inspect_context.py    # dump a thread's context as html
```

slash commands in the TUI: `/help`, `/model`, `/name`, `/load`, `/new`, `/delete`,
`/settings` (settings list — keys on the left, values on the right; enter edits the
selected one).

## rag / ingest

`ingest.py` builds a hybrid (dense + bm25) milvus index from a directory of PDFs into
the live store at `db/milvus.db`. If a store already exists, it is moved to
`db/backups/` (only the last backup is kept) before the rebuild — no manual copying.
`restore.py` swaps a backup back into place. run this from the root of the project:

embeddings are provider-agnostic too: the `embeddings` section of `config/models.json`
(names a `baseUrl`, `apiKey`, and `model`) — any `/v1/embeddings`-compatible server
works: ollama, TensorX, OpenRouter, e.g. `qwen/Qwen3-Embedding-8B`.
