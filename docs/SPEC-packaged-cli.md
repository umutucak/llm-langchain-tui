# SPEC: packaged CLI binary, config-dir state, first-boot workspace

Issue #38. Branch: `feat/packaged-cli-binary`.

Typing `gm` in any terminal, from any directory, starts the GM assistant. No repo
directory, no `uv run python app.py` ritual. Configuration lives in the platform config
dir; everything the assistant produces lives in a workspace directory the user picks on
first boot.

#37 (file tools and rendered artifacts) depends on this issue: its `templates/` and
`artifacts/` conventions and its write sandbox target the workspace defined here, and
its Obsidian removal rides on the config relocation done here.

## Goal

`gm` anywhere, every time. First run on a fresh machine: install, run `gm`, answer one
question about where artifacts go, use the app.

## Non-goals

- No subcommands. `gm` is the whole interface; `/model`, `/load`, `/new` and friends
  stay slash commands inside the TUI. (A `gm --help` comes free from argparse-like
  handling only if it costs nothing.)
- No per-campaign folders or campaign registry. The workspace is flat: `templates/`,
  `artifacts/`, and whatever the agent creates under them. Campaign separation, if
  wanted later, is thread metadata, not paths.
- No daemon, no autostart, no shell integration.

## The binary

### Entry point

`pyproject.toml` gains:

```toml
[project.scripts]
gm = "llmtui.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

and `package = false` under `[tool.uv]` is removed. The flat layout already works for
hatchling: `llmtui/` is the import package and gets packaged; `scripts/`, `tests/`,
`db/`, `resources/`, `prompts/` stay out of the wheel — they are either dev-only or
replaced by config-dir content (see below).

`app.py` becomes a two-line shim kept for the dev loop (`uv run python app.py`), or is
deleted outright in favour of `uv run gm`. Either is acceptable; the shim is the smaller
diff. `main()` moves to `llmtui/cli.py`, unchanged in body.

### Install and update

```sh
uv tool install .          # from the repo
uv tool upgrade llmtui     # after pulling
```

`uv tool` gives `gm` an isolated venv on the PATH. This is the same mechanism pi, ruff
and pipx use; nothing novel to build.

### What must change for this to work

The only structural change: **paths stop resolving against `PROJECT_ROOT`.** A `gm`
installed in an isolated venv has no repo beside it. `prompts/` and `resources/`
therefore move into the package (hatchling `force-include` or a package-data entry), so
`SYSTEM_PROMPT_PATH` resolves against `importlib.resources` on `llmtui` instead of the
filesystem root. The user's own system prompt, if they have customized one, becomes an
optional override path in the config dir that wins over the packaged default.

## Config: platform dirs

`llmtui/config.py` switches from `PROJECT_ROOT`-relative resolution to `platformdirs`:

| constant | today | after | |---|---|---| | config.json, models.json |
`PROJECT_ROOT/config/` | `platformdirs.user_config_dir("llmtui")` | | sqlite, milvus,
backups | `PROJECT_ROOT/db/` | `platformdirs.user_data_dir("llmtui")` | | system prompt
| `PROJECT_ROOT/prompts/` | packaged default, overridable from config dir | | pdf
library | config key, repo-relative | config key, absolute or `~`-relative | |
artifacts, templates | — (new in #37) | workspace (below) |

- `platformdirs.user_config_dir("llmtui")` → `~/.config/llmtui` (Linux),
  `~/Library/Application Support/llmtui` (macOS), `%APPDATA%\llmtui` (Windows).
- Existing keys, validation behaviour (fail at import, offending key named) and the
  `LLMTUI_CONFIG` env override are unchanged — only the base directory moves.
- `models.json` sits next to `config.json` in the config dir; the README's setup step
  becomes "copy the examples into the config dir" (paths printed by the error message,
  same as now).
- Migration of the user's current `config/config.json` is a one-time manual copy, noted
  in the README; nothing auto-migrates. There is exactly one machine to migrate and the
  user is its operator.

## First boot: the workspace

On startup, before the TUI renders, `main()` checks config for `workspace` (an absolute
path).

- Missing or no-longer-exists → prompt, inside the TUI, before the chat screen: propose
  `~/Documents/gm-assistant`, accept a custom path, Enter confirms, Esc quits. The
  choice is written to config.json and never asked again.
- The prompt creates the skeleton on confirm: the workspace directory plus `templates/`
  and `artifacts/` (children of those two are created lazily by the agent per #37's
  conventions).
- If the workspace directory exists but is empty, that is not an error and not a
  re-prompt — an empty workspace is a valid workspace.

The workspace path is exported as `llmtui.config.WORKSPACE_DIR` and is the single knob
#37 reads. Nothing else in the codebase learns what a workspace is in this issue.

## Threads and checkpoints

One sqlite file in the data dir, one milvus db beside it — not per-directory. Threads
are already uuid-keyed (`agent_thread_config`); `/load` lists them as it does today.
Per-campaign separation is deliberately deferred: if it is wanted, it becomes a label on
threads, not a directory layout.

## Testing

- Config dir resolution: platformdirs patched per platform; `LLMTUI_CONFIG` override
  still wins; missing keys still raise at import with the offending key named.
- First boot: no workspace → prompt fires once and persists the choice; workspace
  configured and present → no prompt; configured but deleted → re-prompt, not a crash;
  empty-but-existing workspace → no prompt.
- Entry point: after `uv sync`, `uv run gm` starts the TUI (smoke-level; the existing
  TUI harness covers behaviour).
- Packaging: `uv build` produces a wheel whose `llmtui/` contains the packaged prompt;
  `uv tool install` from that wheel works in a clean venv.

## Work order

1. `pyproject.toml`: build-system, `[project.scripts]`, package data; `llmtui/cli.py`
   with `main()` moved from `app.py`.
1. `config.py`: platformdirs base dirs, packaged system prompt with config-dir override,
   `WORKSPACE_DIR` config key.
1. First-boot workspace prompt in `main()`/TUI.
1. README: install/update, config-dir paths, first-boot behaviour.
1. Tests per above; `make test`, then `uv tool install .` and a manual `gm`-from-`/tmp`
   run before opening the PR.
