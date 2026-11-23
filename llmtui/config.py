"""App configuration, loaded from config.json and validated at import.

A missing key, a mistyped value, or a whole missing section raises at import
with the offending key in the message — the same fail-fast contract the old
.env reader had, nothing travels far before it breaks.

Everything lives under one root directory: the platform config dir
(~/.config/llmtui on Linux and macOS, %APPDATA%\\llmtui on Windows),
config.json and models.json directly inside it, the databases under db/.
Relative paths in config.json resolve against that root, so `gm` works from
any directory and an installed binary finds its state the same way the dev
loop does.

config.json and models.json are gitignored (they can hold secrets); the
*.example.json files in the repo are the committed templates. Copy them into
the config dir to run.

The system prompts are packaged inside llmtui/prompts and used by default; a
system_prompt / self_name_prompt key in config.json points at an optional
override file that wins over the packaged one.

Everything provider/model related lives in models.json (see
llmtui/models_config.py, step 2 of SPEC-provider-migration.md) — nothing
here knows about providers.
"""

import importlib.resources
import json
import os
import sys
from pathlib import Path

# milvus-lite forks a server subprocess out of this process, and gRPC writes a
# fork-guard warning to stderr the first time that happens -- which lands mid-TUI,
# rendered straight into the prompt area. the warning is harmless and gRPC's C
# core reads this at import time, so it is set here, before anything imports it
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

# dev-only anchor for scripts/inspect_context.py's output; everything the app
# itself uses resolves against ROOT below
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# one root for config and data alike (issue #38): ~/.config/llmtui on Linux and
# macOS, %APPDATA%\llmtui on Windows. hand-rolled because platformdirs picks the
# macOS Library dir, which we deliberately do not want here
ROOT: Path = (
    Path.home() / ".config" / "llmtui"
    if sys.platform != "win32"
    else Path(os.getenv("APPDATA", "~\\AppData\\Roaming")) / "llmtui"
).expanduser()

# overridable so tests can point at their own config
CONFIG_PATH: Path = Path(os.getenv("LLMTUI_CONFIG", ROOT / "config.json"))


def _load() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"{CONFIG_PATH} not found. Copy config/config.example.json to"
            " the config dir and fill in your values."
        )
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{CONFIG_PATH} is not valid JSON: {e}") from None


_cfg = _load()


def _section(name: str) -> dict:
    section = _cfg.get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f"{CONFIG_PATH}: missing or malformed section {name!r}")
    return section


def _get(section: dict, key: str) -> str:
    value = section.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"{CONFIG_PATH}: {key!r} is missing or empty")
    return value.strip() if isinstance(value, str) else value


def _path(section: dict, key: str) -> str:
    """Absolute path as a string; relative values resolve against ROOT."""

    raw = Path(_get(section, key)).expanduser()
    return str(raw if raw.is_absolute() else ROOT / raw)


def _optional_path(section: dict, key: str) -> str | None:
    """Optional override path; None falls back to the packaged default."""

    value = section.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _path(section, key)


def _int(section: dict, key: str) -> int:
    value = _get(section, key)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{CONFIG_PATH}: {key!r} must be a whole number, got {value!r}"
        ) from None


# packaged prompts, always present; config keys below are optional overrides
_PACKAGED_PROMPTS = importlib.resources.files("llmtui") / "prompts"

# paths
_paths = _section("paths")
SQLITE_DB_PATH: str = _path(_paths, "sqlite_db")
# single live vector store + backup dir (issue #24, replaces the dev/prod split)
MILVUS_DB_PATH: str = _path(_paths, "milvus_db")
MILVUS_BACKUP_DIR: str = _path(_paths, "backup_dir")
SYSTEM_PROMPT_PATH: str = _optional_path(_paths, "system_prompt") or str(
    _PACKAGED_PROMPTS / "system.txt"
)
SELF_NAME_PROMPT_PATH: str = _optional_path(_paths, "self_name_prompt") or str(
    _PACKAGED_PROMPTS / "self_name.txt"
)
OBSIDIAN_CERT_PATH: str = _path(_paths, "obsidian_cert")

# where the agent's artifacts live. optional: unset means the first-boot
# workspace prompt in the TUI asks once and writes it back (issue #38)
WORKSPACE_DIR: str | None = (
    str(Path(_cfg["workspace"]).expanduser()) if "workspace" in _cfg else None
)


def save_workspace(path: str) -> None:
    """Persist the workspace choice to config.json and the live module state."""

    global WORKSPACE_DIR
    WORKSPACE_DIR = path
    _cfg["workspace"] = path
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(_cfg, f, indent=2)


# milvus index configuration, shared by ingest and search so the schemas match
_milvus = _section("milvus")
DENSE_METRIC_TYPE: str = _get(_milvus, "dense_metric_type")
DENSE_INDEX_TYPE: str = _get(_milvus, "dense_index_type")
SPARSE_METRIC_TYPE: str = _get(_milvus, "sparse_metric_type")
SPARSE_INDEX_TYPE: str = _get(_milvus, "sparse_index_type")

# fixed by langchain_milvus, not user-configurable: the name it gives its
# collection, and the property key ingest.py uses to record which embeddings
# entry built the store. shared by ingest and search so the two can't drift
MILVUS_COLLECTION_NAME: str = "LangChainCollection"
EMBEDDING_MODEL_KEY: str = "embedding_model"

MILVUS_GRPC_OPTIONS: dict = {
    "grpc.keepalive_permit_without_calls": False,
    "grpc.keepalive_time_ms": 300_000,
}

# agent behaviour
_agent = _section("agent")
# searches allowed per user turn
MAX_TOOL_CALLS: int = _int(_agent, "max_tool_calls")
DISPLAY_THINKING: bool = bool(_get(_agent, "display_thinking"))

# obsidian / mcp
_obsidian = _section("obsidian")
OBSIDIAN_API_KEY: str = _get(_obsidian, "api_key")

# the pdf library itself. citations resolve against this directory; ingest.py
# reads the same directory from its argv, so the two must stay in sync. Milvus
# metadata holds basenames only, so same-named PDFs in different subdirs collide
_library = _section("library")
LIBRARY_DIR: str = _path(_library, "pdf_dir")

# where inspect_context.py leaves its rendered pages. not in config.json
# because nothing else reads it and it is not a knob worth turning
CONTEXT_DUMP_DIR: str = str(PROJECT_ROOT / "resources" / "context_inspection")
