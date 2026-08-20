"""Every setting read from .env, resolved and type-checked in one place.

os.getenv returns None for a key that is missing or misspelled, and that None
then travels a long way before something breaks in a message that does not
name the key. Reading them all here means a bad .env fails at import, saying
which key.

Relative paths in .env are resolved against the project root, so app.py and
anything in scripts/ find db/ and prompts/ regardless of the directory they
were launched from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def _text(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise RuntimeError(f"{key} is missing or empty in {PROJECT_ROOT / '.env'}")
    return value.strip()


def _path(key: str) -> str:
    """Absolute path as a string, the type os.getenv used to hand back."""

    return str(PROJECT_ROOT / _text(key))


def _int(key: str) -> int:
    try:
        return int(_text(key))
    except ValueError:
        raise RuntimeError(f"{key} must be a whole number, got {_text(key)!r}") from None


def _float(key: str) -> float:
    try:
        return float(_text(key))
    except ValueError:
        raise RuntimeError(f"{key} must be a number, got {_text(key)!r}") from None


def _flag(key: str) -> bool:
    """A 0/1 switch. Anything else is a typo worth failing on."""

    value = _text(key)
    if value not in ("0", "1"):
        raise RuntimeError(f"{key} must be 0 or 1, got {value!r}")
    return value == "1"


# models
MODEL: str = _text("MODEL")
EMBEDDING_MODEL: str = _text("EMBEDDING_MODEL")

# stores
SQLITE_DB_PATH: str = _path("SQLITE_DB_PATH")
MILVUS_DEV_DB_PATH: str = _path("MILVUS_DEV_DB_PATH")
MILVUS_PROD_DB_PATH: str = _path("MILVUS_PROD_DB_PATH")

# milvus index configuration, shared by ingest and search so the schemas match
DENSE_METRIC_TYPE: str = _text("DENSE_METRIC_TYPE")
DENSE_INDEX_TYPE: str = _text("DENSE_INDEX_TYPE")
SPARSE_METRIC_TYPE: str = _text("SPARSE_METRIC_TYPE")
SPARSE_INDEX_TYPE: str = _text("SPARSE_INDEX_TYPE")

# prompts
SYSTEM_PROMPT_PATH: str = _path("SYSTEM_PROMPT_PATH")
SELF_NAME_PROMPT_PATH: str = _path("SELF_NAME_PROMPT_PATH")

# agent behaviour
# searches allowed per user turn
MAX_TOOL_CALLS: int = _int("MAX_TOOL_CALLS")
CONTEXT_SIZE: int = _int("CONTEXT_SIZE")
DISPLAY_THINKING: bool = _flag("DISPLAY_THINKING")

# sampling
IS_REASONING: bool = _flag("IS_REASONING")
TEMPERATURE: float = _float("TEMPERATURE")
TOP_K: int = _int("TOP_K")
TOP_P: float = _float("TOP_P")
REPETITION_PENALTY: float = _float("REPETITION_PENALTY")

# where inspect_context.py leaves its rendered pages. not in .env because
# nothing else reads it and it is not a knob worth turning
CONTEXT_DUMP_DIR: str = str(PROJECT_ROOT / "resources" / "context_inspection")
