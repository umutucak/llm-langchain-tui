"""Update plumbing for the packaged `gm` binary (issue #38).

The release channel is git tags: `gm update` reinstalls from the latest tag
via uv (which the user never sees otherwise), and the TUI checks at most once
a day whether a newer tag exists, staying silent on any failure.
"""

import json
import re
import subprocess
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from llmtui import config

REPO = "umutucak/llm-langchain-tui"
UPDATE_HINT = "new version available — run gm update"

# timestamp of the last startup check, next to the config file
_CHECK_FILE: Path = config.ROOT / "last_update_check"
_CHECK_INTERVAL = 24 * 60 * 60


def installed_version() -> str:
    try:
        return version("llmtui")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def latest_tag() -> tuple[str, str | None]:
    """(status, tag). status is "ok" | "empty" | "error"; the tag is the
    repo's newest one (GitHub returns them newest-first), None when there
    isn't one or the API call failed.
    """

    try:
        with urllib.request.urlopen(
            f"https://api.github.com/repos/{REPO}/tags", timeout=3
        ) as r:
            tags = json.load(r)
        return "ok", tags[0]["name"] if tags else None
    except Exception:
        return "error", None


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v))


def is_newer(latest: str, installed: str) -> bool:
    return _key(latest) > _key(installed)


def check_for_update() -> str | None:
    """Once a day, compare the installed version against the latest tag.

    Returns the one-line hint for the transcript, or None when there is
    nothing to say: no newer tag, or any failure (offline, rate-limited,
    timeout) — the check must never make noise or block.
    """

    now = time.time()
    try:
        if now - _CHECK_FILE.stat().st_mtime < _CHECK_INTERVAL:
            return None
    except OSError:
        pass

    _, tag = latest_tag()
    try:
        _CHECK_FILE.write_text(str(now))
    except OSError:
        pass  # a read-only config dir must not break the TUI over a hint
    if tag and is_newer(tag, installed_version()):
        return UPDATE_HINT
    return None


def update(nightly: bool = False) -> int:
    """`gm update`: reinstall from the latest tag; with --nightly, from the
    latest commit on main. Returns an exit code."""

    if nightly:
        print("updating gm to the latest commit on main ...")
        ref = "main"
    else:
        status, tag = latest_tag()
        if tag is None:
            print(
                "gm update: no tagged release found on GitHub"
                if status == "empty"
                else "gm update: could not reach GitHub to find the latest version"
            )
            return 1
        print(f"updating gm to {tag} ...")
        ref = tag
    try:
        return subprocess.run(
            [
                "uv",
                "tool",
                "install",
                "--reinstall",
                f"git+https://github.com/{REPO}.git@{ref}",
            ],
            check=False,
        ).returncode
    except FileNotFoundError:
        print("gm update: uv was not found on PATH; install uv and retry")
        return 1
