"""Config-dir resolution, packaged prompts and the first-boot workspace (issue #38)."""

import asyncio
import importlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    """A temp config dir with the example config copied in; LLMTUI_CONFIG set.

    llmtui.config reads the env at import, so tests that need a fresh import
    reload the module after the patch.
    """

    example = Path(__file__).parents[1] / "config" / "config.example.json"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(json.loads(example.read_text())))
    monkeypatch.setenv("LLMTUI_CONFIG", str(cfg))
    return tmp_path


@pytest.fixture
def cfg(config_env):
    import llmtui.config

    return importlib.reload(llmtui.config)


# ---------- config resolution ----------


def test_root_is_one_dir(cfg):
    # single root on every platform: ~/.config/llmtui on Linux and macOS
    assert cfg.ROOT.name == "llmtui"
    assert cfg.CONFIG_PATH == Path(os.environ["LLMTUI_CONFIG"])


def test_missing_config_names_the_path(monkeypatch, tmp_path):
    import llmtui.config

    monkeypatch.setenv("LLMTUI_CONFIG", str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError, match="nope.json"):
        importlib.reload(llmtui.config)
    monkeypatch.delenv("LLMTUI_CONFIG")
    importlib.reload(llmtui.config)  # restore for later tests


def test_relative_paths_resolve_against_root(cfg):
    assert cfg.SQLITE_DB_PATH == str(cfg.ROOT / "db" / "checkpoints.sqlite")
    assert cfg.MILVUS_BACKUP_DIR == str(cfg.ROOT / "db" / "backups")


def test_absolute_path_in_config_stays_absolute(cfg, config_env):
    custom = str(config_env / "custom.db")
    data = json.loads(Path(cfg.CONFIG_PATH).read_text())
    data["paths"]["sqlite_db"] = custom
    with open(cfg.CONFIG_PATH, "w") as f:
        json.dump(data, f)

    importlib.reload(cfg)
    assert cfg.SQLITE_DB_PATH == custom


def test_packaged_prompt_default(cfg):
    # packaged prompts always resolve, without any config key
    assert "system.txt" in cfg.SYSTEM_PROMPT_PATH
    assert Path(cfg.SYSTEM_PROMPT_PATH).read_text()


# ---------- workspace ----------


def test_workspace_unset_then_saved(cfg, config_env):
    assert cfg.WORKSPACE_DIR is None
    ws = config_env / "ws"
    cfg.save_workspace(str(ws))
    assert cfg.WORKSPACE_DIR == str(ws)
    assert json.loads(Path(cfg.CONFIG_PATH).read_text())["workspace"] == str(ws)


def _prompt_returns(monkeypatch, module, path, record=None):
    class FakePrompt(module.WorkspacePrompt):
        async def run_async(self):
            if record is not None:
                record["ran"] = True
            # the real prompt creates the skeleton on confirm; mirror that
            for child in ("templates", "artifacts", "memory"):
                (path / child).mkdir(parents=True, exist_ok=True)
            return str(path)

    monkeypatch.setattr(module, "WorkspacePrompt", FakePrompt)


def test_ensure_workspace_persists_and_creates_skeleton(cfg, config_env, monkeypatch):
    import llmtui.workspace as w

    ws = config_env / "ws"
    seen = {}
    _prompt_returns(monkeypatch, w, ws, seen)

    # first boot: prompt fires, skeleton is created, choice persisted
    assert asyncio.run(w.ensure_workspace()) == str(ws)
    assert seen
    assert sorted(os.listdir(ws)) == ["artifacts", "memory", "templates"]
    assert json.loads(Path(cfg.CONFIG_PATH).read_text())["workspace"] == str(ws)

    # second boot: configured and present -> no prompt
    seen.clear()
    assert asyncio.run(w.ensure_workspace()) == str(ws)
    assert not seen


def test_ensure_workspace_reprompts_when_deleted(cfg, config_env, monkeypatch):
    import llmtui.workspace as w

    ws = config_env / "gone"
    cfg.save_workspace(str(ws))  # configured...
    # ...but the directory no longer exists -> prompt fires again
    new = config_env / "new-ws"
    _prompt_returns(monkeypatch, w, new)

    assert asyncio.run(w.ensure_workspace()) == str(new)


def test_ensure_workspace_empty_dir_is_valid(cfg, config_env, monkeypatch):
    import llmtui.workspace as w

    ws = config_env / "empty"
    ws.mkdir()
    cfg.save_workspace(str(ws))

    class Boom(w.WorkspacePrompt):
        async def run_async(self):
            raise AssertionError("prompt must not fire for an existing workspace")

    monkeypatch.setattr(w, "WorkspacePrompt", Boom)
    assert asyncio.run(w.ensure_workspace()) == str(ws)


# ---------- workspace migration ----------


def _make_ws(root: Path) -> Path:
    ws = root / "ws"
    for child in ("templates", "artifacts", "memory/deep"):
        (ws / child).mkdir(parents=True)
    (ws / "memory/notes.md").write_text("hello dmg")
    (ws / "memory/deep/nested.bin").write_bytes(os.urandom(4096))
    return ws


def test_migrate_happy_path_then_old_removed(cfg, tmp_path):
    import llmtui.workspace as w

    src = _make_ws(tmp_path)
    payload = (src / "memory/deep/nested.bin").read_bytes()
    dst = tmp_path / "elsewhere"

    w.migrate_workspace(str(src), str(dst))

    assert not src.exists()
    assert (dst / "memory/notes.md").read_text() == "hello dmg"
    assert (dst / "memory/deep/nested.bin").read_bytes() == payload


def test_migrate_refuses_nonempty_target_and_nesting(cfg, tmp_path):
    import llmtui.workspace as w

    src = _make_ws(tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "user-file.txt").write_text("precious")
    with pytest.raises(RuntimeError, match="not empty"):
        w.migrate_workspace(str(src), str(busy))
    assert (busy / "user-file.txt").read_text() == "precious"
    assert src.exists()

    with pytest.raises(RuntimeError, match="nests inside"):
        w.migrate_workspace(str(src), str(src / "sub"))


def test_migrate_rolls_back_on_verification_failure(cfg, tmp_path, monkeypatch):
    import llmtui.workspace as w

    src = _make_ws(tmp_path)
    dst = tmp_path / "elsewhere"
    monkeypatch.setattr(w.filecmp, "cmp", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="cancelled, nothing changed"):
        w.migrate_workspace(str(src), str(dst))

    # pre-migration state restored: target gone, source untouched
    assert not dst.exists()
    assert src.exists()
    assert (src / "memory/notes.md").read_text() == "hello dmg"


def test_migrate_into_empty_existing_target(cfg, tmp_path):
    import llmtui.workspace as w

    src = _make_ws(tmp_path)
    dst = tmp_path / "empty"
    dst.mkdir()
    w.migrate_workspace(str(src), str(dst))
    assert not src.exists()
    assert dst.exists()
    assert (dst / "memory/notes.md").read_text() == "hello dmg"


# ---------- updates ----------


def test_update_check_once_a_day(cfg, tmp_path, monkeypatch):
    import llmtui.updates as u

    monkeypatch.setattr(u, "_CHECK_FILE", tmp_path / "stamp")
    monkeypatch.setattr(u, "latest_tag", lambda: ("ok", "v9.9.9"))
    assert u.check_for_update() == u.UPDATE_HINT
    assert (tmp_path / "stamp").exists()

    # within the interval: silent, tag not even fetched
    calls = []

    def no_fetch():
        calls.append(1)
        return "error", None

    monkeypatch.setattr(u, "latest_tag", no_fetch)
    assert u.check_for_update() is None
    assert not calls


def test_update_check_silent_on_failure(cfg, tmp_path, monkeypatch):
    import llmtui.updates as u

    monkeypatch.setattr(u, "_CHECK_FILE", tmp_path / "stamp")
    monkeypatch.setattr(u, "latest_tag", lambda: ("error", None))
    assert u.check_for_update() is None


def test_is_newer():
    import llmtui.updates as u

    assert u.is_newer("v0.2.0", "0.1.0")
    assert not u.is_newer("v0.1.0", "0.1.0")
    assert not u.is_newer("v0.1.0", "0.2.0")


def test_gm_update_shells_out_to_uv(cfg, monkeypatch):
    import llmtui.updates as u

    monkeypatch.setattr(u, "latest_tag", lambda: ("ok", "v0.2.0"))
    ran = {}

    def fake_run(cmd, **kwargs):
        ran["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(u.subprocess, "run", fake_run)
    assert u.update() == 0
    assert "git+https://github.com/umutucak/llm-langchain-tui.git@v0.2.0" in ran["cmd"]
    assert "--reinstall" in ran["cmd"]


def test_gm_update_nightly_grabs_main(cfg, monkeypatch):
    import llmtui.updates as u

    ran = {}

    def fake_run(cmd, **kwargs):
        ran["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(u.subprocess, "run", fake_run)
    assert u.update(nightly=True) == 0
    assert any("@main" in a for a in ran["cmd"])
