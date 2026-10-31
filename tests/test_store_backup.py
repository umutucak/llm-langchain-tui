"""store_backup: move-based backup, last-1 retention, reversible restore."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milvus_lite.server_manager import server_manager_instance
from pymilvus import DataType, MilvusClient

from llmtui import store_backup
from llmtui.config import MILVUS_COLLECTION_NAME

results = {}


def check(label, conditions):
    ok = True
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for desc, passed in conditions:
        print(f"    {'PASS' if passed else 'FAIL'}  {desc}")
        ok = ok and passed
    results[label] = ok


def test_store_backup_cases():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        live = tmp / "milvus.db"
        backup_dir = tmp / "backups"

        # ---- 1. backup is a move, live store is gone ----
        live.mkdir()
        (live / "LOCK").write_text("old")
        first = store_backup.backup_store(live, backup_dir)
        check(
            "backup_store moves the live store",
            [
                ("live store is gone after the move", not live.exists()),
                ("backup dir exists with the store", first.is_dir()),
                ("contents came along", (first / "LOCK").read_text() == "old"),
                ("name is milvus-<timestamp>-<tag>", first.name.startswith("milvus-")),
            ],
        )

        # ---- 2. last-1 retention ----
        live.mkdir()
        (live / "LOCK").write_text("newer")
        second = store_backup.backup_store(live, backup_dir)
        names = {p.name for p in backup_dir.iterdir()}
        check(
            "last-1 retention",
            [
                ("only one backup left", len(names) == 1),
                ("the newest one is it", second.is_dir() and second.name in names),
            ],
        )

        # ---- 3. restore swaps the backup into place, reversibly ----
        live.mkdir()
        (live / "LOCK").write_text("current")
        store_backup.restore_backup(second, live, backup_dir)
        check(
            "restore_backup swaps into the live location",
            [
                (
                    "live store is the restored one",
                    (live / "LOCK").read_text() == "newer",
                ),
                ("restored dir is no longer a backup", not second.exists()),
                (
                    "the old live store was backed up first (reversible)",
                    len(store_backup.list_backups(backup_dir)) == 1,
                ),
            ],
        )

        # ---- 4. edge cases ----
        try:
            store_backup.backup_store(tmp / "nope", backup_dir)
            missing = False
        except FileNotFoundError:
            missing = True
        check(
            "edge cases",
            [
                ("missing live store raises FileNotFoundError", missing),
                (
                    "missing backup dir lists empty",
                    store_backup.list_backups(tmp / "no") == [],
                ),
            ],
        )

    # ---- 5. the move leaves no milvus-lite server holding the old path ----
    # a server left running keeps the moved-away db open and writes its dirs back
    # into the old location on the next use (issue #27)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        live = tmp / "milvus.db"

        client = MilvusClient(uri=str(live))
        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
        client.create_collection(MILVUS_COLLECTION_NAME, schema=schema)
        client.close()

        store_backup.backup_store(live, tmp / "backups")
        held = {Path(path).resolve() for path in server_manager_instance._servers}
        reopened = MilvusClient(uri=str(live))
        stale = reopened.has_collection(MILVUS_COLLECTION_NAME)
        reopened.close()
        store_backup._close_server(live)
        check(
            "the move leaves no server holding the live path",
            [
                ("no server still holds the live path", live.resolve() not in held),
                ("a reconnect gets a fresh store, not the moved-away one", not stale),
            ],
        )

    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"FAILED: {failed}")
        print(f"ALL {len(results)} PASS")
    failed = [label for label, ok in results.items() if not ok]
    assert not failed, f"failing cases: {failed}"
