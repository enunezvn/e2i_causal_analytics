"""Shard 05 Task 1 (regression lock) — triggers.brand_id + the WS2-TR-003 arm
columns must stay in the loader allowlist, or the loader strips them at load
(batch_loader.py:344) and kisqali_oncologist_reach / action_rate_uplift read 0.

These were registered by Shard 02; this test locks them so a future edit cannot
silently drop brand_id (the INDEX "loader silently drops columns" landmine)."""

from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS


def test_triggers_brand_id_registered():
    assert "brand_id" in TABLE_COLUMNS["triggers"]


def test_triggers_is_synthetic_registered():
    assert "is_synthetic" in TABLE_COLUMNS["triggers"]


def test_triggers_ws2tr003_arm_columns_registered():
    for c in ("action_taken", "control_group_flag", "outcome_tracked"):
        assert c in TABLE_COLUMNS["triggers"], f"{c} stripped -> action_rate_uplift breaks"


def test_triggers_false_positive_flag_registered():
    # #1118 WS2-TR-005: if the loader strips false_positive_flag, every row
    # reverts to the schema default FALSE and False Alert Rate is vacuously GOOD.
    assert "false_positive_flag" in TABLE_COLUMNS["triggers"], (
        "false_positive_flag stripped -> TR-005 stays vacuous after reseed"
    )


def test_triggers_view_timestamp_registered():
    # #1387 view-stage realism: if the loader strips view_timestamp, the column
    # stays 100% NULL after the backfill and the funnel's view stage is
    # timestamp-less again (the exact silent-drop landmine this file locks).
    assert "view_timestamp" in TABLE_COLUMNS["triggers"], (
        "view_timestamp stripped -> #1387 backfill silently no-ops the column"
    )
