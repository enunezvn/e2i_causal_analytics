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
