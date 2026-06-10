"""Shard 09 WS2-TR-008 (Change-Fail Rate): some synthetic triggers must reference a
previous_trigger_id (a superseding change) with a change_type and a change_failed flag
so CFR = count(change_failed) / count(previous_trigger_id IS NOT NULL) is non-NULL."""

import pandas as pd

from src.ml.synthetic.generators.change_tracking import stamp_change_tracking


def test_change_tracking_produces_supersession_chain():
    df = pd.DataFrame(
        {
            "trigger_id": [f"trg_{i:04d}" for i in range(60)],
            "is_synthetic": [True] * 60,
        }
    )
    out = stamp_change_tracking(df, seed=2, change_fraction=0.4, fail_fraction=0.25)
    for c in ("previous_trigger_id", "change_type", "change_failed", "change_outcome_delta"):
        assert c in out.columns
    # at least some rows are "changes" (previous_trigger_id set) so CFR denom > 0
    changes = out[out["previous_trigger_id"].notna()]
    assert len(changes) > 0
    # previous_trigger_id must reference an actual trigger_id in the frame
    assert changes["previous_trigger_id"].isin(out["trigger_id"]).all()
    # at least one failed change so the numerator is > 0 (CFR non-degenerate)
    assert bool(changes["change_failed"].any())
    # changes carry a change_type
    assert changes["change_type"].notna().all()


def test_change_tracking_does_not_mutate_input():
    df = pd.DataFrame({"trigger_id": ["t0"], "is_synthetic": [True]})
    _ = stamp_change_tracking(df, seed=1)
    assert "previous_trigger_id" not in df.columns
