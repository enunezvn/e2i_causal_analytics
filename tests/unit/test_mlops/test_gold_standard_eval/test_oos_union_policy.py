"""OOS-union eval-window policy pins (2026-07-23).

The gold-standard holdout headline scores the champion on EVERY row outside its
training data — ``data_split IN ('test', 'holdout')`` — instead of the old
``holdout``-only window. Measured rationale (2026-07-23, live cohorts):
single-window draws at the old sizes (patient n~850, hcp n=250) made the
calibration-slope KPI a window lottery — random same-size windows of the Remi
persistence OOS pool span slope ~1.0-1.24 around a true OOS slope of ~1.12 —
and a Platt calibrator fitted on the (then-unused) test split is ~identity, so
no recalibration could stabilize the old headline. The union doubles n,
tightens every bootstrap CI, and makes the HCP eval invariant to the
test/holdout boundary (whose live DB ratios still predate the #44 quota).

These pins lock the three definition sites (per-brand runner, superseded pooled
initiation runner, holdout re-record script) to the SAME window and keep it
disjoint from the champion's training splits — a drifted copy silently
re-shrinks the window or, far worse, leaks training rows into the headline.
"""

from scripts.backfill_goldstd_holdout_metrics import (
    LEGACY_EVAL_SPLITS,
    OOS_EVAL_SPLITS,
)
from src.mlops.gold_standard_eval.run_initiation_eval import (
    _OOS_EVAL_SPLITS as INITIATION_SPLITS,
)
from src.mlops.gold_standard_eval.run_persistence_eval import (
    _CHAMPION_TRAIN_SPLITS,
    _OOS_EVAL_SPLITS,
)


class TestOOSUnionWindow:
    def test_eval_window_is_test_union_holdout(self):
        assert _OOS_EVAL_SPLITS == ("test", "holdout")

    def test_all_definition_sites_agree(self):
        assert INITIATION_SPLITS == _OOS_EVAL_SPLITS
        assert OOS_EVAL_SPLITS == _OOS_EVAL_SPLITS

    def test_eval_window_disjoint_from_training(self):
        # The headline must stay strictly out-of-sample: no split may be both
        # a champion-training split and part of the eval window.
        assert not set(_CHAMPION_TRAIN_SPLITS) & set(_OOS_EVAL_SPLITS)

    def test_legacy_window_is_a_subset(self):
        # The backfill's faithfulness guard replays the pre-policy window; it
        # must remain a strict subset of the union or the guard is meaningless.
        assert set(LEGACY_EVAL_SPLITS) < set(OOS_EVAL_SPLITS)
