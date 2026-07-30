"""Unit tests for ``scripts/promote_hcp_adoption_champions.py`` (#1354 lane P).

Covers the PURE decision logic of the promotion script — metric helpers
(positive-class score extraction, calibration-in-the-large intercept), the
calibration pathology gate, the faithfulness check, the exact registry update
payload, and the promote/hold decision — plus the single-row write helper
against a fake async client. No DB, no docker, no model artifacts: fixtures
only. The live-DB scoring path is exercised by the script's own dry-run
against the real registry (evidence in PR #1354's body), never by CI.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.promote_hcp_adoption_champions as promo  # noqa: E402

# ---------------------------------------------------------------------------
# Module constants — the promotion scope is EXACTLY the 3 hcp_adoption models.
# ---------------------------------------------------------------------------


def test_model_allowlist_is_exactly_the_three_hcp_adoption_models():
    assert promo.MODEL_ALLOWLIST == (
        "hcp_adoption_fabhalta_goldstd_lr_v1",
        "hcp_adoption_kisqali_goldstd_lr_v1",
        "hcp_adoption_remibrutinib_goldstd_lr_v1",
    )


def test_eval_windows_match_goldstd_oos_union_policy():
    # Lockstep with scripts/backfill_goldstd_holdout_metrics.py: the held-out
    # window is the OOS union (test+holdout); the encoder fits on train+validation.
    assert promo.OOS_EVAL_SPLITS == ("test", "holdout")
    assert promo.TRAIN_SPLITS == ("train", "validation")


# ---------------------------------------------------------------------------
# positive_class_scores — classes_ ordering must be respected.
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, x):
        return self._proba


def test_positive_class_scores_picks_the_positive_column():
    proba = [[0.3, 0.7], [0.9, 0.1]]
    m = _FakeModel([0, 1], proba)
    np.testing.assert_allclose(promo.positive_class_scores(m, None), [0.7, 0.1])


def test_positive_class_scores_respects_reversed_class_order():
    proba = [[0.3, 0.7], [0.9, 0.1]]
    m = _FakeModel([1, 0], proba)  # positive class is column 0
    np.testing.assert_allclose(promo.positive_class_scores(m, None), [0.3, 0.9])


# ---------------------------------------------------------------------------
# calibration_intercept — calibration-in-the-large (offset-logit MLE).
# ---------------------------------------------------------------------------


def test_calibration_intercept_zero_when_scores_match_outcome_rate():
    y = np.array([0] * 5 + [1] * 5)
    p = np.full(10, 0.5)
    a = promo.calibration_intercept(y, p)
    assert a is not None
    assert abs(a) < 1e-8


def test_calibration_intercept_recovers_logit_shift():
    # All scores 0.5 (offset 0) but 80% positives: the offset-logit MLE
    # intercept is exactly logit(0.8).
    y = np.array([1] * 8 + [0] * 2)
    p = np.full(10, 0.5)
    a = promo.calibration_intercept(y, p)
    assert a is not None
    assert abs(a - math.log(0.8 / 0.2)) < 1e-8


def test_calibration_intercept_none_for_single_class_labels():
    assert promo.calibration_intercept(np.ones(10), np.full(10, 0.6)) is None


# ---------------------------------------------------------------------------
# pathology_gate — HOLD on mis-scaled calibration or no skill over base rate.
# ---------------------------------------------------------------------------

_HEALTHY = {"calibration_slope": 1.0, "brier_score": 0.18}


def test_pathology_gate_passes_a_healthy_model():
    ok, reasons = promo.pathology_gate(_HEALTHY, prevalence=0.4)
    assert ok is True
    assert reasons == []


@pytest.mark.parametrize("slope", [0.4, 2.5, 0.0, -1.0])
def test_pathology_gate_holds_slope_outside_band(slope):
    ok, reasons = promo.pathology_gate({**_HEALTHY, "calibration_slope": slope}, prevalence=0.4)
    assert ok is False
    assert any("calibration_slope" in r for r in reasons)


def test_pathology_gate_holds_when_slope_unfittable():
    m = {"brier_score": 0.18}  # scorer omits the key when unfittable
    ok, reasons = promo.pathology_gate(m, prevalence=0.4)
    assert ok is False
    assert any("calibration_slope" in r for r in reasons)


def test_pathology_gate_holds_brier_at_or_above_prevalence_baseline():
    # prevalence 0.4 -> constant base-rate Brier = 0.4*0.6 = 0.24
    ok, reasons = promo.pathology_gate({**_HEALTHY, "brier_score": 0.24}, prevalence=0.4)
    assert ok is False
    assert any("brier" in r for r in reasons)


def test_pathology_gate_boundary_slopes_pass():
    for slope in (0.5, 2.0):
        ok, reasons = promo.pathology_gate({**_HEALTHY, "calibration_slope": slope}, prevalence=0.4)
        assert ok is True, reasons


# ---------------------------------------------------------------------------
# faithfulness_check — the loaded artifact must reproduce the stored holdout.
# ---------------------------------------------------------------------------

_STORED = {"auc_roc": 0.791084, "accuracy": 0.715}


def test_faithfulness_check_passes_within_tolerance():
    computed = {"auc_roc": 0.791084, "accuracy": 0.715}
    ok, d_auc, d_acc = promo.faithfulness_check(computed, _STORED)
    assert ok is True
    assert d_auc < 1e-9 and d_acc < 1e-9


def test_faithfulness_check_fails_on_auc_drift():
    computed = {"auc_roc": 0.781, "accuracy": 0.715}
    ok, d_auc, _ = promo.faithfulness_check(computed, _STORED)
    assert ok is False
    assert d_auc > promo.TOL


def test_faithfulness_check_fails_on_missing_inputs():
    assert promo.faithfulness_check(None, _STORED)[0] is False
    assert promo.faithfulness_check({"auc_roc": 0.79, "accuracy": 0.7}, {})[0] is False


# ---------------------------------------------------------------------------
# build_registry_update — the EXACT write payload (schema-checked).
# ---------------------------------------------------------------------------


def test_build_registry_update_payload_shape_and_rounding():
    metrics = {
        "auc_roc": 0.791084,
        "accuracy": 0.715,
        "pr_auc": 0.7348414,
        "brier_score": 0.1813125,
        "calibration_slope": 1.0086021,
    }
    update = promo.build_registry_update(metrics, "2026-07-30T00:00:00+00:00")
    assert update == {
        "pr_auc": 0.7348,
        "brier_score": 0.1813,
        "calibration_slope": 1.0086,
        "stage": "production",
        "is_champion": True,
        "promoted_at": "2026-07-30T00:00:00+00:00",
    }


def test_build_registry_update_never_touches_out_of_scope_columns():
    metrics = {
        "auc_roc": 0.79,
        "accuracy": 0.7,
        "pr_auc": 0.73,
        "brier_score": 0.18,
        "calibration_slope": 1.0,
    }
    update = promo.build_registry_update(metrics, "2026-07-30T00:00:00+00:00")
    # auc is already recorded (and re-verified via faithfulness); data_split,
    # model_name, experiment_id etc. must never appear in the write payload.
    for forbidden in ("auc", "data_split", "model_name", "experiment_id", "is_synthetic"):
        assert forbidden not in update


# ---------------------------------------------------------------------------
# decide — promote/hold wiring of faithfulness + gate + payload.
# ---------------------------------------------------------------------------

_COMPUTED_OK = {
    "auc_roc": 0.791084,
    "accuracy": 0.715,
    "pr_auc": 0.734841,
    "brier_score": 0.181312,
    "calibration_slope": 1.008602,
}


def test_decide_promotes_faithful_healthy_model():
    action, reasons, update = promo.decide(
        _STORED, _COMPUTED_OK, prevalence=0.407, promoted_at_iso="T"
    )
    assert action == "promote"
    assert reasons == []
    assert update is not None and update["is_champion"] is True


def test_decide_holds_on_pathological_calibration_without_payload():
    computed = {**_COMPUTED_OK, "calibration_slope": 0.3}
    action, reasons, update = promo.decide(_STORED, computed, prevalence=0.407, promoted_at_iso="T")
    assert action == "hold"
    assert update is None
    assert any("calibration_slope" in r for r in reasons)


def test_decide_holds_on_unfaithful_artifact_without_payload():
    computed = {**_COMPUTED_OK, "auc_roc": 0.70}
    action, reasons, update = promo.decide(_STORED, computed, prevalence=0.407, promoted_at_iso="T")
    assert action == "hold"
    assert update is None
    assert any("unfaithful" in r for r in reasons)


# ---------------------------------------------------------------------------
# _apply_update — single-row, PK-scoped write via the async client.
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, log, table):
        self._log = log
        self._table = table
        self._update = None
        self._eq = []

    def update(self, payload):
        self._update = payload
        return self

    def eq(self, col, val):
        self._eq.append((col, val))
        return self

    async def execute(self):
        self._log.append((self._table, self._update, list(self._eq)))
        return type("R", (), {"data": [{}]})()


class _FakeClient:
    def __init__(self):
        self.log = []

    def table(self, name):
        return _FakeQuery(self.log, name)


def test_apply_update_targets_exactly_one_row_by_id():
    client = _FakeClient()
    update = {"stage": "production", "is_champion": True}
    asyncio.run(promo._apply_update(client, "row-uuid-1", update))
    assert client.log == [("ml_model_registry", update, [("id", "row-uuid-1")])]
