"""#577 WS1-MP-009 (feature_drift / PSI) faithful e2e: the metric computes the RIGHT value
against the LIVE DB over the coherently-seeded ml_drift_history (migration 053) — not just
"runs".

These assert MEANING (the #574 lesson):
- avg PSI is a real float in the low-moderate band (~0.094 live) — status GOOD under YAML
  lower-is-better (target 0.10 / warning 0.20) — NOT a suspicious 0.0 and NOT alarmingly high.
- The calculator's value EQUALS AVG(test_statistic) over the live psi/data rows — proving it
  reads the seeded aggregate, not a constant.
- ANTI-FABRICATION self-reproducibility: each per-feature row's stored test_statistic recomputes
  from its OWN stored baseline/current mean/std (via the shared _compute_feature_psi) to <5e-7,
  and their average equals calculate().value. PSI is COMPUTED, not hardcoded.
- NO-DRIFT DISPROOF: for every seeded feature, _compute_feature_psi(baseline, baseline) == 0.0,
  and a larger injected shift strictly increases PSI — so PSI responds ONLY to real drift.
- The seeded baseline_mean/std EQUAL ml_preprocessing_metadata.feature_distributions (the YAML
  source) — the "source" and the calculator's table are the SAME distribution.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the model_performance_feature_drift query
exists (migration 053 applied) — NOT a 044-era query.
"""

import os

import numpy as np
import pytest

from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.ml.data_generator import _compute_feature_psi

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

_QUERY_ID = "model_performance_feature_drift"
_MP009 = "WS1-MP-009"


@pytest.fixture
def calc():
    c = ModelPerformanceCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc("kpi_query", {"query_id": _QUERY_ID, "params": []}).execute()
    except Exception as e:
        pytest.skip(f"#577 feature_drift query unavailable (migration 053 not applied?): {e}")
    return c


def _live_feature_rows(calc):
    """The per-feature seeded rows (baseline/current mean+std + stored test_statistic)."""
    # The registry query returns only the aggregate; read the raw rows directly for the
    # self-reproducibility proof (same DB, same client).
    res = (
        calc.db_client.table("ml_drift_history")
        .select("feature_name,baseline_mean,baseline_std,current_mean,current_std,test_statistic")
        .eq("test_type", "psi")
        .eq("drift_type", "data")
        .execute()
    )
    return res.data


def test_feature_drift_value_is_low_moderate_band(calc):
    """calculate(WS1-MP-009).value is a real PSI float in (0, 0.10] -> GOOD under lower-is-better.
    Range assertion (the realization is one deterministic seed), never the exact float."""
    value, error = calc._calc_feature_drift({"model_name": "default_model"})
    assert error is None, f"expected SQL leg to succeed, got error={error}"
    assert value is not None
    assert 0.0 < value <= 0.10, f"avg PSI {value} not in the low-moderate GOOD band"


def test_feature_drift_equals_live_aggregate(calc):
    """The calculator's value EQUALS AVG(test_statistic) over the live psi/data rows — proving
    it reads the seeded aggregate (not a constant)."""
    rows = _live_feature_rows(calc)
    assert rows, "no seeded psi/data rows in ml_drift_history"
    expected = float(np.mean([float(r["test_statistic"]) for r in rows]))
    value, _ = calc._calc_feature_drift({"model_name": "default_model"})
    assert abs(value - expected) < 1e-6


def test_feature_drift_rows_are_self_reproducible(calc):
    """ANTI-FABRICATION (core): each row's stored test_statistic recomputes from its OWN stored
    baseline/current mean/std via the shared _compute_feature_psi to <5e-7 — the PSI is
    COMPUTED and self-auditable, not reverse-engineered to a target."""
    rows = _live_feature_rows(calc)
    assert rows
    for r in rows:
        recomputed = _compute_feature_psi(
            float(r["baseline_mean"]),
            float(r["baseline_std"]),
            float(r["current_mean"]),
            float(r["current_std"]),
        )
        assert abs(recomputed - float(r["test_statistic"])) < 5e-7, (
            f"{r['feature_name']}: stored {r['test_statistic']} != recomputed {recomputed}"
        )


def test_feature_drift_no_drift_disproof(calc):
    """NO-DRIFT DISPROOF: for every seeded feature, PSI(baseline, baseline) is EXACTLY 0.0, and
    a larger injected shift strictly increases PSI — PSI responds ONLY to injected drift."""
    rows = _live_feature_rows(calc)
    assert rows
    for r in rows:
        bm, bs = float(r["baseline_mean"]), float(r["baseline_std"])
        assert _compute_feature_psi(bm, bs, bm, bs) == 0.0
        small = _compute_feature_psi(bm, bs, bm + 0.2 * bs, bs)
        large = _compute_feature_psi(bm, bs, bm + 0.6 * bs, bs)
        assert 0.0 < small < large


def test_feature_drift_baseline_equals_feature_distributions(calc):
    """COHERENCE: the seeded baseline_mean/std EQUAL ml_preprocessing_metadata.feature_distributions
    (the YAML 'source') — the source and the calculator's table are the SAME distribution."""
    fd = (
        calc.db_client.table("ml_preprocessing_metadata")
        .select("feature_distributions")
        .limit(1)
        .execute()
        .data
    )
    if not fd:
        pytest.skip("no ml_preprocessing_metadata row")
    dist = fd[0]["feature_distributions"]
    for r in _live_feature_rows(calc):
        f = r["feature_name"]
        if f not in dist:
            continue
        assert abs(float(r["baseline_mean"]) - dist[f]["mean"]) < 1e-6
        assert abs(float(r["baseline_std"]) - dist[f]["std"]) < 1e-6


def test_feature_drift_latest_run_scoping_excludes_older_runs(calc):
    """REGRESSION LOCK (codex MEDIUM): ml_drift_history is a HISTORY table that accumulates runs,
    so the KPI scopes to current_end=MAX(current_end). An OLDER-window high-PSI run must NOT
    dilute the KPI. Self-protecting: the injected row uses an older current_end, so even if the
    finally-cleanup failed, the scoping itself keeps it out of the real KPI."""
    before, err = calc._calc_feature_drift({"model_name": "default_model"})
    assert err is None and before is not None
    sentinel = "stale_run_regression_test"
    injected = {
        "model_id": None,
        "drift_type": "data",
        "feature_name": "age",
        "test_type": "psi",
        "test_statistic": 0.99,  # alarmingly high — would wreck a lifetime average
        "threshold": 0.10,
        "drift_detected": True,
        "severity": "low",
        "baseline_start": "2024-01-01T00:00:00+00:00",
        "baseline_end": "2024-02-01T00:00:00+00:00",
        "current_start": "2024-02-01T00:00:00+00:00",
        "current_end": "2024-03-01T00:00:00+00:00",  # far OLDER than the seed's NOW() window
        "detected_by": sentinel,
    }
    try:
        calc.db_client.table("ml_drift_history").insert(injected).execute()
        after, _ = calc._calc_feature_drift({"model_name": "default_model"})
        assert abs(after - before) < 1e-9, (
            f"older high-PSI run leaked into the KPI: {before} -> {after} (latest-run scoping broken)"
        )
    finally:
        calc.db_client.table("ml_drift_history").delete().eq("detected_by", sentinel).execute()


def test_feature_drift_status_is_lower_is_better_good(calc):
    """End-to-end banding: the full calculate() path yields a GOOD status under YAML
    lower-is-better bands (avg ~0.094 <= target 0.10), exercising the tuple unwrap +
    lower_is_better membership for MP-009."""
    from src.kpi.models import KPIStatus
    from src.kpi.registry import get_registry

    kpi = get_registry().get(_MP009)
    if kpi is None or kpi.threshold is None:
        pytest.skip("WS1-MP-009 not in the loaded KPI registry / no threshold")
    result = calc.calculate(kpi, {"model_name": "default_model"})
    assert result.value is not None and 0.0 < result.value <= kpi.threshold.warning
    assert result.status == KPIStatus.GOOD
