"""Unit tests for issue #778: cohort_builder + refutation_runner as REAL tools.

The tool-composer-remediation (PRs #774-777) made these two composable tools
fail closed (honest) rather than fabricate. This follow-up makes them real:

- ``cohort_builder`` routes through the #779 ``cohort_resolution`` service to get
  a real (brand, region) cohort DataFrame (or uses an injected ``estimation_data``
  frame), returns REAL patient IDs, and applies simple inclusion/exclusion
  criteria. It NEVER fabricates ``P001/P002/P003`` placeholder IDs.
- ``refutation_runner`` reuses the R6/F1 DoWhy refutation path (invoking
  ``DoWhyExecutor`` with ``run_refutation=True``) on the in-context DataFrame.
  Fail-closed paths (no DataFrame / no treatment+outcome / missing columns) are
  covered here; the real-suite success path is an integration test.

Anti-mocking discipline: every "no data" path must raise a descriptive
RuntimeError ("Refusing to fabricate"), never a plausible-but-fake result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr


def _cohort_df(n: int = 6, id_col: str = "patient_id") -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_col: [f"PJ{i:04d}" for i in range(n)],
            "brand": ["Kisqali"] * n,
            "geographic_region": ["northeast"] * n,
            "age_at_diagnosis": [40 + i for i in range(n)],
            "treatment_initiated": [i % 2 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# cohort_builder — real cohort via #779 service / injected frame
# ---------------------------------------------------------------------------
def test_cohort_builder_resolves_via_cohort_resolution_service(monkeypatch):
    df = _cohort_df(5)
    called = {}

    def _fake_resolve(brand, region, **kwargs):
        called["args"] = (brand, region)
        return df

    monkeypatch.setattr(tr.cohort_resolution, "resolve_cohort_frame", _fake_resolve)
    out = tr.cohort_builder(brand="Kisqali", region="Northeast")
    assert out.total_evaluated == 5
    assert out.total_eligible == 5
    assert out.eligible_patient_ids == [f"PJ{i:04d}" for i in range(5)]
    assert called["args"] == ("Kisqali", "Northeast")
    # Anti-mock regression: NO fabricated placeholder IDs.
    assert "P001" not in out.eligible_patient_ids
    assert "P002" not in out.eligible_patient_ids


def test_cohort_builder_prefers_injected_dataframe(monkeypatch):
    injected = _cohort_df(4)

    def _boom(*a, **k):
        raise AssertionError("resolve_cohort_frame must NOT be called when a frame is injected")

    monkeypatch.setattr(tr.cohort_resolution, "resolve_cohort_frame", _boom)
    out = tr.cohort_builder(brand="Kisqali", estimation_data=injected)
    assert out.total_evaluated == 4
    assert out.eligible_patient_ids == [f"PJ{i:04d}" for i in range(4)]


def test_cohort_builder_fails_closed_when_no_cohort(monkeypatch):
    monkeypatch.setattr(tr.cohort_resolution, "resolve_cohort_frame", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Refusing to fabricate"):
        tr.cohort_builder(brand="Kisqali", region="Northeast")


def test_cohort_builder_fails_closed_when_no_id_column(monkeypatch):
    no_id = pd.DataFrame({"age_at_diagnosis": [50, 60], "brand": ["Kisqali", "Kisqali"]})
    monkeypatch.setattr(tr.cohort_resolution, "resolve_cohort_frame", lambda *a, **k: no_id)
    with pytest.raises(RuntimeError, match="patient-id|Refusing to fabricate"):
        tr.cohort_builder(brand="Kisqali")


def test_cohort_builder_applies_inclusion_criteria():
    df = _cohort_df(6)  # age_at_diagnosis = 40..45
    out = tr.cohort_builder(
        brand="Kisqali",
        inclusion_criteria=["age_at_diagnosis >= 43"],
        estimation_data=df,
    )
    assert out.total_evaluated == 6
    assert out.total_eligible == 3  # ages 43,44,45
    assert out.eligible_patient_ids == ["PJ0003", "PJ0004", "PJ0005"]


def test_cohort_builder_applies_exclusion_criteria():
    df = _cohort_df(6)
    out = tr.cohort_builder(
        brand="Kisqali",
        exclusion_criteria=["treatment_initiated == 1"],
        estimation_data=df,
    )
    # rows with treatment_initiated==1 (odd indices) are excluded -> 0,2,4 remain
    assert out.eligible_patient_ids == ["PJ0000", "PJ0002", "PJ0004"]


def test_cohort_builder_unparseable_criterion_recorded_not_silently_applied():
    df = _cohort_df(4)
    out = tr.cohort_builder(
        brand="Kisqali",
        inclusion_criteria=["nonexistent_col >= 5", "age_at_diagnosis >= 40"],
        estimation_data=df,
    )
    # The unparseable/unknown-column criterion must NOT silently drop everyone;
    # it is recorded honestly and the valid criterion still applies.
    assert out.total_eligible == 4  # all ages >= 40
    assert any("unapplied" in k.lower() for k in out.criteria_breakdown)


def test_cohort_builder_empty_rhs_criterion_recorded_unapplied():
    df = _cohort_df(4)
    out = tr.cohort_builder(
        brand="Kisqali",
        inclusion_criteria=["age_at_diagnosis == "],  # empty RHS
        estimation_data=df,
    )
    # Empty-RHS criterion is recorded unapplied (honest), not a wrong comparison.
    assert out.total_eligible == 4
    assert out.criteria_breakdown.get("_unapplied_criteria") == 1


# ---------------------------------------------------------------------------
# refutation_runner — fail-closed paths (real-suite success is integration)
# ---------------------------------------------------------------------------
def test_refutation_runner_fails_closed_without_dataframe():
    with pytest.raises(RuntimeError, match="Refusing to fabricate"):
        tr.refutation_runner(estimate_id="est-123")


def test_refutation_runner_fails_closed_without_treatment_outcome():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(RuntimeError, match="treatment|outcome"):
        tr.refutation_runner(estimate_id="est-123", estimation_data=df)


def test_refutation_runner_fails_closed_on_missing_columns():
    df = pd.DataFrame({"engagement": [0, 1, 0], "converted": [1.0, 2.0, 1.5]})
    with pytest.raises(RuntimeError, match="not in|Refusing to fabricate"):
        tr.refutation_runner(
            estimate_id="est-123",
            estimation_data=df,
            treatment="engagement",
            outcome="converted",
            confounders=["does_not_exist"],
        )
