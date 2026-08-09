"""Red-first real-computation assertions for the six tool_composer registry tools
flagged by GH #621 (incomplete #354 anti-mock cleanup).

These tools previously returned hardcoded demo entities/values:

- ``risk_scorer``           -> fixed ``E001/E002/E003`` IDs + hardcoded risk scores/tiers.
- ``cate_analyzer``         -> hardcoded ``high_volume_academic`` segment effects.
- ``segment_ranker``        -> hardcoded ranking of ``high_volume_academic`` etc.
- ``gap_calculator``        -> hardcoded ``northeast/midwest`` region values.
- ``roi_estimator``         -> hardcoded ``estimated_roi=3.2``.
- ``propensity_estimator``  -> hardcoded ``mean_propensity=0.35`` distribution.

The Tier 1-5 keyless harness anti-fabrication gate
(``src/testing/agent_quality_gates.py:146``) correctly rejects the ``E\\d{3}``
fabricated IDs. The fix is to STOP fabricating: each tool now computes its
output from a caller-supplied real ``pandas.DataFrame`` (threaded through the
executor context as ``$context.estimation_data`` exactly like
``causal_effect_estimator``), and FAILS CLOSED (``RuntimeError``) when no real
DataFrame / required upstream result is supplied -- never substituting a
plausible-but-fake placeholder.

These assertions FAIL on the hardcoded placeholder bodies and PASS only once
the tools compute from real inputs. Tests build their OWN DataFrames (the
anti-pattern is synthetic data fabricated INSIDE the tool body; tests are
allowed to construct the frame the tool then consumes via kwargs).

Cross-refs:
- Precedent: ``test_causal_effect_estimator.py`` (#354 Phase C-7).
- Anti-fab gate: ``src/testing/agent_quality_gates.py``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.tool_registrations import (
    cate_analyzer,
    gap_calculator,
    propensity_estimator,
    risk_scorer,
    roi_estimator,
    segment_ranker,
)

# ============================================================================
# Fixtures: real DataFrames the tools consume via kwargs.
# ============================================================================


def _build_cohort_df(*, n: int = 300, seed: int = 7) -> pd.DataFrame:
    """A cohort frame mirroring the tier0 fixture's real columns.

    Includes a real entity id column (``patient_id``), categorical segment
    columns (``geographic_region``, ``age_group``), numeric features, a binary
    treatment, and a binary outcome with a genuine treatment effect so the
    causal/propensity tools recover a non-degenerate signal.
    """
    rng = np.random.default_rng(seed)
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    age_group = rng.choice(["<50", "50-65", ">65"], size=n)
    days_on_therapy = rng.integers(30, 400, size=n)
    prior_treatments = rng.integers(0, 5, size=n)
    hcp_visits = rng.integers(1, 20, size=n)
    high_engagement = (hcp_visits >= np.median(hcp_visits)).astype(int)
    # Outcome correlated with treatment + a covariate -> real, recoverable signal.
    logit = -0.5 + 1.2 * high_engagement - 0.01 * days_on_therapy + 0.2 * prior_treatments
    prob = 1.0 / (1.0 + np.exp(-logit))
    outcome = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "geographic_region": regions,
            "age_group": age_group,
            "days_on_therapy": days_on_therapy,
            "prior_treatments": prior_treatments,
            "hcp_visits": hcp_visits,
            "high_engagement": high_engagement,
            "discontinuation_flag": outcome,
        }
    )


def _dump(out):
    return out.model_dump() if hasattr(out, "model_dump") else out


# ============================================================================
# Anti-fabrication invariants (catch the OLD hardcoded bodies).
# ============================================================================

_FORBIDDEN_LITERALS = (
    "E001",
    "E002",
    "E003",
    "high_volume_academic",
    "community_practice",
    "integrated_health",
    "v2.3.1",
    "2024-01-15T10:30:00Z",
)


def _executable_source(func) -> str:
    """Return the function source with docstrings and comments stripped.

    The anti-fab gate scans the tool's OUTPUT (``response_text``) for forbidden
    literals -- it does NOT forbid mentioning what was replaced. These source
    scans mirror that intent: a docstring documenting "replaces the fabricated
    E001/E002/E003 placeholder" is desirable provenance, NOT a fabrication. So
    we strip docstrings/comments and assert the EXECUTABLE code never emits the
    forbidden literals.
    """
    import ast

    src = inspect.getsource(func)
    tree = ast.parse(src)
    fn = tree.body[0]
    # Drop the docstring node if present.
    body = fn.body  # type: ignore[attr-defined]
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    # ast.unparse reproduces executable code without comments or docstrings.
    return "\n".join(ast.unparse(node) for node in body)


@pytest.mark.parametrize(
    "func",
    [
        cate_analyzer,
        segment_ranker,
        gap_calculator,
        roi_estimator,
        risk_scorer,
        propensity_estimator,
    ],
)
def test_no_hardcoded_demo_literals_in_executable_body(func):
    """The EXECUTABLE tool body must not contain the hardcoded demo literals/IDs."""
    src = _executable_source(func)
    for lit in _FORBIDDEN_LITERALS:
        assert lit not in src, f"{func.__name__} still emits hardcoded literal {lit!r}"


def test_no_fabricated_entity_ids_in_executable_bodies():
    """No ``E\\d{3}`` fabricated entity IDs in the rewired executable bodies."""
    for func in (risk_scorer, segment_ranker, gap_calculator, roi_estimator):
        src = _executable_source(func)
        ids = re.findall(r"\bE\d{3}\b", src)
        assert not ids, f"{func.__name__} fabricates entity IDs {ids}"


def test_no_random_fabrication_in_executable_bodies():
    for func in (
        cate_analyzer,
        segment_ranker,
        gap_calculator,
        roi_estimator,
        risk_scorer,
        propensity_estimator,
    ):
        src = _executable_source(func)
        assert "np.random" not in src, f"{func.__name__} uses np.random"
        assert "random.uniform" not in src, f"{func.__name__} uses random.uniform"


# ============================================================================
# Fail-closed: no DataFrame -> RuntimeError (never a placeholder).
# ============================================================================


def test_gap_calculator_fails_closed_without_dataframe():
    with pytest.raises(RuntimeError):
        gap_calculator(metric="discontinuation_flag", entity_type="region", entities=[])


def test_cate_analyzer_fails_closed_without_dataframe():
    with pytest.raises(RuntimeError):
        cate_analyzer(
            treatment="high_engagement",
            outcome="discontinuation_flag",
            segments=["age_group"],
        )


def test_risk_scorer_fails_closed_without_dataframe():
    with pytest.raises(RuntimeError):
        risk_scorer(entity_type="patient", risk_type="discontinuation")


def test_propensity_estimator_fails_closed_without_dataframe():
    with pytest.raises(RuntimeError):
        propensity_estimator(treatment="high_engagement", covariates=["days_on_therapy"])


# ============================================================================
# gap_calculator: real per-entity gap computation.
# ============================================================================


def test_gap_calculator_computes_real_region_values():
    df = _build_cohort_df()
    out = gap_calculator(
        metric="days_on_therapy",
        entity_type="region",
        entities=[],
        estimation_data=df,
        group_by="geographic_region",
    )
    d = _dump(out)
    real_regions = set(df["geographic_region"].unique())
    assert set(d["entity_values"].keys()) == real_regions
    expected = df.groupby("geographic_region")["days_on_therapy"].mean().to_dict()
    for region, val in d["entity_values"].items():
        assert val == pytest.approx(expected[region], rel=1e-6)
    assert d["top_performer"] == max(expected, key=expected.get)
    assert d["bottom_performer"] == min(expected, key=expected.get)
    assert d["gap"] == pytest.approx(max(expected.values()) - min(expected.values()), rel=1e-6)


# ============================================================================
# cate_analyzer: real per-segment treatment effects.
# ============================================================================


def test_cate_analyzer_computes_real_segment_effects():
    df = _build_cohort_df()
    out = cate_analyzer(
        treatment="high_engagement",
        outcome="discontinuation_flag",
        segments=["age_group"],
        estimation_data=df,
    )
    d = _dump(out)
    real_segments = set(df["age_group"].unique())
    seg_names = {s["name"] for s in d["segments"]}
    assert seg_names == real_segments
    for seg in d["segments"]:
        sub = df[df["age_group"] == seg["name"]]
        treated = sub[sub["high_engagement"] == 1]["discontinuation_flag"].mean()
        control = sub[sub["high_engagement"] == 0]["discontinuation_flag"].mean()
        assert seg["cate"] == pytest.approx(treated - control, rel=1e-6, abs=1e-9)
        assert seg["n"] == len(sub)
    assert set(d["high_responders"]) <= real_segments
    assert set(d["effect_by_segment"].keys()) == real_segments


# ============================================================================
# segment_ranker: ranks the REAL upstream cate/gap result it consumes.
# ============================================================================


def test_segment_ranker_ranks_real_upstream_result():
    cate_results = {"effect_by_segment": {"alpha": 0.30, "beta": 0.05, "gamma": 0.18}}
    out = segment_ranker(cate_results=cate_results)
    d = _dump(out)
    ranked = d["ranking"]
    assert [r["segment"] for r in ranked] == ["alpha", "gamma", "beta"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert ranked[0]["score"] == pytest.approx(0.30)


def test_segment_ranker_fails_closed_without_upstream():
    with pytest.raises(RuntimeError):
        segment_ranker(cate_results={})


# ============================================================================
# roi_estimator: real arithmetic from the gap result it consumes.
# ============================================================================


def test_roi_estimator_computes_from_gap_result():
    gap_analysis = {"gap": 100.0, "entity_values": {"a": 50.0, "b": 150.0}}
    out = roi_estimator(gap_analysis=gap_analysis, investment=1000.0)
    d = _dump(out)
    assert isinstance(d["estimated_roi"], float)
    assert d["estimated_roi"] >= 0.0
    # ROI must vary with the gap (not a frozen 3.2 constant).
    out2 = roi_estimator(
        gap_analysis={"gap": 200.0, "entity_values": {"a": 50.0, "b": 250.0}},
        investment=1000.0,
    )
    assert _dump(out2)["estimated_roi"] != d["estimated_roi"]


def test_roi_estimator_fails_closed_without_gap():
    with pytest.raises(RuntimeError):
        roi_estimator(gap_analysis={}, investment=1000.0)


def test_roi_estimator_fails_closed_on_nonpositive_investment():
    with pytest.raises(RuntimeError):
        roi_estimator(gap_analysis={"gap": 10.0}, investment=0.0)


# ============================================================================
# risk_scorer: real per-entity scores using REAL entity IDs from the frame.
# ============================================================================


def test_risk_scorer_uses_real_entity_ids_and_scores():
    df = _build_cohort_df()
    out = risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=df,
        id_column="patient_id",
        outcome="discontinuation_flag",
    )
    d = _dump(out)
    scores = d["scores"]
    assert len(scores) == len(df)
    real_ids = set(df["patient_id"])
    for s in scores:
        assert s["entity_id"] in real_ids  # NEVER E001/E002/E003
        assert 0.0 <= s["risk_score"] <= 1.0
        assert s["risk_tier"] in {"low", "medium", "high"}
    assert not any(re.fullmatch(r"E\d{3}", s["entity_id"]) for s in scores)


# ============================================================================
# propensity_estimator: real propensity scores from covariates.
# ============================================================================


def test_propensity_estimator_computes_real_scores():
    df = _build_cohort_df()
    out = propensity_estimator(
        treatment="high_engagement",
        covariates=["days_on_therapy", "prior_treatments", "hcp_visits"],
        estimation_data=df,
    )
    d = _dump(out)
    dist = d["propensity_distribution"]
    assert 0.0 <= dist["min"] <= dist["median"] <= dist["max"] <= 1.0
    assert dist["q25"] <= dist["median"] <= dist["q75"]
    assert 0.0 <= d["mean_propensity"] <= 1.0
    assert 0.0 <= d["common_support"] <= 1.0
    # Must not be the hardcoded placeholder distribution.
    assert not (
        dist["min"] == 0.05
        and dist["q25"] == 0.22
        and dist["median"] == 0.34
        and dist["q75"] == 0.48
        and dist["max"] == 0.92
    )


def test_module_imports_cleanly():
    assert Path(tr.__file__).exists()


# ============================================================================
# roi_estimator uncertainty: MEASURED from entity spread, never a constant band.
#
# The interval was previously a hardcoded +/-25% of the opportunity value. That
# is information-free -- it moves with the point estimate and with nothing else,
# so two gaps with wildly different entity spreads reported identically-shaped
# "confidence intervals". These tests pin the replacement to real data.
# ============================================================================


def test_roi_estimator_interval_responds_to_entity_spread():
    """Same gap and same investment, different interior spread -> different band.

    This is the property the +/-25% constant could never satisfy: both cases below
    have gap=100 and 4 entities, so the point ROI is identical, but the tight
    cluster is far less sensitive to dropping one entity than the case where a
    single entity creates the whole spread.
    """
    tight = roi_estimator(
        gap_analysis={"gap": 100.0, "entity_values": {"a": 0.0, "b": 49.0, "c": 51.0, "d": 100.0}},
        investment=1000.0,
    )
    fragile = roi_estimator(
        gap_analysis={"gap": 100.0, "entity_values": {"a": 0.0, "b": 1.0, "c": 2.0, "d": 100.0}},
        investment=1000.0,
    )

    tight_ci = _dump(tight)["confidence_interval"]
    fragile_ci = _dump(fragile)["confidence_interval"]

    assert len(tight_ci) == 2 and len(fragile_ci) == 2
    assert _dump(tight)["estimated_roi"] == _dump(fragile)["estimated_roi"]
    # The fragile set collapses much further when its outlier is dropped.
    assert fragile_ci[0] < tight_ci[0]
    # And the bands are genuinely different -- not a fixed ratio of the estimate.
    assert tight_ci != fragile_ci


def test_roi_estimator_interval_is_not_a_fixed_ratio_of_the_estimate():
    """Guards the specific regression: a band that is always estimate*(1-+0.25)."""
    out = _dump(
        roi_estimator(
            gap_analysis={
                "gap": 100.0,
                "entity_values": {"a": 0.0, "b": 1.0, "c": 2.0, "d": 100.0},
            },
            investment=1000.0,
        )
    )
    roi = out["estimated_roi"]
    lo, hi = out["confidence_interval"]
    assert not (lo == pytest.approx(roi * 0.75) and hi == pytest.approx(roi * 1.25))


def test_roi_estimator_omits_interval_when_spread_unmeasurable():
    """n<3 entities -> no honest range exists; omit rather than substitute one."""
    out = _dump(
        roi_estimator(
            gap_analysis={"gap": 100.0, "entity_values": {"a": 50.0, "b": 150.0}},
            investment=1000.0,
        )
    )
    assert out["confidence_interval"] == []
    assert any("fewer than 3" in a for a in out["assumptions"])


def test_roi_estimator_discloses_band_is_not_a_confidence_interval():
    """The field is named confidence_interval; assumptions must not let that stand
    as an unqualified claim of coverage."""
    out = _dump(
        roi_estimator(
            gap_analysis={
                "gap": 100.0,
                "entity_values": {"a": 0.0, "b": 40.0, "c": 60.0, "d": 100.0},
            },
            investment=1000.0,
        )
    )
    joined = " ".join(out["assumptions"]).lower()
    assert "leave-one-out" in joined
    assert "not a sampling confidence interval" in joined


def test_roi_estimator_no_hardcoded_band_constant_remains():
    """Source-level guard: the 0.25 band constant must be gone."""
    src = inspect.getsource(tr.roi_estimator)
    assert "0.25" not in src
    assert "25%" not in src
