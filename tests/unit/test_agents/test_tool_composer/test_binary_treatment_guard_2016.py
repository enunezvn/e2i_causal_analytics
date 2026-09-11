"""``propensity_estimator`` and ``cate_analyzer`` refuse a treatment that is not 0/1 (#2016).

Both tools read the planner-bound ``treatment`` as a binary assignment, and neither
checked that it was one:

* ``propensity_estimator`` cast it with ``astype(int)``. A count such as
  ``prior_therapy_lines`` (0..3 on the live ``patient_journeys``) became a multi-class
  fit whose P(class == 1) was reported as "the propensity", and every unit with a
  value >= 2 fell out of the treated arm. A 0-1 rate such as ``adherence_rate``
  truncated to one class and was refused as "fewer than 2 classes", which names the
  wrong defect.
* ``cate_analyzer`` split on ``== 1`` vs ``== 0``, so a count silently estimated
  "exactly 1 vs exactly 0" and ignored every other value.

The treatment is reachable with such a column: an LLM plan binds any column, and the
deterministic KPI plan binds the first ``numeric-continuous`` driver when the frame has
no binary one (``composer._build_kpi_causal_plan``). #2003 / PR #2012 fixed the same
cast for ``risk_scorer.outcome``; all three tools now share one check.

Every test runs the REAL tools on real DataFrames.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.errors import ToolRefusalError
from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionStatus,
    SubQuestion,
)
from src.tool_registry.registry import get_registry

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _cohort(n: int = 400, seed: int = 2016) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    severity = rng.normal(size=n)
    engagement = rng.normal(size=n)
    treated = (severity + rng.normal(size=n) > 0).astype(int)
    outcome = 0.4 * treated + 0.5 * severity + rng.normal(size=n)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "age_group": rng.choice(["<50", "50-65", ">65"], size=n),
            "disease_severity": severity,
            "engagement_score": engagement,
            "copay_support": treated,
            "prior_therapy_lines": rng.integers(0, 4, size=n),
            "visits": rng.integers(0, 13, size=n),
            "adherence_rate": rng.uniform(0.2, 1.0, size=n),
            "outcome": outcome,
        }
    )


def _propensity(df: pd.DataFrame, treatment: str) -> Any:
    return tr.propensity_estimator(
        treatment=treatment,
        covariates=["disease_severity", "engagement_score"],
        estimation_data=df,
    )


def _cate(df: pd.DataFrame, treatment: str) -> Any:
    return tr.cate_analyzer(
        treatment=treatment,
        outcome="outcome",
        segments=["age_group"],
        estimation_data=df,
    )


# ---------------------------------------------------------------------------
# A count or a 0-1 rate is refused, with a reason that names the real defect
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", [_propensity, _cate], ids=["propensity", "cate"])
@pytest.mark.parametrize("treatment", ["visits", "prior_therapy_lines"])
def test_a_count_treatment_is_refused(tool, treatment):
    df = _cohort()
    with pytest.raises(ToolRefusalError) as excinfo:
        tool(df, treatment)
    reason = str(excinfo.value)
    assert f"treatment column {treatment!r} is not a binary 0/1 column" in reason
    distinct = int(df[treatment].nunique())
    assert f"{distinct} distinct non-null values" in reason
    # The observed values are named, in numeric order, so a reader sees it is a count.
    assert "[0, 1, 2, 3" in reason


@pytest.mark.parametrize("tool", [_propensity, _cate], ids=["propensity", "cate"])
def test_a_rate_treatment_is_refused_for_being_non_binary_not_single_class(tool):
    with pytest.raises(ToolRefusalError) as excinfo:
        tool(_cohort(), "adherence_rate")
    reason = str(excinfo.value)
    assert "treatment column 'adherence_rate' is not a binary 0/1 column" in reason
    # The pre-fix reasons: truncation to one class, or no segment with a 0/1 contrast.
    assert "fewer than 2 classes" not in reason
    assert "yields a measured CATE" not in reason


@pytest.mark.parametrize("tool", [_propensity, _cate], ids=["propensity", "cate"])
def test_a_rate_that_reaches_exactly_0_and_1_is_refused(tool):
    """The live shape: Kisqali ``adherence_rate`` holds exact 0 and 1 among 6,121 values.

    Pre-fix this was not refused at all. On the deployed image (main 56f8b8589) over the
    real 8,730-row frame, propensity_estimator returned mean_propensity 0.038 (every rate
    below 1 cast to 0), and cate_analyzer returned effects for 3 regions from only the
    rows at exactly 1 vs exactly 0.
    """
    df = _cohort()
    df.loc[: len(df) // 3, "adherence_rate"] = 1.0
    df.loc[len(df) // 3 : len(df) // 2, "adherence_rate"] = 0.0
    with pytest.raises(ToolRefusalError, match="'adherence_rate' is not a binary 0/1 column"):
        tool(df, "adherence_rate")


@pytest.mark.parametrize("tool", [_propensity, _cate], ids=["propensity", "cate"])
def test_a_duplicated_treatment_column_name_is_refused_structurally(tool):
    """``df[name]`` on a duplicated name is a DataFrame, whose ``unique()`` does not exist.

    The guard must refuse the ambiguous column rather than raise ``AttributeError`` into
    the executor's retrying arm (codex r1 MEDIUM).
    """
    base = _cohort()
    df = pd.concat([base, base[["visits"]].rename(columns={"visits": "copay_support"})], axis=1)
    assert list(df.columns).count("copay_support") == 2
    with pytest.raises(ToolRefusalError, match="2 columns are named 'copay_support'"):
        tool(df, "copay_support")


def test_the_observed_values_in_a_refusal_are_bounded():
    """Each rendered value is clipped, so a long value cannot push the reason past the
    composer's 2000-char carry limit, which truncates from the END (codex r1 LOW)."""
    df = _cohort(n=60)
    df["dose"] = [tuple(range(i, i + 2000)) for i in range(len(df))]
    with pytest.raises(ToolRefusalError) as excinfo:
        _cate(df, "dose")
    reason = str(excinfo.value)
    assert "treatment column 'dose' is not a binary 0/1 column" in reason
    assert len(reason) < 700, len(reason)


@pytest.mark.parametrize("tool", [_propensity, _cate], ids=["propensity", "cate"])
def test_a_two_valued_treatment_that_is_not_0_1_is_refused(tool):
    """{1, 2} is 'binary' to the planner's column profile (2 distinct numeric values).

    Pre-fix, propensity_estimator crashed on an empty control arm (np.min of an empty
    array, retried as a tool failure) and cate_analyzer blamed the segments.
    """
    df = _cohort()
    df["arm_code"] = df["copay_support"] + 1
    with pytest.raises(ToolRefusalError, match="'arm_code' is not a binary 0/1 column"):
        tool(df, "arm_code")


def test_a_list_valued_treatment_is_refused_structurally():
    """The guard itself must not raise ``TypeError: unhashable type`` into the retry arm."""
    df = _cohort(n=40)
    df["codes"] = [[i % 2] for i in range(len(df))]
    with pytest.raises(ToolRefusalError, match="'codes' is not a binary 0/1 column"):
        _propensity(df, "codes")


# ---------------------------------------------------------------------------
# Every 0/1 encoding still works, and gives the same answer as plain int
# ---------------------------------------------------------------------------
_ENCODINGS = {
    "int": lambda s: s.astype(int),
    "float": lambda s: s.astype(float),
    "bool": lambda s: s.astype(bool),
    "Int64": lambda s: s.astype("Int64"),
}


@pytest.mark.parametrize("encoding", sorted(_ENCODINGS))
def test_a_0_1_treatment_is_estimated_under_every_encoding(encoding):
    base = _cohort()
    df = base.copy()
    df["copay_support"] = _ENCODINGS[encoding](base["copay_support"])

    expected_ps = _propensity(base, "copay_support").model_dump()
    got_ps = _propensity(df, "copay_support").model_dump()
    assert got_ps == expected_ps

    expected_cate = _cate(base, "copay_support").model_dump()
    got_cate = _cate(df, "copay_support").model_dump()
    assert got_cate == expected_cate
    assert len(got_cate["effect_by_segment"]) == 3


def test_cate_ignores_null_treatment_rows_under_nullable_int64_as_under_float():
    """Null treatment rows sit in neither arm, under ``Int64`` exactly as under float.

    A regression pin, not a red test: it passed before the fix too (measured). The new
    guard runs ``dropna`` before the check, so a nullable column must stay estimable.
    """
    base = _cohort()
    as_float = base.copy()
    as_float["copay_support"] = base["copay_support"].astype(float)
    as_float.loc[::7, "copay_support"] = np.nan
    as_int64 = as_float.copy()
    as_int64["copay_support"] = as_float["copay_support"].astype("Int64")

    assert (
        _cate(as_int64, "copay_support").model_dump()
        == _cate(as_float, "copay_support").model_dump()
    )


@pytest.mark.parametrize("encoding", ["float", "Int64"])
def test_propensity_refuses_a_null_treatment_instead_of_crashing(encoding):
    """A propensity is P(treated | X) for units whose assignment is known.

    Pre-fix, ``astype(int)`` raised ``IntCastingNaNError`` (a plain ``ValueError``, so
    retried and charged to the circuit breaker). Dropping the rows would silently shrink
    the population the reported distribution describes, and the output has no field to
    disclose it, so the tool refuses and names the count.
    """
    df = _cohort()
    df["copay_support"] = df["copay_support"].astype(float)
    df.loc[::10, "copay_support"] = np.nan
    if encoding == "Int64":
        df["copay_support"] = df["copay_support"].astype("Int64")
    with pytest.raises(ToolRefusalError) as excinfo:
        _propensity(df, "copay_support")
    reason = str(excinfo.value)
    assert "'copay_support'" in reason
    assert "40 null" in reason


# ---------------------------------------------------------------------------
# One shared check across risk_scorer, propensity_estimator and cate_analyzer
# ---------------------------------------------------------------------------
def test_the_three_tools_share_one_binary_check():
    df = _cohort()
    df["discontinued"] = df["visits"]
    reasons: Dict[str, str] = {}
    for name, call in {
        "risk_scorer": lambda: tr.risk_scorer(
            entity_type="patient", risk_type="x", estimation_data=df, outcome="discontinued"
        ),
        "propensity_estimator": lambda: _propensity(df, "discontinued"),
        "cate_analyzer": lambda: _cate(df, "discontinued"),
    }.items():
        with pytest.raises(ToolRefusalError) as excinfo:
            call()
        reasons[name] = str(excinfo.value)

    distinct = int(df["visits"].nunique())
    shared = (
        f"column 'discontinued' is not a binary 0/1 column: its non-null values must be a "
        f"subset of {{0, 1}} (bool, int, float or nullable Int64), but it carries {distinct} "
        "distinct non-null values"
    )
    for name, reason in reasons.items():
        assert reason.startswith(f"{name}: "), reason
        assert shared in reason, (name, reason)


# ---------------------------------------------------------------------------
# Reachability: the deterministic KPI plan binds a count driver as the treatment
# ---------------------------------------------------------------------------
@pytest.fixture
def _clean_bounded_pool():
    from src.api.dependencies.compute import _reset_limiter_cache_for_tests

    _reset_limiter_cache_for_tests()
    yield
    _reset_limiter_cache_for_tests()


@pytest.mark.asyncio
async def test_a_kpi_plan_binding_a_count_driver_gets_a_structured_cate_refusal(
    _clean_bounded_pool,
):
    """A KPI frame with no binary driver: the plan binds the count ``visits``.

    The plan's ``cate_analyzer`` step runs through the real executor and registry, and
    fails once with the binary reason instead of reporting a "1 vs 0" CATE.
    """
    rng = np.random.default_rng(7)
    n = 240
    frame = pd.DataFrame(
        {
            "converted": rng.integers(0, 2, size=n),
            "visits": rng.integers(0, 13, size=n),
            "delivery_channel": rng.choice(["email", "crm", "phone"], size=n),
        }
    )
    decomposition = DecompositionResult(
        original_query="what drove conversion and which segments respond best",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question="which segments respond best?",
                intent="COMPARATIVE",
                entities=[],
                depends_on=[],
            )
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    context = {"estimation_data": frame, "kpi_outcome": "converted"}
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    plan = composer._build_kpi_causal_plan(decomposition, context, "converted")
    assert plan is not None
    cate_step = next(s for s in plan.steps if s.tool_name == "cate_analyzer")
    assert cate_step.input_mapping["treatment"] == "visits"

    plan.steps = [cate_step]
    plan.parallel_groups = [[cate_step.step_id]]
    executor = PlanExecutor(
        tool_registry=get_registry(),
        max_retries=2,
        backoff_base_delay=0.0,
        backoff_max_delay=0.0,
        enable_caching=False,
    )
    trace = await executor.execute(plan, context)

    result = trace.step_results[0]
    assert result.status == ExecutionStatus.FAILED
    assert result.output.success is False
    assert "treatment column 'visits' is not a binary 0/1 column" in (result.output.error or "")
