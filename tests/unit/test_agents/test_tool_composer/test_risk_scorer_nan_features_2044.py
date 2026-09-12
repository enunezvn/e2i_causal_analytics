"""``risk_scorer`` handles a numeric feature carrying missing values (#2044).

The tool is otherwise carefully fail-closed: no DataFrame, a missing outcome column, a
missing ``id_column``, a non-binary outcome, fewer than 2 outcome classes and no usable
numeric features each raise ``ToolRefusalError`` with an honest, specific reason. NaN
features were the one unguarded condition: ``work[feature_cols].astype(float)`` went
straight into ``LogisticRegression.fit`` and sklearn raised a raw
``ValueError: Input X contains NaN`` that reached the user, taking three dependent steps
down with it.

The fixtures mirror the shape MEASURED on the real 8,730 x 85 Kisqali cohort
(``cohort_resolution.resolve_cohort_frame("Kisqali", None)``, 2026-09-12): 29 numeric
columns, of which exactly two carry NaN, and BOTH are structurally missing --
``days_to_treatment`` is NaN for exactly the 5,712 rows with ``treatment_initiated == 0``
(no days-to-treatment for a patient who never started) and ``gap_days`` for exactly the
7 rows with ``adherence_rate == 0.0`` (no refill gap for a patient who never refilled).

Two properties are pinned here, because a percentage alone gets both wrong:

* **Structural exclusion is independent of the share.** A column undefined for a knowable
  subpopulation is excluded whether it is missing for 65% of rows or 20%, because keeping
  it deletes exactly that subpopulation. A threshold-only rule retains it at 20% and
  silently deletes every never-treated patient.
* **Aggregate retention is gated, not just per-column missingness.** Five features each
  under the per-column limit, missing on disjoint rows, can complete-case away most of
  the cohort while every individual column looks fine.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError
from src.agents.tool_composer.synthesizer import (
    SYNTHESIS_OUTPUT_BUDGET_CHARS,
    project_tool_output,
)

N_SPORADIC_NAN = 5


def _kisqali_shaped_cohort(n: int = 800, seed: int = 2044) -> pd.DataFrame:
    """A cohort with the real frame's two NaN patterns: structural and sporadic."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(size=n)
    # ~35% initiate treatment, matching the measured 3,018 / 8,730.
    initiated = (rng.random(n) < 0.35).astype(int)

    # STRUCTURAL: undefined wherever treatment never started (65% NaN on the real frame).
    days_to_treatment = np.where(initiated == 1, rng.uniform(1, 90, size=n), np.nan)

    # SPORADIC: a handful of genuinely missing values explained by no other column.
    gap_days = rng.uniform(0, 30, size=n)
    gap_days[rng.choice(n, size=N_SPORADIC_NAN, replace=False)] = np.nan

    event = (rng.random(n) < 1.0 / (1.0 + np.exp(-severity))).astype(int)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "disease_severity": severity,
            "engagement_score": rng.normal(size=n),
            "age_at_diagnosis": rng.integers(30, 85, size=n),
            "treatment_initiated": initiated,
            "days_to_treatment": days_to_treatment,
            "gap_days": gap_days,
            "discontinued_180d": event,
        }
    )


def _score(df: pd.DataFrame, **kwargs):
    return tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=df,
        outcome="discontinued_180d",
        **kwargs,
    )


# ============================================================================
# The defect: a NaN feature crashed the fit with a raw sklearn ValueError.
# ============================================================================


def test_a_missing_feature_value_no_longer_crashes_the_fit():
    """Today: ``ValueError: Input X contains NaN`` escapes to the user."""
    out = _score(_kisqali_shaped_cohort())
    assert all(0.0 <= s["risk_score"] <= 1.0 for s in out.scores)
    assert {s["risk_tier"] for s in out.scores} <= {"low", "medium", "high"}


def test_the_structurally_missing_column_is_dropped_not_the_rows():
    """Columns-first keeps the never-treated population; rows-first would delete it."""
    df = _kisqali_shaped_cohort()
    out = _score(df)

    # Only the sporadically-missing rows are lost -- NOT the never-treated.
    assert out.n_scored == len(df) - N_SPORADIC_NAN
    assert out.n_rows_dropped == N_SPORADIC_NAN
    assert len(out.scores) == out.n_scored

    # The surviving entity set is EXACTLY the rows with no sporadic gap.
    expected = set(df.loc[df["gap_days"].notna(), "patient_id"].astype(str))
    assert {s["entity_id"] for s in out.scores} == expected
    # The never-treated population survives: only its share of the sporadic gaps is lost,
    # never the whole subpopulation the way a rows-first complete-case would take it.
    never_treated = set(df.loc[df["treatment_initiated"] == 0, "patient_id"].astype(str))
    assert len(never_treated & expected) >= len(never_treated) - N_SPORADIC_NAN


# ============================================================================
# Structural missingness is identified by APPLICABILITY, not by a percentage.
#
# A column undefined for a knowable subpopulation must be excluded at ANY share.
# With a threshold-only rule, a cohort of 100 with 20 never-treated patients puts
# ``days_to_treatment`` at exactly 20% -- under the limit, so retained -- and the
# row stage then deletes all 20 never-treated patients. Because triage runs AFTER
# the ``entity_ids`` filter, a requested subset reaches this on live data.
# ============================================================================


def _subset_cohort(n: int = 100, n_never: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    initiated = np.array([0] * n_never + [1] * (n - n_never))
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:03d}" for i in range(n)],
            "disease_severity": rng.normal(size=n),
            "treatment_initiated": initiated,
            "days_to_treatment": np.where(initiated == 1, rng.uniform(1, 90, size=n), np.nan),
            "discontinued_180d": rng.integers(0, 2, size=n),
        }
    )


def test_a_structural_column_is_excluded_even_below_the_threshold():
    df = _subset_cohort()
    assert df["days_to_treatment"].isna().mean() == pytest.approx(0.20)
    out = _score(df)
    assert out.n_scored == len(df)
    assert out.n_rows_dropped == 0
    assert "days_to_treatment" in out.missing_data_disclosure
    assert "undefined wherever treatment_initiated == 0" in out.missing_data_disclosure


def test_the_structural_rule_names_the_subpopulation_not_just_the_count():
    """A count is not a disclosure of WHAT was set aside."""
    out = _score(_subset_cohort())
    assert "undefined wherever" in out.missing_data_disclosure
    assert "20 of 100" in out.missing_data_disclosure


def test_a_sporadically_missing_column_under_the_limit_is_retained():
    """Non-structural gaps cost rows, not the feature: 5 rows beats losing a column."""
    df = _kisqali_shaped_cohort()
    out = _score(df)
    # gap_days is retained (its NaNs are explained by nothing), so rows were dropped.
    assert out.n_rows_dropped == N_SPORADIC_NAN
    assert "gap_days" not in out.missing_data_disclosure


# ---- boundary: pins the 20% limit, so a 50% cutoff cannot pass ----------------


def _sporadic_cohort(n: int = 200, n_missing: int = 40, seed: int = 7) -> pd.DataFrame:
    """``spotty`` is missing on rows no other column identifies."""
    rng = np.random.default_rng(seed)
    spotty = rng.normal(size=n)
    spotty[rng.choice(n, size=n_missing, replace=False)] = np.nan
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:03d}" for i in range(n)],
            "disease_severity": rng.normal(size=n),
            "engagement_score": rng.normal(size=n),
            "spotty": spotty,
            "discontinued_180d": rng.integers(0, 2, size=n),
        }
    )


def test_a_non_structural_column_at_the_limit_is_retained():
    """Exactly 20% missing is NOT over the limit -- the column stays, 40 rows go."""
    df = _sporadic_cohort(n_missing=40)
    assert df["spotty"].isna().mean() == pytest.approx(0.20)
    out = _score(df)
    assert "spotty" not in out.missing_data_disclosure
    assert out.n_rows_dropped == 40


def test_a_non_structural_column_just_over_the_limit_is_excluded():
    """21% is over -- a 50% cutoff would leave this green while violating the policy."""
    df = _sporadic_cohort(n_missing=42)
    assert df["spotty"].isna().mean() > 0.20
    out = _score(df)
    assert "spotty" in out.missing_data_disclosure
    assert out.n_rows_dropped == 0
    assert out.n_scored == len(df)


# ============================================================================
# Aggregate retention: per-column limits cannot see what the fit ends up on.
# ============================================================================


def _disjoint_gaps_cohort(n: int = 100, k: int = 5, per: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    cols = {}
    for j in range(k):
        values = rng.normal(size=n)
        values[j * per : (j + 1) * per] = np.nan
        cols[f"f{j}"] = values
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:03d}" for i in range(n)],
            "discontinued_180d": np.array([0, 1] * (n // 2)),
            **cols,
        }
    )


def test_it_refuses_when_triage_would_score_only_a_small_minority():
    """5 features at 19% on disjoint rows retains 5 of 100 -- a different cohort."""
    with pytest.raises(ToolRefusalError) as exc:
        _score(_disjoint_gaps_cohort())
    message = str(exc.value)
    assert "5 of 100" in message
    assert "Refusing" in message


def test_a_majority_of_the_cohort_still_scores():
    """The adequacy gate must not fire on ordinary sporadic missingness."""
    out = _score(_sporadic_cohort(n_missing=40))
    assert out.n_scored == 160


def test_it_refuses_when_triage_leaves_a_single_outcome_class():
    """Class loss caused BY the triage must refuse, not fit a degenerate model.

    The gap sits on 10% of rows, so ``spotty`` is RETAINED (under the limit, and
    explained by no feature column) and the ROW stage is what removes the class.
    """
    n = 100
    rng = np.random.default_rng(11)
    outcome = np.array([0] * 90 + [1] * 10)
    feature = rng.normal(size=n)
    feature[outcome == 1] = np.nan  # every positive row is incomplete
    df = pd.DataFrame(
        {
            "patient_id": [f"pt-{i:03d}" for i in range(n)],
            "disease_severity": rng.normal(size=n),
            "spotty": feature,
            "discontinued_180d": outcome,
        }
    )
    with pytest.raises(ToolRefusalError) as exc:
        _score(df)
    assert "class" in str(exc.value).lower() or "Refusing" in str(exc.value)


# ============================================================================
# Ambiguous feature names: the triage must not introduce a NEW crash.
# ============================================================================


def test_duplicate_numeric_feature_names_are_refused_not_crashed():
    """Two numeric columns named ``x`` made ``nan_counts[c]`` a Series -> raw ValueError."""
    frame = pd.DataFrame(np.column_stack([np.arange(10.0), np.arange(10.0)]), columns=["x", "x"])
    frame["patient_id"] = [f"p{i}" for i in range(10)]
    frame["discontinued_180d"] = [0, 1] * 5
    with pytest.raises(ToolRefusalError) as exc:
        _score(frame)
    assert "x" in str(exc.value)
    assert "duplicat" in str(exc.value).lower()


# ============================================================================
# The denominator is the population the CALLER asked about.
# ============================================================================


def _complete_ids(df: pd.DataFrame, k: int = 100) -> list[str]:
    """``k`` entity IDs whose rows survive the NaN triage untouched."""
    return df.loc[df["gap_days"].notna(), "patient_id"].astype(str).tolist()[:k]


def test_the_disclosure_counts_requested_rows_not_the_whole_frame():
    df = _kisqali_shaped_cohort()
    ids = _complete_ids(df)
    out = _score(df, entity_ids=ids)
    assert out.n_scored == len(ids)
    assert out.n_rows_dropped == 0
    disclosure = out.missing_data_disclosure
    assert f"scored {len(ids)} of {len(ids)} requested row(s)" in disclosure
    assert str(len(df)) not in disclosure, disclosure


def test_entity_ids_that_match_nothing_are_reported_not_silently_narrowed():
    """A partial ID match is a silent narrowing unless the tool says so."""
    df = _kisqali_shaped_cohort()
    ids = _complete_ids(df)
    absent = [f"absent-{i:03d}" for i in range(40)]
    out = _score(df, entity_ids=ids + absent)
    assert out.n_scored == len(ids)
    assert f"{len(ids)} of {len(ids) + len(absent)} requested entity ID(s) matched" in (
        out.missing_data_disclosure
    )


def test_a_repeated_entity_id_is_counted_once():
    """The filter is set-based, so counting the raw list would fake a shortfall."""
    df = _kisqali_shaped_cohort()
    ids = _complete_ids(df, k=50)
    out = _score(df, entity_ids=ids + ids)
    assert out.n_scored == len(ids)
    assert f"{len(ids)} of {len(ids)} requested entity ID(s) matched" in (
        out.missing_data_disclosure
    )


def test_a_matched_id_dropped_for_missing_values_is_visible_in_both_counts():
    """Matched-then-dropped: the ID was present, the row still did not survive."""
    df = _kisqali_shaped_cohort()
    incomplete = df.loc[df["gap_days"].isna(), "patient_id"].astype(str).tolist()
    ids = _complete_ids(df, k=20) + incomplete
    out = _score(df, entity_ids=ids)
    assert out.n_rows_dropped == len(incomplete)
    assert out.n_scored == 20
    disclosure = out.missing_data_disclosure
    # All requested IDs matched; the loss is the row stage, and both are stated.
    assert f"{len(ids)} of {len(ids)} requested entity ID(s) matched" in disclosure
    assert f"scored 20 of {len(ids)} requested row(s)" in disclosure


# ============================================================================
# Clean path, refusal backstops, and the trip into the synthesis prompt (#2019).
# ============================================================================


def test_a_complete_cohort_reports_no_drops():
    df = _kisqali_shaped_cohort().drop(columns=["days_to_treatment"]).dropna()
    out = _score(df)
    assert out.n_scored == len(df)
    assert out.n_rows_dropped == 0
    assert "no feature column excluded" in out.missing_data_disclosure
    assert "no row dropped" in out.missing_data_disclosure


def test_it_refuses_when_every_feature_is_too_incomplete_to_use():
    df = _kisqali_shaped_cohort()
    for col in ("disease_severity", "engagement_score", "age_at_diagnosis", "gap_days"):
        df.loc[df.index[: int(len(df) * 0.9)], col] = np.nan
    df = df.drop(columns=["treatment_initiated"])
    with pytest.raises(ToolRefusalError) as exc:
        _score(df)
    message = str(exc.value)
    assert "days_to_treatment" in message and "disease_severity" in message
    assert "refus" in message.lower()


def test_it_refuses_when_no_row_is_complete_across_the_usable_features():
    """Each feature is individually usable, but their gaps tile the whole cohort."""
    n = 800
    rng = np.random.default_rng(2044)
    cols = {}
    block = n // 8
    for k in range(8):
        values = rng.normal(size=n)
        values[k * block : (k + 1) * block] = np.nan
        cols[f"feature_{k}"] = values
    df = pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "discontinued_180d": rng.integers(0, 2, size=n),
            **cols,
        }
    )
    with pytest.raises(ToolRefusalError) as exc:
        _score(df)
    assert "0" in str(exc.value)
    assert "complete" in str(exc.value).lower()


def test_the_disclosure_survives_the_synthesis_projection_as_intact_fields():
    """#2019 keeps every SCALAR field in full; ``scores`` is what gets trimmed."""
    out = _score(_kisqali_shaped_cohort())
    rendered = project_tool_output(out.model_dump(), SYNTHESIS_OUTPUT_BUDGET_CHARS)
    # Parse the JSON body rather than substring-matching, so unrelated score text
    # in the trimmed ``scores`` array cannot satisfy the assertion.
    body = json.loads(rendered[: rendered.rindex("}") + 1])
    assert body["n_scored"] == out.n_scored
    assert body["n_rows_dropped"] == out.n_rows_dropped
    assert body["missing_data_disclosure"] == out.missing_data_disclosure
    assert len(body["scores"]) < out.n_scored  # the bulky container IS trimmed
