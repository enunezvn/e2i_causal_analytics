"""``risk_scorer`` handles a numeric feature carrying missing values (#2044).

The tool is otherwise carefully fail-closed: no DataFrame, a missing outcome column, a
missing ``id_column``, a non-binary outcome, fewer than 2 outcome classes and no usable
numeric features each raise ``ToolRefusalError`` with an honest, specific reason. NaN
features were the one unguarded condition: ``work[feature_cols].astype(float)`` went
straight into ``LogisticRegression.fit`` and sklearn raised a raw
``ValueError: Input X contains NaN`` that reached the user, taking three dependent steps
down with it.

The fixtures mirror the shape MEASURED on the real 8,730 x 85 Kisqali cohort
(``cohort_resolution.resolve_cohort_frame("Kisqali", None)``, 2026-09-12):

* 29 numeric columns, of which exactly two carry NaN;
* ``days_to_treatment`` is NaN for 5,712 of 8,730 rows (65.4%) and its missingness is
  PERFECTLY STRUCTURAL -- NaN for exactly the rows with ``treatment_initiated == 0`` and
  for no other row. There is no "days to treatment" for a patient who never started, so
  every imputed value would be fabricated, and complete-casing on it would silently
  delete the entire never-treated population (outcome prevalence 0.4204 over the cohort
  vs 0.4338 over the complete cases);
* ``gap_days`` is NaN for 7 of 8,730 rows (0.08%) -- ordinary sporadic missingness that
  costs nothing to complete-case away.

Hence the ordering under test: triage FEATURE COLUMNS first, then complete-case the rows
that remain. Rows-first would have scored 3,015 of 8,730 patients; columns-first scores
8,723 of 8,730.
"""

from __future__ import annotations

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

    # STRUCTURAL: undefined wherever treatment never started (65% NaN on this frame).
    days_to_treatment = np.where(initiated == 1, rng.uniform(1, 90, size=n), np.nan)

    # SPORADIC: a handful of genuinely missing values, as gap_days carries live.
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


def _score(df: pd.DataFrame):
    return tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=df,
        outcome="discontinued_180d",
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

    # Only the sporadically-missing rows are lost -- NOT the 65% never-treated.
    assert out.n_scored == len(df) - N_SPORADIC_NAN
    assert out.n_rows_dropped == N_SPORADIC_NAN
    assert len(out.scores) == out.n_scored

    # The never-treated cohort is still represented in the scored entities.
    never_treated = set(df.loc[df["treatment_initiated"] == 0, "patient_id"])
    scored = {s["entity_id"] for s in out.scores}
    assert len(never_treated & scored) > 0.9 * len(never_treated)


def test_the_dropped_column_and_rows_are_disclosed_not_silent():
    """A drop the caller cannot see is indistinguishable from a fabricated answer."""
    out = _score(_kisqali_shaped_cohort())
    disclosure = out.missing_data_disclosure
    assert "days_to_treatment" in disclosure
    assert str(N_SPORADIC_NAN) in disclosure
    # The kept feature must NOT be reported as excluded.
    assert "gap_days" not in disclosure.split("excluded")[-1].split(";")[0]


def test_a_complete_cohort_reports_no_drops():
    """The no-missing-data path is unchanged and says so explicitly."""
    df = _kisqali_shaped_cohort().drop(columns=["days_to_treatment"]).dropna()
    out = _score(df)
    assert out.n_scored == len(df)
    assert out.n_rows_dropped == 0
    assert "no " in out.missing_data_disclosure.lower()


# ============================================================================
# The denominator is the population the CALLER asked about (#2044).
#
# ``entity_ids`` narrows ``work`` before any of this runs. Reporting the scored
# count against the whole frame turns a complete answer over 100 requested
# patients into "scored 100 of 8730" -- a fabricated 98.9% data loss in the one
# field whose job is to stop a caller mistaking a partial cohort for the whole.
# ============================================================================


def _complete_ids(df: pd.DataFrame, k: int = 100) -> list[str]:
    """``k`` entity IDs whose rows survive the NaN triage untouched."""
    return df.loc[df["gap_days"].notna(), "patient_id"].astype(str).tolist()[:k]


def test_the_disclosure_counts_requested_rows_not_the_whole_frame():
    df = _kisqali_shaped_cohort()
    ids = _complete_ids(df)
    out = tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=df,
        outcome="discontinued_180d",
        entity_ids=ids,
    )
    assert out.n_scored == len(ids)
    assert out.n_rows_dropped == 0
    disclosure = out.missing_data_disclosure
    # The full-frame row count must not appear as the denominator.
    assert str(len(df)) not in disclosure, disclosure
    assert "requested" in disclosure, disclosure


def test_entity_ids_that_match_nothing_are_reported_not_silently_narrowed():
    """A partial ID match is a silent narrowing unless the tool says so."""
    df = _kisqali_shaped_cohort()
    ids = _complete_ids(df)
    absent = [f"absent-{i:03d}" for i in range(40)]
    out = tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=df,
        outcome="discontinued_180d",
        entity_ids=ids + absent,
    )
    assert out.n_scored == len(ids)
    assert f"{len(ids)} of {len(ids) + len(absent)}" in out.missing_data_disclosure


# ============================================================================
# Fail-closed backstops: when nothing usable survives, refuse with the numbers.
# ============================================================================


def test_it_refuses_when_every_feature_is_too_incomplete_to_use():
    df = _kisqali_shaped_cohort()
    for col in ("disease_severity", "engagement_score", "age_at_diagnosis", "gap_days"):
        df.loc[df.index[: int(len(df) * 0.9)], col] = np.nan
    df = df.drop(columns=["treatment_initiated"])
    with pytest.raises(ToolRefusalError) as exc:
        _score(df)
    message = str(exc.value)
    assert "days_to_treatment" in message and "disease_severity" in message
    assert "Refusing" in message or "refus" in message.lower()


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


# ============================================================================
# The disclosure has to survive the trip into the synthesis prompt (#2019).
# ============================================================================


def test_the_disclosure_survives_the_synthesis_projection():
    """#2019 keeps every SCALAR field in full; ``scores`` is what gets trimmed."""
    out = _score(_kisqali_shaped_cohort())
    rendered = project_tool_output(out.model_dump(), SYNTHESIS_OUTPUT_BUDGET_CHARS)
    assert len(rendered) > SYNTHESIS_OUTPUT_BUDGET_CHARS * 0  # sanity: non-empty
    assert "days_to_treatment" in rendered
    assert str(out.n_scored) in rendered
    assert str(out.n_rows_dropped) in rendered
