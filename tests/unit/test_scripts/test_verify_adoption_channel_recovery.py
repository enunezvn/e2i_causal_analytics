"""The recovery-probe gate of scripts/verify_adoption_channel_recovery.py (lane T1).

The gate is a pure function of a fits table so its pass and fail cases can be pinned without
a forest fit. Per brand: CI covers the planted RD 8/8, |ATE - planted| <= 0.06 8/8,
Spearman(ATE, planted) >= 0.8, and the null channel |ATE| <= 0.06 with its CI covering 0.
"""

from __future__ import annotations

import pandas as pd

from scripts.verify_adoption_channel_recovery import evaluate_recovery_gate, planted_rd_by_column
from src.data.per_hcp_cohort_columns import (
    ADOPTION_CHANNEL_PLANTED_RD,
    ADOPTION_NULL_CHANNEL,
    INTERVENTION_TREATMENT_MAP,
)

_NULL_COL = INTERVENTION_TREATMENT_MAP[ADOPTION_NULL_CHANNEL]


def _fits(brand: str, ate_offset: float = 0.0, overrides: dict | None = None) -> pd.DataFrame:
    planted = planted_rd_by_column()
    rows = []
    for col, rd in planted.items():
        ate = rd + ate_offset
        rows.append(
            {
                "brand": brand,
                "channel": col,
                "planted_rd": rd,
                "ate": ate,
                "ci_lower": ate - 0.17,
                "ci_upper": ate + 0.17,
                "n": 3400,
                "error": None,
            }
        )
    df = pd.DataFrame(rows)
    for (col, field), value in (overrides or {}).items():
        df.loc[df["channel"] == col, field] = value
    return df


def test_planted_rd_by_column_maps_the_intervention_table_onto_the_planted_columns():
    by_col = planted_rd_by_column()
    assert set(by_col) == set(INTERVENTION_TREATMENT_MAP.values())
    for k, col in INTERVENTION_TREATMENT_MAP.items():
        assert by_col[col] == ADOPTION_CHANNEL_PLANTED_RD[k]
    assert by_col[_NULL_COL] == 0.0


def test_gate_passes_when_every_clause_holds():
    fits = pd.concat([_fits("Remibrutinib", 0.01), _fits("Fabhalta", -0.02), _fits("Kisqali")])
    result = evaluate_recovery_gate(fits)
    assert result.passed
    for brand, g in result.per_brand.items():
        assert g.covers == 8 and g.within_tol == 8 and g.spearman >= 0.8 and g.null_ok, (brand, g)
        assert g.failures == []


def test_gate_fails_when_a_ci_misses_the_planted_rd():
    fits = _fits(
        "Kisqali", overrides={("engagement_score", "ci_lower"): 0.15}
    )  # planted 0.138 < 0.15
    result = evaluate_recovery_gate(fits)
    assert not result.passed
    assert result.per_brand["Kisqali"].covers == 7
    assert any("covers" in f for f in result.per_brand["Kisqali"].failures)


def test_gate_fails_when_a_point_estimate_is_off_by_more_than_the_tolerance():
    fits = _fits("Kisqali", overrides={("speaker_program_count", "ate"): 0.113 + 0.07})
    result = evaluate_recovery_gate(fits)
    assert not result.passed
    assert result.per_brand["Kisqali"].within_tol == 7


def test_gate_fails_when_the_planted_ordering_is_not_recovered():
    planted = planted_rd_by_column()
    reversed_ate = {col: 0.14 - rd for col, rd in planted.items()}  # reverse the ordering
    fits = _fits("Fabhalta")
    fits["ate"] = fits["channel"].map(reversed_ate)
    fits["ci_lower"], fits["ci_upper"] = fits["ate"] - 0.2, fits["ate"] + 0.2
    result = evaluate_recovery_gate(fits, tol=1.0)
    assert not result.passed
    assert result.per_brand["Fabhalta"].spearman < 0.8


def test_gate_fails_when_the_null_channel_reads_as_an_effect():
    fits = _fits(
        "Remibrutinib", overrides={(_NULL_COL, "ate"): 0.08, (_NULL_COL, "ci_lower"): 0.01}
    )
    result = evaluate_recovery_gate(fits)
    assert not result.passed
    assert not result.per_brand["Remibrutinib"].null_ok


def test_gate_accepts_the_seed_artefact_null_inside_the_tolerance():
    # Remibrutinib's null read +0.05 before any planting (twinad_q3_fits.csv); the |ATE| <= 0.06
    # clause with a CI covering 0 accepts it. This pins that the clause is the tolerance, not 0.
    fits = _fits(
        "Remibrutinib",
        overrides={
            (_NULL_COL, "ate"): 0.053,
            (_NULL_COL, "ci_lower"): -0.11,
            (_NULL_COL, "ci_upper"): 0.21,
        },
    )
    assert evaluate_recovery_gate(fits).passed


def test_gate_fails_loud_on_an_errored_or_missing_fit():
    fits = _fits("Kisqali", overrides={("sample_volume", "error"): "TOO_FEW_USABLE_ROWS"})
    result = evaluate_recovery_gate(fits)
    assert not result.passed
    assert any("error" in f for f in result.per_brand["Kisqali"].failures)
    missing = _fits("Kisqali").iloc[:-1]
    result = evaluate_recovery_gate(missing)
    assert not result.passed


def test_gate_verdict_text_starts_with_the_verdict_word():
    assert evaluate_recovery_gate(_fits("Kisqali")).verdict().startswith("PASS")
    assert evaluate_recovery_gate(_fits("Kisqali", 0.2)).verdict().startswith("FAIL")
