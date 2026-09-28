"""The recovery-probe gate of scripts/verify_adoption_channel_recovery.py (lanes T1, T2).

The gate is a pure function of a fits table so its pass and fail cases can be pinned without
a forest fit. Point gate per brand: |ATE - planted| <= 0.06 8/8 and Spearman(ATE, planted)
>= 0.8. Lane T2 (the estimator's interval is now the forest's doubly-robust one): the four
focus channels' CIs exclude 0 in every brand; the null channel has |ATE| <= 0.06 in every
brand and a CI covering 0 in >= 2 of 3 brands, and any null CI excluding 0 is printed. CI
coverage of the planted RD is reported, not gated.
"""

from __future__ import annotations

import pandas as pd
import pytest

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
                # The DR interval's half-width on the live cohort is ~0.04-0.06.
                "ci_lower": ate - 0.045,
                "ci_upper": ate + 0.045,
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


def test_gate_passes_when_every_clause_holds_in_all_three_brands():
    fits = pd.concat([_fits("Remibrutinib", 0.01), _fits("Fabhalta", -0.02), _fits("Kisqali")])
    result = evaluate_recovery_gate(fits)
    assert set(result.per_brand) == {"Remibrutinib", "Fabhalta", "Kisqali"}
    assert result.passed
    for brand, g in result.per_brand.items():
        assert g.covers == 8 and g.within_tol == 8 and g.spearman >= 0.8 and g.null_ok, (brand, g)
        assert g.focus_significant == 4 and g.null_covers_zero, (brand, g)
        assert g.failures == []


def test_gate_fails_when_a_required_brand_is_missing():
    """codex r1 (MED): a --brands subset must never certify. The gate iterates the REQUIRED
    brands, not the brands present, so a one-brand table fails with the two others missing."""
    result = evaluate_recovery_gate(_fits("Kisqali"))
    assert not result.passed
    assert set(result.per_brand) == {"Remibrutinib", "Fabhalta", "Kisqali"}
    assert result.per_brand["Kisqali"].passed
    for brand in ("Remibrutinib", "Fabhalta"):
        assert any("missing" in f for f in result.per_brand[brand].failures), brand
    assert result.verdict().startswith("FAIL")


_ONE = ("Kisqali",)


def test_a_ci_missing_the_planted_rd_is_reported_not_gated():
    """Lane T2: coverage of the planted RD was a gate against ``ate_interval``'s over-wide
    bound. Against a calibrated 95% interval, 24 cells all covering happens ~0.95**24 = 29%
    of the time, so gating on it would fail a correct estimator most runs. Reported only."""
    fits = _fits(
        "Kisqali", overrides={("engagement_score", "ci_lower"): 0.15}
    )  # planted 0.138 < 0.15
    result = evaluate_recovery_gate(fits, required_brands=_ONE)
    assert result.passed
    assert result.per_brand["Kisqali"].covers == 7
    assert "covers 7/8" in result.verdict()


def test_gate_fails_when_a_point_estimate_is_off_by_more_than_the_tolerance():
    fits = _fits("Kisqali", overrides={("speaker_program_count", "ate"): 0.113 + 0.07})
    result = evaluate_recovery_gate(fits, required_brands=_ONE)
    assert not result.passed
    assert result.per_brand["Kisqali"].within_tol == 7


def test_gate_fails_when_the_planted_ordering_is_not_recovered():
    planted = planted_rd_by_column()
    reversed_ate = {col: 0.14 - rd for col, rd in planted.items()}  # reverse the ordering
    fits = _fits("Fabhalta")
    fits["ate"] = fits["channel"].map(reversed_ate)
    fits["ci_lower"], fits["ci_upper"] = fits["ate"] - 0.2, fits["ate"] + 0.2
    result = evaluate_recovery_gate(fits, tol=1.0, required_brands=("Fabhalta",))
    assert not result.passed
    assert result.per_brand["Fabhalta"].spearman < 0.8


def test_gate_fails_when_the_null_channel_reads_as_an_effect():
    fits = _fits(
        "Remibrutinib", overrides={(_NULL_COL, "ate"): 0.08, (_NULL_COL, "ci_lower"): 0.01}
    )
    result = evaluate_recovery_gate(fits)
    assert not result.passed
    assert not result.per_brand["Remibrutinib"].null_ok


def _three(**null_by_brand):
    """All three brands passing, with the null channel's (ate, lo, hi) overridden per brand."""
    parts = []
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        over = {}
        if brand in null_by_brand:
            ate, lo, hi = null_by_brand[brand]
            over = {
                (_NULL_COL, "ate"): ate,
                (_NULL_COL, "ci_lower"): lo,
                (_NULL_COL, "ci_upper"): hi,
            }
        parts.append(_fits(brand, overrides=over))
    return pd.concat(parts)


def test_a_focus_channel_ci_covering_zero_in_any_brand_fails():
    """Lane T2 significance gate: engagement / speaker / peer / PSP exclude 0 in 3/3 brands."""
    fits = _fits("Kisqali", overrides={("patient_support_enrollment", "ci_lower"): -0.01})
    result = evaluate_recovery_gate(fits, required_brands=_ONE)
    assert not result.passed
    assert result.per_brand["Kisqali"].focus_significant == 3
    assert any("patient_support_enrollment" in f for f in result.per_brand["Kisqali"].failures)


def test_one_null_ci_excluding_zero_passes_the_two_of_three_clause_but_is_printed():
    """The live Remibrutinib null (+0.047, CI +0.008..+0.086) is inside the tolerance and one
    brand of three; the gate passes, and the verdict names it on its own line so the >= 2/3
    clause can never hide it."""
    result = evaluate_recovery_gate(_three(Remibrutinib=(0.047, 0.008, 0.086)))
    assert result.passed
    assert result.null_covers == 2
    assert not result.per_brand["Remibrutinib"].null_covers_zero
    lines = [ln for ln in result.verdict().splitlines() if "null CI excludes 0" in ln]
    assert len(lines) == 1
    assert "Remibrutinib" in lines[0] and "+0.047" in lines[0]
    assert "known per-brand bias" in lines[0]


def test_two_null_cis_excluding_zero_fail_the_two_of_three_clause():
    result = evaluate_recovery_gate(
        _three(Remibrutinib=(0.047, 0.008, 0.086), Kisqali=(-0.05, -0.09, -0.01))
    )
    assert not result.passed
    assert result.null_covers == 1
    assert "null covers 0 in 1/3" in result.verdict()
    excl = [ln for ln in result.verdict().splitlines() if "null CI excludes 0" in ln]
    assert len(excl) == 2 and "known per-brand bias" not in excl[1]


def test_the_null_tolerance_binds_in_every_brand_even_with_a_covering_ci():
    result = evaluate_recovery_gate(_three(Fabhalta=(0.07, -0.02, 0.16)))
    assert not result.passed
    assert not result.per_brand["Fabhalta"].null_ok


def test_gate_accepts_the_seed_artefact_null_inside_the_tolerance():
    # Remibrutinib's null read +0.05 before any planting (twinad_q3_fits.csv); the |ATE| <= 0.06
    # clause with a CI covering 0 accepts it. This pins that the clause is the tolerance, not 0.
    fits = _three(Remibrutinib=(0.053, -0.11, 0.21))
    assert evaluate_recovery_gate(fits).passed


def test_gate_fails_loud_on_an_errored_or_missing_fit():
    fits = _fits("Kisqali", overrides={("sample_volume", "error"): "TOO_FEW_USABLE_ROWS"})
    result = evaluate_recovery_gate(fits, required_brands=_ONE)
    assert not result.passed
    assert any("error" in f for f in result.per_brand["Kisqali"].failures)
    missing = _fits("Kisqali").iloc[:-1]
    result = evaluate_recovery_gate(missing, required_brands=_ONE)
    assert not result.passed


def test_gate_verdict_text_starts_with_the_verdict_word():
    assert (
        evaluate_recovery_gate(_fits("Kisqali"), required_brands=_ONE).verdict().startswith("PASS")
    )
    assert (
        evaluate_recovery_gate(_fits("Kisqali", 0.2), required_brands=_ONE)
        .verdict()
        .startswith("FAIL")
    )


def test_certifying_thresholds_are_not_cli_overridable(tmp_path):
    """codex r2 (MED): `--tol 1 --min-spearman -1` would certify a structural null. The gate's
    tolerance, Spearman floor and seed are constants, not flags; argparse must reject them."""
    from scripts.verify_adoption_channel_recovery import main

    frame = tmp_path / "frame.parquet"
    for flag in ("--tol", "1"), ("--min-spearman", "-1"), ("--seed", "7"):
        with pytest.raises(SystemExit) as exc:
            main(["--frame", str(frame), *flag])
        assert exc.value.code == 2, flag


def test_via_twin_loader_requires_live():
    from scripts.verify_adoption_channel_recovery import main

    with pytest.raises(SystemExit) as exc:
        main(["--frame", "x.parquet", "--via-twin-loader"])
    assert exc.value.code == 2


class _Result:
    def __init__(self, data, count):
        self.data, self.count = data, count


class _Query:
    """PostgREST-shaped fake per table: eq filters, exact count, range windows."""

    def __init__(self, rows):
        self._rows, self._filters, self._window = rows, {}, None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a):
        return self

    def range(self, a, b):
        self._window = (a, b)
        return self

    async def execute(self):
        rows = [r for r in self._rows if all(r.get(k, v) == v for k, v in self._filters.items())]
        n = len(rows)
        if self._window is not None:
            rows = rows[self._window[0] : self._window[1] + 1]
        return _Result(rows, n)


class _Client:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _Query(self._tables.get(name, []))


def test_the_twin_loader_frame_is_what_simulate_estimates_on():
    """--via-twin-loader builds the frame with the twin's own load_cohort_frame (collapse,
    channel-row filter, synthetic-only, paged adoption join) for every brand."""
    import asyncio

    from scripts.verify_adoption_channel_recovery import frame_from_twin_loader

    channels = sorted(planted_rd_by_column())
    rollups, adoption = [], []
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        for i in range(30):
            row = {
                "hcp_id": f"h{i:03d}",
                "brand": brand,
                "metric_date": "2026-06-01",
                "region": "west",
                "market_share": 0.5,
                "triggers_total_count": 10.0,
                **{c: float(i) for c in channels},
            }
            rollups.append(row)
            # a post-plant ETL row for the same pair: no channel, must not be collapsed
            rollups.append(
                {
                    **{k: v for k, v in row.items() if k not in channels},
                    "metric_date": "2026-09-28",
                    "triggers_total_count": 999.0,
                }
            )
            adoption.append({"hcp_id": row["hcp_id"], "brand": brand, "adopted": i % 2})
    frame = asyncio.run(
        frame_from_twin_loader(
            _Client({"business_metrics": rollups, "hcp_brand_adoption": adoption})
        )
    )
    assert len(frame) == 90 and set(frame["brand"]) == {"Remibrutinib", "Fabhalta", "Kisqali"}
    assert frame["adopted"].notna().all()
    assert (frame["triggers_total_count"] == 10.0).all()
    assert {"hcp_id", "brand", "adopted", "region", *channels} <= set(frame.columns)
