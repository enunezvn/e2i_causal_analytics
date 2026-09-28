"""The recovery-probe gate of scripts/verify_adoption_channel_recovery.py (lanes T1, T2).

The gate is a pure function of a fits table so its pass and fail cases can be pinned without
a forest fit. Point gate per brand: |ATE - planted| <= 0.06 8/8 (the null included, at 0) and
Spearman(ATE, planted) >= 0.8. Lane T2 (the estimator's interval is the forest's doubly-robust
one): the four focus channels' CIs exclude 0 in every brand. The null channel's INTERVAL is
gated at the family level (owner option A, 2026-09-28): the null is refit on the live design
under fresh DGP seeds, and the pooled false-positive rate must be <= 0.10 and the reported SE
>= 0.9x the empirical SD. A realised null CI excluding 0 is printed as a draw, never gated.
CI coverage of the planted RD is reported, not gated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.verify_adoption_channel_recovery import (
    BRANDS,
    MAX_NULL_FP_RATE,
    MIN_SE_RATIO,
    NULL_CALIBRATION_SEEDS,
    evaluate_null_calibration,
    evaluate_recovery_gate,
    planted_rd_by_column,
    redraw_adopted,
    wilson_interval,
)
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


_Z = 1.959963984540054


def _null_fits(
    ates_by_brand: dict | None = None, *, stderr: float = 0.02, brands=BRANDS, seeds=None
) -> pd.DataFrame:
    """A null-calibration fits table: one row per brand x seed, CI = ate +- z * stderr.

    Default ATEs are the normal quantiles at SD ``stderr`` (deterministic, so the FP rate and
    the SE ratio are exact): a calibrated null, ~5% of CIs excluding 0 and SE/empSD ~1."""
    from statistics import NormalDist

    seeds = list(NULL_CALIBRATION_SEEDS if seeds is None else seeds)
    k = len(seeds)
    calibrated = [stderr * NormalDist().inv_cdf((i + 0.5) / k) for i in range(k)]
    rows = []
    for brand in brands:
        ates = (ates_by_brand or {}).get(brand, calibrated)
        for seed, ate in zip(seeds, ates, strict=True):
            rows.append(
                {
                    "brand": brand,
                    "seed": seed,
                    "ate": ate,
                    "stderr": stderr,
                    "ci_lower": ate - _Z * stderr,
                    "ci_upper": ate + _Z * stderr,
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


def _calibration_ok():
    return evaluate_null_calibration(_null_fits(), reproduction=1.0)


def test_planted_rd_by_column_maps_the_intervention_table_onto_the_planted_columns():
    by_col = planted_rd_by_column()
    assert set(by_col) == set(INTERVENTION_TREATMENT_MAP.values())
    for k, col in INTERVENTION_TREATMENT_MAP.items():
        assert by_col[col] == ADOPTION_CHANNEL_PLANTED_RD[k]
    assert by_col[_NULL_COL] == 0.0


def test_gate_passes_when_every_clause_holds_in_all_three_brands():
    fits = pd.concat([_fits("Remibrutinib", 0.01), _fits("Fabhalta", -0.02), _fits("Kisqali")])
    result = evaluate_recovery_gate(fits, null_calibration=_calibration_ok())
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
    assert result.per_brand["Kisqali"].passed
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


def test_the_null_tolerance_binds_in_every_brand_even_with_a_covering_ci():
    result = evaluate_recovery_gate(_three(Fabhalta=(0.07, -0.02, 0.16)))
    assert not result.passed
    assert not result.per_brand["Fabhalta"].null_ok


def test_gate_fails_loud_on_an_errored_or_missing_fit():
    fits = _fits("Kisqali", overrides={("sample_volume", "error"): "TOO_FEW_USABLE_ROWS"})
    result = evaluate_recovery_gate(fits, required_brands=_ONE)
    assert not result.passed
    assert any("error" in f for f in result.per_brand["Kisqali"].failures)
    missing = _fits("Kisqali").iloc[:-1]
    result = evaluate_recovery_gate(missing, required_brands=_ONE)
    assert not result.passed


def test_gate_verdict_text_starts_with_the_verdict_word():
    all_three = pd.concat([_fits(b) for b in BRANDS], ignore_index=True)
    assert (
        evaluate_recovery_gate(all_three, null_calibration=_calibration_ok())
        .verdict()
        .startswith("PASS")
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


def test_a_brand_subset_never_certifies():
    """codex r6 (MED): a subset is diagnostic, never a PASS -- even with a passing family-level
    null calibration, a per-brand table missing a required brand cannot certify."""
    one = evaluate_recovery_gate(
        _fits("Kisqali"), required_brands=_ONE, null_calibration=_calibration_ok()
    )
    assert one.per_brand["Kisqali"].passed
    assert not one.passed
    assert one.verdict().startswith("FAIL")
    two = pd.concat([_fits(b) for b in BRANDS[:2]], ignore_index=True)
    assert not evaluate_recovery_gate(
        two, required_brands=BRANDS[:2], null_calibration=_calibration_ok()
    ).passed


# ---------------------------------------------------------------------------
# Family-level null calibration (owner option A, 2026-09-28)
# ---------------------------------------------------------------------------


def test_the_calibration_constants_are_the_owner_decision():
    """The certifying thresholds and the replayable seed list are module constants."""
    assert MAX_NULL_FP_RATE == 0.10
    assert MIN_SE_RATIO == 0.9
    assert len(NULL_CALIBRATION_SEEDS) == len(set(NULL_CALIBRATION_SEEDS)) == 100
    assert 427 not in NULL_CALIBRATION_SEEDS  # the live draw is not part of its own null


def test_a_calibrated_null_family_passes_and_reports_its_wilson_interval():
    cal = _calibration_ok()
    assert cal.passed, cal.failures
    n = 3 * len(NULL_CALIBRATION_SEEDS)
    assert cal.n_fits == n
    assert 0.0 < cal.fp_rate <= MAX_NULL_FP_RATE
    lo, hi = wilson_interval(cal.n_fp, n)
    assert (cal.wilson_lo, cal.wilson_hi) == (lo, hi)
    assert 0.95 <= cal.se_ratio <= 1.1
    text = "\n".join(cal.lines())
    assert f"{cal.n_fp}/{n}" in text and "Wilson" in text


def test_a_null_false_positive_rate_above_ten_percent_fails():
    """A biased null (mean +1.5 SE) excludes 0 ~32% of the time with a calibrated SE."""
    k = len(NULL_CALIBRATION_SEEDS)
    shifted = [0.03 + a for a in _null_fits()["ate"].iloc[:k]]
    cal = evaluate_null_calibration(
        _null_fits({"Remibrutinib": shifted, "Fabhalta": shifted, "Kisqali": shifted}),
        reproduction=1.0,
    )
    assert cal.fp_rate > MAX_NULL_FP_RATE
    assert cal.se_ratio >= MIN_SE_RATIO
    assert not cal.passed
    assert any("false-positive rate" in f for f in cal.failures)
    fits = pd.concat([_fits(b) for b in BRANDS], ignore_index=True)
    gate = evaluate_recovery_gate(fits, null_calibration=cal)
    assert not gate.passed and gate.verdict().startswith("FAIL")


def test_an_anti_conservative_interval_fails_with_the_linear_dml_fallback_label():
    """98 null ATEs at 0 and two at +-0.2: the FP rate is 2%, but the empirical SD (0.028)
    exceeds the reported SE (0.02), ratio ~0.70 < 0.9 -- the plan's reversal clause."""
    k = len(NULL_CALIBRATION_SEEDS)
    ates = [0.0] * (k - 2) + [0.2, -0.2]
    cal = evaluate_null_calibration(_null_fits(dict.fromkeys(BRANDS, ates)), reproduction=1.0)
    assert cal.fp_rate <= MAX_NULL_FP_RATE
    assert cal.se_ratio < MIN_SE_RATIO
    assert not cal.passed
    msgs = [f for f in cal.failures if "SE/empirical SD" in f]
    assert len(msgs) == 1 and "fall back to LinearDML" in msgs[0]


def test_the_calibration_never_certifies_on_missing_or_errored_cells():
    """A brand x seed cell that is absent or errored fails: the FP rate of a subset is not
    the FP rate of the family."""
    missing = _null_fits().iloc[:-1]
    cal = evaluate_null_calibration(missing, reproduction=1.0)
    assert not cal.passed and any("missing" in f for f in cal.failures)
    errored = _null_fits()
    errored.loc[0, "error"] = "TOO_FEW_USABLE_ROWS"
    cal = evaluate_null_calibration(errored, reproduction=1.0)
    assert not cal.passed and any("error" in f for f in cal.failures)
    one_brand = _null_fits(brands=("Kisqali",))
    assert not evaluate_null_calibration(one_brand, reproduction=1.0).passed


def test_the_calibration_fails_when_the_redraw_does_not_reproduce_the_live_labels():
    """The fresh-seed redraw is the live DGP only if the live seed reproduces the live
    labels on the gated frame; otherwise it calibrates some other process."""
    cal = evaluate_null_calibration(_null_fits(), reproduction=0.97)
    assert not cal.passed
    assert any("reproduce" in f for f in cal.failures)
    assert not evaluate_null_calibration(_null_fits(), reproduction=None).passed


def test_the_gate_never_certifies_without_the_null_calibration():
    fits = pd.concat([_fits(b) for b in BRANDS], ignore_index=True)
    gate = evaluate_recovery_gate(fits)
    assert not gate.passed
    assert "null calibration not run" in gate.verdict()


def test_a_realised_null_excluding_zero_is_a_draw_line_not_a_bias_line():
    """The live Remibrutinib null (+0.047, CI +0.008..+0.086) is one realised draw: the gate
    passes on the family calibration and prints it without the word "bias"."""
    parts = [
        _fits(
            "Remibrutinib",
            overrides={
                (_NULL_COL, "ate"): 0.047,
                (_NULL_COL, "ci_lower"): 0.008,
                (_NULL_COL, "ci_upper"): 0.086,
            },
        ),
        _fits("Fabhalta"),
        _fits("Kisqali"),
    ]
    cal = _calibration_ok()
    gate = evaluate_recovery_gate(pd.concat(parts), null_calibration=cal)
    assert gate.passed
    lines = [ln for ln in gate.verdict().splitlines() if "null CI excludes 0" in ln]
    assert len(lines) == 1
    line = lines[0]
    assert "Remibrutinib" in line and "+0.047" in line
    assert "realised draw (seed 427)" in line
    assert f"family FP rate {cal.n_fp}/{cal.n_fits}" in line
    assert "bias" not in gate.verdict().lower()


def test_wilson_interval_matches_the_closed_form():
    lo, hi = wilson_interval(9, 300)
    assert lo == pytest.approx(0.01586, abs=5e-5)
    assert hi == pytest.approx(0.05602, abs=5e-5)
    assert wilson_interval(0, 10)[0] == 0.0


def test_redraw_adopted_replaces_labels_only_on_the_frame_rows_that_had_one():
    """The design (rows, treatments, confounders, usable set) is held fixed; only the label
    moves, keyed on (hcp_id, brand). A pair without a live label stays without one."""
    frame = pd.DataFrame(
        {
            "hcp_id": ["a", "b", "c", "a"],
            "brand": ["Kisqali", "Kisqali", "Kisqali", "Fabhalta"],
            "adopted": [1.0, 0.0, None, 1.0],
            "rep_training_score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    derived = pd.DataFrame(
        {
            "hcp_id": ["a", "b", "c", "a"],
            "brand": ["Fabhalta", "Kisqali", "Kisqali", "Kisqali"],
            "adopted": [0, 1, 1, 0],
        }
    )
    out = redraw_adopted(frame, derived)
    assert out["adopted"].tolist()[:2] == [0.0, 1.0]
    assert pd.isna(out["adopted"].iloc[2])
    assert out["adopted"].iloc[3] == 0.0
    assert out["rep_training_score"].tolist() == frame["rep_training_score"].tolist()
    assert frame["adopted"].tolist()[:2] == [1.0, 0.0]  # input untouched


def test_every_realised_null_exclusion_is_printed_and_none_is_gated():
    """codex r1 (MED): the removed ">= 2/3 cover 0" clause must stay removed -- with a passing
    family calibration, three realised exclusions still pass, each on its own draw line."""
    parts = [
        _fits(
            b,
            overrides={
                (_NULL_COL, "ate"): a,
                (_NULL_COL, "ci_lower"): a - 0.03,
                (_NULL_COL, "ci_upper"): a + 0.03,
            },
        )
        for b, a in zip(BRANDS, (0.047, -0.04, 0.035), strict=True)
    ]
    gate = evaluate_recovery_gate(pd.concat(parts), null_calibration=_calibration_ok())
    assert gate.passed
    lines = [ln for ln in gate.verdict().splitlines() if "null CI excludes 0" in ln]
    assert len(lines) == 3 and all("realised draw" in ln for ln in lines)


def test_an_fp_rate_of_exactly_the_ceiling_passes_on_the_point_rate():
    """The gate is on the point FP rate (<= 0.10), not on its Wilson upper bound: 30/300 passes
    although the Wilson interval reaches past 0.10 (the bound is printed as K's precision)."""
    k = len(NULL_CALIBRATION_SEEDS)
    base = list(_null_fits()["ate"].iloc[:k])
    # 10 of 100 per brand beyond +-1.96 SE (the calibrated quantiles put 4 there already).
    ates = sorted(base, key=abs)[: k - 10] + [0.05] * 5 + [-0.05] * 5
    cal = evaluate_null_calibration(_null_fits(dict.fromkeys(BRANDS, ates)), reproduction=1.0)
    assert cal.n_fp == 30 and cal.fp_rate == pytest.approx(0.10)
    assert cal.wilson_hi > MAX_NULL_FP_RATE
    assert cal.passed, cal.failures


def test_the_se_ratio_uses_the_within_brand_spread_not_the_spread_of_brand_means():
    """Brands have their own designs, so their null means differ; the empirical SD is pooled
    WITHIN brand. Shifting a brand's whole null distribution must not move the ratio."""
    k = len(NULL_CALIBRATION_SEEDS)
    base = list(_null_fits()["ate"].iloc[:k])
    shifted = {
        "Remibrutinib": [a + 0.03 for a in base],
        "Fabhalta": base,
        "Kisqali": [a - 0.03 for a in base],
    }
    ref = evaluate_null_calibration(_null_fits(), reproduction=1.0)
    cal = evaluate_null_calibration(_null_fits(shifted), reproduction=1.0)
    assert cal.se_ratio == pytest.approx(ref.se_ratio)


def test_an_unevaluated_or_incomplete_calibration_never_passes():
    """codex r1 (HIGH): a default-constructed calibration, non-finite cells, or an
    unmeasurable reproduction must fail, not vanish."""
    from scripts.verify_adoption_channel_recovery import NullCalibration

    assert not NullCalibration().passed
    fits = pd.concat([_fits(b) for b in BRANDS], ignore_index=True)
    assert not evaluate_recovery_gate(fits, null_calibration=NullCalibration()).passed
    nan_cells = _null_fits()
    nan_cells.loc[nan_cells["brand"] == "Kisqali", "ate"] = float("nan")
    cal = evaluate_null_calibration(nan_cells, reproduction=1.0)
    assert not cal.passed and any("non-finite" in f for f in cal.failures)
    assert not evaluate_null_calibration(_null_fits(), reproduction=float("nan")).passed
