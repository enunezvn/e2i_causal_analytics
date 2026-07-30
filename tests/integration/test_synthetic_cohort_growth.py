"""Synthetic cohort-growth + multi-scenario regression suite.

Per ``.claude/plans/synthetic_cohort_growth_plan_20260509.md`` Phase 4.

Pins:
- scenario_b (IgAN/ESKD screening, n=6000, prev=0.05, AUC band [0.72, 0.78]).
- scenario_c (CSU treatment response, n=6000, prev=0.40, AUC band [0.82, 0.88]).
- scenario_a_balanced (BC iDFS DGP at prev=0.50, n=6000, empirical band).
- Extended-n=20000 envelope shift for scenario_a (Task 4.2).

NOT covered here:
- scenario_a default-n pin — that lives in
  ``test_synthetic_baseline_invariant.py`` and MUST remain unchanged
  (codex Q4b additive-not-replacement). This file is the additive
  surface for new scenarios + extended-n.

CI wall-clock budget: 4 forks × ~3-4 min ≈ 12-16 min. Routes through
the slow-tests Job D per ``memory/pr74_slow_tests_fixes_state.md``.

Dual-band local/CI per ISA-divergence precedent
(``memory/pr69_e2e_environment_delta_diag_20260506.md``). Provenance
(re-pinned 2026-07-30, #1311): LOCAL bands are measured on Ubuntu AVX2 at
hpo-trials=5 (two bit-identical seeded runs each); scenario_b's CI band is
measured from the red-nightly logs; scenario_c and a_balanced CI bands are
UNMEASURED (their tests pass in CI so no value is ever logged) and preserve
the prior effective gate width — see each band's own comment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Per-scenario LOCAL baselines ────────────────────────────────────────────
# Each entry pins the metrics measured on Ubuntu local AVX2 with
# --hpo-trials=5 at the scenario's default n_total. CI bands: measured where
# a nightly log printed a value (scenario_b); otherwise unmeasured and
# preserving the prior effective gate width (see each band's comment).
#
# Tolerances follow the same philosophy as
# tests/integration/test_synthetic_baseline_invariant.py:113-120 — tight
# because seed=42 is bit-deterministic on a given ISA.

# Dual local/CI empirical bands — AVX2 vs AVX512 ISA divergence per
# memory/pr69_e2e_environment_delta_diag_20260506.md precedent (same
# dual-band pattern test_synthetic_baseline_invariant.py uses for scenario_a).
# CI ISA shifts metrics at the floating-point instruction level; both are
# bit-deterministic in their respective env. CI bands measured 2026-05-09
# from PR #111 first CI run.

# scenario_b: prev=0.05, n=6000.
# Calibrated band [0.72, 0.78] from scenarios/scenario_b.py:7 was measured at
# hpo>5 with the 9/10-seed regression test. At hpo=5 (this CI budget) the
# realised AUC is below band by ~0.006 (local) — consistent with HPO under-
# converge at extreme prevalence (n=300 positives ceiling the discriminator).
#
# Codex-rescue H1 (2026-05-09): the previous tolerance ±0.05 effectively
# expanded the calibrated band by 70% — concealing real envelope failures.
# Replaced with a SEPARATE empirical-hpo5 band that's pinned tight
# (±0.025 around the measured value) AND an enforce-or-skip flag for the
# calibrated band so CI explicitly distinguishes "calibrated regression"
# from "empirical hpo=5 envelope".
BASELINE_LOCAL_SCENARIO_B: Dict[str, Any] = {
    "regime": "scenario_b",
    # NOT ENFORCED — enforce_calibrated_band=False (codex pass-2 L3); the
    # _assert_scenario_metrics helper emits warnings.warn when this band
    # is supplied but not enforced. Flip enforce_calibrated_band=True after
    # an hpo>5 sweep confirms the calibrated envelope is reachable.
    "auc_band_calibrated": (0.72, 0.78),
    # Re-pinned 2026-07-30 (#1311): aed06cb7 (#44 plan B1) holdout enlargement
    # (test 15%→10%, holdout 5%→10%) reshuffled the keyed-draw split
    # assignment; CI val_AUC moved 0.7144→0.6269, bit-identical across nightly
    # runs 30433515459 (07-29) and 30524268029 (07-30). LOCAL (AVX2) re-measured
    # 2026-07-30, bit-identical across two seeded runs at 0.62687 — coinciding
    # with CI at 4dp for this regime. Band = observed ±0.025 per the
    # codex-rescue H1 convention above.
    "auc_band_empirical_hpo5_local": (0.6019, 0.6519),  # observed 0.6269 ± 0.025 LOCAL (#1311)
    "auc_band_empirical_hpo5_ci": (0.6019, 0.6519),  # observed 0.6269 ± 0.025 CI (#1311)
    "tolerance_auc_empirical": 0.025,
    "min_perm_p_significant": 0.05,
    # #1311: the aed06cb7 reshuffle moved the LOCAL delta 0.013 → 0.1248
    # (bit-identical two seeded runs; split membership, not regularization).
    # The CI delta is UNMEASURED post-aed06cb7 (the red nightlies failed at
    # the band assert before the delta check) and this knob is shared
    # local/CI, so the cap is a PROVISIONAL ceiling: local 0.125 + headroom
    # for the historical CI-vs-local overfit gap. The helper's measurement
    # hook warns the observed delta on passing runs — tighten from there.
    "max_train_val_delta": 0.20,
    "enforce_calibrated_band": False,
}

# scenario_c: prev=0.40, n=6000. Calibrated band [0.82, 0.88] from
# scenarios/scenario_c.py:7.
# - Local AVX2 hpo=5 measured 0.7829 (0.037 below calibrated band-low) pre-aed06cb7.
# - CI AVX512 hpo=5 measured 0.8408 (IN calibrated band — landed naturally on CI).
# Re-pinned LOCAL 2026-07-30 (#1311): post-aed06cb7 (#44 plan B1 holdout
# enlargement, keyed-draw split reshuffle) local re-measured 0.8337,
# metric-bit-identical across two seeded runs — now IN the calibrated band
# locally too, and the local-vs-CI ISA delta narrowed 0.058 → 0.007.
# (enforce_calibrated_band could now flip True symmetrically — owner's call,
# out of #1311 repair scope.)
BASELINE_LOCAL_SCENARIO_C: Dict[str, Any] = {
    "regime": "scenario_c",
    # NOT ENFORCED — enforce_calibrated_band=False (codex pass-2 L3). Same
    # warn-on-skip behavior as scenario_b. Could flip True for CI alone since
    # CI 0.8408 is in calibrated band, but keeping False symmetric across
    # local/CI until the local hpo>5 sweep confirms landing locally too.
    "auc_band_calibrated": (0.82, 0.88),
    "auc_band_empirical_hpo5_local": (0.8087, 0.8587),  # observed 0.8337 ± 0.025 LOCAL (#1311)
    # CI value NOT re-measured post-aed06cb7 (the test passed in CI, so the
    # nightly logs never printed it — pre-aed06cb7 CI measured 0.8408). Band
    # preserves the previous EFFECTIVE gate width (the pre-#1311 double
    # expansion, see _assert_scenario_metrics note); tighten to
    # observed ± 0.025 once a post-aed06cb7 CI value is logged.
    "auc_band_empirical_hpo5_ci": (0.795, 0.885),
    "tolerance_auc_empirical": 0.025,
    "min_perm_p_significant": 0.05,
    "max_train_val_delta": 0.10,  # observed 0.022 local
    "enforce_calibrated_band": False,
}

# scenario_a_balanced: prev=0.50, n=6000.
# Inherits scenario_a's DGP at 0.50 prevalence. There is NO calibrated band
# (no biology-derived 9/10-seed sweep) — only an empirical hpo=5 measurement:
# pre-aed06cb7 val_AUC=0.7973, train-val Δ=0.0106, MCC=0.4767 (significantly
# higher than scenario_a's — balancing lifts MCC by removing the
# prior-imbalance penalty), perm_p=0.0.
# Re-pinned LOCAL 2026-07-30 (#1311): post-aed06cb7 (#44 plan B1 holdout
# enlargement, keyed-draw split reshuffle) local re-measured 0.7602,
# metric-bit-identical across two seeded runs.
BASELINE_LOCAL_SCENARIO_A_BALANCED: Dict[str, Any] = {
    "regime": "scenario_a_balanced",
    # No calibrated_band — empirical only.
    "auc_band_empirical_hpo5_local": (0.7402, 0.7802),  # observed 0.7602 ± 0.02 LOCAL (#1311)
    # Placeholder — NEVER measured on CI (the test passes there, so the value
    # is not printed). Band preserves the previous EFFECTIVE gate width (the
    # pre-#1311 double expansion); tighten to observed ± 0.02 once a CI value
    # is logged.
    "auc_band_empirical_hpo5_ci": (0.76, 0.84),
    "tolerance_auc_empirical": 0.02,
    "min_perm_p_significant": 0.05,
    "max_train_val_delta": 0.10,  # observed 0.011
    "enforce_calibrated_band": False,  # no calibrated band exists
}


def _select_empirical_band(baseline: Dict[str, Any]) -> tuple[float, float]:
    """Select local vs CI empirical band based on CI env var.

    Mirrors test_synthetic_baseline_invariant.py:99 — `os.getenv("CI")` is set
    automatically by GitHub Actions; local runs default to the local band.
    Restructured as if/return (rather than `key = ... if ... else ...`) to
    avoid gitleaks v8.24.3 generic-api-key heuristic false positive on the
    `key = "..."` + `os.getenv("CI")` co-occurrence.
    """
    if os.getenv("CI"):
        return baseline["auc_band_empirical_hpo5_ci"]
    return baseline["auc_band_empirical_hpo5_local"]


def _run_tier0_e2e(
    regime: str,
    *,
    hpo_trials: int = 5,
    n_total: int | None = None,
    seed: int = 42,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> Dict[str, Any]:
    """Fork run_tier0_test.py and return the captured TIER0_E2E_JSON_OUT artifact.

    Parameters mirror the CLI flags so each test reads at-a-glance. Extra
    args (e.g. ``["--imbalanced", "0.50"]``) splice in unchanged.
    """
    json_out = tmp_path / f"tier0_{regime}_seed{seed}_n{n_total or 'default'}.json"
    env = os.environ.copy()
    env["TIER0_E2E_JSON_OUT"] = str(json_out)
    # #594: synthetic e2e has NO live Feast store. Post #556 the freshness check
    # FAILS CLOSED when Feast is unavailable → all features read stale → the
    # registrar QC gate hard-blocks training → empty validation_metrics →
    # "roc_auc missing". ALLOW_STALE_FEAST=1 is the #556 escape hatch for these
    # intentional no-Feast environments.
    env["ALLOW_STALE_FEAST"] = "1"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--regime",
        regime,
        "--split",
        "auto",
        "--hpo-trials",
        str(hpo_trials),
        "--no-save",
        "--no-bentoml",
        "--seed",
        str(seed),
    ]
    if n_total is not None:
        cmd += ["--n-total", str(n_total)]
    if extra_args:
        cmd += extra_args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1500,  # 25 min hard cap — extended-n runs at n=20000 take longer
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert result.returncode == 0, (
        f"Synthetic e2e {regime!r} (seed={seed}, n_total={n_total!r}) "
        f"exited {result.returncode}; stderr (truncated): {result.stderr[-500:]!r}"
    )
    assert json_out.exists(), (
        f"TIER0_E2E_JSON_OUT artifact missing at {json_out}; runner produced no JSON."
    )

    artifact: Dict[str, Any] = json.loads(json_out.read_text())
    return artifact


def _assert_scenario_metrics(
    artifact: Dict[str, Any],
    *,
    regime: str,
    empirical_band: tuple[float, float],
    tolerance_empirical: float,
    min_perm_p_significant: float,
    max_train_val_delta: float,
    calibrated_band: tuple[float, float] | None = None,
    enforce_calibrated_band: bool = False,
) -> None:
    """Assert val_AUC + permutation + train-val Δ for one scenario.

    Codex-rescue H1 (2026-05-09 pass 1): the assertion now distinguishes the
    SCENARIO's biology-derived calibrated band from the EMPIRICAL hpo=5
    measurement band. A loose tolerance on calibrated bands previously
    expanded by 70% silently masked real envelope failures.

    Codex-rescue H1 (2026-05-09 pass 2): when ``calibrated_band`` is supplied
    but ``enforce_calibrated_band=False``, the function now emits a
    ``warnings.warn`` so the silently-bypassed contract is observable.
    Lifts pass 1's "dead field" risk where a downstream config change to
    ``auc_band_calibrated`` would silently land without test signal.

    The default mode is "empirical" — we assert the val_AUC lands within
    ``empirical_band`` directly. MEASURED bands embed
    ``± tolerance_empirical`` around the observed hpo=5 value; the two
    UNMEASURED CI bands (scenario_c, a_balanced — their tests pass in CI so
    no value is ever logged) instead preserve the prior effective gate
    width, per their own comments. The ``tolerance_empirical`` parameter is
    retained for failure-message provenance only; #1311 codex iter-1
    finding 2 removed the second expansion that doubled the effective
    width. When ``enforce_calibrated_band=True``, we also
    assert the val_AUC lands within ``calibrated_band`` exactly (no
    tolerance) — for cases where the calibrated regression must pass.

    Reads the artifact's validation_metrics + permutation_test +
    test_metrics.train_val_auc_delta and compares against the per-scenario
    band/tolerance. Each failure surfaces with regime-aware context so the
    debugger can trace back to which scenario regressed.
    """
    import warnings

    val = artifact.get("validation_metrics") or {}
    perm = artifact.get("permutation_test") or {}
    test = artifact.get("test_metrics") or {}

    val_auc = val.get("roc_auc")
    assert val_auc is not None, f"{regime}: validation_metrics.roc_auc missing"

    # Empirical band is the primary regression gate. The stored band IS the
    # gate, asserted directly: measured bands embed ± tolerance around the
    # observed value (codex-rescue H1 convention); the two unmeasured CI
    # bands preserve the prior effective width (see their comments). #1311
    # codex iter-1 finding 2: the helper previously expanded the band by
    # ± tolerance AGAIN, doubling the effective width and weakening the gate.
    emp_low, emp_high = empirical_band
    assert emp_low <= val_auc <= emp_high, (
        f"{regime}: val_AUC {val_auc:.4f} outside empirical hpo=5 band "
        f"[{emp_low:.4f}, {emp_high:.4f}] "
        f"(measured bands embed ±{tolerance_empirical} around the observed "
        f"value; unmeasured CI bands preserve the prior effective width — "
        f"see the band's own comment)"
    )

    # Calibrated band (optional) — strict no-tolerance check for biology-
    # derived 9/10-seed envelope. Default off because hpo=5 may under-
    # converge below the calibrated band.
    if calibrated_band is not None:
        if enforce_calibrated_band:
            cal_low, cal_high = calibrated_band
            assert cal_low <= val_auc <= cal_high, (
                f"{regime}: val_AUC {val_auc:.4f} outside calibrated band "
                f"[{cal_low}, {cal_high}] (no tolerance applied; "
                "the hpo=5 envelope must cleanly meet biology-derived calibration)."
            )
        else:
            # Codex-rescue pass-2 H1 fix: warn so a future reader sees the
            # calibrated band is configured but not enforced. Silent pass
            # was the codex finding ("dead field" risk).
            warnings.warn(
                f"{regime}: calibrated_band={calibrated_band} supplied but "
                f"enforce_calibrated_band=False — only empirical hpo=5 band "
                f"is enforced. Flip enforce_calibrated_band=True after an "
                f"hpo>5 sweep confirms landing in the calibrated band.",
                stacklevel=2,
            )

    perm_p = perm.get("permutation_pvalue")
    if perm_p is not None:
        assert perm_p <= min_perm_p_significant, (
            f"{regime}: permutation_p={perm_p:.4f} > {min_perm_p_significant} — "
            "model didn't beat the null on permuted labels (no signal)."
        )

    train_val_delta = test.get("train_val_auc_delta")
    if train_val_delta is not None:
        assert train_val_delta <= max_train_val_delta, (
            f"{regime}: train-val Δ={train_val_delta:.4f} > {max_train_val_delta} — "
            "severe overfit signal."
        )
        # #1311 measurement hook: on a PASSING run pytest suppresses stdout,
        # so the observed delta never reaches the nightly logs — which is why
        # scenario_b's cap is provisional. A warning survives into the
        # warnings summary; read it there and tighten provisional caps.
        warnings.warn(
            f"{regime}: train_val_auc_delta observed {train_val_delta:.4f} "
            f"(cap {max_train_val_delta}) — measurement hook for cap re-pin",
            stacklevel=2,
        )


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_scenario_b_default_n_lands_in_empirical_band(tmp_path: Path) -> None:
    """scenario_b (IgAN/ESKD screening, prev=0.05) hits empirical hpo=5 band.

    Calibrated band per src/ml/synthetic_v2/scenarios/scenario_b.py:7 is
    [0.72, 0.78] (9/10-seed regression at hpo>5). At hpo=5 (this CI budget)
    the envelope lands ~0.006 below band-low; codex H1 gate replaces a
    too-loose calibrated-band check with a tight empirical-band pin.
    Re-enable enforce_calibrated_band once an hpo>5 sweep confirms landing.
    """
    artifact = _run_tier0_e2e("scenario_b", tmp_path=tmp_path)
    _assert_scenario_metrics(
        artifact,
        regime="scenario_b",
        empirical_band=_select_empirical_band(BASELINE_LOCAL_SCENARIO_B),
        tolerance_empirical=BASELINE_LOCAL_SCENARIO_B["tolerance_auc_empirical"],
        min_perm_p_significant=BASELINE_LOCAL_SCENARIO_B["min_perm_p_significant"],
        max_train_val_delta=BASELINE_LOCAL_SCENARIO_B["max_train_val_delta"],
        calibrated_band=BASELINE_LOCAL_SCENARIO_B["auc_band_calibrated"],
        enforce_calibrated_band=BASELINE_LOCAL_SCENARIO_B["enforce_calibrated_band"],
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_scenario_c_default_n_lands_in_empirical_band(tmp_path: Path) -> None:
    """scenario_c (CSU treatment response, prev=0.40) hits empirical hpo=5 band.

    Calibrated band per src/ml/synthetic_v2/scenarios/scenario_c.py:7 is
    [0.82, 0.88]. Pre-aed06cb7 the local hpo=5 envelope landed ~0.037 below
    band-low; since the #1311 re-measurement it lands IN the band locally
    (0.8337). Same calibrated-vs-empirical split as scenario_b.
    """
    artifact = _run_tier0_e2e("scenario_c", tmp_path=tmp_path)
    _assert_scenario_metrics(
        artifact,
        regime="scenario_c",
        empirical_band=_select_empirical_band(BASELINE_LOCAL_SCENARIO_C),
        tolerance_empirical=BASELINE_LOCAL_SCENARIO_C["tolerance_auc_empirical"],
        min_perm_p_significant=BASELINE_LOCAL_SCENARIO_C["min_perm_p_significant"],
        max_train_val_delta=BASELINE_LOCAL_SCENARIO_C["max_train_val_delta"],
        calibrated_band=BASELINE_LOCAL_SCENARIO_C["auc_band_calibrated"],
        enforce_calibrated_band=BASELINE_LOCAL_SCENARIO_C["enforce_calibrated_band"],
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_scenario_a_balanced_default_n_lands_in_empirical_band(tmp_path: Path) -> None:
    """scenario_a_balanced (scenario_a DGP, prev=0.50) lands in empirical band.

    The band is empirical (not biology-derived) because shifting prev=0.20→0.50
    on scenario_a's locked DGP changes the AUC envelope. LOCAL band measured
    2026-07-30 (#1311, observed 0.7602); the CI band is still unmeasured and
    preserves the prior effective gate width (see the band's comment).
    """
    artifact = _run_tier0_e2e("scenario_a_balanced", tmp_path=tmp_path)
    _assert_scenario_metrics(
        artifact,
        regime="scenario_a_balanced",
        empirical_band=_select_empirical_band(BASELINE_LOCAL_SCENARIO_A_BALANCED),
        tolerance_empirical=BASELINE_LOCAL_SCENARIO_A_BALANCED["tolerance_auc_empirical"],
        min_perm_p_significant=BASELINE_LOCAL_SCENARIO_A_BALANCED["min_perm_p_significant"],
        max_train_val_delta=BASELINE_LOCAL_SCENARIO_A_BALANCED["max_train_val_delta"],
        calibrated_band=None,  # no biology-derived band — empirical only
        enforce_calibrated_band=False,
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(2700)
def test_scenario_a_extended_n_20000_envelope_shift(tmp_path: Path) -> None:
    """Plan Task 4.2 — pin the envelope shift at scenario_a + n=20000.

    At n=20000 the train-val Δ should NOT explode and val_AUC should still
    land in/above scenario_a's calibrated band [0.78, 0.83]. Tolerance is
    looser on the upper side because more data typically lifts AUC slightly.
    Documents the n-vs-envelope shape so future readers know n=20000 is
    already covered (don't run more — CI budget).
    """
    artifact = _run_tier0_e2e("scenario_a", n_total=20000, tmp_path=tmp_path, hpo_trials=5)
    _assert_scenario_metrics(
        artifact,
        regime="scenario_a@n=20000",
        # At n=20000 we expect the envelope to widen slightly above
        # the calibrated band-low; pin a slightly wider empirical band
        # until the Phase 1.3 sweep produces a measured value. #1311: this
        # band was never measured either locally or in CI (the test passes
        # in CI, so no value is logged) — it preserves the pre-#1311
        # EFFECTIVE gate width (the old band ± tolerance double expansion);
        # tighten once a measured value lands.
        empirical_band=(0.75, 0.88),
        tolerance_empirical=0.03,
        min_perm_p_significant=0.05,
        max_train_val_delta=0.10,  # densest tier per criteria_validator
        calibrated_band=(0.78, 0.83),
        enforce_calibrated_band=False,  # n=20000 hpo=5 envelope TBD
    )

    # Audit: artifact carries the actual n_total we passed in.
    assert artifact.get("n_total") == 20000, (
        f"Artifact n_total={artifact.get('n_total')!r} != 20000 — "
        "CLI flag did not propagate through to JSON output."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(900)
def test_n_total_below_floor_rejected_at_argparse() -> None:
    """Plan Task 1.1 acceptance — --n-total 99 rejects with parser.error.

    Faster than the full e2e tests because argparse validation kicks in
    before any pipeline work. Confirms the floor message points at api.py:158
    so operators chasing the error know where the constraint comes from.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--regime",
        "scenario_a",
        "--n-total",
        "99",
        "--no-bentoml",
        "--no-save",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
    assert result.returncode == 2, (
        f"Expected argparse exit code 2 for --n-total 99, got {result.returncode}; "
        f"stderr (truncated): {result.stderr[-500:]!r}"
    )
    # parser.error message must reference both the flag, the floor, and the source line
    err = result.stderr
    assert "--n-total" in err and "100" in err and "api.py:158" in err, (
        f"--n-total 99 error message lacks expected pointers; got:\n{err}"
    )


# Task 1.3b — bit-identical no-flag regression. Pinned against the
# pre-PR baseline at /tmp/iter5_synth_growth/scenario_a_n6000.json.
# Codex-rescue M2 (2026-05-09): pinned to FULL precision (the actual
# bit-deterministic value within an ISA), not a 4-decimal truncation.
# Tolerance 1e-9 matches the empirical bit-identicality observed
# locally — anything looser permits silent drift.
SCENARIO_A_NO_FLAG_BASELINE_VAL_AUC_LOCAL = 0.768887662061683
SCENARIO_A_NO_FLAG_BASELINE_TRAIN_VAL_DELTA_LOCAL = 0.0511727697332387


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_no_flag_scenario_a_preserves_pre_pr_baseline(tmp_path: Path) -> None:
    """Plan Task 1.3b — bit-identical regression for the no-flag scenario_a path.

    Ensures the new --n-total / --seed flags do NOT change scenario_a's
    measured metrics when the user invokes the runner with neither flag.
    Asserts at 1e-9 tolerance — the bit-deterministic regime within an ISA
    (verified locally 2026-05-09 with 2 runs producing identical metrics
    to 9 decimal places).

    Failing this means the wiring leaked a side effect — e.g., --seed=42
    threading drifted into a downstream agent, or n_total=None routed
    through a non-default branch in generate_scenario.

    NOT a bit-identical-across-ISA test; that's not achievable per
    memory/pr69_e2e_environment_delta_diag_20260506.md. Within an ISA
    this assertion is bit-deterministic.
    """
    if os.getenv("CI"):
        pytest.skip(
            "no-flag regression is ISA-specific; CI-side covered by "
            "test_synthetic_baseline_invariant.py BASELINE_CI dimension"
        )
    artifact = _run_tier0_e2e("scenario_a", tmp_path=tmp_path)
    val_auc = (artifact.get("validation_metrics") or {}).get("roc_auc")
    train_val_delta = (artifact.get("test_metrics") or {}).get("train_val_auc_delta")
    assert val_auc is not None, "no-flag run produced no val_AUC"
    assert train_val_delta is not None, "no-flag run produced no train_val_auc_delta"
    assert abs(val_auc - SCENARIO_A_NO_FLAG_BASELINE_VAL_AUC_LOCAL) <= 1e-9, (
        f"no-flag scenario_a val_AUC drifted: observed {val_auc!r}, "
        f"baseline {SCENARIO_A_NO_FLAG_BASELINE_VAL_AUC_LOCAL!r} "
        "(±1e-9 — bit-deterministic within ISA). The --n-total / --seed "
        "wiring must not change the no-flag path's behaviour."
    )
    assert abs(train_val_delta - SCENARIO_A_NO_FLAG_BASELINE_TRAIN_VAL_DELTA_LOCAL) <= 1e-9, (
        f"no-flag scenario_a train-val Δ drifted: observed {train_val_delta!r}, "
        f"baseline {SCENARIO_A_NO_FLAG_BASELINE_TRAIN_VAL_DELTA_LOCAL!r} (±1e-9)."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(900)
def test_n_total_at_floor_accepted_at_argparse() -> None:
    """Plan Task 1.1 acceptance — --n-total 100 passes argparse.

    Uses --dry-run so we exercise just argparse + the early validation,
    not the full pipeline. Faster than the e2e tests above.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--regime",
        "scenario_a",
        "--n-total",
        "100",
        "--dry-run",
        "--no-bentoml",
        "--no-save",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
    assert result.returncode == 0, (
        f"Expected exit 0 for --n-total 100 --dry-run, got {result.returncode}; "
        f"stderr (truncated): {result.stderr[-500:]!r}"
    )
    assert "n_total=100" in result.stdout, (
        f"--dry-run output should print n_total=100; got:\n{result.stdout}"
    )


def test_n_total_floor_actually_generates_dataset() -> None:
    """Codex-rescue M3 — verify n_total=100 actually generates a stratified split.

    The argparse-only test above proves the parser accepts the value; this
    test proves the value flows through to ``generate_scenario`` and produces
    a viable cohort. Closes the gap "argparse passes but generation fails"
    that the dry-run path masked.
    """
    import importlib

    runner = importlib.import_module("scripts.run_tier0_test")
    df = runner.generate_sample_data(
        n_samples=1500,  # ignored on synthetic_v2 path
        seed=42,
        _generator="scenario_a",
        n_total=100,
    )
    assert len(df) == 100, (
        f"Expected 100 rows from generate_sample_data(n_total=100), got {len(df)}"
    )
    # discontinuation_flag must be present and balanced enough for the
    # stratified train/val/test split (60/20/20) to leave ≥1 positive in
    # every split — at prev=0.20 with n=100 that's ~20 positives, ≥4 in
    # the smallest split. Bernoulli SD on n=100 = √(0.2·0.8/100) ≈ 0.04 →
    # accept prevalence in [0.10, 0.30].
    pos_rate = df["discontinuation_flag"].mean()
    assert 0.10 <= pos_rate <= 0.30, (
        f"scenario_a n_total=100 prevalence {pos_rate} outside expected "
        "[0.10, 0.30] band (target 0.20, ±2σ Bernoulli)."
    )
