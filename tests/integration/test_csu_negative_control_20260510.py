"""CSU negative-control regression — Plan v4 §2 Gate G1 acceptance criterion.

Closes the **NEW** sub-criterion of Plan v4 §2 Gate G1 (Tier 1B Gate B1):

    "CSU n=9607 negative-control regression test pinning deployer verdict
    UNCHANGED at val_AUC=0.66 (MARGINAL, perm p=0.0) after HBLP default-path
    wiring lands"

This test is the **negative-control half** of G1: it asserts that the v4
HBLP wiring (lands in Phase C, Gate G3) does NOT change the CSU pipeline's
behaviour. CSU is the high-positive-count cohort (n_train_pos≈98) where
the HBLP variance-inflation factor is 1.0 (no relaxation) — the wiring
should be a NO-OP relative to the legacy 5σ z-threshold path. If a future
PR introduces a regression that silently drops a previously-kept feature,
flips the deployer's MARGINAL verdict, or shifts val_AUC outside the
honest band, this test fails loudly.

What this test pins
-------------------
1. Pipeline runs to completion (matches existing
   ``test_csu_val_auc_measurement.test_pipeline_runs_to_completion``)
2. ``val_AUC ∈ [0.62, 0.68]`` honest band (CSU literature anchor;
   PR #106 closure record)
3. ``permutation_test.permutation_pvalue ≤ 0.01`` (effective floor for
   100-perm test; "indistinguishable from <0.001")
4. The five canonical CSU post-anchor journey-metadata features
   (``CSU_POST_INDEX_JOURNEY_FEATURES``) all appear in
   ``leakage_dropped_features`` AND have a corresponding Layer 1
   ``severity=high, remediation=drop`` verdict.

Why this is "negative-control"
------------------------------
CSU is the cohort where HBLP wiring is expected to be a NO-OP. By
contrast, the Optum cohort (low-N, n_train_pos≈22) is where HBLP
relaxation matters. The Optum companion test
(``test_optum_held_out_noninferiority_20260510``) is the **positive
non-inferiority** test; this one is the **CSU negative-control**.

Together the two tests close the v4 G1 acceptance criterion that "HBLP
wiring does NOT silently degrade real-cohort performance".

Real-data dependency
--------------------
This test requires CSU patient_journeys.json to be present at
``data/rwd/csu/e2i_ml_v3_patient_journeys.json``. In CI/worktrees
without real data, the test is **skipped** (not failed); the local
data_dir invariant is checked in the existing
``test_csu_val_auc_measurement`` battery.

Wall-clock budget
-----------------
~5-10 min on real CSU. Marked ``slow``; not in default ``pytest -x``.

References
----------
- Plan v4 §2 Gate G1 acceptance criterion (NEW)
- Plan v3 §6 Tier 1B Gate B1
- ``docs/results/optum_initiation_revalidation_20260510.md``
  (empirical anchor: CSU stable in [0.62, 0.68])
- Companion: ``test_optum_held_out_noninferiority_20260510.py``
- Companion: ``test_g1_lineage_audit_sweep.py``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CSU_DATA_DIR = REPO_ROOT / "data" / "rwd" / "csu"
CSU_JOURNEYS_PATH = CSU_DATA_DIR / "e2i_ml_v3_patient_journeys.json"

# ── Acceptance band per the plan ───────────────────────────────────────────
# Plan v4 §2 G1 explicit pin: val_AUC=0.66 (MARGINAL, perm p=0.0).
# Honest band [0.62, 0.68] per plan v3 §6 Tier 1B + the CSU literature
# anchors codex CSU-benchmark research at PR #106 (psoriasis 0.67,
# AD 0.63, severe asthma 0.66).
VAL_AUC_MIN = 0.62
VAL_AUC_MAX = 0.68

# Permutation null p-value ceiling. Mirrors
# ``test_csu_val_auc_measurement.PERMUTATION_P_MAX``: the 100-perm
# evaluator's smallest empirical p is 0.005 (1/200); the original 0.01
# ceiling remains conservative under DEFAULT_PERMUTATION_COUNT=200.
PERMUTATION_P_MAX = 0.01

# CSU post-anchor journey-metadata features. Must all be dropped by
# Layer 1 manifest verdict. See Plan v3 §6 + backlog #7 CSU sub-gap
# (closure record at memory/backlog_7_csu_subgap_close_20260509.md).
CSU_POST_INDEX_JOURNEY_FEATURES = [
    "journey_start_date",
    "journey_end_date",
    "journey_duration_days",
    "journey_stage",
    "journey_status",
]

# Codex pass-1 HIGH-2 + MED-10 (PR #137 v4 G1): canonical deployer
# verdict the CSU negative-control run MUST emit. Plan v4 §2 G1 cites
# "MARGINAL" as the verdict label, but the empirical anchor at PR #106
# (val_AUC=0.6592, recall ≈ 0.30+, precision ≥ 0.05) maps to "ACCEPTABLE"
# under the runner's _compute_verdict logic at scripts/run_tier0_test.py:
# auc_roc >= 0.65 + recall >= 0.3 + precision >= 0.05 → "ACCEPTABLE".
# We pin EXACT match here so the verdict assertion is decoupled from
# numerical AUC tolerance (codex MED-10).
CSU_EXPECTED_DEPLOYER_VERDICT = "ACCEPTABLE"

# Codex pass-1 HIGH-3 (PR #137 v4 G1): canonical CSU cohort size at
# PR #116 closure (default-window cohort). Pinned so a cohort-build
# regression that silently shifts cohort size fires this gate.
CSU_EXPECTED_COHORT_SIZE = 9607


@pytest.fixture(scope="module")
def csu_negative_control_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Run the full tier0 pipeline on real CSU data, return parsed artifact.

    Module-scoped: one subprocess invocation amortized across every
    G1-negative-control assertion in this file.

    Pattern mirrors ``test_csu_val_auc_measurement.csu_artifact``
    fixture so behaviour stays identical across the two harnesses; if
    one regresses, both regress, and the failure messages stay legible.
    """
    if not CSU_JOURNEYS_PATH.exists():
        # Codex pass-1 HIGH-1: default to hard-fail when real CSU data is
        # absent. CI green without proving real-cohort regression is the
        # exact failure mode this gate guards against. To explicitly opt
        # into a skip (CI environments that intentionally lack data
        # fixtures), set ``ALLOW_MISSING_REAL_DATA=1``.
        if os.environ.get("ALLOW_MISSING_REAL_DATA") == "1":
            pytest.skip(
                f"CSU journeys file not present at {CSU_JOURNEYS_PATH} "
                "and ALLOW_MISSING_REAL_DATA=1 set; G1 CSU negative-control "
                "skip explicitly opted into. Locally: regenerate cohort "
                "via the documented CSU recipe before re-running."
            )
        pytest.fail(
            f"CSU journeys file not present at {CSU_JOURNEYS_PATH}; "
            "G1 CSU negative-control requires real CSU data and "
            "ALLOW_MISSING_REAL_DATA != '1'. Set the env var to opt into "
            "skip behaviour, OR regenerate the cohort via the documented "
            "CSU recipe before re-running."
        )

    out_dir = tmp_path_factory.mktemp("csu_g1_neg_control")
    json_out = out_dir / "csu_g1_neg_control.json"

    env = os.environ.copy()
    env["TIER0_E2E_JSON_OUT"] = str(json_out)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--data-dir",
        str(CSU_DATA_DIR),
        "--brand",
        "competitor",
        "--target",
        "treatment_initiated",
        "--indication",
        "Chronic Spontaneous Urticaria (CSU)",
        "--feature-manifest-source",
        "csu",
        "--hpo-trials",
        "5",  # determinism + speed; matches the val_AUC measurement test
        "--no-bentoml",
        "--no-save",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert result.returncode == 0, (
        f"CSU G1 negative-control tier0 e2e exited {result.returncode}. "
        f"stderr (truncated): {result.stderr[-1500:]!r}"
    )
    assert json_out.exists(), (
        f"TIER0_E2E_JSON_OUT artifact missing at {json_out}; runner produced no JSON."
    )
    return json.loads(json_out.read_text())


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_pipeline_completes_negative_control(
    csu_negative_control_artifact: dict,
) -> None:
    """The full tier0 pipeline must complete without halting on real CSU.

    Negative-control invariant: Plan v4 §2 G1 demands "deployer verdict
    UNCHANGED" — that ONLY makes sense if the pipeline reaches the
    deployer at all. A halt at any earlier step is itself a verdict
    shift relative to the empirical anchor at PR #106.

    Note: ``trained_model_present`` is the runner's "did model_trainer
    produce a model" flag — see ``scripts/run_tier0_test.py:5991``.
    """
    assert not csu_negative_control_artifact.get("pipeline_halted"), (
        f"Pipeline halted: halt_reason="
        f"{csu_negative_control_artifact.get('halt_reason')!r}. "
        "G1 negative-control demands a fully-completed run on CSU. "
        "Re-audit the closure path back to PR #106 / backlog #13."
    )
    assert csu_negative_control_artifact.get("trained_model_present"), (
        "trained_model_present is False — model_trainer halted; "
        "G1 negative-control invariant violated."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_val_auc_pinned_in_honest_band(
    csu_negative_control_artifact: dict,
) -> None:
    """val_AUC ∈ [0.62, 0.68] per plan v4 §2 G1 (CSU pin).

    Plan v4 §2 G1 cites "val_AUC=0.66" as the negative-control anchor.
    The honest band [0.62, 0.68] is the slack for normal seed-to-seed
    variation; an out-of-band value here is a deployer-verdict-class
    shift the G1 invariant explicitly forbids.

    Below 0.62 → Layer 1 manifest may be over-aggressive (a useful
    feature got dropped). Above 0.68 → residual leakage may not be
    fully caught. Either is a regression.
    """
    val_metrics = csu_negative_control_artifact.get("validation_metrics") or {}
    val_auc = val_metrics.get("roc_auc")
    assert val_auc is not None, (
        f"validation_metrics.roc_auc missing. Keys present: {list(val_metrics.keys())}"
    )
    assert VAL_AUC_MIN <= val_auc <= VAL_AUC_MAX, (
        f"val_AUC = {val_auc:.4f} outside CSU honest band "
        f"[{VAL_AUC_MIN}, {VAL_AUC_MAX}] per plan v4 §2 G1 "
        f"negative-control pin (anchor: val_AUC=0.66, MARGINAL).\n"
        f"This is a deployer-verdict-class shift G1 forbids. "
        f"If intentional, update the band IN THIS COMMIT with a "
        f"PR/SHA reference + signed update memo."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_permutation_p_at_genuine_floor(
    csu_negative_control_artifact: dict,
) -> None:
    """Permutation null p-value ≤ 0.01 (floor of 100-perm test).

    Plan v4 §2 G1 cites "perm p=0.0" as the CSU pin. The 100-perm
    permutation null cannot directly resolve below 1/200=0.005; we
    treat ≤ 0.01 as "indistinguishable from < 0.001" and pin that
    ceiling. A CSU run that produces p > 0.01 indicates statistical
    signal regression (e.g., Layer 1 dropped a useful feature, or
    the cohort itself shifted).

    Hard-fail (not skip) on missing payload — same rationale as
    ``test_csu_val_auc_measurement.test_permutation_p_value_significant``
    (codex M7): a missing perm payload means the gate didn't run, not
    that the test is inapplicable.
    """
    perm = csu_negative_control_artifact.get("permutation_test") or {}
    p_value = perm.get("permutation_pvalue", perm.get("p_value"))
    assert p_value is not None, (
        f"permutation_test.permutation_pvalue missing from artifact. "
        f"Keys present: {sorted(perm.keys())!r}. "
        "G1 negative-control demands the permutation gate fired."
    )
    assert p_value <= PERMUTATION_P_MAX, (
        f"Permutation p_value = {p_value:.4f} exceeds CSU negative-"
        f"control ceiling {PERMUTATION_P_MAX} (plan v4 §2 G1 pin: "
        f"perm p=0.0). Statistical signal regression — re-audit the "
        f"surviving feature set + leakage_dropped_features."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_deployer_verdict_unchanged(
    csu_negative_control_artifact: dict,
) -> None:
    """Codex pass-1 HIGH-2 (PR #137 v4 G1): deployer verdict EXACT pin.

    Plan v4 §2 G1 demands "deployer verdict UNCHANGED" — the canonical
    closure of the negative-control invariant. The artifact carries the
    label emitted by ``_compute_verdict`` (one of EXCELLENT, GOOD,
    ACCEPTABLE, THRESHOLD_NEEDED, MARGINAL, POOR). Any shift here is a
    deployer-class regression independent of AUC magnitude.

    Failure modes this gate catches:
    - Plan v4 G3 HBLP wiring causes recall to drop below 0.3 → verdict
      shifts ACCEPTABLE → MARGINAL.
    - Layer 1 over-aggression drops a useful feature → AUC < 0.65 →
      verdict shifts ACCEPTABLE → MARGINAL.
    - Test-set leakage that wasn't there at PR #106 surfaces → AUC > 0.85
      and recall > 0.7 → verdict shifts ACCEPTABLE → EXCELLENT or GOOD.

    Decoupled from numerical AUC tolerance per codex MED-10: the AUC
    band [0.62, 0.68] is the documented numerical tolerance; this gate
    pins the verdict label only.
    """
    actual = csu_negative_control_artifact.get("deployer_verdict")
    assert actual is not None, (
        "deployer_verdict missing from artifact — runner did not "
        "compute the canonical verdict label. Either test_metrics is "
        "empty (early halt) or the runner's artifact-emission block "
        "regressed. Re-audit scripts/run_tier0_test.py artifact emission."
    )
    assert actual == CSU_EXPECTED_DEPLOYER_VERDICT, (
        f"G1 negative-control: deployer verdict shifted from "
        f"{CSU_EXPECTED_DEPLOYER_VERDICT!r} (PR #106 baseline) to "
        f"{actual!r}. Plan v4 §2 G1 demands the verdict UNCHANGED.\n"
        f"This is a deployer-class regression independent of AUC. "
        f"Triage:\n"
        f"  - artifact.validation_metrics: "
        f"{csu_negative_control_artifact.get('validation_metrics')}\n"
        f"  - artifact.test_metrics: "
        f"{csu_negative_control_artifact.get('test_metrics')}\n"
        f"  - artifact.deployer_verdict_description: "
        f"{csu_negative_control_artifact.get('deployer_verdict_description')!r}\n"
        f"If the shift is intentional, update "
        f"CSU_EXPECTED_DEPLOYER_VERDICT IN THIS COMMIT with PR/SHA "
        f"reference + signed memo."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_cohort_size_pinned(
    csu_negative_control_artifact: dict,
) -> None:
    """Codex pass-1 HIGH-3 (PR #137 v4 G1): pin CSU cohort_size = 9607.

    Plan v4 §2 G1 explicitly cites "CSU n=9607". The artifact carries
    ``cohort_size`` (the assembled-cohort row count). A cohort-build
    regression that silently shifts the cohort size — even by 1 — fires
    this gate; the val_AUC honest band is sample-size dependent and a
    silent cohort-size shift voids the empirical anchor.
    """
    actual = csu_negative_control_artifact.get("cohort_size")
    assert actual is not None, (
        "cohort_size missing from artifact — runner did not record "
        "the assembled-cohort row count. Either eligible_df + patient_df "
        "are both absent in state, or the runner's artifact-emission "
        "block regressed."
    )
    assert actual == CSU_EXPECTED_COHORT_SIZE, (
        f"G1 negative-control: CSU cohort_size shifted from "
        f"{CSU_EXPECTED_COHORT_SIZE} (PR #116 closure baseline) to "
        f"{actual}. The CSU pipeline cohort assembly produced a "
        f"different patient count than the empirical anchor. This may "
        f"indicate cohort-build path regression, eligibility-filter "
        f"changes, or upstream data shifts. Re-audit "
        f"src/agents/data/cohort_constructor and the CSU eligibility "
        f"filters.\n"
        f"If the shift is intentional, update CSU_EXPECTED_COHORT_SIZE "
        f"IN THIS COMMIT with PR/SHA reference + signed memo."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_csu_layer_1_post_anchor_drops_unchanged(
    csu_negative_control_artifact: dict,
) -> None:
    """All five CSU post-anchor journey-metadata features are dropped.

    Plan v4 §2 G1 negative-control invariant: HBLP wiring must NOT
    change the Layer 1 manifest-driven drop set on CSU. The five
    ``CSU_POST_INDEX_JOURNEY_FEATURES`` (declared
    ``knowable_at=post_index`` in
    ``src/data/manifests/csu_feature_manifest.py``) MUST all appear
    in ``leakage_dropped_features`` AND have a corresponding Layer 1
    ``severity=high, remediation=drop`` verdict.

    Mirrors the existing
    ``test_csu_val_auc_measurement.test_csu_post_index_journey_features_dropped_via_layer_1``
    invariant; this test pins the same closure under the v4 G1
    negative-control framing so a single CSU regression fires both
    test_csu_val_auc_measurement.py AND
    test_csu_negative_control_20260510.py with consistent messaging.

    A regression here proves the HBLP wiring (when it lands in G3) is
    not a NO-OP for CSU and the v4 G1 invariant is violated.
    """
    dropped = set(csu_negative_control_artifact.get("leakage_dropped_features") or [])
    verdicts = csu_negative_control_artifact.get("adaptive_verdicts") or []

    layer_1_drops = {
        v["feature"]
        for v in verdicts
        if isinstance(v, dict)
        and v.get("layer") == "1"
        and v.get("severity") == "high"
        and v.get("remediation") == "drop"
        and v.get("feature")
    }

    missing_from_dropped = [f for f in CSU_POST_INDEX_JOURNEY_FEATURES if f not in dropped]
    missing_layer_1 = [f for f in CSU_POST_INDEX_JOURNEY_FEATURES if f not in layer_1_drops]

    assert not missing_from_dropped, (
        f"G1 negative-control: CSU post-anchor features missing from "
        f"leakage_dropped_features: {missing_from_dropped}. "
        f"Saw dropped={sorted(dropped)}. "
        f"HBLP wiring must NOT change the Layer 1 drop set on CSU."
    )
    assert not missing_layer_1, (
        f"G1 negative-control: CSU post-anchor features missing a "
        f"Layer 1 drop verdict: {missing_layer_1}. "
        f"Layer 1 verdicts seen: {sorted(layer_1_drops)}. "
        f"HBLP wiring must NOT change Layer 1 manifest verdicts on CSU."
    )
