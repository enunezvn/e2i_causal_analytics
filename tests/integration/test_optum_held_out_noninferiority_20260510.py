"""Optum held-out non-inferiority — Plan v4 §2 Gate G1 acceptance criterion.

Closes the **NEW** sub-criterion of Plan v4 §2 Gate G1 (Tier 1B Gate B1):

    "Optum held-out non-inferiority test — held-out test AUC on Optum is
    no WORSE than baseline within ``epsilon=0.02`` slack (held-out, NOT val)"

This test is the **non-inferiority half** of G1: it asserts that the v4
HBLP wiring (lands in Phase C, Gate G3) does NOT degrade the Optum
pipeline's held-out test AUC by more than the pre-specified slack
``epsilon = 0.02``. Optum is the low-positive-count cohort
(n_train_pos≈22 default / ≈34 relaxed) where HBLP variance-inflation
matters most — relaxation could in principle keep more Layer 3 features
that survive the inflated z-threshold, lifting AUC. But it could also
keep noise features that degrade AUC. This test pins **non-inferiority**:
the wiring may improve Optum, but it must not silently degrade it.

Held-out vs val
---------------
Plan v4 §2 G1 explicitly demands "held-out, NOT val". The runner emits
``test_metrics`` (the held-out test split, see
``scripts/run_tier0_test.py:5977``); ``validation_metrics`` is the
training-loop val split. We read ``test_metrics.roc_auc`` here.

Why ``epsilon = 0.02``
----------------------
Plan v4 §2 G1 hardcodes ``epsilon = 0.02``. This is the slack for
seed-to-seed noise on a held-out test set with ~12 positives at
Optum-default n=1294 (per
``docs/results/optum_initiation_revalidation_20260510.md`` — n_test=195
of which ~6 are positives). On so few positives a single misclassified
case shifts AUC by ~0.05+. The 0.02 slack is therefore best-case noise
floor; v4 G1 explicitly allows non-inferiority within this band but
forbids regression beyond it.

Baseline source
---------------
Baseline = the empirical held-out test AUC observed at PR #116 closure
(see ``docs/results/optum_initiation_revalidation_20260510.md`` — Test
AUC = 0.4347 at default n=1294 PRE=360/POST=180 enrollment regime). The
baseline is intentionally pinned at the default-window cohort (NOT the
relaxed n=1697 cohort), because:

1. The default-window cohort is the production methodology pending
   N3 sign-off (Plan v4 §2 N3, awaiting domain-expert review).
2. Plan v4 §5 explicitly forbids encoding the data-snooped n=1697
   GENUINE outcome as a regression test until N3 sign-off (the
   relaxed-window adoption is "data_snooped" per codex-rescue
   CLAIM-D).
3. Non-inferiority against the default-window baseline is the
   conservative reading: HBLP wiring must not degrade what the
   currently-approved methodology already achieves.

On the n=22-positives Optum default-window cohort, AUC≈0.43 is BELOW
random (perm p=0.67, MARGINAL). The non-inferiority gate is therefore
"not silently MORE broken than the empirical anchor" — i.e., the
held-out test AUC must remain ≥ 0.43 - 0.02 = 0.41. A run producing
held-out AUC < 0.41 indicates HBLP wiring made the small-positive-count
problem worse, which is the failure mode this test catches.

Real-data dependency
--------------------
This test requires Optum initiation cohort to be present at
``data/rwd/optum/initiation/``. In CI/worktrees without real data, the
test is **skipped** (not failed); local invariants are exercised by
``test_optum_initiation_arm_guard.py`` which uses a renamed synthetic
frame.

Wall-clock budget
-----------------
~2-4 min on real Optum (smaller cohort than CSU). Marked ``slow``;
not in default ``pytest -x``.

References
----------
- Plan v4 §2 Gate G1 acceptance criterion (NEW)
- Plan v4 §5 ("data_snooped" risk on n=1697)
- Plan v3 §6 Tier 1B Gate B1
- ``docs/results/optum_initiation_revalidation_20260510.md`` (baseline)
- Companion: ``test_csu_negative_control_20260510.py``
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
OPTUM_DATA_DIR = REPO_ROOT / "data" / "rwd" / "optum" / "initiation"

# ── G1 baseline + non-inferiority slack ────────────────────────────────────
# Baseline is the held-out test AUC at PR #116 closure (default-window
# cohort, n=1294, n_train_pos≈22). Source:
# docs/results/optum_initiation_revalidation_20260510.md table line 24.
#
# Why 0.4347 specifically (not 0.43 rounded): the empirical anchor is
# pinned to 4 decimal places to give the non-inferiority slack
# (epsilon=0.02) a concrete numerical floor. The implied minimum is
# 0.4347 - 0.02 = 0.4147.
OPTUM_BASELINE_HELDOUT_AUC = 0.4347
OPTUM_NONINFERIORITY_EPSILON = 0.02
OPTUM_HELDOUT_AUC_FLOOR = OPTUM_BASELINE_HELDOUT_AUC - OPTUM_NONINFERIORITY_EPSILON

# Optum cohort target column — must match
# COHORT_TARGETS["initiation"] in scripts/run_optum_tier0_test.py.
OPTUM_INITIATION_TARGET = "initiated_biologic_180d"

# Codex pass-1 HIGH-3 (PR #137 v4 G1): canonical Optum default-window
# cohort size at PR #116 closure (n=1294 PRE=360/POST=180; NOT the
# data-snooped n=1697 relaxed-window cohort, per plan v4 §5 forbidding
# encoding the snooped outcome). Pinned so a silent shift to relaxed
# windows or upstream data changes fires this gate.
OPTUM_EXPECTED_DEFAULT_COHORT_SIZE = 1294

# Codex pass-1 MED-9 (PR #137 v4 G1): the baseline artifact at
# docs/calibration/g1_optum_baseline_20260510.md is the single source
# of truth for the empirical anchor. test_optum_baseline_artifact_present
# validates the document exists + contains required fields. The numeric
# constants above must match the artifact's "Baseline metrics" table.
OPTUM_BASELINE_ARTIFACT_PATH = REPO_ROOT / "docs" / "calibration" / "g1_optum_baseline_20260510.md"
OPTUM_BASELINE_REQUIRED_FIELDS = (
    "auc_value",
    "split_name",
    "cohort_n",
    "target",
    "window_regime",
    "pr_reference",
)


@pytest.fixture(scope="module")
def optum_held_out_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    g1_artifact_registry: dict,
) -> dict:
    """Run the full tier0 pipeline on real Optum initiation data.

    Module-scoped: one subprocess invocation amortized across every
    G1 non-inferiority assertion in this file.

    Pattern mirrors ``test_csu_val_auc_measurement.csu_artifact``:
    sets ``TIER0_E2E_JSON_OUT`` so the runner emits the structured
    JSON artifact at scripts/run_tier0_test.py:5977 (which includes
    ``test_metrics.roc_auc`` — the held-out test split AUC G1 demands).
    """
    if not OPTUM_DATA_DIR.exists():
        # Codex pass-1 HIGH-1: default to hard-fail when real Optum data is
        # absent. CI green without proving real-cohort regression is the
        # exact failure mode this gate guards against. To explicitly opt
        # into a skip (CI environments that intentionally lack data
        # fixtures), set ``ALLOW_MISSING_REAL_DATA=1``.
        if os.environ.get("ALLOW_MISSING_REAL_DATA") == "1":
            pytest.skip(
                f"Optum initiation cohort not present at {OPTUM_DATA_DIR} "
                "and ALLOW_MISSING_REAL_DATA=1 set; G1 Optum non-inferiority "
                "skip explicitly opted into. Locally: run "
                "scripts/convert_optum_rwd.py --cohort initiation."
            )
        pytest.fail(
            f"Optum initiation cohort not present at {OPTUM_DATA_DIR}; "
            "G1 Optum non-inferiority requires real Optum data and "
            "ALLOW_MISSING_REAL_DATA != '1'. Set the env var to opt into "
            "skip behaviour, OR run scripts/convert_optum_rwd.py "
            "--cohort initiation before re-running."
        )

    out_dir = tmp_path_factory.mktemp("optum_g1_noninf")
    json_out = out_dir / "optum_g1_noninf.json"

    env = os.environ.copy()
    env["TIER0_E2E_JSON_OUT"] = str(json_out)

    # Use the canonical run_tier0_test.py harness (not run_optum_tier0_test.py)
    # so the JSON artifact emission path is identical to CSU's. The
    # Optum-specific wrapper sets CONFIG overrides which run_tier0_test.py
    # also accepts via --target / --indication / --brand flags.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--data-dir",
        str(OPTUM_DATA_DIR),
        "--brand",
        "competitor",
        "--target",
        OPTUM_INITIATION_TARGET,
        # Codex pass-1 LOW-11: pass --indication explicitly so the
        # cohort target is unambiguous in the artifact metadata, not
        # derived from a default. The
        # test_optum_indication_recorded_in_artifact test below verifies
        # the runner echoed this back.
        "--indication",
        "initiation",
        "--feature-manifest-source",
        "optum",
        "--hpo-trials",
        "5",  # determinism + speed
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

    # Note: We do NOT assert returncode == 0 here. Plan v4 §2 G1's
    # baseline (0.4347) is itself from a halted-at-Step-7 run (deployer
    # MARGINAL → halt). The runner returns non-zero on halt, but the
    # JSON artifact IS emitted (the e2e_out branch runs in the finally
    # block per scripts/run_tier0_test.py:5942). What G1 demands is
    # the held-out AUC value, not a clean exit. We still surface the
    # stderr in the artifact-missing branch for triage.
    if not json_out.exists():
        pytest.fail(
            f"TIER0_E2E_JSON_OUT artifact missing at {json_out}; "
            f"runner exit={result.returncode}. "
            f"stderr (truncated): {result.stderr[-1500:]!r}"
        )
    # Codex pass-2 HIGH-4 (PR #137 v4 G1): register the artifact path
    # in the session-shared registry so the lineage-audit sweep
    # (test_g1_lineage_audit_sweep.py::test_g1_lineage_audit_on_actual_relaxed_features)
    # can consume it in the SAME pytest run without env-var indirection.
    g1_artifact_registry["optum"] = json_out
    return json.loads(json_out.read_text())


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_optum_pipeline_emitted_test_metrics(
    optum_held_out_artifact: dict,
) -> None:
    """The artifact MUST contain a ``test_metrics`` block with ``roc_auc``.

    G1 non-inferiority is computed against held-out test AUC, NOT val
    AUC (plan v4 §2 G1 explicitly: "held-out, NOT val"). A missing
    ``test_metrics.roc_auc`` field means either (a) the evaluator
    skipped held-out evaluation, or (b) the JSON serializer dropped
    the value. Both are regressions worth failing on, not silent
    skips.

    Distinct from ``trained_model_present``: the Optum default-window
    cohort can produce a model that the deployer halts on (MARGINAL),
    but the evaluator's held-out test set IS still scored. ``test_metrics``
    being non-empty even when the deployer halts is the load-bearing
    behaviour this test guards.
    """
    test_metrics = optum_held_out_artifact.get("test_metrics") or {}
    assert isinstance(test_metrics, dict) and test_metrics, (
        "test_metrics is missing or empty; G1 non-inferiority requires "
        "held-out AUC, which lives at validation step's test_metrics."
    )
    assert "roc_auc" in test_metrics, (
        f"test_metrics.roc_auc missing. Keys present: {sorted(test_metrics.keys())}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_optum_held_out_auc_non_inferior(
    optum_held_out_artifact: dict,
) -> None:
    """Held-out test AUC ≥ baseline - epsilon (plan v4 §2 G1).

    Plan v4 §2 G1 acceptance criterion (Optum half):

        held-out test AUC on Optum is no WORSE than baseline within
        ``epsilon=0.02`` slack

    Baseline is pinned at ``OPTUM_BASELINE_HELDOUT_AUC = 0.4347``
    (PR #116 closure; default-window cohort n=1294). Floor =
    ``baseline - epsilon = 0.4147``.

    Failure modes this test catches:
    - HBLP wiring drops a small-N feature that was carrying signal
      (test AUC drops below floor)
    - Layer 3 z-threshold gets unintentionally tightened (more drops →
      less signal → AUC degrades)
    - Statistical regression of any kind that surfaces as held-out
      degradation
    """
    test_metrics = optum_held_out_artifact["test_metrics"]
    held_out_auc = test_metrics["roc_auc"]
    assert held_out_auc is not None, (
        "test_metrics.roc_auc is None; held-out evaluator did not produce a value."
    )
    held_out_auc_f = float(held_out_auc)
    assert held_out_auc_f >= OPTUM_HELDOUT_AUC_FLOOR, (
        f"G1 non-inferiority FAIL: held-out AUC = {held_out_auc_f:.4f} < "
        f"floor {OPTUM_HELDOUT_AUC_FLOOR:.4f} "
        f"(baseline {OPTUM_BASELINE_HELDOUT_AUC} - epsilon "
        f"{OPTUM_NONINFERIORITY_EPSILON}).\n"
        f"This indicates HBLP wiring degraded held-out performance on "
        f"Optum beyond the slack the plan allows. Re-audit the surviving "
        f"feature set, leakage_dropped_features, and Layer 3 verdicts. "
        f"If the regression is intentional (e.g., a deliberate Layer 1 "
        f"tightening), update OPTUM_BASELINE_HELDOUT_AUC IN THIS COMMIT "
        f"with a PR/SHA reference + signed update memo."
    )


# Codex pass-1 MED-9: baseline-artifact integrity test runs in CI without
# real data (no fixture dep). Marked integration only; not real_data.
@pytest.mark.integration
def test_optum_baseline_artifact_present_and_complete() -> None:
    """Codex pass-1 MED-9 (PR #137 v4 G1): baseline artifact must exist
    and carry all required fields.

    The hardcoded magic baseline number (``OPTUM_BASELINE_HELDOUT_AUC``)
    is now anchored to a documented artifact at
    ``docs/calibration/g1_optum_baseline_20260510.md``. This test
    validates the artifact is present and carries the required fields
    so a future caller can triage a non-inferiority failure against
    documented metadata, not a magic number.

    Required fields (per the baseline artifact's "Baseline metrics" table):
      - auc_value (numeric, must match OPTUM_BASELINE_HELDOUT_AUC)
      - split_name (held_out_test)
      - cohort_n (1294, must match OPTUM_EXPECTED_DEFAULT_COHORT_SIZE)
      - target (initiated_biologic_180d, must match OPTUM_INITIATION_TARGET)
      - window_regime (default)
      - pr_reference (PR #116)
    """
    assert OPTUM_BASELINE_ARTIFACT_PATH.exists(), (
        f"G1 MED-9: baseline artifact missing at "
        f"{OPTUM_BASELINE_ARTIFACT_PATH}. The non-inferiority test reads "
        f"this document for empirical-anchor metadata; without it the "
        f"hardcoded baseline magic number is un-anchored."
    )
    text = OPTUM_BASELINE_ARTIFACT_PATH.read_text()
    missing = [f for f in OPTUM_BASELINE_REQUIRED_FIELDS if f not in text]
    assert not missing, (
        f"G1 MED-9: baseline artifact at {OPTUM_BASELINE_ARTIFACT_PATH} "
        f"is missing required fields: {missing}.\n"
        f"Add a row for each missing field in the Baseline metrics "
        f"table. Required: {list(OPTUM_BASELINE_REQUIRED_FIELDS)}"
    )
    # Confirm the AUC value, cohort size, and target match the test
    # constants — the artifact is the source of truth, but the test's
    # constants must not silently drift.
    assert "0.4347" in text, (
        "G1 MED-9: baseline artifact does not record auc_value=0.4347. "
        "Constants in the test file must match the artifact."
    )
    assert "1294" in text, (
        "G1 MED-9: baseline artifact does not record cohort_n=1294. "
        "Constants in the test file must match the artifact."
    )
    assert OPTUM_INITIATION_TARGET in text, (
        f"G1 MED-9: baseline artifact does not record "
        f"target={OPTUM_INITIATION_TARGET!r}. Constants in the test "
        f"file must match the artifact."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_optum_indication_recorded_in_artifact(
    optum_held_out_artifact: dict,
) -> None:
    """Codex pass-1 LOW-11 (PR #137 v4 G1): runner artifact MUST
    record the ``--indication initiation`` flag value.

    The non-inferiority gate is specific to the initiation cohort.
    A future regression where the runner is invoked with a different
    indication (or without the flag, falling through to a default)
    would silently void the gate's empirical anchor. This test
    asserts the artifact carries the explicit indication value the
    fixture commanded.
    """
    indication = optum_held_out_artifact.get("indication")
    assert indication == "initiation", (
        f"G1 LOW-11: expected indication='initiation' in artifact "
        f"(passed via ``--indication initiation`` in the fixture's "
        f"subprocess command), but got {indication!r}. The runner "
        f"may have regressed in echoing CONFIG.indication into the "
        f"TIER0_E2E_JSON_OUT artifact."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_optum_default_window_cohort_size_pinned(
    optum_held_out_artifact: dict,
) -> None:
    """Codex pass-1 HIGH-3 (PR #137 v4 G1): pin default-window cohort
    size = 1294, NOT the data-snooped relaxed n=1697.

    Plan v4 §5 explicitly forbids encoding the data-snooped n=1697
    GENUINE outcome as a regression test until N3 sign-off. The
    baseline non-inferiority gate is therefore pinned at the
    default-window cohort. A silent shift from PRE=360/POST=180
    (default) to PRE=180/POST=90 (relaxed) — which would change cohort
    size to 1697 — fires this gate.
    """
    actual = optum_held_out_artifact.get("cohort_size")
    assert actual is not None, (
        "cohort_size missing from Optum artifact — runner did not "
        "record the assembled-cohort row count. Either eligible_df + "
        "patient_df are both absent in state, or the runner's "
        "artifact-emission block regressed."
    )
    assert actual == OPTUM_EXPECTED_DEFAULT_COHORT_SIZE, (
        f"G1 non-inferiority: Optum cohort_size = {actual} but expected "
        f"default-window {OPTUM_EXPECTED_DEFAULT_COHORT_SIZE} (PR #116 "
        f"closure baseline). If actual=1697 the relaxed-window cohort "
        f"is being used — plan v4 §5 forbids encoding that data-snooped "
        f"outcome as a regression test until N3 sign-off. Either revert "
        f"to PRE=360/POST=180 enrollment regime, OR if N3 sign-off has "
        f"landed, update OPTUM_EXPECTED_DEFAULT_COHORT_SIZE IN THIS "
        f"COMMIT with PR/SHA reference + signed memo."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
def test_optum_artifact_records_baseline_anchor(
    optum_held_out_artifact: dict,
) -> None:
    """Sanity check: artifact carries enough metadata to triage failures.

    A non-inferiority test that reports ONLY pass/fail is hard to
    triage when it fires. This test pins the minimum metadata that
    the artifact must carry so a future failure can be diagnosed
    against the empirical anchor at PR #116 without re-running the
    pipeline:

    - ``feature_manifest_source`` = "optum" (Layer 1 fired)
    - ``adaptive_verdicts`` non-empty (Layer 5 fired)
    - ``test_metrics`` carries at least ``roc_auc`` (above test)

    Mirrors the existing ``test_csu_val_auc_measurement.test_feature_manifest_source_threaded``
    check; this one applies it to Optum.
    """
    assert optum_held_out_artifact.get("feature_manifest_source") == "optum", (
        f"feature_manifest_source != 'optum'; Optum manifest contracts "
        f"did not fire. Got "
        f"{optum_held_out_artifact.get('feature_manifest_source')!r}."
    )
    verdicts = optum_held_out_artifact.get("adaptive_verdicts") or []
    assert verdicts, (
        "adaptive_verdicts is empty — Layer 5 did not produce a single "
        "verdict on the real Optum run. Either the runner regressed back "
        "to the SampleDataGenerator path, or the Optum manifest contracts "
        "no longer match the on-disk schema."
    )
