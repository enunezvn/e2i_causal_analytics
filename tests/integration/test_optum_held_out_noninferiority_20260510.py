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
# Canonical baseline values are sourced from the structured sidecar:
#   docs/calibration/g1_optum_baseline_20260510.json
#
# The sidecar is the single source of truth for the empirical anchor at
# PR #116 closure (default-window cohort, n=1294, n_train_pos≈22). The
# .json file holds auc, split, cohort_n, target, window_regime, pr_number,
# and commit_sha as typed fields — no substring search over markdown.
#
# To update the baseline, edit the sidecar JSON (with a PR reference and
# domain-expert sign-off), not this test file.
_BASELINE_SIDECAR = REPO_ROOT / "docs" / "calibration" / "g1_optum_baseline_20260510.json"
_REQUIRED_SIDECAR_KEYS = frozenset(
    {
        "auc",
        "split",
        "cohort_n",
        "target",
        "window_regime",
        "pr_number",
        "commit_sha",
        "noninferiority_epsilon",
        "_schema_version",
    }
)

# Independent pinned anchors — NOT derived from the sidecar at runtime.
# These literals are the load-bearing regression pins for the PR #116
# baseline. If the sidecar is updated, these constants must be changed
# IN THE SAME COMMIT with a PR reference + domain-expert sign-off.
# They exist so test_g1_baseline_sidecar_field_types can verify the
# sidecar has not been silently downgraded without a code review.
_EXPECTED_SIDECAR_VALUES: dict = {
    "auc": 0.4347,
    "split": "held_out_test",
    "cohort_n": 1294,
    "target": "initiated_biologic_180d",
    "window_regime": "PRE=360d/POST=180d",
    "pr_number": 116,
    "commit_sha": "0dc85a4",
    "noninferiority_epsilon": 0.02,
    "_schema_version": 1,
}


def _load_baseline() -> dict:
    """Load and minimally validate the G1 Optum baseline sidecar.

    Raises ``FileNotFoundError`` if the sidecar is missing (indicates a
    broken repo state, not a data-availability skip) and ``KeyError`` if
    required fields are absent (guards against partial edits).
    """
    if not _BASELINE_SIDECAR.exists():
        raise FileNotFoundError(
            f"G1 baseline sidecar missing at {_BASELINE_SIDECAR}. "
            "This file must be present in the repo — it is not a data "
            "dependency. Check that docs/calibration/ was committed."
        )
    data = json.loads(_BASELINE_SIDECAR.read_text())
    missing = _REQUIRED_SIDECAR_KEYS - data.keys()
    if missing:
        raise KeyError(
            f"G1 baseline sidecar {_BASELINE_SIDECAR} is missing required "
            f"fields: {sorted(missing)}. Do not edit the sidecar without "
            "preserving all required keys."
        )
    return data


_BASELINE = _load_baseline()

OPTUM_BASELINE_HELDOUT_AUC: float = float(_BASELINE["auc"])
OPTUM_NONINFERIORITY_EPSILON: float = float(_BASELINE["noninferiority_epsilon"])
OPTUM_HELDOUT_AUC_FLOOR: float = OPTUM_BASELINE_HELDOUT_AUC - OPTUM_NONINFERIORITY_EPSILON

# Optum cohort target column — must match
# COHORT_TARGETS["initiation"] in scripts/run_optum_tier0_test.py.
OPTUM_INITIATION_TARGET: str = _BASELINE["target"]

# Codex pass-1 HIGH-3 (PR #137 v4 G1): canonical Optum default-window
# cohort size at PR #116 closure (n=1294 PRE=360/POST=180; NOT the
# data-snooped n=1697 relaxed-window cohort, per plan v4 §5 forbidding
# encoding the snooped outcome). Sourced from the sidecar so a sidecar
# update triggers a single change point, not a scattered constant hunt.
OPTUM_EXPECTED_DEFAULT_COHORT_SIZE: int = int(_BASELINE["cohort_n"])

# Codex pass-2 NEW-LOW (PR #137 v4 G1 iter-3): the structured sidecar
# at docs/calibration/g1_optum_baseline_20260510.json is the single
# source of truth for the empirical anchor (not the .md file).
# The .md file (docs/calibration/g1_optum_baseline_20260510.md) is
# documentation only — do not perform substring searches on it.
# Structured sidecar integrity is validated by test_g1_baseline_sidecar_*
# below. The .md path is kept here for cross-reference only.
OPTUM_BASELINE_ARTIFACT_MD_PATH = (
    REPO_ROOT / "docs" / "calibration" / "g1_optum_baseline_20260510.md"
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


# Codex pass-2 NEW-LOW (PR #137 v4 G1 iter-3): replaced substring search
# on .md with structured JSON sidecar check. The .md is documentation only.
@pytest.mark.integration
def test_optum_baseline_artifact_present_and_complete() -> None:
    """Codex pass-2 NEW-LOW (PR #137 v4 G1 iter-3): structured sidecar
    replaces substring search on .md artifact.

    Previously (MED-9) this test searched the .md text for field names
    like "auc_value" appearing anywhere in the document. A stale duplicate
    value in the markdown could satisfy the test without representing the
    actual baseline. Replaced with:
      - Structured JSON sidecar at ``docs/calibration/g1_optum_baseline_20260510.json``
      - Typed field equality checks (auc==0.4347, cohort_n==1294, ...)
      - .md remains for documentation/human-readable cross-reference only

    This test validates the JSON sidecar, NOT the .md file. Field-level
    equality guards are in ``test_g1_baseline_sidecar_*`` below.
    """
    assert _BASELINE_SIDECAR.exists(), (
        f"G1 baseline sidecar missing at {_BASELINE_SIDECAR}. "
        "The JSON sidecar must be checked into the repo — it is not a "
        "data dependency. Check that docs/calibration/ was committed."
    )
    data = json.loads(_BASELINE_SIDECAR.read_text())
    # Typed equality — not substring search. A tester cannot satisfy these
    # by adding a duplicate value elsewhere in the file.
    assert data["auc"] == pytest.approx(OPTUM_BASELINE_HELDOUT_AUC, abs=1e-9), (
        f"Sidecar auc={data['auc']!r} != test constant "
        f"OPTUM_BASELINE_HELDOUT_AUC={OPTUM_BASELINE_HELDOUT_AUC}. "
        "Update the sidecar with a PR reference + domain-expert sign-off; "
        "the test constant derives from the sidecar automatically."
    )
    assert data["cohort_n"] == OPTUM_EXPECTED_DEFAULT_COHORT_SIZE, (
        f"Sidecar cohort_n={data['cohort_n']} != "
        f"OPTUM_EXPECTED_DEFAULT_COHORT_SIZE={OPTUM_EXPECTED_DEFAULT_COHORT_SIZE}."
    )
    assert data["target"] == OPTUM_INITIATION_TARGET, (
        f"Sidecar target={data['target']!r} != OPTUM_INITIATION_TARGET={OPTUM_INITIATION_TARGET!r}."
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


# ── Sidecar schema tests (no real data required) ──────────────────────────


def test_g1_baseline_sidecar_loads_cleanly() -> None:
    """Sidecar exists, is valid JSON, and contains all required keys.

    This test runs in CI without real cohort data. It guards against:
    - The sidecar file being accidentally deleted from the repo
    - Partial edits that remove required keys
    - JSON syntax errors introduced when updating baseline values
    """
    assert _BASELINE_SIDECAR.exists(), (
        f"G1 baseline sidecar missing at {_BASELINE_SIDECAR}. "
        "The sidecar must be checked into the repo under docs/calibration/."
    )
    data = json.loads(_BASELINE_SIDECAR.read_text())
    missing = _REQUIRED_SIDECAR_KEYS - data.keys()
    assert not missing, (
        f"G1 baseline sidecar is missing required keys: {sorted(missing)}. "
        f"Present keys: {sorted(data.keys())}"
    )


def test_g1_baseline_sidecar_field_types() -> None:
    """Sidecar fields have the expected types and value ranges.

    Catches silently-wrong edits like setting auc="0.4347" (string) or
    auc=0 (zero-init placeholder that would make the floor trivially
    satisfy any positive AUC).
    """
    data = json.loads(_BASELINE_SIDECAR.read_text())
    assert isinstance(data["auc"], (int, float)), (
        f"sidecar 'auc' must be numeric; got {type(data['auc'])}"
    )
    assert 0.0 < data["auc"] < 1.0, (
        f"sidecar 'auc' = {data['auc']} is outside (0, 1); likely a placeholder or corrupt value."
    )
    assert isinstance(data["noninferiority_epsilon"], (int, float)), (
        f"sidecar 'noninferiority_epsilon' must be numeric; got "
        f"{type(data['noninferiority_epsilon'])}"
    )
    assert 0.0 < data["noninferiority_epsilon"] <= 0.10, (
        f"sidecar 'noninferiority_epsilon' = {data['noninferiority_epsilon']} "
        "is outside (0, 0.10]; check for copy-paste error."
    )
    assert isinstance(data["cohort_n"], int), (
        f"sidecar 'cohort_n' must be int; got {type(data['cohort_n'])}"
    )
    assert data["cohort_n"] > 0, f"sidecar 'cohort_n' = {data['cohort_n']} must be positive."
    assert isinstance(data["pr_number"], int), (
        f"sidecar 'pr_number' must be int; got {type(data['pr_number'])}"
    )
    assert isinstance(data["commit_sha"], str) and len(data["commit_sha"]) >= 7, (
        f"sidecar 'commit_sha' must be a non-empty string (≥7 chars); got {data['commit_sha']!r}"
    )
    assert isinstance(data["target"], str) and data["target"], (
        f"sidecar 'target' must be a non-empty string; got {data['target']!r}"
    )
    assert isinstance(data["split"], str) and data["split"], (
        f"sidecar 'split' must be a non-empty string; got {data['split']!r}"
    )
    assert isinstance(data["window_regime"], str) and data["window_regime"], (
        f"sidecar 'window_regime' must be a non-empty string; got {data['window_regime']!r}"
    )
    # Codex pass-3 LOW-1 fix: independent pinned invariants, separate from
    # sidecar-derived runtime constants. These literal values are the
    # regression pins for the PR #116 baseline. A tester cannot satisfy
    # these by editing only the sidecar (both constants AND sidecar must
    # change, which forces a code review).
    for key, expected in _EXPECTED_SIDECAR_VALUES.items():
        actual = data.get(key)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"Sidecar field {key!r} = {actual!r} does not match the "
                f"independent expected value {expected!r} (PR #116 anchor). "
                f"To update the baseline, change _EXPECTED_SIDECAR_VALUES "
                f"in this file AND the sidecar JSON in the same commit with "
                f"a PR reference + domain-expert sign-off."
            )
        else:
            assert actual == expected, (
                f"Sidecar field {key!r} = {actual!r} does not match the "
                f"independent expected value {expected!r} (PR #116 anchor). "
                f"To update the baseline, change _EXPECTED_SIDECAR_VALUES "
                f"in this file AND the sidecar JSON in the same commit with "
                f"a PR reference + domain-expert sign-off."
            )


def test_g1_baseline_sidecar_consistent_with_module_constants() -> None:
    """Module-level constants derived from the sidecar match the sidecar values.

    Guards against a future edit that updates the sidecar but forgets
    to reload the module, or a cached import with stale values.

    Note: this test is NOT a substitute for the independent pinned
    invariants in ``test_g1_baseline_sidecar_field_types`` —
    ``OPTUM_BASELINE_HELDOUT_AUC`` et al. are derived from the sidecar,
    so asserting ``constant == sidecar field`` only checks self-consistency.
    The load-bearing regression pin is the ``_EXPECTED_SIDECAR_VALUES``
    dict in ``test_g1_baseline_sidecar_field_types``.
    """
    data = json.loads(_BASELINE_SIDECAR.read_text())
    assert OPTUM_BASELINE_HELDOUT_AUC == float(data["auc"]), (
        f"Module constant OPTUM_BASELINE_HELDOUT_AUC={OPTUM_BASELINE_HELDOUT_AUC} "
        f"!= sidecar auc={data['auc']}. Module must derive from sidecar, not override it."
    )
    assert OPTUM_NONINFERIORITY_EPSILON == float(data["noninferiority_epsilon"]), (
        f"Module constant OPTUM_NONINFERIORITY_EPSILON={OPTUM_NONINFERIORITY_EPSILON} "
        f"!= sidecar noninferiority_epsilon={data['noninferiority_epsilon']}."
    )
    assert OPTUM_HELDOUT_AUC_FLOOR == pytest.approx(
        float(data["auc"]) - float(data["noninferiority_epsilon"]), abs=1e-9
    ), (
        f"OPTUM_HELDOUT_AUC_FLOOR={OPTUM_HELDOUT_AUC_FLOOR} does not equal "
        f"auc - epsilon = {float(data['auc']) - float(data['noninferiority_epsilon'])}."
    )
