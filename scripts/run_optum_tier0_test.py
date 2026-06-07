#!/usr/bin/env python3
"""Tier-0 MLOps pipeline runner for Optum RWD cohorts.

Thin wrapper around ``scripts/run_tier0_test.py`` that:
  1. Selects one of the three Optum cohorts (initiation / discontinuation /
     persistence) produced by ``scripts/convert_optum_rwd.py``.
  2. Sets the appropriate target column, brand, and AUC threshold per cohort.
  3. Invokes the shared ``run_pipeline`` step functions with
     ``--data-dir data/rwd/optum/<cohort>``.

Usage:
    # Full pipeline on the initiation cohort
    python scripts/run_optum_tier0_test.py --cohort initiation

    # Specific step only
    python scripts/run_optum_tier0_test.py --cohort initiation --step 2

    # Dry-run
    python scripts/run_optum_tier0_test.py --cohort initiation --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and reuse the canonical tier-0 runner. We override its CONFIG before
# calling into run_pipeline.
import scripts.run_tier0_test as tier0  # noqa: E402
from src.data.manifests import MANIFEST_SOURCES  # noqa: E402

COHORT_TARGETS: dict[str, str] = {
    "initiation": "initiated_biologic_180d",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_at_180d",
    # Mart-sourced cohorts (entity-stacked Optum drop -> convert_optum_mart.py).
    # disc/persistence derive TRUE 180d targets from the mart's coverage/gap cols
    # (Option B); apply_overrides pushes the right label into tier0.CONFIG so the
    # pipeline trains/evaluates on the per-cohort outcome (the runner-target footgun).
    "initiation_mart": "initiated_biologic_180d",
    "discontinuation_mart": "discontinued_180d",
    "persistence_mart": "persistent_at_180d",
    # HCP-grain commercial-targeting cohort (entity-stacked Optum drop ->
    # convert_optum_hcp_adoption.py). Native shape of "commercial HCP targeting";
    # the only mart grain where a strong target + admissible features co-exist.
    "hcp_adoption": "adopted_target_brand",
}

COHORT_DIR: dict[str, str] = {
    "initiation": "data/rwd/optum/initiation",
    "discontinuation": "data/rwd/optum/discontinuation",
    "persistence": "data/rwd/optum/persistence",
    # Non-``optum`` paths so the ``optum_mart`` feature-manifest override resolves
    # without an autodetect (M2) conflict against the ``optum`` source.
    "initiation_mart": "data/rwd/mart/initiation",
    "discontinuation_mart": "data/rwd/mart/discontinuation",
    "persistence_mart": "data/rwd/mart/persistence",
    "hcp_adoption": "data/rwd/mart/hcp_adoption",
}

_MART_SUFFIX = "_mart"


def _convert_hint(cohort: str) -> str:
    """Suggested converter command when a cohort's data dir is missing.

    Mart cohorts (``*_mart``) are built by the entity-stacked-mart adapter
    (``convert_optum_mart.py``), whose ``--cohort`` takes the BASE name and whose
    ``--output`` is the exact dir. Legacy optum cohorts use the raw-claims
    converter (``convert_optum_rwd.py``). The previous hint always named
    ``convert_optum_rwd.py --cohort <cohort>`` — wrong for every mart cohort
    (that converter has no ``*_mart`` cohort), a footgun for whoever hits a
    missing dir.
    """
    if cohort == "hcp_adoption":
        # HCP-grain commercial-targeting cohort: its own entity-stacked-mart adapter.
        return f"python scripts/convert_optum_hcp_adoption.py --output {COHORT_DIR[cohort]}"
    if cohort.endswith(_MART_SUFFIX):
        base = cohort[: -len(_MART_SUFFIX)]
        return f"python scripts/convert_optum_mart.py --cohort {base} --output {COHORT_DIR[cohort]}"
    return f"python scripts/convert_optum_rwd.py --cohort {cohort}"


def _mart_manifest_warning(cohort: str, feature_manifest_source: str | None) -> str | None:
    """Warn when a ``*_mart`` cohort runs without a resolved feature manifest.

    Mart dirs deliberately autodetect to ``None`` (so an explicit ``optum_mart``
    override never M2-conflicts), which means forgetting
    ``--feature-manifest-source optum_mart`` silently drops the Layer-5 leakage
    verdicts. The converter's positive-enumeration allow-list is the PRIMARY
    leakage defense (forbidden columns are never even emitted), so this is a loud
    WARNING, not a fail-close. Returns ``None`` for non-mart cohorts (they
    autodetect their own manifest) and when a source is resolved.
    """
    if not cohort.endswith(_MART_SUFFIX) or feature_manifest_source is not None:
        return None
    return (
        f"cohort '{cohort}' is running WITHOUT a feature manifest — Layer 5 "
        f"leakage verdicts will NOT fire (defense-in-depth disabled). The "
        f"converter's allow-list is still active, but pass "
        f"'--feature-manifest-source optum_mart' to restore the full guard."
    )


def _single_class_error(data_dir: Path, cohort: str, target: str) -> str | None:
    """Pre-flight fail-closed guard: return an actionable message if the cohort's
    target column has < 2 classes.

    A single-class target crashes tier0's stratified split deep in the pipeline
    with a cryptic sklearn error; this surfaces it up front instead. Reads ONLY
    the target column. Returns ``None`` — deferring to the downstream gates — when
    the journeys file or target column is absent/unreadable: the converter is
    contractually allowed to emit an empty / zero-positive cohort (see
    test_convert_zero_positive_cohort_no_error), and the deployer fail-closes a
    weak model. This guard only catches the unmodelable single-class case.
    """
    journeys = data_dir / "e2i_ml_v3_patient_journeys.parquet"
    if not journeys.exists():
        return None
    try:
        import pandas as pd

        col = pd.read_parquet(journeys, columns=[target])[target]
    except Exception:
        return None  # column absent / unreadable -> defer to downstream gates
    n_classes = int(col.nunique(dropna=True))
    if n_classes >= 2:
        return None
    return (
        f"Cohort '{cohort}' target '{target}' has {n_classes} class(es) over "
        f"{len(col)} rows — classification needs >=2 classes, so this cohort is "
        f"unusable for tier-0 modeling (stratified split would fail). Rebuild with "
        f"more data or a different window. Hint: {_convert_hint(cohort)}"
    )


@dataclass
class OptumTestConfig:
    """Overrides applied to tier0.CONFIG before running the pipeline.

    These mirror the structure of ``tier0.TestConfig`` but with values tuned
    for the leakage-safe Optum V2-style data: AUC threshold raised to 0.65
    (V2 data should be cleaner than CSU V1), minority-class recall/precision
    kept low because CSU biologic initiation is a rare event in claims RWD.
    """

    brand: str = "competitor"
    indication: str = "Chronic Spontaneous Urticaria (CSU)"
    problem_type: str = "binary_classification"
    hpo_trials: int = 10
    min_eligible_patients: int = 30
    min_auc_threshold: float = 0.65
    min_minority_recall: float = 0.10
    min_minority_precision: float = 0.05
    enable_mlflow: bool = True
    enable_opik: bool = False
    min_samples_per_split: int = 10
    # Harness cohort QC gate threshold (field-adaptive; see tier0._build_cohort_config).
    cohort_min_data_quality: float = 0.5
    # Tier C: require the bootstrap AUC CI lower bound > 0.5 (significantly
    # better than chance), not just the point estimate. Surfaced either way.
    auc_gate_require_significance: bool = False
    # Tier D memory lever: False (--single-model) skips Step 5b alternative
    # training (champion = primary) so the run avoids the multi-model peak that
    # OOMs on a memory-constrained host.
    train_alternatives: bool = True


def apply_overrides(cohort: str, overrides: OptumTestConfig) -> None:
    """Mutate ``tier0.CONFIG`` with cohort-specific values."""
    tier0.CONFIG.brand = overrides.brand
    tier0.CONFIG.indication = overrides.indication
    tier0.CONFIG.problem_type = overrides.problem_type
    tier0.CONFIG.hpo_trials = overrides.hpo_trials
    tier0.CONFIG.min_eligible_patients = overrides.min_eligible_patients
    tier0.CONFIG.min_auc_threshold = overrides.min_auc_threshold
    tier0.CONFIG.min_minority_recall = overrides.min_minority_recall
    tier0.CONFIG.min_minority_precision = overrides.min_minority_precision
    tier0.CONFIG.enable_mlflow = overrides.enable_mlflow
    tier0.CONFIG.enable_opik = overrides.enable_opik
    tier0.CONFIG.min_samples_per_split = overrides.min_samples_per_split
    tier0.CONFIG.cohort_min_data_quality = overrides.cohort_min_data_quality
    tier0.CONFIG.auc_gate_require_significance = overrides.auc_gate_require_significance
    tier0.CONFIG.train_alternatives = overrides.train_alternatives
    tier0.CONFIG.target_outcome = COHORT_TARGETS[cohort]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-0 pipeline runner for Optum RWD cohorts.")
    parser.add_argument(
        "--cohort",
        required=True,
        choices=tuple(COHORT_TARGETS.keys()),
        help="Which Optum cohort subdir to load",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8). Default: full pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-bentoml", action="store_true")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking (enabled by default)",
    )
    parser.add_argument("--enable-opik", action="store_true")
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (used to resolve data/rwd/optum/<cohort>)",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.65,
        help="Minimum validation AUC for success (default 0.65)",
    )
    parser.add_argument(
        "--min-samples-per-split",
        type=int,
        default=10,
        help=(
            "Minimum viable samples per split for split_enforcer gate "
            "(default: 10; set to 5 for discontinuation/persistence cohorts at n=47)"
        ),
    )
    parser.add_argument(
        "--cohort-min-quality",
        type=float,
        default=0.5,
        help=(
            "Harness cohort QC gate threshold on data_quality_score (default 0.5). "
            "Field-adaptive: a no-op when the cohort frame carries no "
            "data_quality_score (the adapter already did the real cohorting)."
        ),
    )
    parser.add_argument(
        "--auc-significance-gate",
        action="store_true",
        help=(
            "Make the AUC gate CI-aware: require the bootstrap AUC CI lower bound "
            "to exceed the 0.5 no-skill floor (significantly better than chance), "
            "not just the point estimate. The bootstrap CI is surfaced either way."
        ),
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help=(
            "Train only the selected primary model (skip the Step 5b champion "
            "comparison of alternatives). Memory lever for constrained hosts: "
            "avoids holding multiple trained models + bootstrap arrays at once."
        ),
    )
    parser.add_argument(
        "--feature-manifest-source",
        type=str,
        choices=tuple(sorted(MANIFEST_SOURCES)),
        default=None,
        help=(
            "Opt this run into a cohort-specific feature manifest so Layer 5 "
            "(adaptive_validity_check) consults the matching FeatureContract "
            "registry for layer='1' verdicts. When omitted the value is "
            "auto-detected from the resolved Optum cohort dir "
            "('data/rwd/optum/<cohort>' → 'optum'); pass explicitly to override "
            "the auto-detection. A conflicting override (e.g. 'csu' against an "
            "Optum data dir) fails fast (M2) rather than silently applying the "
            "wrong manifest."
        ),
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help=(
            "Skip the full pipeline (Steps 5-7) and run only data-loading + scope-definer "
            "as a converter smoke test. Recommended for the n=47 discontinuation/persistence "
            "cohorts per tier0_evaluation_vs_distilled_mlops.md:242,266 ('n=47 is a converter "
            "smoke test, NOT methodology validation') — added by tier0_quality_remediation_arc "
            "Shard C 2026-05-06."
        ),
    )
    parser.add_argument(
        "--deployment-intent",
        type=str,
        choices=("clinical", "commercial"),
        default="clinical",
        help=(
            "Deployment use case — recalibrates the deployment AUC bar. "
            "'clinical' (default): literature floor AUC 0.75 (Vickers 2019; "
            "Cook 2007), for published / site-of-care decision models. "
            "'commercial': HCP targeting / propensity (never used at site of "
            "care) — separately-cited floor AUC 0.65 + prevalence-aware "
            "operating gates (recall 0.50, MCC 0.10, net-benefit p_t 0.10). "
            "Use for the optum-mart commercial cohorts (e.g. discontinuation)."
        ),
    )

    args = parser.parse_args()

    cfg = OptumTestConfig(
        min_auc_threshold=args.min_auc,
        enable_mlflow=not args.disable_mlflow,
        enable_opik=args.enable_opik,
        hpo_trials=args.hpo_trials,
        min_samples_per_split=args.min_samples_per_split,
        cohort_min_data_quality=args.cohort_min_quality,
        auc_gate_require_significance=args.auc_significance_gate,
        train_alternatives=not args.single_model,
    )
    if args.enable_opik:
        os.environ["OPIK_ENABLED"] = "true"

    apply_overrides(args.cohort, cfg)

    data_dir = (args.data_root / COHORT_DIR[args.cohort]).resolve()
    if not data_dir.exists() and not args.dry_run:
        print(
            f"ERROR: Optum cohort directory not found: {data_dir}\n"
            f"Run: {_convert_hint(args.cohort)}",
            file=sys.stderr,
        )
        return 2

    # Pre-flight fail-closed guard: a single-class target would crash tier0's
    # stratified split with a cryptic error deep in the pipeline. Surface it here
    # with an actionable message. Defers (no-op) when the cohort file/column is
    # absent — the converter may emit an empty/zero-positive cohort by contract and
    # the deployer fail-closes a weak model; this only catches the unmodelable case.
    if not args.dry_run:
        single_class = _single_class_error(data_dir, args.cohort, tier0.CONFIG.target_outcome)
        if single_class is not None:
            print(f"ERROR: {single_class}", file=sys.stderr)
            return 2

    # Resolve which feature manifest Layer 5 should consult. Optum cohorts live
    # under 'data/rwd/optum/<cohort>', so auto-detection yields 'optum'; an
    # explicit --feature-manifest-source overrides it (and a conflicting choice
    # fails fast via the M1/M2/M3 contract). Without this, run_pipeline left
    # feature_manifest_source unset and Layer 5's manifest-driven Layer 1
    # verdicts (post-index leak catch + declared-safe σ-inflation, PR #544)
    # silently no-op'd for every Optum run. Mirrors the CSU runner.
    feature_manifest_source = tier0._resolve_feature_manifest_source(
        str(data_dir), args.feature_manifest_source
    )

    print("\n=== Optum Tier-0 Pipeline Runner ===")
    print(f"  Cohort: {args.cohort}")
    print(f"  Target: {tier0.CONFIG.target_outcome}")
    print(f"  Data dir: {data_dir}")
    print(f"  Feature manifest: {feature_manifest_source}")
    print(f"  AUC threshold: {tier0.CONFIG.min_auc_threshold}")
    print(
        f"  Deployment intent: {args.deployment_intent} "
        f"(commercial → AUC 0.65 bar; clinical → 0.75)"
    )
    print(f"  MLflow: {tier0.CONFIG.enable_mlflow}, Opik: {tier0.CONFIG.enable_opik}")

    manifest_warning = _mart_manifest_warning(args.cohort, feature_manifest_source)
    if manifest_warning is not None:
        print(f"  ⚠️  WARNING: {manifest_warning}", file=sys.stderr)

    # Tier-2 SMOKE_TEST_ONLY (per tier0_quality_remediation_arc Shard C, 2026-05-06):
    # n=47 disc/pers cohorts are documented as converter smoke tests, NOT methodology
    # validation (tier0_evaluation_vs_distilled_mlops.md:242,266). When this flag is
    # set, validate the data dir is loadable, list cohort files, print summary, exit 0
    # without invoking model trainer/evaluator/deployer.
    if args.smoke_test_only:
        print("\n[OPTUM SMOKE_TEST_ONLY MODE]")
        print(f"  Cohort: {args.cohort} ({COHORT_DIR[args.cohort]})")
        cohort_files = sorted(p.name for p in data_dir.glob("*.json")) if data_dir.exists() else []
        print(f"  Files in cohort dir: {len(cohort_files)}")
        for fname in cohort_files[:10]:
            fpath = data_dir / fname
            print(f"    {fname} ({fpath.stat().st_size} bytes)")
        if len(cohort_files) > 10:
            print(f"    ... and {len(cohort_files) - 10} more")
        print("\nSkipping Steps 5-7 (model training/eval/deploy) per --smoke-test-only.")
        print("Per tier0_evaluation_vs_distilled_mlops.md:242,266: n=47 cohorts are NOT")
        print("methodology validation — this run is a converter smoke test only.")
        return 0

    state = asyncio.run(
        tier0.run_pipeline(
            step=args.step,
            dry_run=args.dry_run,
            imbalance_ratio=None,
            include_bentoml=not args.no_bentoml,
            data_dir=str(data_dir),
            deployment_intent=args.deployment_intent,
            feature_manifest_source=feature_manifest_source,
        )
    )

    # Tier-1 Optum Task 5.2 carve-out (per tier0_quality_remediation_arc Shard C,
    # 2026-05-06): when permutation test is RANDOM at Optum scale (n>=200), invoke
    # the literal carve-out at tier0_evaluation_vs_distilled_mlops.md:703 — document
    # the run but flag it as NOT production-grade. This is reframing language only;
    # no CI-failing exit code (deployer's success_criteria_not_met already handles
    # the actual gate).
    if not args.dry_run and not args.step:
        try:
            permutation_test = (state or {}).get("permutation_test", {}) or {}
            signal_genuine = permutation_test.get("signal_genuine")
            shuffled_auc = permutation_test.get("shuffled_auc")
            p_value = permutation_test.get("p_value")
            cohort_size = (state or {}).get("data_quality_metrics", {}).get("cohort_size")
            if cohort_size is None:
                training_samples = (state or {}).get("training_samples")
                cohort_size = training_samples if training_samples is not None else 0

            if signal_genuine is False and (cohort_size or 0) >= 200:
                print(f"\n{'=' * 70}")
                print("[OPTUM TASK 5.2 CARVE-OUT INVOKED]")
                print(f"{'=' * 70}")
                print(
                    "Per tier0_evaluation_vs_distilled_mlops.md:703 —\n"
                    '  "If permutation test is RANDOM at Optum scale, document and do\n'
                    '   not publish as production-grade."'
                )
                pct = (
                    f"{(permutation_test.get('positive_rate') or 0) * 100:.1f}%"
                    if permutation_test.get("positive_rate") is not None
                    else "?"
                )
                print(
                    f"\nVerdict tag: PENDING — RANDOM at Optum scale "
                    f"(n={cohort_size}, p={p_value}, shuffled AUC={shuffled_auc}, prevalence={pct})."
                )
                print(
                    "Acceptance criterion (Task 5.2: all R-grades >= B): NOT MET.\n"
                    "This run is documented but is NOT production-grade.\n"
                    "Remediation path: Task 5.2 is data-gated on cohort prevalence growth."
                )
                print(f"{'=' * 70}")
        except Exception as e:
            # Reframing print is best-effort; never fail the runner over it.
            print(f"\n[Optum Task 5.2 reframing skipped: {type(e).__name__}: {e}]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
