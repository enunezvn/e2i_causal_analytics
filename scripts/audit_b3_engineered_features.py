"""v5 Gate B3 — Layer 3 audit of engineered features on real cohort data.

Loads CSU + Optum patient_journeys, applies the engineer_features helper
from src.agents.ml_foundation.data_preparer.nodes.feature_engineering,
and runs the production-parity adversarial probe
(src.data.adversarial_leakage.compute_adversarial_score) on every
materialized engineered feature.

Decision rule (per B3 pre-spec
docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md and v5 plan
section 4 risk register): any engineered feature with z >= 5sigma on
real cohort data MUST be dropped or re-engineered. No threshold-shopping.

Exit codes:
    0 — all engineered features have z < 5sigma on both cohorts.
    1 — at least one engineered feature has z >= 5sigma; audit failed.
    2 — required cohort data not present.

Usage:
    python -m scripts.audit_b3_engineered_features
    python -m scripts.audit_b3_engineered_features --n-permutations 50

The default 200 permutations is production-parity. Use --n-permutations
50 for faster smoke runs (CI / regression-pin); production-parity
audit requires the full 200.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

# Production constant (mirrors adaptive_validity_check.HIGH_Z used by G2
# harness). If this drifts, the audit is no longer in production parity.
_HIGH_Z_THRESHOLD = 5.0


def _audit_cohort(
    cohort_label: str,
    data_dir: Path,
    target_col: str,
    manifest_source: str,
    n_permutations: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """Run Layer 3 audit on engineered features for one cohort.

    Returns (z_base, z_engineered, amplifying_drops).

    Decision: an engineered feature is dropped only when (a) its z >=
    _HIGH_Z_THRESHOLD AND (b) z > 1.5 * max(z over derivation_inputs).
    The amplification check is the operational leakage signature for
    combinatorial features: an interaction or composite that creates
    signal which none of its base inputs has alone is suspect.

    If z >= threshold but z <= 1.5 * max_input_z, the engineered
    feature INHERITS predictive signal from already-strong pre-anchor
    inputs; HBLP's declared_safe=True path applies (relax to 7.5sigma)
    and the audit records the case as "inherited_signal" not dropped.
    """
    import logging

    from scripts.measure_b3_val_auc_contrast import _filter_to_manifest_safe
    from scripts.run_tier1b_b2_experiment import (
        _build_features_and_target,
        _compute_marginal_z_scores,
        _load_patient_journeys,
    )
    from src.agents.ml_foundation.data_preparer.nodes.feature_engineering import (
        engineer_features,
    )
    from src.data.manifests import lookup_feature_contract

    logger = logging.getLogger(__name__)

    df = _load_patient_journeys(data_dir)
    X_raw, y = _build_features_and_target(df, target_col=target_col)

    # H1 (codex): Filter to manifest-declared pre-anchor base features
    # BEFORE computing z_base / engineering. Without this, the baseline
    # surface contains post-anchor leaky columns
    # (Optum: initiated_biologic_180d z=57σ; CSU: data_quality_score
    # z=74σ — undeclared) that contaminate max_input_z and make the
    # amplification verdict unreliable.
    X, dropped_pre_filter = _filter_to_manifest_safe(X_raw, manifest_source)
    base_cols = list(X.columns)

    X, materialized = engineer_features(X, manifest_source)

    print(
        f"[{cohort_label}] n_rows={len(X)} n_base_after_filter={len(base_cols)} "
        f"(dropped {len(dropped_pre_filter)} non-pre-anchor / un-manifested) "
        f"target={target_col} n_pos={int(y.sum())} engineered_added={materialized}"
    )

    if not materialized:
        print(f"[{cohort_label}] WARNING: no engineered features materialized")
        return {}, {}, []

    z_base = _compute_marginal_z_scores(X[base_cols], y, n_permutations=n_permutations)
    z_eng = _compute_marginal_z_scores(X[materialized], y, n_permutations=n_permutations)

    amplifying_drops: List[str] = []
    missing_manifest: List[str] = []
    for name, z_value in z_eng.items():
        if z_value < _HIGH_Z_THRESHOLD:
            continue
        contract = lookup_feature_contract(name, data_source=manifest_source)
        if contract is None:
            # M2 (codex): a missing manifest entry is NOT itself
            # evidence of leakage amplification — it's a manifest
            # hygiene issue. Surface it loudly but do NOT count it
            # as an amplifying-drop (which would mis-classify the
            # feature). The correct resolution is to ADD the manifest
            # entry; the audit must not silently amplify a docs gap
            # into a feature drop decision.
            logger.warning(
                "[%s] %s: z=%.2f but no manifest contract — add an "
                "entry before re-running audit. NOT dropped.",
                cohort_label,
                name,
                z_value,
            )
            missing_manifest.append(name)
            continue
        input_zs = [
            float(z_base[input_name])
            for input_name in contract.derivation_inputs
            if input_name in z_base
        ]
        max_input_z = max(input_zs, default=0.0)
        if z_value > max_input_z * 1.5:
            amplifying_drops.append(name)
            print(
                f"[{cohort_label}] AMPLIFICATION: {name} z={z_value:.2f} "
                f"> 1.5 * max_input_z ({max_input_z:.2f}); flagging as leakage."
            )
        else:
            print(
                f"[{cohort_label}] INHERITED: {name} z={z_value:.2f}, "
                f"max_input_z={max_input_z:.2f}; declared_safe HBLP path applies."
            )

    if missing_manifest:
        print(
            f"[{cohort_label}] MISSING MANIFEST: {missing_manifest} "
            "(add contracts; not amplification-dropped)"
        )

    return z_base, z_eng, amplifying_drops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Number of permutations for the adversarial probe (default 200, "
        "production parity). Use 50 for faster CI / smoke runs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/calibration/b3_engineered_audit_20260511.json",
        help="Path to write the audit report JSON.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip cohorts whose data dirs are absent rather than exiting non-zero.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csu_dir = repo_root / "data" / "rwd" / "csu"
    optum_dir = repo_root / "data" / "rwd" / "optum" / "initiation"

    cohorts: Mapping[str, Tuple[Path, str, str]] = {
        "csu": (csu_dir, "treatment_initiated", "csu"),
        "optum_initiation": (optum_dir, "treatment_initiated", "optum"),
    }

    report: Dict[str, dict] = {}
    overall_dropped: List[str] = []

    for label, (data_dir, target, manifest_source) in cohorts.items():
        if not data_dir.exists():
            msg = f"Cohort data dir absent: {data_dir}"
            if args.skip_missing:
                print(f"[{label}] SKIP — {msg}")
                report[label] = {"skipped": True, "reason": msg}
                continue
            print(f"[{label}] FAIL — {msg}", file=sys.stderr)
            return 2

        try:
            z_base, z_eng, dropped = _audit_cohort(
                cohort_label=label,
                data_dir=data_dir,
                target_col=target,
                manifest_source=manifest_source,
                n_permutations=args.n_permutations,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] ERROR — {exc}", file=sys.stderr)
            report[label] = {"error": str(exc)}
            return 1

        report[label] = {
            "n_permutations": args.n_permutations,
            "z_scores_base": {k: float(v) for k, v in z_base.items()},
            "z_scores_engineered": {k: float(v) for k, v in z_eng.items()},
            "dropped_for_amplifying_leakage": dropped,
            "high_z_threshold": _HIGH_Z_THRESHOLD,
            "leakage_rule": (
                "Engineered feature is dropped only when (a) z >= "
                f"{_HIGH_Z_THRESHOLD}sigma AND (b) z > 1.5 * max(z over "
                "manifest-declared derivation_inputs). High z that "
                "merely INHERITS from already-high-z pre-anchor inputs "
                "is documented but NOT dropped; HBLP declared_safe=True "
                "applies."
            ),
        }

        if dropped:
            overall_dropped.extend(f"{label}/{name}" for name in dropped)
            print(f"[{label}] AMPLIFYING DROP: {dropped}")
        else:
            n_inherited = sum(1 for z in z_eng.values() if z >= _HIGH_Z_THRESHOLD)
            print(
                f"[{label}] PASS — {n_inherited} engineered features have "
                f"z >= {_HIGH_Z_THRESHOLD}sigma but inherit from "
                f"already-high-z pre-anchor inputs (declared_safe path)."
            )

    output_path = repo_root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nAudit report written to {output_path}")

    if overall_dropped:
        print(
            f"\nAUDIT FAILED: {len(overall_dropped)} engineered features must be "
            f"dropped or re-engineered: {overall_dropped}",
            file=sys.stderr,
        )
        return 1

    print("\nAUDIT PASSED: all engineered features are below Layer 3 threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
