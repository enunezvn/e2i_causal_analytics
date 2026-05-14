"""Layer 5 (k, epsilon) joint-threshold calibration sweep — issue #194.

Closes the n-dependent false-positive blowup in
``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py``.

Problem: Layer 5 uses a 5σ z-threshold on the permutation-anchored AUC drift
(``z = (actual_auc - null_mean) / null_std``) to flag features as suspicious.
By the CLT the permutation-null std shrinks as ``~1/sqrt(n)``, so at large n
even a benign feature with single-feature AUC just 0.005 above chance lands
many σ above the null — false positives on benign features become predictable
at n ≥ 10k.

Decision (set by the user in issue #194; this script does NOT re-litigate):
adopt a JOINT threshold ``(z > k) AND (|delta_AUC| > epsilon)`` where
``delta_AUC = actual_auc - null_mean`` (folded AUC-ROC, same scale as
``compute_adversarial_score``'s ``actual_auc`` / ``null_mean``). The absolute-
effect floor ``epsilon`` is interpretable in the pharma domain: less than 1
AUC-ROC point above chance is not an actionable leakage signal.

This sweep characterises the FPR over benign features at multiple cohort
sizes for a grid of ``(k, epsilon)`` candidates and a TPR floor that pins
the known leak patterns (post_index_aggregation, post_hoc_termination,
treatment_leaked_code, spurious_correlation) at signal_scale=1.0.

Run via::

    PYTHONPATH=src .venv/bin/python \\
        scripts/calibration/run_layer5_joint_threshold_sweep.py \\
        --output-jsonl calibration_runs/issue_194_sweep.jsonl

The default grid is ``k in {3.0, 3.5, 4.0, 4.5, 5.0}`` and
``epsilon in {0.005, 0.01, 0.015, 0.02}`` per the issue body; the runner
also evaluates the legacy ``z > 5σ`` rule alone (epsilon=0.0) as a
baseline. Seed is pinned so the sweep is deterministic.

The output JSONL is keyed by ``(n, k, epsilon)`` with the per-cell FPR,
TPR, and the underlying counters (n_benign_flagged, n_benign_tested,
n_leak_flagged, n_leak_tested).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data.adversarial_leakage import compute_adversarial_score
from src.repositories.synthetic_rwd_realistic import (
    RwdRealisticConfig,
    generate_rwd_realistic,
)

# Pinned for sweep reproducibility — codex pass-1 question (c).
DEFAULT_SEED = 7
DEFAULT_N_PERMUTATIONS = 200

# Cohort sizes to sweep. n=10000 is the primary target (issue body); the
# smaller sizes ensure we don't over-tighten and lose TPR at small n
# (the existing 14/14 ``synthetic_rwd_realistic`` tests run at n=2000 and
# n=5000), and n=50000 stress-tests the joint-threshold ceiling — even
# at very large n the joint check should hold FPR ≤ 1% on benign features.
DEFAULT_N_GRID: tuple[int, ...] = (1000, 5000, 10000, 50000)

# Default sweep grids. Issue #194 body originally proposed
# ``k in {3.0, 3.5, 4.0, 4.5, 5.0}`` × ``epsilon in {0.005, 0.01,
# 0.015, 0.02}`` as the starting point, but the calibration sweep
# (2026-05-14) showed the legitimate-weak-predictor p99 |delta_AUC|
# at n=10000 is 0.0913 — so ``epsilon`` must extend above 0.02 to
# discriminate benign weak signals from real leaks. The grid below
# spans both the issue-body range and the wider band that the
# calibration explored, with ``epsilon=0.10`` (the production-
# chosen floor) explicitly included so a reader of the sweep
# output can verify the chosen value's FPR ledger at every n.
# The (k=5.0, epsilon=0.0) cell reproduces the LEGACY behaviour
# (z > HIGH_Z alone) as the FPR-vs-alternative baseline.
#
# Codex pass-1 MEDIUM-2: prior default ``epsilon`` grid stopped at
# 0.02, so re-running this script in its DEFAULT mode would NOT
# evaluate the production-chosen ``epsilon=0.10``. Extended below.
DEFAULT_K_GRID: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0)
DEFAULT_EPSILON_GRID: tuple[float, ...] = (
    0.0,
    0.005,
    0.01,
    0.015,
    0.02,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
)

# How many independent benign-feature replicates to evaluate at each n.
# Each replicate generates a fresh i.i.d. standard normal column on a
# synthetic cohort and scores it against the target. At n=10000 a benign
# feature has z ~ N(0, 1) on the folded-AUC null, so the legacy 5σ rule
# yields theoretical FPR <1 in 1.7M (one-sided), but the FOLDED AUC scale
# inflates the lower-tail contributions, AND at low n the realized null
# std is itself noisy enough that empirical 5σ rejections happen far more
# often than the asymptotic prediction. 200 replicates × 4 cohort sizes
# = 800 i.i.d. benign-feature evaluations; the n=10k FPR has Monte-Carlo
# half-width ≈ sqrt(0.01·0.99/200) ≈ 0.7pp at the 1% target — adequate
# to discriminate between the candidate (k, epsilon) pairs.
DEFAULT_N_BENIGN_REPLICATES = 200

# Leak patterns the joint threshold MUST continue to flag (TPR preservation).
# Matches the 4 injection patterns in ``synthetic_rwd_realistic`` plus
# the pure-noise CONTROL (which is benign — we evaluate it under the
# benign FPR ledger, not the leak TPR ledger).
LEAK_PATTERNS: tuple[str, ...] = (
    "post_index_aggregation",
    "post_hoc_termination",
    "treatment_leaked_code",
    "spurious_correlation",
)


def _score_one(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    n_permutations: int,
) -> tuple[float, float, float, float]:
    """Score one feature; return (z_score, delta_auc, actual_auc, null_mean).

    ``delta_auc = actual_auc - null_mean`` (folded AUC-ROC scale).
    Returns NaNs if the score is degenerate.
    """
    result = compute_adversarial_score(
        feature,
        target,
        n_permutations=n_permutations,
        seed=seed,
    )
    z = float(result.get("z_score", float("nan")))
    actual_auc = float(result.get("actual_auc", float("nan")))
    null_mean = float(result.get("null_mean", float("nan")))
    delta_auc = actual_auc - null_mean if np.isfinite(actual_auc) and np.isfinite(null_mean) else float("nan")
    return z, delta_auc, actual_auc, null_mean


def _decide_joint(z: float, delta_auc: float, *, k: float, epsilon: float) -> bool:
    """Apply the joint ``(z > k) AND (|delta_AUC| > epsilon)`` decision.

    Returns False for non-finite z (degenerate score — treat as not-flagged,
    consistent with ``hblp_classify``'s severity=info fallback).
    """
    if not (np.isfinite(z) and np.isfinite(delta_auc)):
        return False
    return (z > k) and (abs(delta_auc) > epsilon)


def _benign_evidence_for_n(
    n: int, *, n_replicates: int, base_seed: int, n_permutations: int
) -> list[tuple[float, float]]:
    """Collect (z, delta_auc) for ``n_replicates`` benign-feature evaluations.

    Issue #194 — codex pass-1 MEDIUM-2 fix: the calibration must reproduce
    the SAME failure mode that motivated the issue, namely the legitimate
    weak demographic predictors (``age``, ``eligibility_duration_days``)
    used in ``synthetic_rwd_realistic._generate_target``. These features
    are NOT leaks — they are real, weak, causal predictors with empirical
    single-feature AUC ~0.54 — but the legacy 5σ z-threshold flagged them
    at large n. A pure i.i.d. Gaussian benign feature has effective
    AUC ≈ 0.50 and trips the legacy threshold ~0% of the time; it would
    not surface the issue.

    The sweep collects evidence for BOTH demographic features per cohort
    (2× evaluations per replicate) so the FPR ledger is over the full
    "legitimate weak signal" population the issue body refers to.

    The synthetic cohort itself is regenerated per replicate — at
    fixed ``signal_scale=1.0`` and ``prevalence=0.024``, with the cohort
    seed offset by ``base_seed * 10000 + replicate_idx``. Both
    demographic-feature scorings on a single cohort share the same
    target labels (they are NOT independent draws), but they are
    independent feature axes. The FPR is computed over the joint
    population.
    """
    evidence: list[tuple[float, float]] = []
    for r in range(n_replicates):
        cohort_seed = base_seed * 10000 + r
        cohort = generate_rwd_realistic(
            RwdRealisticConfig(
                n_patients=n,
                prevalence=0.024,
                missing_demo_rate=0.0,
                signal_scale=1.0,
                seed=cohort_seed,
            )
        )
        target = cohort["treatment_initiated"].to_numpy(dtype=int)
        # Skip pathological cohorts with only one target class — the
        # permutation null is undefined and the FPR ledger should not
        # count them.
        if len(np.unique(target)) < 2:
            continue
        # Per codex pass-1 MEDIUM-2: use the SAME demographic features
        # that were calibrated against (``age`` + eligibility_duration_days``)
        # so the script reproduces the calibration regime, not a
        # different (cleaner) feature distribution.
        for feat_name in ("age", "eligibility_duration_days"):
            feat = cohort[feat_name].to_numpy(dtype=float)
            z, delta_auc, _, _ = _score_one(
                feat, target, seed=base_seed, n_permutations=n_permutations
            )
            evidence.append((z, delta_auc))
    return evidence


def _leak_evidence_for_n(
    n: int, *, n_replicates: int, base_seed: int, n_permutations: int
) -> dict[str, list[tuple[float, float]]]:
    """Collect (z, delta_auc) per leak pattern for ``n_replicates`` cohorts.

    Returns a dict keyed by leak pattern → list of (z, delta_auc) tuples.
    """
    by_pattern: dict[str, list[tuple[float, float]]] = {p: [] for p in LEAK_PATTERNS}
    for r in range(n_replicates):
        cohort_seed = base_seed * 10000 + r
        for pattern in LEAK_PATTERNS:
            cohort = generate_rwd_realistic(
                RwdRealisticConfig(
                    n_patients=n,
                    prevalence=0.024,
                    missing_demo_rate=0.0,
                    signal_scale=1.0,
                    leakage_pattern=pattern,
                    seed=cohort_seed,
                )
            )
            target = cohort["treatment_initiated"].to_numpy(dtype=int)
            if len(np.unique(target)) < 2:
                continue
            # The leak column is named ``*_LEAK`` for every pattern
            # except ``borderline_genuine`` (which we don't sweep here —
            # it's the v5 Gate C2 sanity-check, separately handled).
            leak_col = [c for c in cohort.columns if c.endswith("_LEAK")][0]
            feature = cohort[leak_col].to_numpy(dtype=float)
            z, delta_auc, _, _ = _score_one(
                feature, target, seed=base_seed, n_permutations=n_permutations
            )
            by_pattern[pattern].append((z, delta_auc))
    return by_pattern


def run_sweep(
    *,
    output_jsonl: Path,
    n_grid: Iterable[int] = DEFAULT_N_GRID,
    k_grid: Iterable[float] = DEFAULT_K_GRID,
    epsilon_grid: Iterable[float] = DEFAULT_EPSILON_GRID,
    n_benign_replicates: int = DEFAULT_N_BENIGN_REPLICATES,
    n_leak_replicates: int = 10,
    base_seed: int = DEFAULT_SEED,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> list[dict[str, float]]:
    """Run the full (n, k, epsilon) sweep; emit JSONL rows.

    Returns the row list as well (caller-convenience for tests).
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for n in n_grid:
        # Collect benign + leak evidence ONCE per n (the slow step), then
        # iterate the (k, epsilon) grid as a pure decision over the cached
        # scores. Saves ~|K|·|E| seconds per n.
        benign_evidence = _benign_evidence_for_n(
            n,
            n_replicates=n_benign_replicates,
            base_seed=base_seed,
            n_permutations=n_permutations,
        )
        leak_evidence = _leak_evidence_for_n(
            n,
            n_replicates=n_leak_replicates,
            base_seed=base_seed,
            n_permutations=n_permutations,
        )

        for k in k_grid:
            for epsilon in epsilon_grid:
                n_benign = len(benign_evidence)
                n_benign_flagged = sum(
                    1 for z, d in benign_evidence if _decide_joint(z, d, k=k, epsilon=epsilon)
                )
                fpr = n_benign_flagged / n_benign if n_benign > 0 else float("nan")

                tpr_by_pattern: dict[str, float] = {}
                for pattern, evidence in leak_evidence.items():
                    n_pat = len(evidence)
                    n_flagged = sum(
                        1 for z, d in evidence if _decide_joint(z, d, k=k, epsilon=epsilon)
                    )
                    tpr_by_pattern[pattern] = n_flagged / n_pat if n_pat > 0 else float("nan")

                row = {
                    "n": int(n),
                    "k": float(k),
                    "epsilon": float(epsilon),
                    "fpr": float(fpr),
                    "n_benign_flagged": int(n_benign_flagged),
                    "n_benign_tested": int(n_benign),
                    "tpr_post_index_aggregation": float(
                        tpr_by_pattern.get("post_index_aggregation", float("nan"))
                    ),
                    "tpr_post_hoc_termination": float(
                        tpr_by_pattern.get("post_hoc_termination", float("nan"))
                    ),
                    "tpr_treatment_leaked_code": float(
                        tpr_by_pattern.get("treatment_leaked_code", float("nan"))
                    ),
                    "tpr_spurious_correlation": float(
                        tpr_by_pattern.get("spurious_correlation", float("nan"))
                    ),
                    "n_permutations": int(n_permutations),
                    "n_benign_replicates": int(n_benign_replicates),
                    "n_leak_replicates": int(n_leak_replicates),
                    "base_seed": int(base_seed),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
                rows.append(row)

    with output_jsonl.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL file (overwritten on each run).",
    )
    parser.add_argument(
        "--n-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_N_GRID),
        help="Comma-separated cohort sizes (default: 1000,5000,10000,50000).",
    )
    parser.add_argument(
        "--k-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_K_GRID),
        help="Comma-separated z-thresholds (default: 3.0,3.5,4.0,4.5,5.0).",
    )
    parser.add_argument(
        "--epsilon-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_EPSILON_GRID),
        help="Comma-separated |delta_AUC| floors (default: 0.0,0.005,0.01,0.015,0.02).",
    )
    parser.add_argument(
        "--n-benign-replicates", type=int, default=DEFAULT_N_BENIGN_REPLICATES
    )
    parser.add_argument("--n-leak-replicates", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = run_sweep(
        output_jsonl=args.output_jsonl,
        n_grid=[int(x) for x in args.n_grid.split(",")],
        k_grid=[float(x) for x in args.k_grid.split(",")],
        epsilon_grid=[float(x) for x in args.epsilon_grid.split(",")],
        n_benign_replicates=args.n_benign_replicates,
        n_leak_replicates=args.n_leak_replicates,
        base_seed=args.base_seed,
        n_permutations=args.n_permutations,
    )
    # Brief stdout summary — full details in the JSONL.
    print(f"Sweep complete: {len(rows)} rows -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
