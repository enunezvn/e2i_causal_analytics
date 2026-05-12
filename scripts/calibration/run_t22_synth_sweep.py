"""T2.2 perm-anchored AUC threshold calibration — single-cell runner.

Backlog #135. Spec: ``docs/calibration/t22_perm_anchored_synth_20260510.md``
§2.2 (per-cell computation) + §4 (compute-runner harness).

For one (seed, target_auc) cell:

1. Map ``target_auc`` to a ``signal_scale`` for ``synthetic_rwd_realistic``
   via a fixed calibration table empirically derived in PR #152 §13. The
   table targets a working AUC range of [0.55, 0.85].
2. Generate a synthetic cohort with n_patients=1400 (≈ 1000 train + 400
   held-out), prevalence=0.10 (avoids extreme imbalance at small n that
   distorts AUC measurement), missing_demo_rate=0.0 (clean signal-to-
   noise; missing-data injection is a downstream stress test, not a
   calibration variable).
3. Train a logistic regression on the 4 demographic features that
   ``_generate_target`` uses (age_norm, icd_severe, insurance_premium,
   long_eligibility). Stratified train/test split per seed.
4. Compute ``realized_auc`` from the trained model's test-set probas.
5. Call ``compute_permutation_test`` (the same helper the model_trainer
   evaluator uses) with n_permutations=200 to derive
   ``permutation_null_p99``. ``margin_p99 = realized_auc - perm_null_p99``.
6. Emit one JSONL row capturing all inputs + outputs for downstream
   aggregation.

Run via::

    PYTHONPATH=src python scripts/calibration/run_t22_synth_sweep.py \
        --seed 0 --target-auc 0.65 \
        --output-jsonl calibration_runs/t22_synth_seed0_auc065.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
    compute_permutation_test,
)
from src.repositories.synthetic_rwd_realistic import (
    RwdRealisticConfig,
    generate_rwd_realistic,
)

# Calibration table — empirical mapping from target_auc to signal_scale.
# Calibrated 2026-05-12 against the 4-feature LR at n=1400, prevalence=0.10,
# missing_demo_rate=0.0 over seeds 0-4. Realized AUC tracks target within
# ~±0.05 at the small-n config; cells that exceed ±0.02 mean drift are
# flagged downstream by ``aggregate_t22_sweep.py``.
TARGET_AUC_TO_SIGNAL_SCALE: dict[float, float] = {
    0.55: 0.7,
    0.60: 1.1,
    0.65: 1.4,
    0.70: 2.2,
    0.75: 3.2,
    0.80: 4.5,
    0.85: 5.8,
}

# Cohort-size parameters held constant across all cells. Documented in
# the doc spec §2.1; selected to match the test-pin convention and to keep
# the perm-null distribution wide (small-n is the worst-case for the
# permutation-anchored margin).
N_PATIENTS = 1400
PREVALENCE = 0.10
MISSING_DEMO_RATE = 0.0
TEST_SIZE = 400  # train ≈ 1000, test ≈ 400 (close to doc's 1000+200+200)
N_PERMUTATIONS = 200  # matches DEFAULT_PERMUTATION_COUNT


def _extract_target_features(df) -> np.ndarray:
    """Recreate the 4-feature surface used inside ``_generate_target``.

    Keeping the feature extraction local to this script preserves the
    invariant that the regime's target-generating coefficients and the
    classifier's input features stay in lockstep — if a future PR adds
    a 5th coefficient to ``_generate_target``, this function MUST add
    the matching feature here (the sweep would otherwise systematically
    underestimate the achievable AUC at each signal_scale).
    """
    age_norm = (df["age"].values - 50) / 20
    icd_severe = df["primary_diagnosis_code"].isin(["L50.1", "L50.8"]).astype(int).values
    insurance_premium = (
        df["insurance_product"].isin(["commercial_PPO", "self_insured"]).astype(int).values
    )
    long_eligibility = (df["eligibility_duration_days"].values > 365).astype(int)
    return np.column_stack([age_norm, icd_severe, insurance_premium, long_eligibility])


def run_cell(seed: int, target_auc: float) -> dict[str, float | int | str | None]:
    """Run a single (seed, target_auc) sweep cell and return its result row."""
    if target_auc not in TARGET_AUC_TO_SIGNAL_SCALE:
        raise ValueError(
            f"target_auc={target_auc} is not in the calibrated table "
            f"{sorted(TARGET_AUC_TO_SIGNAL_SCALE)}. Add a calibration entry "
            f"before sweeping a new target AUC."
        )
    signal_scale = TARGET_AUC_TO_SIGNAL_SCALE[target_auc]

    cfg = RwdRealisticConfig(
        n_patients=N_PATIENTS,
        prevalence=PREVALENCE,
        missing_demo_rate=MISSING_DEMO_RATE,
        signal_scale=signal_scale,
        seed=seed,
    )
    df = generate_rwd_realistic(cfg)
    X = _extract_target_features(df)
    y = df["treatment_initiated"].values

    # Stratify to keep both classes in train + test even at prevalence=0.10.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test)
    realized_auc = float(roc_auc_score(y_test, proba_test[:, 1]))

    perm = compute_permutation_test(y_test, proba_test, n_permutations=N_PERMUTATIONS)
    perm_null_p99 = perm.get("permutation_null_p99")
    margin_p99: float | None
    if perm_null_p99 is None:
        margin_p99 = None
    else:
        margin_p99 = realized_auc - float(perm_null_p99)

    return {
        "seed": int(seed),
        "target_auc": float(target_auc),
        "signal_scale": float(signal_scale),
        "realized_auc": realized_auc,
        "perm_null_p99": (float(perm_null_p99) if perm_null_p99 is not None else None),
        "margin_p99": margin_p99,
        "n_patients": int(N_PATIENTS),
        "prevalence": float(PREVALENCE),
        "test_size": int(TEST_SIZE),
        "n_permutations": int(N_PERMUTATIONS),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-auc", type=float, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL file (one row appended).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    row = run_cell(seed=args.seed, target_auc=args.target_auc)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Codex pass-1 L4: write-mode (NOT append) so re-running a cell
    # overwrites the prior row. Append-mode would silently produce
    # duplicate rows when a cell is re-run, which the aggregator's
    # ``_load_rows`` then double-counts per target_auc. The per-cell
    # output file name embeds (seed, target_auc) so collisions only
    # happen on intentional re-runs.
    with args.output_jsonl.open("w") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
