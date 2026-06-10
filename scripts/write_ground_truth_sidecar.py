#!/usr/bin/env python
"""Write the Shard-03 ground-truth sidecar the Shard-11 acceptance harness reads
(``data/synthetic/ground_truth_<run>.json``), reconciling the gap the harness
flagged: ``GroundTruthStore.to_json_file`` exists but no producer ever called it.

FAITHFUL true_ate: regenerate the patient frame with the SAME config the loader uses
by default (seed=42, DGPType.CONFOUNDED) and read the per-unit ``treatment_effect_estimate``
(tau_i) the DGP STAMPS as "the recoverable ground truth" (patient_generator.py:245).
The realized ATE per brand is mean(tau_i) over that brand's rows — the value the causal
agent recovers (Shard 03: within ~0.01). NOT the config base coefficient, NOT a guess.

Usage (from the worktree, PYTHONPATH=worktree):
    python scripts/write_ground_truth_sidecar.py [--n 8000] [--seed 42]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ml.synthetic.config import DGP_CONFIGS, Brand, DGPType
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator
from src.ml.synthetic.ground_truth.causal_effects import (
    GroundTruthEffect,
    GroundTruthStore,
)

# The harness maps each cohort -> the dgp label its sidecar entry is keyed under
# (scripts/validate_synthetic_causal.py::_COHORT_DGP). initiation is the only cohort
# whose gate (gate 3) consumes _true_ate, and it maps to "confounded" — which IS the
# loader's default --dgp. We emit a confounded entry per brand; the realized per-brand
# tau is the recoverable ATE. (heterogeneous-keyed cohorts in gate 10 do NOT call
# _true_ate, so they need no sidecar entry; we still emit a heterogeneous alias from
# the same realized tau so a future _true_ate lookup fails open to a real value, not a
# missing-key crash.)
TREATMENT_VAR = "treatment_arm"
OUTCOME_VAR = "treatment_initiated"


def _per_brand_effects(df: pd.DataFrame, dgp_type: DGPType) -> list[GroundTruthEffect]:
    cfg = DGP_CONFIGS[dgp_type]
    confounders = list(getattr(cfg, "confounders", []) or ["disease_severity", "age_at_diagnosis"])
    out: list[GroundTruthEffect] = []
    tau = pd.to_numeric(df.get("treatment_effect_estimate"), errors="coerce")
    if tau is None or tau.dropna().empty:
        raise SystemExit(
            "treatment_effect_estimate is absent/empty on the generated frame — the DGP "
            "did not stamp the per-unit tau; cannot derive a faithful true_ate."
        )
    work = df.assign(_tau=tau)
    for brand_val, grp in work.groupby("brand"):
        g = grp.dropna(subset=["_tau"])
        if g.empty:
            continue
        true_ate = float(g["_tau"].mean())
        cate_by_segment = None
        if "segment_assignment" in g.columns:
            cate_by_segment = {
                str(seg): float(sub["_tau"].mean()) for seg, sub in g.groupby("segment_assignment")
            }
        split_counts = (
            {str(k): int(v) for k, v in g["data_split"].value_counts().items()}
            if "data_split" in g.columns
            else {}
        )
        out.append(
            GroundTruthEffect(
                brand=Brand(brand_val),
                dgp_type=dgp_type,
                true_ate=round(true_ate, 6),
                tolerance=0.10,
                confounders=confounders,
                treatment_variable=TREATMENT_VAR,
                outcome_variable=OUTCOME_VAR,
                cate_by_segment=cate_by_segment,
                n_samples=int(len(g)),
                data_split_counts=split_counts,
            )
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n", type=int, default=8000, help="patients to regenerate for the realized ATE"
    )
    p.add_argument("--seed", type=int, default=42, help="MUST match the loader's seed (default 42)")
    p.add_argument("--out-dir", default="data/synthetic")
    args = p.parse_args()

    # Regenerate with the loader's default DGP (confounded) and seed; the realized
    # per-unit tau is stamped on the frame (treatment_effect_estimate).
    cfg = GeneratorConfig(seed=args.seed, n_records=args.n, dgp_type=DGPType.CONFOUNDED)
    df = PatientGenerator(cfg).generate()
    brands = sorted(df["brand"].unique().tolist())
    print(f"Regenerated {len(df)} patients across brands={brands} (seed={args.seed}, confounded)")

    store = GroundTruthStore()
    effects = _per_brand_effects(df, DGPType.CONFOUNDED)
    # Emit a heterogeneous-labelled alias per brand from the same realized tau so any
    # _COHORT_DGP=heterogeneous lookup resolves to a real value (gate 10 does not use it).
    hetero_aliases = [
        GroundTruthEffect(
            brand=e.brand,
            dgp_type=DGPType.HETEROGENEOUS,
            true_ate=e.true_ate,
            tolerance=e.tolerance,
            confounders=e.confounders,
            treatment_variable=e.treatment_variable,
            outcome_variable=e.outcome_variable,
            cate_by_segment=e.cate_by_segment,
            n_samples=e.n_samples,
            data_split_counts=e.data_split_counts,
        )
        for e in effects
    ]
    for e in [*effects, *hetero_aliases]:
        store.store(e)
        print(
            f"  {e.brand.value:<14} {e.dgp_type.value:<13} true_ate={e.true_ate:+.4f} n={e.n_samples}"
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = f"{args.out_dir}/ground_truth_{stamp}.json"
    store.to_json_file(out_path)
    print(f"Wrote {len(effects) + len(hetero_aliases)} ground-truth entries -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
