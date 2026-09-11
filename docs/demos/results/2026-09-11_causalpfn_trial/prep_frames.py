"""Host-side frame prep for the #1990 CausalPFN trial (spec §11 item 1).

DGP frames: 3 brands x 20 seeds, n_records=3000, heterogeneous DGP, planted
true_ate + segment CATE map for treatment_arm -> treatment_initiated.
Live frames: the two 1,500-row spec §2 pairs, pulled the way the route pulls them.
"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path
REPO = Path("/home/enunez/Projects/e2i_causal_analytics"); sys.path.insert(0, str(REPO))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"

SEEDS = [21, 7, 99, 123] + [s for s in range(1, 30) if s not in (7, 21)][:16]  # 20 seeds, the probe's four first
assert len(SEEDS) == 20 and len(set(SEEDS)) == 20

def dgp():
    from src.ml.synthetic.config import Brand, DGPType
    from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS
    from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator
    cols = [*ARM_CONFOUNDERS, "treatment_arm", "treatment_initiated", "segment_assignment"]
    t0 = time.time(); n = 0
    for brand in (Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI):
        for seed in SEEDS:
            name = f"dgp_{brand.value}_s{seed}"
            if (OUT / f"{name}.parquet").exists():
                continue
            cfg = GeneratorConfig(seed=seed, n_records=3000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
            df = PatientGenerator(cfg).generate()
            df[cols].to_parquet(OUT / f"{name}.parquet", index=False)
            meta = {
                "kind": "dgp", "brand": brand.value, "seed": seed, "n": int(len(df)),
                "treatment": "treatment_arm", "outcome": "treatment_initiated",
                "covariates": list(ARM_CONFOUNDERS), "segment_col": "segment_assignment",
                "true_ate": float(df.attrs["true_ate"]),
                "cate_by_segment": {k: float(v) for k, v in df.attrs["cate_by_segment"].items()},
            }
            (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
            n += 1
    print(f"dgp: wrote {n} frames in {time.time()-t0:.1f}s")

async def live():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _brand_scoped_covariates, _load_agent_estimation_frame
    pairs = [
        ("live_all_treatment_arm_persistent_180d", "treatment_arm", "persistent_180d", None, (0.019, 0.152)),
        ("live_remi_treatment_arm_treatment_initiated", "treatment_arm", "treatment_initiated", "Remibrutinib", (0.117, 0.236)),
    ]
    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    for name, t, o, brand, ref_ci in pairs:
        covariates = [c for c in _brand_scoped_covariates(list(spec["covariate"]), brand) if c not in (t, o)]
        df, select_cols = await _load_agent_estimation_frame(
            dataset="patient_journeys", treatment_var=t, outcome_var=o, covariates=covariates, limit=1500, brand=brand)
        covs = [c for c in select_cols if c not in (t, o) and c in df.columns]
        keep = [t, o, *covs]
        df[keep].to_parquet(OUT / f"{name}.parquet", index=False)
        meta = {"kind": "live", "brand": brand, "n": int(len(df)), "treatment": t, "outcome": o,
                "covariates": covs, "dtypes": {c: str(df[c].dtype) for c in keep},
                "reported_ci_spec_s2": list(ref_ci)}
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"live: {name} n={len(df)} covs={covs}")

if MODE in ("all", "dgp"):
    dgp()
if MODE in ("all", "live"):
    asyncio.run(live())
