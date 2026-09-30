"""#2335 disproof: does scoping detect_leakage to the contract covariates change the route?

Before the fix a sweep retrain's scope_spec.required_features was scope_builder's generic
placeholder list, which no table carries, so detect_leakage's structural checks (perfect
separation, zero variance, MI, logical dependency, single-feature AUC, Cramer's V) ran on
ZERO features. After it they run on the contract's covariates. This measures whether that
newly-real coverage flags anything on the hcp_adoption goldstd frame.

Pure and local: no DB / Redis / network. The frame is rebuilt from the committed DGP
(HCPGenerator seed 42 + generate_hcp_brand_adoption_frame seed 427, the aggregate-faithful
seed per scripts/load_hcp_brand_adoption.py; per-row labels are NOT bit-identical to live,
label rates 0.394/0.418/0.405 here vs live 0.389/0.407/0.394). Variants: the placeholder list
("default"), the contract covariates ("contract", the fix), and no list ("empty" = every
column). The patient contracts' full-column coverage was measured on LIVE data for Kisqali
in docs/demos/results/2026-09-29_2294_live_verify (detect_leakage severity none x3).

    PYTHONPATH=$PWD .venv/bin/python docs/demos/results/2026-09-30_2335_required_features/\
probe_detect_leakage_contract_features_2335.py
"""

import asyncio
from datetime import date

import numpy as np

import src

print("src from", src.__file__)
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage
from src.agents.ml_foundation.scope_definer.nodes.scope_builder import _define_excluded_features
from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator
from src.ml.synthetic.generators.hcp_brand_adoption_generator import (
    generate_hcp_brand_adoption_frame,
)

hcp = HCPGenerator(GeneratorConfig(id_prefix="scv", seed=42, n_records=5000)).generate()
if "peer_influence_score" not in hcp:
    hcp["peer_influence_score"] = np.log1p(hcp["influence_network_size"]).round(2)
adopt = generate_hcp_brand_adoption_frame(hcp, seed=427, end_date=date(2026, 6, 1), n_months=37)
COLS = [
    "peer_influence_score",
    "influence_network_size",
    "years_experience",
    "specialty",
    "geographic_region",
]
DEFAULT = [
    "hcp_specialty",
    "patient_count",
    "prescription_history",
    "brand_affinity_score",
    "engagement_score",
    "channel_response_rate",
]
for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
    f = adopt[adopt.brand == brand].merge(hcp[["hcp_id"] + COLS], on="hcp_id")[
        COLS + ["adopted", "data_split"]
    ]
    splits = {
        s: f[f.data_split == s].reset_index(drop=True)
        for s in ("train", "validation", "test", "holdout")
    }
    for label, req in (("default", DEFAULT), ("contract", COLS), ("empty", [])):
        state = {
            "experiment_id": "probe_2335",
            "train_df": splits["train"],
            "validation_df": splits["validation"],
            "test_df": splits["test"],
            "holdout_df": splits["holdout"],
            "scope_spec": {
                "prediction_target": "adopted",
                "required_features": req,
                "excluded_features": _define_excluded_features({}),
                "feature_manifest_source": "synthetic_csu",
                "problem_type": "binary_classification",
            },
        }
        out = asyncio.run(detect_leakage(state))
        print(
            brand,
            label,
            "rows",
            len(f),
            "label_rate",
            round(f.adopted.mean(), 4),
            "severity",
            out["leakage_severity"],
            "leaked",
            out["leaked_features"],
            "findings",
            [
                (d.get("feature"), d.get("severity"), d.get("check") or d.get("check_name"))
                for d in out["leakage_findings"]
            ],
            "blocking",
            out.get("blocking_issues"),
        )
