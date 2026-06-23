"""FAITHFUL calibration gate for the HCP-adoption cohort (the 4th cohort × 3 brands).

DB-BACKED (E2I_DB_INTEGRATION) by necessity: HCP adoption is a JOIN cohort —
``hcp_brand_adoption`` ⋈ ``hcp_profiles``, with the 5 leakage-safe predictive
covariates (peer_influence_score, influence_network_size, years_experience,
specialty, geographic_region) embedded at load time by FeatureBuilder's HCP path.
There is NO faithful in-memory frame: the artifact generator
(``generate_hcp_adoption_frame``) emits only centrality + treatment_arm (2 encoded
features), so a hermetic gate would test a 2-feature PROXY, not the ~19-feature
model that actually deploys. This gate therefore routes through the EXACT real
pipeline ``_run_one_cohort`` uses — ``FeatureBuilder(make_hcp_spec(brand))
.load_frame(client)`` → ``train_cohort_model`` → holdout-split AUC — against the
docker DB that holds the synthetic HCP rows. No hand-rolled encoder.

WHY this gate exists: HCP-adoption was NOT part of the T9/T11 patient-cohort DGP
enrichment, and its per-brand holdout AUC sits BELOW the 0.80 patient target
(2026-06-22 registry: Remibrutinib 0.765 / Fabhalta 0.810 / Kisqali 0.786). That
sub-0.80 is ACCEPTED — the adoption signal is genuinely thinner than the
persistence/initiation outcomes. This gate INCLUDES HCP adoption in the
regression-coverage regime (previously it had only artifact-shape tests, no AUC
gate) and locks the deployed models against a silent reseed/retrain regression,
WITHOUT imposing the patient 0.80 floor: it asserts each brand trains the full
~19-feature model and clears a realistic adoption floor (>=0.70).

All three brands run in ONE event loop on ONE resolved client (mirroring
``run_hcp_cohorts``' single ``asyncio.run`` pass) — a per-brand parametrize would
spawn a fresh loop per case and the async DB client's teardown would race across
loops ("Event loop is closed").

Skipped unless ``E2I_DB_INTEGRATION=1`` (the join cohort cannot be generated
in-memory).
"""

from __future__ import annotations

import os

import pytest
from sklearn.metrics import roc_auc_score

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import BRANDS, make_hcp_spec
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.gold_standard_eval.run_persistence_eval import _resolve_client

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="HCP adoption is a JOIN cohort with no faithful in-memory frame; the "
    "calibration gate needs the docker DB. Run with E2I_DB_INTEGRATION=1.",
)

_CHAMPION_TRAIN_SPLITS = ("train", "validation")
_HOLDOUT_SPLIT = "holdout"
# User-accepted band: HCP-adoption AUC is allowed BELOW the 0.80 patient floor.
# Floor 0.70 sits clearly under the lowest observed (0.765) with headroom; the
# ceiling is a sanity bound. This guards REGRESSION, not the 0.80 target.
_HCP_AUC_FLOOR = 0.70
_HCP_AUC_CEIL = 0.86


async def _faithful_hcp(client, brand: str) -> dict:
    """Train the REAL gold-standard HCP-adoption model exactly as ``_run_one_cohort``
    does (FeatureBuilder JOIN-load → calibrated LR → holdout-split AUC)."""
    spec = make_hcp_spec(brand)
    assert spec.label_column == "adopted" and spec.grain == "hcp"
    frame = await FeatureBuilder(spec).load_frame(client, splits=None)
    assert not frame.empty, f"{brand}: HCP load_frame returned an empty frame"
    train = frame[frame["data_split"].isin(_CHAMPION_TRAIN_SPLITS)]
    holdout = frame[frame["data_split"] == _HOLDOUT_SPLIT]
    assert not train.empty and not holdout.empty, f"{brand}: missing train/holdout split"

    fb = FeatureBuilder(spec)
    x_train, y_train = fb.build_from_frame(train)  # FIT
    model = train_cohort_model(spec, x_train, y_train)
    x_te = fb.transform(holdout)
    y_te = holdout[spec.label_column].astype(int).to_numpy()
    pos = list(model.classes_).index(1) if 1 in model.classes_ else 0
    auc = float(roc_auc_score(y_te, model.predict_proba(x_te.to_numpy(dtype=float))[:, pos]))
    return {
        "auc": auc,
        "n_features": len(fb.feature_columns),
        "n_holdout": int(len(holdout)),
        "prev": float(y_te.mean()),
    }


@pytest.mark.asyncio
async def test_hcp_adoption_faithful_holdout_auc_all_brands():
    client = await _resolve_client(None)  # one faithful docker client, one loop
    results = {brand: await _faithful_hcp(client, brand) for brand in BRANDS}

    for brand, m in results.items():
        # The JOIN embeds the 5 hcp_profiles covariates → ~19 encoded features (the
        # deployed model). A ~2-feature result means the join silently dropped them.
        assert m["n_features"] >= 15, (
            f"{brand}: only {m['n_features']} encoded HCP features (expected ~19 from "
            f"the hcp_profiles JOIN); the profile covariates may have been dropped"
        )
        # Adoption prevalence sane (the DGP bands adoption ~0.05-0.60).
        assert 0.05 <= m["prev"] <= 0.60, (
            f"{brand}: adoption prevalence {m['prev']:.3f} out of band"
        )
        # Sub-0.80 is OK (user-accepted); this floor guards regression of the
        # thinner adoption signal, it does NOT impose the patient 0.80 target.
        assert _HCP_AUC_FLOOR <= m["auc"] <= _HCP_AUC_CEIL, (
            f"{brand}: HCP holdout AUC {m['auc']:.4f} outside accepted "
            f"[{_HCP_AUC_FLOOR}, {_HCP_AUC_CEIL}] (n_holdout={m['n_holdout']})"
        )
