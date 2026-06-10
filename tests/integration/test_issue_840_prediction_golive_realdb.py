"""Faithful end-to-end go-live test for prediction_synthesizer (#840).

Gated on E2I_DB_INTEGRATION=1 (writes real ml_model_registry / ml_experiments
rows against the live Supabase, then cleans up). Proves the acceptance:

    A structured prediction request returns a REAL ensemble prediction
    (status="completed", >=2 models succeeded) via the production factory
    wiring — not status="failed", no UNVALIDATED model_clients={} fallback.

Run:
    E2I_DB_INTEGRATION=1 LOKY_MAX_CPU_COUNT=1 pytest -n0 \
        tests/integration/test_issue_840_prediction_golive_realdb.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("E2I_DB_INTEGRATION"),
    reason="requires E2I_DB_INTEGRATION=1 + live Supabase",
)

# Isolated namespace so this test never clobbers the persistent go-live deploy.
_SUFFIX = "_itest840"
_EXPERIMENT = "csu_treatment_initiation_itest840"


@pytest.mark.asyncio
async def test_chat_prediction_returns_real_ensemble(tmp_path: Path, monkeypatch):
    from src.agents.factory import _prediction_synthesizer_kwargs
    from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.prediction_synthesizer_deploy import deploy

    manifest = tmp_path / "deployment_manifest.json"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", str(manifest))

    # 1. DEPLOY real models (small cohort for speed) + register champion rows.
    result = await deploy(
        n_total=1500,
        seed=11,
        artifact_dir=artifacts,
        manifest_path=manifest,
        register=True,
        name_suffix=_SUFFIX,
        experiment_name=_EXPERIMENT,
    )
    assert len(result["registered"]) >= 2, f"deploy did not register >=2 models: {result}"

    client = await get_async_supabase_client()
    assert client is not None, "live Supabase client required for this faithful test"

    try:
        # 2. Construct the agent via the PRODUCTION factory wiring (real registry
        #    + real manifest-loaded clients). Side features off for determinism.
        kwargs = _prediction_synthesizer_kwargs()
        clients = kwargs["model_clients"]
        assert set(result["registered"]).issubset(set(clients.keys())), (
            "factory did not load the deployed model clients from the manifest"
        )

        agent = PredictionSynthesizerAgent(
            model_registry=kwargs["model_registry"],
            model_clients=clients,
            enable_memory=False,
            enable_opik=False,
            enable_dspy=False,
        )

        # 3. The live registry must resolve the deployed champions for the target.
        resolved = await kwargs["model_registry"].get_models_for_target(
            "csu_treatment_initiation", "hcp"
        )
        for name in result["registered"]:
            assert name in resolved, f"{name} not resolved by live registry: {resolved}"

        # 4. A structured request returns a REAL ensemble (not status=failed).
        feature_names = next(iter(clients.values())).feature_names
        features = dict.fromkeys(feature_names, 1.0)
        out = await agent.synthesize(
            entity_id="HCP_INT_840",
            entity_type="hcp",
            prediction_target="csu_treatment_initiation",
            features=features,
            time_horizon="30d",
            ensemble_method="weighted",
            include_context=False,
        )

        assert out.status == "completed", f"expected completed, got {out.status}"
        assert out.models_succeeded >= 2, f"expected >=2 models, got {out.models_succeeded}"
        assert out.ensemble_prediction is not None
        point = out.ensemble_prediction["point_estimate"]
        assert 0.0 <= float(point) <= 1.0, f"point_estimate out of range: {point}"

        # 5. TARGET-AGNOSTIC GUARD (codex HIGH-2): the SAME csu manifest is
        #    loaded, but a request for an unrelated target (no deployable model)
        #    must FAIL CLOSED via the registry — NOT fabricate a prediction from
        #    the csu clients. This exercises the simple-graph path the chat
        #    entrypoints default to (include_context=False).
        bogus = await agent.synthesize(
            entity_id="HCP_INT_840",
            entity_type="hcp",
            prediction_target="nonexistent_target_840",
            features=features,
            time_horizon="30d",
            ensemble_method="weighted",
            include_context=False,
        )
        assert bogus.status == "failed", (
            f"target-agnostic fabrication: unrelated target returned {bogus.status} "
            "instead of failing closed"
        )
    finally:
        # FK-safe cleanup: registry rows first, then the test experiment.
        for name in result["registered"]:
            await client.table("ml_model_registry").delete().eq("model_name", name).execute()
        await client.table("ml_experiments").delete().eq("experiment_name", _EXPERIMENT).execute()
