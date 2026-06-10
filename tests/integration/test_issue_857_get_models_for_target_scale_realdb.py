"""Faithful real-DB regression test for #857 Gap 1.

``MLModelRegistryRepository.get_models_for_target`` must not ``414 URI too
long`` at prod scale. ``csu_treatment_initiation`` carries hundreds of
experiments (the #850 synthetic load); the old implementation fetched every
experiment id for the target and filtered the registry with
``.in_("experiment_id", [253 uuids])`` — a ~10KB PostgREST GET URL that the
server rejects with 414. This test registers an isolated DEPLOYABLE model under
that high-cardinality target and asserts the resolver returns it WITHOUT
raising, exercising the exact fan-out that 414'd.

Gated on E2I_DB_INTEGRATION=1 (writes real rows, then cleans up). Run:

    E2I_DB_INTEGRATION=1 LOKY_MAX_CPU_COUNT=1 .venv/bin/python -m pytest -n0 \
        tests/integration/test_issue_857_get_models_for_target_scale_realdb.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("E2I_DB_INTEGRATION"),
    reason="requires E2I_DB_INTEGRATION=1 + live Supabase",
)

_TARGET = "csu_treatment_initiation"
_EXPERIMENT = "csu_treatment_initiation_itest857"
_VERSION = "857test"
_DEPLOYABLE = "csu_itest857_deployable"  # production + artifact  -> resolved
_STAGING = "csu_itest857_staging"  # staging               -> excluded
_NOART = "csu_itest857_noart"  # production, null art   -> excluded


@pytest.mark.asyncio
async def test_get_models_for_target_resolves_at_scale_no_414():
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.ml_experiment import MLModelRegistryRepository

    client = await get_async_supabase_client()
    assert client is not None, "live Supabase client required for this faithful test"
    repo = MLModelRegistryRepository(client)

    # Precondition: the target must be high-cardinality enough to actually
    # reproduce the 414 with the OLD per-experiment-id IN-list query — otherwise
    # this test passes vacuously and is not a faithful regression guard.
    # MEASURED on this server: the old ``.in_("experiment_id", [N uuids])`` query
    # first returns 414 at N=209 ids (N<=200 stays under the URI limit). The #850
    # synthetic load put 253 experiments on this target, so the live scale is
    # safely in the 414 zone; require >=250 (above the ~209 boundary, with margin
    # for modest count drift). Below this, the env cannot reproduce the bug.
    exp = await (
        client.table("ml_experiments")
        .select("id", count="exact")
        .eq("prediction_target", _TARGET)
        .execute()
    )
    n_exp = exp.count if exp.count is not None else len(exp.data or [])
    assert n_exp >= 250, (
        "target must carry enough experiments to reproduce the 414 with the old "
        f"query (measured boundary ~209 ids; #850 load = 253), got {n_exp} — "
        "this environment cannot faithfully exercise the #857 scale path"
    )

    created = await (
        client.table("ml_experiments")
        .insert(
            {
                "experiment_name": _EXPERIMENT,
                "prediction_target": _TARGET,
                "brand": "Remibrutinib",
                "is_synthetic": False,
                "created_by": "itest857",
                "description": "scale regression for #857 get_models_for_target",
            }
        )
        .execute()
    )
    experiment_id = created.data[0]["id"]
    now = datetime.now(timezone.utc).isoformat()

    def _row(name: str, stage: str, artifact: str | None) -> dict:
        # Only the deployable row is champion (the tr_single_champion trigger
        # caps champions at one per experiment); is_champion is irrelevant to
        # the resolver but must not trip the trigger.
        return {
            "experiment_id": experiment_id,
            "model_name": name,
            "model_version": _VERSION,
            "algorithm": "logistic_regression",
            "stage": stage,
            "is_champion": name == _DEPLOYABLE,
            "is_synthetic": False,
            "artifact_path": artifact,
            "trained_at": now,
            "registered_at": now,
        }

    try:
        await (
            client.table("ml_model_registry")
            .insert(_row(_DEPLOYABLE, "production", "/tmp/itest857_deployable.pkl"))
            .execute()
        )
        await (
            client.table("ml_model_registry")
            .insert(_row(_STAGING, "staging", "/tmp/itest857_staging.pkl"))
            .execute()
        )
        await client.table("ml_model_registry").insert(_row(_NOART, "production", None)).execute()

        # THE DISPROOF: at 253+ experiments, the old query 414'd here. The fixed
        # FK-embed query must return the deployable model and not raise.
        names = await repo.get_models_for_target(_TARGET, "hcp")

        assert _DEPLOYABLE in names, f"deployable model not resolved at scale: {names}"
        assert _STAGING not in names, "staging model must be excluded (not promoted)"
        assert _NOART not in names, "null-artifact model must be excluded (unloadable)"
    finally:
        for name in (_DEPLOYABLE, _STAGING, _NOART):
            await (
                client.table("ml_model_registry")
                .delete()
                .eq("model_name", name)
                .eq("model_version", _VERSION)
                .execute()
            )
        await client.table("ml_experiments").delete().eq("experiment_name", _EXPERIMENT).execute()
