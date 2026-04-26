"""Block 5B (#14): integration round-trip for tier-0 Feast auto-register.

Block 5 wired ``FeatureAnalyzerAgent._auto_register_in_feast`` ->
``FeastClient.register_feature_view`` -> ``FeatureStore.apply([fv])`` end-
to-end, but the helper short-circuits with ``skipped_reason``
when ``feast_registration_config`` isn't passed in ``input_data``. The
existing unit tests cover the wiring against a mocked client; this
integration test exercises the **live** Feast registry to catch wiring
mistakes that mocks cannot reproduce (entity name mismatch, missing
batch source, schema-builder regressions).

Skip behaviour
--------------
This test mirrors ``tests/integration/test_feast_offline_online_parity.py``
exactly:

* Requires the ``feast`` Python SDK (``pytest.importorskip``).
* Requires opting in via ``FEAST_INTEGRATION=1``.
* Requires ``feature_repo/`` on disk and a registry with the ``hcp``
  entity + ``business_metrics_source`` batch source already applied.

Fail-vs-skip discipline
-----------------------
Once a developer has opted in via ``FEAST_INTEGRATION=1``, this module
distinguishes two failure modes:

* **Environment / connection problems** (Feast SDK missing, repo path
  not present, registry not bootstrapped, entity / batch source
  not yet applied) → ``skip``. These are caller-environment issues; the
  opt-in does not assert the registry is up.
* **Registration failure with a connected Feast** (``register_feature_view``
  returns ``{registered: False, error: ...}``, or the round-trip
  ``get_feature_view`` fails after ``registered: True``) → ``fail``.
  We connected, the wiring is wrong, and that is exactly what this test
  is supposed to catch.

Cleanup
-------
Tests apply a uniquely-named FeatureView (UUID-suffixed) and remove it
again via ``store.apply([], objects_to_delete=[fv], partial=True)`` in a
``try/finally`` so the registry doesn't accumulate test FVs across
re-runs. Cleanup failure → low-severity warning, not a test failure;
Feast 0.43.0 ``apply()`` is idempotent (upsert), so a missed cleanup
just gets re-applied on the next run.

Findings reference: Block 5B (#14, auto-register integration round-trip).
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from pathlib import Path
from typing import Any

import pytest

from tests.integration._feast_helpers import feast_integration_available

# Skip the entire module if the Feast Python SDK is not importable.
pytest.importorskip("feast", reason="Feast SDK not installed; skipping auto-register tests.")

# Loading the registry + applying a FeatureView is heavy (Pydantic + dask).
# The 30s default pytest timeout is too tight; mirror the parity test's
# 180s ceiling so the entire module shares a consistent budget.
pytestmark = pytest.mark.timeout(180)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"

# Live-confirmed registry names (Block 5B preflight Q1 — see Tier-0 plan
# for the FeatureStore.list_data_sources() / list_entities() probe). If
# Block 6B's canonical-schema migration renames these, this test must be
# updated alongside the registry change.
ENTITY_NAME = "hcp"
BATCH_SOURCE_NAME = "business_metrics_source"

# The smallest feature set that exercises the schema-builder / PushSource
# round-trip without depending on any real downstream consumers.
SELECTED_FEATURES = ["trx_count", "nrx_count"]


@pytest.fixture(scope="module")
def feature_store() -> Any:
    """Construct a Feast ``FeatureStore`` rooted at ``feature_repo/``.

    Module-scoped so we pay the registry-load cost once. Skips when:

    * Caller hasn't opted in (``FEAST_INTEGRATION`` unset).
    * ``feature_repo/`` is missing on disk.
    * Registry init raises.
    * Either of the required pre-existing registry objects (the ``hcp``
      entity, the ``business_metrics_source`` batch source) isn't applied.
      This last one is a **prerequisite check**, not a test failure: the
      auto-register helper depends on those being there, and the user
      can fix it with ``feast apply``.
    """
    if not feast_integration_available():
        pytest.skip(
            "FEAST_INTEGRATION not set; skipping auto-register integration. "
            "Set FEAST_INTEGRATION=1 on a host with a bootstrapped Feast registry."
        )

    from feast import FeatureStore

    if not FEATURE_REPO.exists():
        pytest.skip(f"feature_repo not found at {FEATURE_REPO}")

    try:
        store = FeatureStore(repo_path=str(FEATURE_REPO))
    except Exception as exc:  # noqa: BLE001 — skip on ANY init failure
        pytest.skip(f"FeatureStore init failed: {exc!s:.200}")

    # Pre-existing registry objects MUST be there. This is the only thing
    # we check before declaring the environment fit-for-purpose.
    try:
        store.get_entity(ENTITY_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"entity {ENTITY_NAME!r} not registered in Feast: {exc!s:.200}. "
            "Run `feast apply` from feature_repo/ first."
        )

    try:
        store.get_data_source(BATCH_SOURCE_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"batch source {BATCH_SOURCE_NAME!r} not registered in Feast: "
            f"{exc!s:.200}. Run `feast apply` from feature_repo/ first."
        )

    return store


def test_auto_register_round_trip(feature_store: Any) -> None:
    """End-to-end: ``_auto_register_in_feast`` -> ``register_feature_view``
    -> ``store.apply`` writes a FeatureView the registry can return.

    Strategy (mirrors the unit test in
    ``tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/test_feature_analyzer_agent.py``,
    but talks to a real Feast registry):

      1. Build a unique experiment_id (UUID-suffixed) so re-runs don't
         collide on FeatureView name.
      2. Construct a fake ``final_state`` with two surviving features.
      3. Construct ``input_data`` with the live-confirmed entity +
         batch source names.
      4. Call ``agent._auto_register_in_feast(state, input_data)``
         directly (not the full ``agent.run()`` path — the LangGraph
         orchestration is not what this test is here to assert).
      5. Assert the helper reports ``registered=True`` with the
         experiment-scoped name.
      6. Round-trip through ``store.get_feature_view`` and assert the
         registry actually returned the new FV with the expected
         entity, feature set, and ``experiment_id`` tag.
      7. Cleanup via ``apply([], objects_to_delete=[fv], partial=True)``
         in a ``try/finally`` so the registry doesn't accumulate test FVs.
    """
    from src.agents.ml_foundation.feature_analyzer.agent import (
        FeatureAnalyzerAgent,
    )

    experiment_id = f"5btest_{uuid.uuid4().hex[:8]}"
    fv_name = f"tier0_{experiment_id}_features"

    agent = FeatureAnalyzerAgent()
    final_state: dict[str, Any] = {
        "experiment_id": experiment_id,
        "selected_features": list(SELECTED_FEATURES),
    }
    input_data: dict[str, Any] = {
        "feast_registration_config": {
            "entity_name": ENTITY_NAME,
            "batch_source_name": BATCH_SOURCE_NAME,
        },
    }

    try:
        # ---- Auto-register call -----------------------------------------
        result = asyncio.run(
            agent._auto_register_in_feast(final_state, input_data=input_data)
        )

        # `registered=False` here means we connected to Feast but the
        # auto-register wiring is wrong — fail-loud, that's exactly
        # what this integration test is supposed to catch (per the
        # module docstring's fail-vs-skip discipline).
        if not result.get("registered"):
            pytest.fail(
                f"_auto_register_in_feast returned registered=False; "
                f"error={result.get('error')!r} "
                f"skipped_reason={result.get('skipped_reason')!r} "
                f"full_result={result!r}"
            )
        assert result["feature_view_name"] == fv_name
        assert result["features_count"] == len(SELECTED_FEATURES)
        assert result["error"] is None

        # ---- Registry round-trip ---------------------------------------
        # ``get_feature_view`` is the clearer failure mode — list_feature_views
        # would let a missing-FV-amongst-many slip into a confusing diff;
        # ``get_feature_view`` raises with a precise name (Q8 resolution).
        try:
            fv_obj = feature_store.get_feature_view(fv_name)
        except Exception as exc:  # noqa: BLE001
            # Registry round-trip failed AFTER registered=True →
            # this is exactly the bug class this test exists to catch.
            pytest.fail(
                f"FeatureView {fv_name!r} reported registered=True but "
                f"get_feature_view raised: {exc!s:.500}"
            )

        # Entity round-trip. Feast 0.43.0 stores entities-by-reference on
        # the FV; the join_key is the canonical identity.
        feature_view_entity_names = list(getattr(fv_obj, "entities", []) or [])
        feature_view_join_keys = list(getattr(fv_obj, "join_keys", []) or [])
        assert (
            ENTITY_NAME in feature_view_entity_names
            or "hcp_id" in feature_view_join_keys
        ), (
            f"FeatureView {fv_name!r} did not bind to entity {ENTITY_NAME!r}; "
            f"got entities={feature_view_entity_names!r} "
            f"join_keys={feature_view_join_keys!r}"
        )

        # Feature schema round-trip.
        registered_feature_names = sorted(f.name for f in fv_obj.features)
        assert registered_feature_names == sorted(SELECTED_FEATURES), (
            f"FeatureView {fv_name!r} feature names diverged from input: "
            f"got {registered_feature_names!r} "
            f"expected {sorted(SELECTED_FEATURES)!r}"
        )

        # experiment_id tag round-trip (Q7 resolution: assert ONLY this
        # tag, not the other auto-injected ones — they're documented in
        # FeastClient.register_feature_view but not load-bearing).
        assert fv_obj.tags.get("experiment_id") == experiment_id, (
            f"experiment_id tag drift: registered tag="
            f"{fv_obj.tags.get('experiment_id')!r} expected={experiment_id!r}"
        )

    finally:
        # ---- Cleanup ----------------------------------------------------
        # Reconstruct a minimal FV-shaped object for the delete call.
        # ``apply([], objects_to_delete=[fv], partial=True)`` is canonical
        # for Feast 0.43.0 (Q3 resolution). Failures here are
        # low-severity: Feast 0.43.0 ``apply()`` is idempotent (upsert),
        # so a missed cleanup just gets re-applied on the next run.
        try:
            asyncio.run(
                _delete_feature_view_async(feature_store, fv_name)
            )
        except Exception as cleanup_exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to clean up auto-registered FeatureView {fv_name!r}: "
                f"{cleanup_exc!s:.200}. Feast 0.43.0 apply() is idempotent so "
                f"this does not break subsequent runs, but the registry will "
                f"accumulate test FVs until manually pruned.",
                stacklevel=2,
            )


async def _delete_feature_view_async(store: Any, fv_name: str) -> None:
    """Async cleanup helper.

    Looks up the FV by name (so we use the registry-resolved entity object,
    not a fresh Entity stub that might not match the canonical reference)
    and asks Feast to delete it via ``apply([], objects_to_delete=[fv],
    partial=True)``. Wrapped in ``asyncio.to_thread`` because
    ``FeatureStore.apply`` is a blocking call.
    """
    fv = store.get_feature_view(fv_name)
    await asyncio.to_thread(
        store.apply, [], objects_to_delete=[fv], partial=True
    )
