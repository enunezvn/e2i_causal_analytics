"""Factory wiring + live registry adapter for prediction_synthesizer (#840).

The factory must inject BOTH a ``model_registry`` (so the orchestrator can
resolve deployable champions for a target) and ``model_clients`` (loaded from
the deployment manifest). Previously only ``model_clients`` was passed, so even
a populated registry would never be queried.

The registry adapter (``LiveChampionModelRegistry``) is constructed
synchronously in the (sync) factory but acquires the async Supabase client
lazily at query time — inside the agent's async context — and FAILS CLOSED
(returns ``[]``) when no client is available, rather than no-op'ing at
construction (the #845 client-less trap) or fabricating model names.
"""

from __future__ import annotations

from typing import List

import pytest


@pytest.mark.asyncio
async def test_live_registry_fails_closed_when_no_async_client(monkeypatch):
    """No Supabase client -> get_models_for_target returns [] (no fabrication)."""
    import src.agents.prediction_synthesizer.registry_adapter as ra

    async def _no_client():
        return None

    monkeypatch.setattr(ra, "get_async_supabase_client", _no_client)
    registry = ra.LiveChampionModelRegistry()
    assert await registry.get_models_for_target("csu_treatment_initiation", "hcp") == []


@pytest.mark.asyncio
async def test_live_registry_delegates_to_repo():
    """Delegates to the underlying repo's get_models_for_target."""
    from src.agents.prediction_synthesizer.registry_adapter import LiveChampionModelRegistry

    class _FakeRepo:
        async def get_models_for_target(self, target: str, entity_type: str = "") -> List[str]:
            assert target == "csu_treatment_initiation"
            return ["csu_model_a", "csu_model_b"]

    registry = LiveChampionModelRegistry(repo=_FakeRepo())
    names = await registry.get_models_for_target("csu_treatment_initiation", "hcp")
    assert names == ["csu_model_a", "csu_model_b"]


@pytest.mark.asyncio
async def test_live_registry_resolves_client_once(monkeypatch):
    """The async client is acquired lazily and only once (cached)."""
    import src.agents.prediction_synthesizer.registry_adapter as ra

    calls = {"n": 0}

    class _Client:
        def table(self, *_a, **_k):  # pragma: no cover - not exercised here
            raise AssertionError("not used in this test")

    async def _get_client():
        calls["n"] += 1
        return _Client()

    monkeypatch.setattr(ra, "get_async_supabase_client", _get_client)

    captured = {}

    class _Repo:
        def __init__(self, supabase_client=None):
            captured["client"] = supabase_client

        async def get_models_for_target(self, target, entity_type=""):
            return []

    monkeypatch.setattr(ra, "MLModelRegistryRepository", _Repo)

    registry = ra.LiveChampionModelRegistry()
    await registry.get_models_for_target("t", "hcp")
    await registry.get_models_for_target("t", "hcp")
    assert calls["n"] == 1, "async client should be acquired once, then cached"
    assert isinstance(captured["client"], _Client)


def test_factory_injects_registry_and_clients(monkeypatch, tmp_path):
    """The factory builds prediction_synthesizer kwargs with a live registry
    AND model_clients (empty when no manifest -> agent fails closed honestly)."""
    import src.agents.factory as factory
    from src.agents.prediction_synthesizer.registry_adapter import LiveChampionModelRegistry

    # point manifest at a nonexistent path -> no clients (fail-closed substrate)
    monkeypatch.setenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", str(tmp_path / "absent.json"))

    kwargs = factory._prediction_synthesizer_kwargs()
    assert isinstance(kwargs["model_registry"], LiveChampionModelRegistry)
    assert kwargs["model_clients"] == {}
