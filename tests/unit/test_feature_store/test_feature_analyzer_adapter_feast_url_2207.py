"""#2207 follow-up (owner decision 2026-09-22): the FeatureAnalyzerAdapter obtains its
Feast client through ``get_feast_client()`` so ``FEAST_URL`` (the e2i_feast sidecar,
#532) is honoured. A bare ``FeastClient()`` was embedded mode, which needs
``import feast`` — impossible on the app/worker image (#307) — so the adapter reported
"Feast not available" on every worker run (measured 2026-09-22).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.feature_store import feast_client as feast_client_module
from src.feature_store.feature_analyzer_adapter import FeatureAnalyzerAdapter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_client_is_remote_when_feast_url_is_set(monkeypatch):
    monkeypatch.setenv("FEAST_URL", "http://feast:6566")
    monkeypatch.setattr(feast_client_module, "_client", None)
    adapter = FeatureAnalyzerAdapter(MagicMock(), enable_feast=True)
    try:
        assert await adapter._ensure_feast_initialized() is True
        assert adapter._feast_client is not None
        assert adapter._feast_client._remote_base_url == "http://feast:6566"
        assert adapter._feast_client._store is None  # no embedded store, no feast import
    finally:
        monkeypatch.setattr(feast_client_module, "_client", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_uses_the_process_singleton(monkeypatch):
    monkeypatch.setenv("FEAST_URL", "http://feast:6566")
    monkeypatch.setattr(feast_client_module, "_client", None)
    try:
        a = FeatureAnalyzerAdapter(MagicMock(), enable_feast=True)
        b = FeatureAnalyzerAdapter(MagicMock(), enable_feast=True)
        await a._ensure_feast_initialized()
        await b._ensure_feast_initialized()
        assert a._feast_client is b._feast_client is feast_client_module._client
    finally:
        monkeypatch.setattr(feast_client_module, "_client", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_injected_client_is_still_respected():
    injected = MagicMock()

    async def _init():
        return None

    injected.initialize = _init
    adapter = FeatureAnalyzerAdapter(MagicMock(), feast_client=injected)
    assert await adapter._ensure_feast_initialized() is True
    assert adapter._feast_client is injected
