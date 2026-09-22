"""#2207 follow-up (owner decision 2026-09-22): the worker's materialize beats POST to the
e2i_feast sidecar's ``/materialize`` and ``/materialize-incremental`` (feast 0.43
feature server, request schema read from its ``/openapi.json`` inside worker_medium on
2026-09-22: ``MaterializeRequest{start_ts, end_ts, feature_views?}``,
``MaterializeIncrementalRequest{end_ts, feature_views?}``) instead of failing at
``import feast`` (#307).

The HTTP seam is faked at ``httpx.AsyncClient`` so every test asserts the EXACT path and
JSON body the sidecar will receive; a non-2xx or unreachable sidecar is a ``failed``
result (never an exception the beat would have to guess at).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import pytest

from src.feature_store import feast_remote_materialize as mod
from src.feature_store.feast_client import FeastClient, FeastConfig

START = datetime(2026, 9, 15, 6, 0, tzinfo=timezone.utc)
END = datetime(2026, 9, 22, 6, 0, tzinfo=timezone.utc)


class _FakeResponse:
    def __init__(self, status_code: int, body: Any = None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://feast:6566/x"),
                response=httpx.Response(self.status_code, text=self.text),
            )


class _FakeAsyncClient:
    """Captures POSTs; `responses` maps path -> response (or an exception to raise)."""

    calls: List[Dict[str, Any]] = []
    responses: Dict[str, Any] = {}

    def __init__(self, *_a, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url: str, json: Any = None):
        path = url.split("6566", 1)[1]
        type(self).calls.append({"url": url, "path": path, "json": json, "timeout": self.timeout})
        resp = type(self).responses.get(path, _FakeResponse(200, None))
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture()
def fake_http(monkeypatch):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = {}
    monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


# ---------------------------------------------------------------------------
# request shape mirrors the sidecar's openapi schema exactly
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_materialize_posts_exact_body(fake_http):
    result = await mod.post_materialize(
        "http://feast:6566/",
        start_date=START,
        end_date=END,
        feature_views=["hcp_profile_features"],
        timeout=12.5,
    )
    (call,) = fake_http.calls
    assert call["url"] == "http://feast:6566/materialize"
    assert call["json"] == {
        "start_ts": START.isoformat(),
        "end_ts": END.isoformat(),
        "feature_views": ["hcp_profile_features"],
    }
    assert call["timeout"] == 12.5
    assert result["status"] == "completed"
    assert result["feature_views"] == ["hcp_profile_features"]
    assert result["start_date"] == START.isoformat()
    assert result["end_date"] == END.isoformat()
    assert result["duration_seconds"] >= 0
    assert result["materializer"] == "remote:http://feast:6566"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incremental_materialize_posts_exact_body_and_null_views_means_all(fake_http):
    result = await mod.post_materialize_incremental(
        "http://feast:6566",
        end_date=END,
        feature_views=None,
        timeout=30.0,
        default_feature_views=["a_view", "b_view"],
    )
    (call,) = fake_http.calls
    assert call["url"] == "http://feast:6566/materialize-incremental"
    assert call["json"] == {"end_ts": END.isoformat(), "feature_views": None}
    assert result["status"] == "completed"
    assert result["incremental"] is True
    # None was sent (the sidecar materializes every online view); the result names
    # the views the caller considers targeted so tracking rows can be written.
    assert result["feature_views"] == ["a_view", "b_view"]


# ---------------------------------------------------------------------------
# failure shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_2xx_is_a_failed_result_with_the_body(fake_http):
    fake_http.responses["/materialize-incremental"] = _FakeResponse(
        422, {"detail": "bad"}, text='{"detail":"bad"}'
    )
    result = await mod.post_materialize_incremental(
        "http://feast:6566", end_date=END, feature_views=["x"], timeout=1.0
    )
    assert result["status"] == "failed"
    assert "422" in result["error"]
    assert "bad" in result["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unreachable_sidecar_is_a_failed_result(fake_http):
    fake_http.responses["/materialize"] = httpx.ConnectError("connection refused")
    result = await mod.post_materialize(
        "http://feast:6566", start_date=START, end_date=END, feature_views=None, timeout=1.0
    )
    assert result["status"] == "failed"
    assert "connection refused" in result["error"]
    assert "http://feast:6566/materialize" in result["error"]


# ---------------------------------------------------------------------------
# FeastClient in remote mode delegates (no feast import, no embedded store)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feast_client_remote_mode_materializes_over_http(fake_http):
    client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))
    await client.initialize()
    assert client._store is None  # remote mode never builds an embedded store

    full = await client.materialize(start_date=START, end_date=END, feature_views=["v1"])
    inc = await client.materialize_incremental(end_date=END, feature_views=None)

    assert [c["path"] for c in fake_http.calls] == ["/materialize", "/materialize-incremental"]
    assert full["status"] == "completed" and inc["status"] == "completed"
    assert fake_http.calls[1]["json"] == {"end_ts": END.isoformat(), "feature_views": None}
    # a None target resolves to the ENABLED views of config/feast_materialization.yaml
    assert "hcp_profile_features" in inc["feature_views"]
    assert "market_dynamics_features" not in inc["feature_views"]  # enabled: false (#556)
    # the in-process timestamps are updated for the views the run covered
    assert "v1" in client._materialization_timestamps


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feast_client_without_url_or_store_still_reports_skipped(monkeypatch):
    """Embedded mode with no store is unchanged: ``skipped`` (the #556 fail-loud key)."""
    client = FeastClient(config=FeastConfig(server_url=None))
    client._initialized = True  # bypass embedded init; no store, no remote url
    result = await client.materialize(start_date=START, end_date=END)
    assert result["status"] == "skipped"
