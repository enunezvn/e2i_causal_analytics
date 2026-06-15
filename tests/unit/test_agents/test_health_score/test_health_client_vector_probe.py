"""WS-BACKEND (codex r2 RESOLUTION-5): the vector-store probe must not treat a
bare 401 as healthy — that masks an absent/inaccessible table. It disambiguates
401/403 with a second probe of a guaranteed-nonexistent table, matching the
measured live PostgREST semantics (table resolved AFTER JWT validation):

    real 200/206                -> healthy (route resolves)
    real 404                    -> unhealthy (PGRST205, table absent)
    real 401/403 + bogus 404    -> healthy (JWT valid; table exists, RLS-restricted)
    real 401/403 + bogus 401/403-> degraded (auth broken; status unverifiable)
"""

import pytest

from src.agents.health_score.health_client import SupabaseHealthClient

_BOGUS = "__vector_health_probe_absent__"


class _Resp:
    def __init__(self, code: int):
        self.status_code = code


class _FakeHTTP:
    """Returns a canned status per URL substring; records call order."""

    def __init__(self, by_url: dict[str, int]):
        self._by_url = by_url
        self.calls: list[str] = []

    async def get(self, url, headers=None):
        self.calls.append(url)
        # Check the bogus probe first so it is not shadowed by a broad match.
        if _BOGUS in url:
            return _Resp(self._by_url.get(_BOGUS, 599))
        if "rag_document_chunks" in url:
            return _Resp(self._by_url.get("rag_document_chunks", 599))
        return _Resp(599)

    async def aclose(self):
        return None


def _client(by_url: dict[str, int]) -> SupabaseHealthClient:
    c = SupabaseHealthClient(supabase_url="http://supabase.test")
    c._http_client = _FakeHTTP(by_url)  # type: ignore[assignment]
    return c


@pytest.mark.asyncio
async def test_vector_present_200_is_ok():
    c = _client({"rag_document_chunks": 200})
    r = await c._check_vector_store()
    assert r["ok"] is True
    # A definitive 200 needs no second probe.
    assert c._http_client.calls == [  # type: ignore[union-attr]
        "http://supabase.test/rest/v1/rag_document_chunks?limit=1"
    ]


@pytest.mark.asyncio
async def test_vector_absent_404_is_unhealthy():
    c = _client({"rag_document_chunks": 404})
    r = await c._check_vector_store()
    assert r["ok"] is False
    assert r.get("degraded") is True


@pytest.mark.asyncio
async def test_vector_401_but_table_exists_is_ok():
    # The live deployment: anon role 401s the RLS-restricted table, but the bogus
    # table 404s -> JWT valid -> rag_document_chunks exists -> healthy (this is the
    # false-alarm fix: it must stay GREEN).
    c = _client({"rag_document_chunks": 401, _BOGUS: 404})
    r = await c._check_vector_store()
    assert r["ok"] is True
    assert len(c._http_client.calls) == 2  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_vector_401_with_broken_auth_is_degraded_not_masked():
    # Both probes 401 -> the JWT itself is invalid/missing, so the table is
    # unverifiable. Surface degraded instead of falsely reporting healthy.
    c = _client({"rag_document_chunks": 401, _BOGUS: 401})
    r = await c._check_vector_store()
    assert r["ok"] is False
    assert r.get("degraded") is True
    assert "unverifiable" in r.get("error", "")


@pytest.mark.asyncio
async def test_vector_no_url_is_degraded():
    # __init__ falls back to the SUPABASE_URL env var when given "", so force the
    # unconfigured branch directly to keep the test hermetic (no network).
    c = SupabaseHealthClient(supabase_url="http://placeholder")
    c.supabase_url = ""
    r = await c._check_vector_store()
    assert r["ok"] is False
    assert r.get("degraded") is True
