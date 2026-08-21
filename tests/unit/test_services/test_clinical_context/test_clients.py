"""Unit tests for the ClinicalTrials.gov v2 + PubMed E-utilities REST clients
via httpx.MockTransport (no live HTTP). Pins the response shapes verified live
2026-06-19."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    ClinicalTrialsError,
    CTGovEndpoint,
    PubMedArticle,
    PubMedClient,
    PubMedError,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _ctgov(handler: Callable[[httpx.Request], httpx.Response]) -> ClinicalTrialsClient:
    return ClinicalTrialsClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _pubmed(handler: Callable[[httpx.Request], httpx.Response]) -> PubMedClient:
    return PubMedClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_clinical_trials_primary_endpoints_dedup_and_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/studies" in request.url.path
        q = request.url.query.decode("utf-8")
        assert "query.intr" in q and "ribociclib" in q
        return httpx.Response(
            200,
            json={
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT01"},
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {"measure": "Overall Survival (OS)"},
                                    {"measure": "Progression-Free Survival (PFS)"},
                                ]
                            },
                        }
                    },
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT02"},
                            "outcomesModule": {
                                "primaryOutcomes": [{"measure": "Overall Survival (OS)"}]
                            },
                        }
                    },
                ]
            },
        )

    with _ctgov(handler) as client:
        eps = client.primary_endpoints("ribociclib", "breast cancer", limit=5)
    # Deduped, first-seen order preserved (compared on the measure text).
    assert [e.measure for e in eps] == [
        "Overall Survival (OS)",
        "Progression-Free Survival (PFS)",
    ]
    # First-seen study's NCT id rides along.
    assert eps[0].nct_id == "NCT01"


def test_clinical_trials_primary_endpoints_carry_time_frame_and_nct_id() -> None:
    """Faithful replay of the live CT.gov v2 shape (verified live 2026-07-14 for
    remibrutinib / CSU): each primary outcome carries a ``timeFrame`` and the study
    carries its ``nctId`` under ``identificationModule`` — both must ride through as
    structured ``CTGovEndpoint`` fields, not be discarded."""

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.query.decode("utf-8")
        # The client must now REQUEST the time-frame field.
        assert "PrimaryOutcomeTimeFrame" in q
        return httpx.Response(
            200,
            json={
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT05030311"},
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {
                                        "measure": (
                                            "Change From Baseline in Weekly Urticaria "
                                            "Score (UAS7) at Week 12 (Scenario 1 With "
                                            "UAS7 as Primary Efficacy Endpoint)"
                                        ),
                                        "timeFrame": "Baseline, Week 12",
                                    }
                                ]
                            },
                        }
                    }
                ]
            },
        )

    with _ctgov(handler) as client:
        eps = client.primary_endpoints("remibrutinib", "chronic spontaneous urticaria", limit=5)
    assert len(eps) == 1
    assert isinstance(eps[0], CTGovEndpoint)
    assert eps[0].measure.startswith("Change From Baseline in Weekly Urticaria Score (UAS7)")
    assert eps[0].time_frame == "Baseline, Week 12"
    assert eps[0].nct_id == "NCT05030311"


def test_clinical_trials_malformed_study_skipped_not_whole_batch() -> None:
    """A malformed study (truthy non-dict protocolSection / nested module) must skip
    only THAT study, not raise and drop the whole batch to the curated fallback. The
    nested isinstance guards keep the well-formed studies' endpoints instead of losing
    an entire live response because one study in it was malformed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "studies": [
                    # protocolSection is a non-dict (truthy string) -> skip study.
                    {"protocolSection": "bad"},
                    # outcomesModule is a non-dict -> skip study.
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT_SKIP"},
                            "outcomesModule": "bad",
                        }
                    },
                    # identificationModule is a non-dict -> measure still surfaces,
                    # nct_id degrades to None (not a crash).
                    {
                        "protocolSection": {
                            "identificationModule": "bad",
                            "outcomesModule": {
                                "primaryOutcomes": [{"measure": "Duration of Response"}]
                            },
                        }
                    },
                    # Fully well-formed -> both fields ride through.
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT_GOOD"},
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {
                                        "measure": "Overall Survival (OS)",
                                        "timeFrame": "Up to 5 years",
                                    }
                                ]
                            },
                        }
                    },
                ]
            },
        )

    with _ctgov(handler) as client:
        eps = client.primary_endpoints("ribociclib", "breast cancer", limit=5)
    # Malformed studies skipped; well-formed endpoints survive, first-seen order kept.
    assert [e.measure for e in eps] == ["Duration of Response", "Overall Survival (OS)"]
    # Bad identificationModule -> measure kept, nct_id None (graceful, not dropped).
    assert eps[0].nct_id is None
    assert eps[1].nct_id == "NCT_GOOD"
    assert eps[1].time_frame == "Up to 5 years"


def test_clinical_trials_empty_inputs_skip_network() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"studies": []})

    with _ctgov(handler) as client:
        assert client.primary_endpoints("", "breast cancer") == []
        assert client.primary_endpoints("ribociclib", "") == []
    assert calls["n"] == 0


def test_clinical_trials_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _ctgov(handler) as client:
        with pytest.raises(ClinicalTrialsError):
            client.primary_endpoints("ribociclib", "breast cancer")


def test_pubmed_top_article_resolves_title_and_doi() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "esearch.fcgi" in path:
            assert "ribociclib" in request.url.query.decode("utf-8").lower()
            return httpx.Response(200, json={"esearchresult": {"idlist": ["35642282"]}})
        assert "esummary.fcgi" in path
        assert "35642282" in request.url.query.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["35642282"],
                    "35642282": {
                        "uid": "35642282",
                        "title": "CDK4/6 inhibitor treatment use in women treated for advanced breast cancer.",
                        "source": "J Oncol Pharm Pract",
                        "pubdate": "2023 Jul",
                        "articleids": [
                            {"idtype": "pubmed", "value": "35642282"},
                            {"idtype": "doi", "value": "10.1177/10781552221102884"},
                        ],
                    },
                }
            },
        )

    with _pubmed(handler) as client:
        art = client.top_article("ribociclib persistence adherence")
    assert isinstance(art, PubMedArticle)
    assert art.pmid == "35642282"
    assert "CDK4/6" in art.title
    assert art.journal == "J Oncol Pharm Pract"
    assert art.doi == "10.1177/10781552221102884"
    assert art.url == "https://pubmed.ncbi.nlm.nih.gov/35642282/"


def test_pubmed_no_hits_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "esearch.fcgi" in request.url.path
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    with _pubmed(handler) as client:
        assert client.top_article("no-such-topic-xyz") is None


def test_pubmed_fetch_by_pmid_resolves_seed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "esummary.fcgi" in request.url.path
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["35642282"],
                    "35642282": {
                        "uid": "35642282",
                        "title": "Seed title",
                        "source": "J Oncol Pharm Pract",
                        "pubdate": "2023 Jul",
                        "articleids": [{"idtype": "doi", "value": "10.1/x"}],
                    },
                }
            },
        )

    with _pubmed(handler) as client:
        art = client.fetch_by_pmid("35642282")
    assert art is not None and art.pmid == "35642282" and art.title == "Seed title"


def test_pubmed_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with _pubmed(handler) as client:
        with pytest.raises(PubMedError):
            client.top_article("ribociclib")


# ---------------------------------------------------------------------------
# HTTP-429 retry (#1612)
#
# NCBI E-utilities allows 3 requests/second without an API key. Measured
# 2026-08-14: 8 rapid esearch calls returned [200, 200, 200, 429, 429, 429,
# 429, 200] — 4 of 8 throttled. PubMedClient had no retry at all, so a burst
# surfaced as PubMedError -> CitationFragment(source="unavailable"), cached as a
# degraded fan-out. ChEMBLClient already retries 429 (chembl.py:_get); these
# tests hold PubMed to the same in-repo standard.
#
# MockTransport is used here only because a 429 cannot be produced on demand
# from the real endpoint; the *live* counterpart in
# tests/integration/test_clinical_context/ proves the retry against real NCBI
# throttling.
# ---------------------------------------------------------------------------


def test_pubmed_retries_on_429_then_succeeds() -> None:
    """A throttled esearch is retried rather than raised straight through."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in request.url.path:
            calls.append("esearch")
            if len(calls) == 1:
                return httpx.Response(429, text="rate limit")
            return httpx.Response(200, json={"esearchresult": {"idlist": ["38507751"]}})
        return httpx.Response(
            200,
            json={
                "result": {
                    "38507751": {
                        "title": "Ribociclib plus Endocrine Therapy in Early Breast Cancer.",
                        "source": "N Engl J Med",
                        "pubdate": "2024 Mar 21",
                        "articleids": [{"idtype": "doi", "value": "10.1056/NEJMoa2305488"}],
                    }
                }
            },
        )

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
        retry_backoff_s=0.0,
    )
    with client:
        article = client.top_article("ribociclib breast cancer")

    assert calls == ["esearch", "esearch"], "429 was not retried"
    assert article is not None and article.pmid == "38507751"


def test_pubmed_raises_after_retry_budget_exhausted() -> None:
    """Persistent throttling still surfaces as an error, not a silent None.

    The provider layer turns this into ``source="unavailable"``; what must not
    happen is an unbounded retry loop against an API that is refusing us.
    """
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="rate limit")

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
        retry_backoff_s=0.0,
    )
    with client:
        with pytest.raises(PubMedError):
            client.top_article("ribociclib")

    # 1 initial attempt + 2 retries, then give up.
    assert len(calls) == 3, f"expected 3 attempts with max_retries=2, got {len(calls)}"


def test_pubmed_does_not_retry_non_429_errors() -> None:
    """A 503 is not a throttle; retrying it just doubles load on a sick endpoint."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="down")

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=3,
        retry_backoff_s=0.0,
    )
    with client:
        with pytest.raises(PubMedError):
            client.top_article("ribociclib")

    assert len(calls) == 1, "a 503 must not be retried"


# ---------------------------------------------------------------------------
# NCBI api_key authentication (#1628)
#
# NCBI E-utilities allows 3 req/s unauthenticated and 10 req/s with an API key.
# Measured live 2026-08-15, 8 rapid esearch calls, same host and same params,
# only the key differing:
#   no key   -> [200, 200, 200, 429, 429, 429, 429, 429]  (5/8 throttled)
#   with key -> [200, 200, 200, 200, 200, 200, 200, 200]  (0/8)
#
# `NCBI_API_KEY` was present and valid in the droplet `.env`, but nothing in
# `src/` ever read it, so every production PubMed call ran on the 3 req/s tier.
# The key must reach BOTH E-utilities calls (esearch and esummary): they are
# separate HTTP requests and either one can be what trips the throttle.
# ---------------------------------------------------------------------------


def _capture_pubmed_params(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    **client_kwargs: object,
) -> list[dict[str, str]]:
    """Drive one top_article() round-trip, returning each request's params."""
    if env_value is None:
        monkeypatch.delenv("NCBI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("NCBI_API_KEY", env_value)

    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if "esearch" in request.url.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": ["33730455"]}})
        return httpx.Response(
            200,
            json={
                "result": {
                    "33730455": {
                        "title": "A title",
                        "source": "J Test",
                        "pubdate": "2021",
                        "articleids": [],
                    }
                }
            },
        )

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **client_kwargs,  # type: ignore[arg-type]
    )
    with client:
        assert client.top_article("ribociclib breast cancer") is not None
    assert len(seen) == 2, "expected an esearch followed by an esummary"
    return seen


def test_pubmed_sends_api_key_on_every_call_when_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCBI_API_KEY reaches both esearch and esummary."""
    seen = _capture_pubmed_params(monkeypatch, "k" * 36)
    for i, params in enumerate(seen):
        assert params.get("api_key") == "k" * 36, f"request {i} unauthenticated: {params}"


def test_pubmed_omits_api_key_entirely_when_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key -> the parameter is ABSENT, not an empty string.

    ``api_key=`` with an empty value is not the same as omitting it: NCBI
    rejects a malformed key rather than serving the anonymous tier.
    """
    seen = _capture_pubmed_params(monkeypatch, None)
    for i, params in enumerate(seen):
        assert "api_key" not in params, f"request {i} sent an empty api_key: {params}"


def test_pubmed_explicit_api_key_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly-passed key wins over the env var (mirrors _OpenFDAClient)."""
    seen = _capture_pubmed_params(monkeypatch, "env-key", api_key="explicit-key")
    for params in seen:
        assert params.get("api_key") == "explicit-key"


# --- #1763 Phase 2: multi-PMID relevance search --------------------------------


def test_pubmed_search_pmids_returns_the_relevance_ranked_ids() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/esearch.fcgi" in request.url.path
        q = request.url.query.decode("utf-8")
        seen["query"] = q
        return httpx.Response(200, json={"esearchresult": {"idlist": ["11", "22", "33"]}})

    pmids = _pubmed(handler).search_pmids("ribociclib breast cancer persistence", retmax=3)
    assert pmids == ["11", "22", "33"]
    assert "retmax=3" in seen["query"]
    assert "sort=relevance" in seen["query"]


def test_pubmed_search_pmids_empty_term_skips_the_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no HTTP call expected for an empty term")

    assert _pubmed(handler).search_pmids("", retmax=3) == []


def test_pubmed_search_pmids_no_hits_is_an_empty_list_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    assert _pubmed(handler).search_pmids("nothing matches this", retmax=5) == []
