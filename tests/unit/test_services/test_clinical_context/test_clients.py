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
    # Deduped, first-seen order preserved.
    assert eps == ["Overall Survival (OS)", "Progression-Free Survival (PFS)"]


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
