"""Unit tests for EuropePMCClient using httpx.MockTransport."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.europe_pmc import EuropePMCClient, EuropePMCError, reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> EuropePMCClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return EuropePMCClient(client=http)


def _sample_payload(*, abstract: str = "Sample abstract.", title: str = "T") -> dict:
    return {
        "resultList": {
            "result": [
                {
                    "abstractText": abstract,
                    "title": title,
                    "journalTitle": "J. Test",
                    "pubYear": "2024",
                    "id": "12345678",
                    "source": "MED",
                }
            ]
        }
    }


def test_fetch_abstract_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/europepmc/webservices/rest/search" in request.url.path
        params = request.url.params
        assert "ext_id:12345678" in (params.get("query") or "")
        assert params.get("format") == "json"
        assert params.get("resultType") == "core"
        return httpx.Response(200, json=_sample_payload(abstract="Atopic dermatitis treats."))

    with _client_with_handler(handler) as client:
        record = client.fetch_abstract("12345678")
        assert record is not None
        assert record.identifier == "12345678"
        assert record.identifier_kind == "pmid"
        assert record.abstract == "Atopic dermatitis treats."
        assert record.source == "europe_pmc"
        assert record.year == 2024
        assert record.journal == "J. Test"


def test_fetch_abstract_returns_none_when_no_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resultList": {"result": []}})

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("99999") is None


def test_fetch_abstract_returns_none_when_abstract_missing() -> None:
    """A search hit with no abstractText must collapse to None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_payload(abstract=""))

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("12345678") is None


def test_fetch_abstract_empty_pmid_returns_none_without_request() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_sample_payload())

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("") is None
        assert call_count["n"] == 0


def test_fetch_abstract_5xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    with _client_with_handler(handler) as client:
        with pytest.raises(EuropePMCError) as exc:
            client.fetch_abstract("12345678")
        assert "503" in str(exc.value)


def test_fetch_abstract_non_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with _client_with_handler(handler) as client:
        with pytest.raises(EuropePMCError) as exc:
            client.fetch_abstract("12345678")
        assert "non-JSON" in str(exc.value)


def test_fetch_abstract_caches() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_sample_payload())

    with _client_with_handler(handler) as client:
        client.fetch_abstract("12345678")
        client.fetch_abstract("12345678")
        assert call_count["n"] == 1


def test_fetch_abstract_handles_non_dict_record_gracefully() -> None:
    """If the result is malformed (e.g., a list of strings), return None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resultList": {"result": ["not a dict"]}})

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("12345678") is None


def test_fetch_abstract_year_falls_back_when_pubyear_invalid() -> None:
    """Non-digit pubYear → record.year is None, not a crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "abstractText": "x",
                            "title": "y",
                            "pubYear": "in press",
                            # Defensive validation requires the pmid + source
                            # to match the request.
                            "pmid": "12345678",
                            "source": "MED",
                        }
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        record = client.fetch_abstract("12345678")
        assert record is not None
        assert record.year is None


def test_fetch_abstract_rejects_record_with_mismatched_pmid() -> None:
    """Codex review MEDIUM (2026-05-08): defensive validation. If Europe PMC
    ever returns a record whose pmid doesn't match the request, reject it
    rather than silently verifying the wrong abstract."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "abstractText": "Wrong record!",
                            "pmid": "99999999",  # Not what we asked for.
                            "source": "MED",
                        }
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("12345678") is None


def test_fetch_abstract_rejects_record_with_non_med_source() -> None:
    """Records from other Europe PMC sources (e.g., AGR, PMC) shouldn't be
    treated as PubMed equivalents."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "abstractText": "Wrong source!",
                            "pmid": "12345678",
                            "source": "AGR",
                        }
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.fetch_abstract("12345678") is None


def test_fetch_abstract_accepts_record_with_id_field_when_pmid_absent() -> None:
    """Some records use ``id`` instead of ``pmid``; the validation should
    fall back to that field."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "abstractText": "Right abstract!",
                            "id": "12345678",  # No pmid field; use id.
                            "source": "MED",
                        }
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        record = client.fetch_abstract("12345678")
        assert record is not None
        assert record.abstract == "Right abstract!"
