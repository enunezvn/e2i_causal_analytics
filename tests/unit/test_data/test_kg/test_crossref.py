"""Unit tests for CrossrefClient using httpx.MockTransport."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.crossref import CrossrefClient, CrossrefError, reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> CrossrefClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return CrossrefClient(client=http, contact_email="test@example.com")


def _sample_message(*, abstract: str = "<jats:p>Atopic dermatitis treats.</jats:p>") -> dict:
    return {
        "message": {
            "abstract": abstract,
            "title": ["A Test Article"],
            "container-title": ["J. Test"],
            "issued": {"date-parts": [[2024, 5, 8]]},
            "DOI": "10.1234/abc.2024.001",
        }
    }


def test_fetch_doi_metadata_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/works/10.1234/abc.2024.001" in request.url.path
        # Polite-pool User-Agent must include mailto:.
        ua = request.headers.get("User-Agent", "")
        assert "mailto:test@example.com" in ua
        return httpx.Response(200, json=_sample_message())

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc.2024.001")
        assert record is not None
        assert record.identifier == "10.1234/abc.2024.001"
        assert record.identifier_kind == "doi"
        # JATS tags must be stripped from the abstract.
        assert "jats:" not in record.abstract
        assert "Atopic dermatitis treats." in record.abstract
        assert record.title == "A Test Article"
        assert record.journal == "J. Test"
        assert record.year == 2024


def test_fetch_doi_metadata_strips_nested_jats_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sample_message(abstract="<jats:p><jats:italic>X</jats:italic> reduces Y.</jats:p>")
        return httpx.Response(200, json=body)

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc")
        assert record is not None
        assert record.abstract.strip() == "X reduces Y."


def test_fetch_doi_metadata_returns_none_on_404() -> None:
    """Crossref 404 is the standard "DOI not found" path; degrade to None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    with _client_with_handler(handler) as client:
        assert client.fetch_doi_metadata("10.0000/missing") is None


def test_fetch_doi_metadata_returns_none_when_abstract_missing() -> None:
    """Many publishers don't deposit abstracts; degrade to None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "title": ["No abstract"],
                    "issued": {"date-parts": [[2020]]},
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.fetch_doi_metadata("10.0000/no-abstract") is None


def test_fetch_doi_metadata_empty_doi_returns_none_without_request() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_sample_message())

    with _client_with_handler(handler) as client:
        assert client.fetch_doi_metadata("") is None
        assert call_count["n"] == 0


def test_fetch_doi_metadata_5xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with _client_with_handler(handler) as client:
        with pytest.raises(CrossrefError) as exc:
            client.fetch_doi_metadata("10.1234/abc")
        assert "502" in str(exc.value)


def test_fetch_doi_metadata_caches() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_sample_message())

    with _client_with_handler(handler) as client:
        client.fetch_doi_metadata("10.1234/abc")
        client.fetch_doi_metadata("10.1234/abc")
        assert call_count["n"] == 1


def test_fetch_doi_metadata_year_falls_back_to_published_print() -> None:
    """When ``issued`` is missing, fall back to ``published-print``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "abstract": "<jats:p>X</jats:p>",
                    "published-print": {"date-parts": [[2018]]},
                }
            },
        )

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc")
        assert record is not None
        assert record.year == 2018


def test_fetch_doi_metadata_handles_malformed_message_gracefully() -> None:
    """If 'message' is missing or non-dict, return None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weird": "no message"})

    with _client_with_handler(handler) as client:
        assert client.fetch_doi_metadata("10.1234/abc") is None


def test_fetch_doi_metadata_url_encodes_reserved_characters() -> None:
    """DOIs containing ``?``, ``#``, ``%``, spaces must be URL-encoded so
    httpx doesn't reinterpret them as query/fragment delimiters.

    Asserts against ``raw_path`` (bytes form) — ``url.path`` decodes
    percent-escapes back for display.
    """

    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["raw_path"] = request.url.raw_path
        return httpx.Response(200, json=_sample_message())

    with _client_with_handler(handler) as client:
        client.fetch_doi_metadata("10.1234/abc?weird#frag")
    raw = captured["raw_path"]
    # ``?`` and ``#`` must have been percent-encoded; ``/`` preserved.
    assert b"%3F" in raw or b"%3f" in raw
    assert b"%23" in raw
    assert b"/works/10.1234/abc" in raw


def test_fetch_doi_metadata_preserves_doi_slashes() -> None:
    """The forward slash separating registrant from suffix must NOT be
    encoded — Crossref expects the canonical DOI shape."""

    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["raw_path"] = request.url.raw_path
        return httpx.Response(200, json=_sample_message())

    with _client_with_handler(handler) as client:
        client.fetch_doi_metadata("10.1234/abc.2024.001")
    raw = captured["raw_path"]
    assert b"/works/10.1234/abc.2024.001" in raw
    assert b"%2F" not in raw.upper()
