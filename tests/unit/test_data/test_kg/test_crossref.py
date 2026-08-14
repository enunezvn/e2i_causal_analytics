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


def test_fetch_doi_metadata_strips_non_jats_xhtml_and_mathml() -> None:
    """Codex review MEDIUM (2026-05-08): non-JATS markup (XHTML, MathML,
    plain XML) was leaking through the old ``jats:`` -only regex."""
    abstract_raw = (
        "<jats:p>Aspirin <i>inhibits</i> COX-2 via "
        "<mml:math><mml:mi>k</mml:mi></mml:math> in <p>vivo</p>.</jats:p>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_message(abstract=abstract_raw))

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc")
    assert record is not None
    # No tag artifacts at all.
    assert "<" not in record.abstract
    assert ">" not in record.abstract
    assert "Aspirin inhibits COX-2 via k in vivo." in record.abstract


def test_fetch_doi_metadata_unescapes_html_entities() -> None:
    """``&amp;``, ``&lt;``, ``&#x2014;`` (em-dash) etc. must be decoded so
    entity matching can find their natural-text forms."""
    abstract_raw = (
        "<jats:p>"
        "Aspirin &amp; ibuprofen reduce inflammation &mdash; "
        "100&#x25; effective in &lt;some&gt; patients."
        "</jats:p>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_message(abstract=abstract_raw))

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc")
    assert record is not None
    assert "Aspirin & ibuprofen" in record.abstract
    assert "—" in record.abstract  # em-dash decoded
    assert "100%" in record.abstract
    assert "<some>" in record.abstract


def test_fetch_doi_metadata_collapses_whitespace_runs() -> None:
    """After tag removal, multiple newlines/spaces left behind must collapse."""
    abstract_raw = "<jats:p>Aspirin\n\n   <i>treats</i>\n   inflammation.</jats:p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_message(abstract=abstract_raw))

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc")
    assert record is not None
    # Collapsed to single spaces.
    assert record.abstract == "Aspirin treats inflammation."


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


def test_tag_removal_does_not_glue_adjacent_text_nodes() -> None:
    """Structured JATS abstracts must not fuse a section title to the next word.

    #1608: ``_ALL_TAGS.sub("", ...)`` deleted tags without leaving a separator,
    so ``<jats:title>Background</jats:title><jats:p>Breast cancer ...`` became
    ``"BackgroundBreast cancer"``. ``CitationResolver._first_match`` matches on
    WORD BOUNDARIES, so the fused term could never match and the first entity
    after every section heading was invisible to verification — a systematic
    source of false "unverified" verdicts, since JATS abstracts are almost
    always structured (Background / Methods / Results / Conclusions).

    Measured against the live Crossref record for 10.1186/s13058-023-01623-6:
    "breast cancer" did not match before this fix and does after.
    """
    structured = (
        "<jats:title>Abstract</jats:title>"
        "<jats:sec><jats:title>Background</jats:title>"
        "<jats:p>Breast cancer treatments may affect return to work.</jats:p></jats:sec>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_message(abstract=structured))

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc.2024.001")

    assert record is not None
    assert "BackgroundBreast" not in record.abstract, (
        f"tag removal fused adjacent text nodes: {record.abstract!r}"
    )
    assert "Background Breast cancer" in record.abstract
    # Whitespace must still be collapsed — no double spaces from the separator.
    assert "  " not in record.abstract


def test_word_boundary_entity_match_survives_a_structured_abstract() -> None:
    """The end-to-end consequence: the matcher must find the fused term."""
    from src.data.kg.citation_resolver import _first_match

    structured = (
        "<jats:title>Background</jats:title>"
        "<jats:p>Breast cancer treatments may affect outcomes.</jats:p>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_message(abstract=structured))

    with _client_with_handler(handler) as client:
        record = client.fetch_doi_metadata("10.1234/abc.2024.001")

    assert record is not None
    assert _first_match(["breast cancer"], record.abstract.lower()) == "breast cancer"
