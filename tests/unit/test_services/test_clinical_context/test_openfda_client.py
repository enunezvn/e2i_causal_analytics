"""Unit tests for the OpenFDA drug label REST client via httpx.MockTransport
(no live HTTP). Pins the response shapes from the openFDA API
https://api.fda.gov/drug/label.json verified 2026-06-20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx
import pytest

from src.services.clinical_context.clients import (
    _OpenFDAClient,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _openfda(handler: Callable[[httpx.Request], httpx.Response]) -> _OpenFDAClient:
    return _OpenFDAClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

_KISQALI_CO_PACK = {
    "openfda": {
        "generic_name": ["letrozole and ribociclib"],
        "brand_name": ["KISQALI FEMARA CO-PACK"],
    },
    "indications_and_usage": [
        "1 INDICATIONS AND USAGE KISQALI FEMARA CO-PACK (ribociclib and letrozole) "
        "is a kinase inhibitor and an aromatase inhibitor combination indicated in "
        "combination with an aromatase inhibitor for adult patients..."
    ],
}

_KISQALI_STANDALONE = {
    "openfda": {
        "generic_name": ["ribociclib"],
        "brand_name": ["KISQALI"],
    },
    "indications_and_usage": [
        "1 INDICATIONS AND USAGE KISQALI is a kinase inhibitor indicated: "
        "for the adjuvant treatment of adult patients with hormone receptor (HR)-positive, "
        "human epidermal growth factor receptor 2 (HER2)-negative early breast cancer..."
    ],
}

_REMIBRUTINIB = {
    "openfda": {
        "generic_name": ["remibrutinib"],
        "brand_name": ["RHAPSIDO"],
    },
    "indications_and_usage": [
        "1 INDICATIONS AND USAGE RHAPSIDO (remibrutinib) is a Bruton tyrosine kinase inhibitor "
        "indicated for the treatment of adults with chronic spontaneous urticaria (CSU) who "
        "remain symptomatic despite H1 antihistamine treatment. "
        "Limitations of Use: RHAPSIDO is not indicated for other forms of urticaria."
    ],
}

_IPTACOPAN = {
    "openfda": {
        "generic_name": ["iptacopan"],
        "brand_name": ["FABHALTA"],
    },
    "boxed_warning": ["WARNING: SERIOUS INFECTIONS ..."],
    "indications_and_usage": [
        "1 INDICATIONS AND USAGE FABHALTA (iptacopan) is a complement factor B inhibitor "
        "indicated for the treatment of adults with paroxysmal nocturnal hemoglobinuria (PNH)."
    ],
}


# ---------------------------------------------------------------------------
# fetch_label — single-ingredient preference
# ---------------------------------------------------------------------------


def test_fetch_label_prefers_single_ingredient_match() -> None:
    """ribociclib search returns co-pack first, standalone second.
    fetch_label MUST return the standalone record (generic_name == ["ribociclib"])."""

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.query.decode("utf-8")
        assert "openfda.generic_name" in q
        assert "ribociclib" in q
        return httpx.Response(
            200,
            json={"results": [_KISQALI_CO_PACK, _KISQALI_STANDALONE]},
        )

    with _openfda(handler) as client:
        result = client.fetch_label("ribociclib")

    assert result is not None
    assert result["openfda"]["generic_name"] == ["ribociclib"]
    assert result["openfda"]["brand_name"] == ["KISQALI"]


def test_fetch_label_returns_first_when_no_single_ingredient_match() -> None:
    """If no result has a single-ingredient generic_name, return the first result."""
    multi_a = {
        "openfda": {"generic_name": ["a and b"]},
        "indications_and_usage": ["1 INDICATIONS AND USAGE ..."],
    }
    multi_b = {
        "openfda": {"generic_name": ["a and b and c"]},
        "indications_and_usage": ["1 INDICATIONS AND USAGE ..."],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [multi_a, multi_b]})

    with _openfda(handler) as client:
        result = client.fetch_label("a")

    # First result returned (no single-ingredient match); use == not `is` since
    # the result comes from JSON-decoded response, not the original dict object.
    assert result == multi_a
    assert result is not None


# ---------------------------------------------------------------------------
# fetch_label — 404 / empty / brand fallback
# ---------------------------------------------------------------------------


def test_fetch_label_returns_none_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

    with _openfda(handler) as client:
        assert client.fetch_label("unknowndrug") is None


def test_fetch_label_returns_none_on_empty_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with _openfda(handler) as client:
        assert client.fetch_label("unknowndrug") is None


def test_fetch_label_brand_fallback_when_generic_empty() -> None:
    """When generic_name search returns empty results, retry with brand_name."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.query.decode("utf-8")
        if "openfda.generic_name" in q:
            calls.append("generic")
            return httpx.Response(200, json={"results": []})
        calls.append("brand")
        assert "openfda.brand_name" in q
        return httpx.Response(200, json={"results": [_KISQALI_STANDALONE]})

    with _openfda(handler) as client:
        result = client.fetch_label("KISQALI")

    assert calls == ["generic", "brand"]
    assert result is not None
    assert result["openfda"]["generic_name"] == ["ribociclib"]


def test_fetch_label_brand_fallback_not_triggered_on_404() -> None:
    """On a 404 from generic search, skip the brand retry and return None."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.query.decode("utf-8")
        calls.append("generic" if "openfda.generic_name" in q else "brand")
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

    with _openfda(handler) as client:
        result = client.fetch_label("unknowndrug")

    # 404 on generic → no results (not "empty"), no brand retry
    assert result is None
    assert "brand" not in calls


def test_fetch_label_returns_none_on_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _openfda(handler) as client:
        assert client.fetch_label("ribociclib") is None


# ---------------------------------------------------------------------------
# Static helpers — approved_indications
# ---------------------------------------------------------------------------


def test_approved_indications_strips_header_and_excludes_lou() -> None:
    indications = _OpenFDAClient.approved_indications(_REMIBRUTINIB)

    # Must not be empty
    assert indications

    full_text = " ".join(indications)
    # Must not contain the section header
    assert "1 INDICATIONS AND USAGE" not in full_text
    # Must not contain the Limitations of Use sentence
    assert "Limitations of Use" not in full_text
    # Must contain the main indication text
    assert "chronic spontaneous urticaria" in full_text


def test_approved_indications_no_lou_returns_full_body() -> None:
    """When there is no Limitations of Use, return the full body (header stripped)."""
    indications = _OpenFDAClient.approved_indications(_IPTACOPAN)

    assert indications
    full_text = " ".join(indications)
    assert "1 INDICATIONS AND USAGE" not in full_text
    assert "paroxysmal nocturnal hemoglobinuria" in full_text


def test_approved_indications_missing_field_returns_empty() -> None:
    assert _OpenFDAClient.approved_indications({}) == []
    assert _OpenFDAClient.approved_indications({"indications_and_usage": []}) == []


# ---------------------------------------------------------------------------
# Static helpers — limitations_of_use
# ---------------------------------------------------------------------------


def test_limitations_of_use_extracts_text() -> None:
    lou = _OpenFDAClient.limitations_of_use(_REMIBRUTINIB)

    assert lou is not None
    assert "not indicated for other forms of urticaria" in lou
    # Should not bleed into the indication body
    assert "chronic spontaneous urticaria (CSU)" not in lou


def test_limitations_of_use_returns_none_when_absent() -> None:
    assert _OpenFDAClient.limitations_of_use(_IPTACOPAN) is None
    assert _OpenFDAClient.limitations_of_use({}) is None


# Path to the captured real-world OpenFDA label fixtures.
_OPENFDA_FIXTURES = Path(__file__).parents[3] / "fixtures" / "openfda_labels"


def _load_fixture(name: str) -> dict:
    return json.loads((_OPENFDA_FIXTURES / name).read_text())


def test_limitations_of_use_trims_to_bounded_clause_real_fixture() -> None:
    """Regression (#1056): the captured RHAPSIDO label concatenates the Highlights
    and full-text indication blocks, so the raw indications field carries a
    duplicated indication sentence and a SECOND Limitations-of-Use copy after the
    first marker. The extractor must return ONLY the bounded Limitations-of-Use
    clause, not the trailing indication / repeated text."""
    label = _load_fixture("remibrutinib.json")

    lou = _OpenFDAClient.limitations_of_use(label)

    assert lou == "Limitations of Use: RHAPSIDO is not indicated for other forms of urticaria."
    # The trailing duplicated indication block must NOT bleed in.
    assert "kinase inhibitor" not in lou
    # The duplicated 2nd Limitations-of-Use copy + reference tag must be gone.
    assert lou.count("Limitations of Use") == 1
    assert "( 1 )" not in lou


def test_limitations_of_use_no_lou_real_fixtures_return_none() -> None:
    """Brands whose captured label carries no Limitations of Use still return None
    (fail-open contract unchanged)."""
    assert _OpenFDAClient.limitations_of_use(_load_fixture("iptacopan.json")) is None
    assert _OpenFDAClient.limitations_of_use(_load_fixture("ribociclib.json")) is None


def test_limitations_of_use_returns_none_for_contentless_marker() -> None:
    """A bare or doubled "Limitations of Use" marker with no actual limitation
    text must fail open to None, not surface a contentless "Limitations of Use:"
    stub (degenerate/malformed labels)."""
    # Marker present but nothing follows it.
    assert (
        _OpenFDAClient.limitations_of_use(
            {
                "indications_and_usage": [
                    "1 INDICATIONS AND USAGE X is indicated for Y. Limitations of Use:"
                ]
            }
        )
        is None
    )
    # Doubled marker with the content only after the duplicate.
    assert (
        _OpenFDAClient.limitations_of_use(
            {
                "indications_and_usage": [
                    "Limitations of Use: Limitations of Use: Not indicated for Z."
                ]
            }
        )
        is None
    )


def test_limitations_of_use_keeps_multi_sentence_limitations() -> None:
    """A Limitations-of-Use clause may legitimately span multiple sentences,
    including a restrictive 'indicated only ... not for' phrasing. Those are
    limitations and must be kept whole — the bounding logic must not truncate at
    the bare word 'indicated' (it only stops at a POSITIVE indication restart)."""
    label = {
        "indications_and_usage": [
            "1 INDICATIONS AND USAGE DRUG is indicated for condition X. "
            "Limitations of Use: DRUG is not for use in children. "
            "DRUG is indicated only after failure of therapy A and not for first-line use."
        ]
    }

    lou = _OpenFDAClient.limitations_of_use(label)

    assert lou is not None
    assert "not for use in children" in lou
    assert "indicated only after failure of therapy A and not for first-line use" in lou


# ---------------------------------------------------------------------------
# Static helpers — boxed_warning
# ---------------------------------------------------------------------------


def test_boxed_warning_returns_text() -> None:
    warning = _OpenFDAClient.boxed_warning(_IPTACOPAN)
    assert warning == "WARNING: SERIOUS INFECTIONS ..."


def test_boxed_warning_returns_none_when_absent() -> None:
    assert _OpenFDAClient.boxed_warning(_KISQALI_STANDALONE) is None
    assert _OpenFDAClient.boxed_warning({}) is None


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def test_api_key_attached_to_request_when_set() -> None:
    seen_params: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(request.url.query.decode("utf-8"))
        return httpx.Response(200, json={"results": [_KISQALI_STANDALONE]})

    client = _OpenFDAClient(
        api_key="test-key-xyz",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with client:
        client.fetch_label("ribociclib")

    assert seen_params
    assert "api_key=test-key-xyz" in seen_params[0]


def test_no_api_key_omits_param() -> None:
    seen_params: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(request.url.query.decode("utf-8"))
        return httpx.Response(200, json={"results": [_KISQALI_STANDALONE]})

    client = _OpenFDAClient(
        api_key=None,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with client:
        client.fetch_label("ribociclib")

    assert seen_params
    assert "api_key" not in seen_params[0]
