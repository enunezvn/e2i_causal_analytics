"""Unit tests for UMLSClient using httpx.MockTransport."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.umls_uts import (
    UMLSAuthError,
    UMLSClient,
    UMLSError,
    UMLSNotFoundError,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "fake-key",
) -> UMLSClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return UMLSClient(api_key=api_key, client=http)


def test_search_returns_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/search/current" in request.url.path
        assert request.url.params.get("string") == "atopic dermatitis"
        assert request.url.params.get("apiKey") == "fake-key"
        return httpx.Response(
            200,
            json={
                "result": {
                    "results": [
                        {"ui": "C0011615", "name": "Dermatitis, Atopic", "rootSource": "MTH"}
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        results = client.search("atopic dermatitis")
        assert len(results) == 1
        assert results[0]["ui"] == "C0011615"


def test_search_collapses_none_sentinel() -> None:
    """UTS returns ``[{ui: 'NONE'}]`` for empty searches; client returns []."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": {"results": [{"ui": "NONE", "name": "NO RESULTS"}]}},
        )

    with _client_with_handler(handler) as client:
        assert client.search("nonsense") == []


def test_cui_lookup_extracts_semantic_types_and_atom_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "name": "Dermatitis, Atopic",
                    "semanticTypes": [{"name": "Disease or Syndrome"}],
                    "atomCount": 576,
                }
            },
        )

    with _client_with_handler(handler) as client:
        concept = client.cui_lookup("C0011615")
        assert concept.cui == "C0011615"
        assert concept.preferred_name == "Dermatitis, Atopic"
        assert concept.semantic_types == ("Disease or Syndrome",)
        assert concept.atom_count == 576


def test_cui_lookup_caches_result() -> None:
    """Second call with the same CUI must NOT hit the network again."""

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={
                "result": {
                    "name": "Dermatitis, Atopic",
                    "semanticTypes": [],
                    "atomCount": 1,
                }
            },
        )

    with _client_with_handler(handler) as client:
        client.cui_lookup("C0011615")
        client.cui_lookup("C0011615")
        assert call_count["n"] == 1


def test_auth_error_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    with _client_with_handler(handler) as client:
        with pytest.raises(UMLSAuthError):
            client.search("anything")


def test_auth_error_on_403() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    with _client_with_handler(handler) as client:
        with pytest.raises(UMLSAuthError):
            client.search("anything")


def test_not_found_error_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    with _client_with_handler(handler) as client:
        with pytest.raises(UMLSNotFoundError):
            client.cui_lookup("C9999999")


def test_500_raises_generic_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    with _client_with_handler(handler) as client:
        with pytest.raises(UMLSError) as exc:
            client.search("x")
        assert "500" in str(exc.value)


def test_non_json_body_raises_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with _client_with_handler(handler) as client:
        with pytest.raises(UMLSError) as exc:
            client.search("x")
        assert "non-JSON" in str(exc.value)


def test_missing_api_key_raises_at_construction() -> None:
    """Constructing without a key in env or argument must fail loudly."""
    import os

    saved = os.environ.pop("UMLS_UTS_API_KEY", None)
    try:
        with pytest.raises(UMLSAuthError):
            UMLSClient()
    finally:
        if saved is not None:
            os.environ["UMLS_UTS_API_KEY"] = saved


def test_code_to_cui_via_search_endpoint() -> None:
    """code_to_cui hits /search with inputType=sourceUi, returnIdType=concept."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/search/current" in request.url.path
        params = request.url.params
        assert params.get("string") == "L20.9"
        assert params.get("inputType") == "sourceUi"
        assert params.get("sabs") == "ICD10CM"
        assert params.get("returnIdType") == "concept"
        assert params.get("searchType") == "exact"
        return httpx.Response(
            200,
            json={
                "result": {
                    "results": [
                        {"ui": "C0011615", "name": "Dermatitis, Atopic", "rootSource": "MTH"}
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        cui = client.code_to_cui("L20.9", source="ICD10CM")
        assert cui == "C0011615"


def test_code_to_cui_returns_none_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    with _client_with_handler(handler) as client:
        assert client.code_to_cui("ZZZ", source="ICD10CM") is None


def test_code_to_cui_returns_none_when_no_results() -> None:
    """Empty results list = no CUI."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"results": []}})

    with _client_with_handler(handler) as client:
        assert client.code_to_cui("ZZZ", source="ICD10CM") is None


def test_code_to_cui_skips_none_sentinel_row() -> None:
    """UTS sometimes returns a single ui='NONE' row; client must skip it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": {"results": [{"ui": "NONE", "name": "NO RESULTS"}]}},
        )

    with _client_with_handler(handler) as client:
        assert client.code_to_cui("ZZZ", source="ICD10CM") is None


def test_code_to_cui_returns_first_when_multiple_cui_rows() -> None:
    """When multiple valid CUIs are returned, deterministically pick the first.

    UTS occasionally returns >1 CUI for ambiguous source codes (e.g., legacy
    codes that map to multiple modern concepts). v1 always picks the first;
    a future revision could rank by atom count or rootSource priority.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "results": [
                        {"ui": "C1111111", "name": "First", "rootSource": "MTH"},
                        {"ui": "C2222222", "name": "Second", "rootSource": "SNOMEDCT_US"},
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.code_to_cui("ambiguous", source="ICD10CM") == "C1111111"


def test_code_to_cui_picks_first_cui_skipping_non_cui_rows() -> None:
    """Rows must start with 'C' to be CUIs; skip non-CUI rows defensively."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "results": [
                        {"ui": "A12345", "name": "atom row"},
                        {"ui": "C1234567", "name": "the CUI"},
                    ]
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.code_to_cui("X", source="LNC") == "C1234567"


def test_crosswalk_returns_atoms() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/crosswalk/current/source/RXNORM/" in request.url.path
        return httpx.Response(
            200,
            json={
                "result": [
                    {"ui": "A123", "rootSource": "MSH", "name": "Atom1"},
                    {"ui": "A124", "rootSource": "SNOMEDCT_US", "name": "Atom2"},
                ]
            },
        )

    with _client_with_handler(handler) as client:
        atoms = client.crosswalk("12345", source="RXNORM")
        assert len(atoms) == 2
        assert atoms[0]["ui"] == "A123"


def test_crosswalk_404_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        assert client.crosswalk("ZZZ", source="ICD10CM") == []


def test_cui_relations_returns_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/content/current/CUI/C0011615/relations" in request.url.path
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "relationLabel": "RB",
                        "additionalRelationLabel": "isa",
                        "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011603",
                        "relatedIdName": "Dermatitis",
                        "rootSource": "MSH",
                    },
                    {
                        "relationLabel": "RN",
                        "additionalRelationLabel": "inverse_isa",
                        "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011620",
                        "relatedIdName": "Atopic Dermatitis (variant)",
                        "rootSource": "SNOMEDCT_US",
                    },
                ]
            },
        )

    with _client_with_handler(handler) as client:
        rows = client.cui_relations("C0011615")
        assert len(rows) == 2
        assert rows[0]["additionalRelationLabel"] == "isa"


def test_cui_relations_returns_empty_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        assert client.cui_relations("C9999999") == []


def test_cui_relations_caches() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"result": []})

    with _client_with_handler(handler) as client:
        client.cui_relations("C0011615")
        client.cui_relations("C0011615")
        assert call_count["n"] == 1


def test_search_empty_string_returns_empty_list_without_request() -> None:
    """Don't waste a network call on empty input."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"result": {"results": []}})

    with _client_with_handler(handler) as client:
        assert client.search("") == []
        assert call_count["n"] == 0
