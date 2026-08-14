"""Unit tests for OpenTargetsClient using httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from src.data.kg.open_targets import (
    OpenTargetsClient,
    OpenTargetsError,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenTargetsClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return OpenTargetsClient(client=http)


def test_drug_disease_evidence_happy_path() -> None:
    """Open Targets v4 returns drug->disease claims via ``drug.indications``.

    The previous version of this test asserted a ``diseaseId`` GraphQL variable
    and an ``evidences`` block. Open Targets REMOVED the top-level ``evidences``
    Query field and renamed ``maxPhaseForIndication`` to ``maxClinicalStage``,
    so the old document returned HTTP 400 on every live call while this mocked
    test stayed green (#1607). The disease is now filtered client-side from the
    drug's full indication list, so ``diseaseId`` is no longer a query variable.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["variables"]["drugId"] == "CHEMBL1234"
        assert "diseaseId" not in body["variables"], (
            "diseaseId is no longer a query variable — the indication list is "
            "returned whole and filtered by the caller"
        )
        assert "maxPhaseForIndication" not in body["query"], (
            "maxPhaseForIndication was renamed upstream; using it makes the "
            "whole query fail with HTTP 400"
        )
        return httpx.Response(
            200,
            json={
                "data": {
                    "drug": {
                        "id": "CHEMBL1234",
                        "name": "TestDrug",
                        "maximumClinicalStage": "APPROVAL",
                        "indications": {
                            "count": 1,
                            "rows": [
                                {
                                    "disease": {"id": "EFO_0001", "name": "TestDisease"},
                                    "maxClinicalStage": "APPROVAL",
                                }
                            ],
                        },
                    }
                }
            },
        )

    with _client_with_handler(handler) as client:
        result = client.drug_disease_evidence("CHEMBL1234", "EFO_0001")
        rows = result["drug"]["indications"]["rows"]
        assert len(rows) == 1
        assert rows[0]["disease"]["id"] == "EFO_0001"
        assert rows[0]["maxClinicalStage"] == "APPROVAL"


def test_search_drug_returns_first_drug_hit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "search": {"hits": [{"id": "CHEMBL999", "name": "Aspirin", "entity": "drug"}]}
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.search_drug("aspirin") == "CHEMBL999"


def test_search_drug_skips_non_drug_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "search": {
                        "hits": [
                            {"id": "EFO_xx", "name": "x", "entity": "disease"},
                            {"id": "CHEMBL_target", "name": "y", "entity": "target"},
                        ]
                    }
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.search_drug("aspirin") is None


def test_search_drug_empty_string_returns_none() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"data": {"search": {"hits": []}}})

    with _client_with_handler(handler) as client:
        assert client.search_drug("") is None
        assert call_count["n"] == 0


def test_search_disease_returns_first_disease_hit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "MONDO_0004980",
                                "name": "atopic dermatitis",
                                "entity": "disease",
                            }
                        ]
                    }
                }
            },
        )

    with _client_with_handler(handler) as client:
        assert client.search_disease("atopic dermatitis") == "MONDO_0004980"


def test_graphql_errors_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"errors": [{"message": "field 'foo' is not a valid query field"}]},
        )

    with _client_with_handler(handler) as client:
        with pytest.raises(OpenTargetsError):
            client.query_raw("{ foo }")


def test_http_5xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    with _client_with_handler(handler) as client:
        with pytest.raises(OpenTargetsError) as exc:
            client.query_raw("{ x }")
        assert "503" in str(exc.value)


def test_caches_drug_disease_response() -> None:
    """Repeated drug_disease_evidence call must hit the network only once."""

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={
                "data": {
                    "drug": {"id": "X", "name": "X", "indications": {"rows": []}},
                    "evidences": {"count": 0, "rows": []},
                }
            },
        )

    with _client_with_handler(handler) as client:
        client.drug_disease_evidence("X", "Y")
        client.drug_disease_evidence("X", "Y")
        assert call_count["n"] == 1


def test_drug_disease_cache_is_keyed_on_the_drug_alone() -> None:
    """Different diseases for one drug must NOT re-fetch the same payload.

    The v4 query takes only ``$drugId`` and returns the drug's whole indication
    list, which the caller filters. Keying the cache on the disease as well
    meant a cache build over N features asked Open Targets for the identical
    payload N times — 74 round-trips where 1 suffices on the Optum manifest.
    """

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={
                "data": {"drug": {"id": "X", "name": "X", "indications": {"count": 0, "rows": []}}}
            },
        )

    with _client_with_handler(handler) as client:
        client.drug_disease_evidence("X", "MONDO_1")
        client.drug_disease_evidence("X", "MONDO_2")
        client.drug_disease_evidence("X", "MONDO_3")
        assert call_count["n"] == 1, (
            "the disease is filtered client-side and is not a query variable, so "
            "it must not multiply network calls"
        )

    # A different drug is a genuinely different query.
    call_count["n"] = 0
    with _client_with_handler(handler) as client:
        client.drug_disease_evidence("A", "MONDO_1")
        client.drug_disease_evidence("B", "MONDO_1")
        assert call_count["n"] == 2


def test_payload_missing_data_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weird_payload": "no data"})

    with _client_with_handler(handler) as client:
        with pytest.raises(OpenTargetsError) as exc:
            client.query_raw("{ x }")
        assert "missing 'data'" in str(exc.value)
