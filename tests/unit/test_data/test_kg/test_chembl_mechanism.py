"""Unit tests for ChEMBLClient.get_mechanism / mechanism_of_action via
httpx.MockTransport. Mirrors test_chembl.py; pins the /mechanism.json shape
verified live 2026-06-19 against ChEMBL REST v34."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.chembl import ChEMBLClient, reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(handler: Callable[[httpx.Request], httpx.Response]) -> ChEMBLClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return ChEMBLClient(client=http)


def test_get_mechanism_returns_actions_for_molecule_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/mechanism.json" in request.url.path
        query = request.url.query.decode("utf-8")
        assert "molecule_chembl_id" in query
        assert "CHEMBL3545110" in query
        return httpx.Response(
            200,
            json={
                "mechanisms": [
                    {
                        "mechanism_of_action": "Cyclin-dependent kinase 4 inhibitor",
                        "action_type": "INHIBITOR",
                        "target_chembl_id": "CHEMBL331",
                    },
                    {
                        "mechanism_of_action": "Cyclin-dependent kinase 6 inhibitor",
                        "action_type": "INHIBITOR",
                        "target_chembl_id": "CHEMBL2508",
                    },
                ]
            },
        )

    with _client_with_handler(handler) as client:
        mechs = client.get_mechanism("CHEMBL3545110")
    assert [m.mechanism_of_action for m in mechs] == [
        "Cyclin-dependent kinase 4 inhibitor",
        "Cyclin-dependent kinase 6 inhibitor",
    ]
    assert mechs[0].action_type == "INHIBITOR"
    assert mechs[0].target_chembl_id == "CHEMBL331"


def test_get_mechanism_empty_id_skips_network() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"mechanisms": []})

    with _client_with_handler(handler) as client:
        assert client.get_mechanism("") == []
    assert calls["n"] == 0


def test_mechanism_of_action_resolves_drug_name_to_first_moa() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/molecule.json" in path:
            assert "ribociclib" in request.url.query.decode("utf-8").lower()
            return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL3545110"}]})
        assert "/mechanism.json" in path
        return httpx.Response(
            200,
            json={"mechanisms": [{"mechanism_of_action": "Cyclin-dependent kinase 4 inhibitor"}]},
        )

    with _client_with_handler(handler) as client:
        assert client.mechanism_of_action("ribociclib") == "Cyclin-dependent kinase 4 inhibitor"


def test_mechanism_of_action_unresolved_drug_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"molecules": []})

    with _client_with_handler(handler) as client:
        assert client.mechanism_of_action("not-a-real-drug") is None
