"""Unit tests for RxNavClient using httpx.MockTransport."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.rxnav import RxNavClient, RxNavError, reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> RxNavClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return RxNavClient(client=http)


def test_rxcui_for_name_returns_first_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/rxcui.json" in request.url.path
        assert request.url.params.get("name") == "ibuprofen"
        # search=2 → approximate fallback enabled
        assert request.url.params.get("search") == "2"
        return httpx.Response(
            200,
            json={"idGroup": {"rxnormId": ["5640"]}},
        )

    with _client_with_handler(handler) as client:
        assert client.rxcui_for_name("ibuprofen") == "5640"


def test_rxcui_for_name_returns_none_when_no_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"idGroup": {}})

    with _client_with_handler(handler) as client:
        assert client.rxcui_for_name("zzzzz") is None


def test_rxcui_for_name_empty_returns_none_without_request() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"idGroup": {}})

    with _client_with_handler(handler) as client:
        assert client.rxcui_for_name("") is None
        assert call_count["n"] == 0


def test_rxcui_for_ndc_extracts_rxcui() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/ndcstatus.json" in request.url.path
        return httpx.Response(
            200,
            json={"ndcStatus": {"rxcui": "1049640", "status": "ACTIVE"}},
        )

    with _client_with_handler(handler) as client:
        assert client.rxcui_for_ndc("12345678901") == "1049640"


def test_properties_returns_property_block() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/rxcui/5640/properties.json" in request.url.path
        return httpx.Response(
            200,
            json={"properties": {"rxcui": "5640", "name": "ibuprofen", "tty": "IN"}},
        )

    with _client_with_handler(handler) as client:
        props = client.properties("5640")
        assert props is not None
        assert props["name"] == "ibuprofen"


def test_properties_returns_none_for_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with _client_with_handler(handler) as client:
        assert client.properties("99999999") is None


def test_caches_name_lookup() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"idGroup": {"rxnormId": ["5640"]}})

    with _client_with_handler(handler) as client:
        client.rxcui_for_name("ibuprofen")
        client.rxcui_for_name("ibuprofen")
        assert call_count["n"] == 1


def test_5xx_raises_rxnav_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with _client_with_handler(handler) as client:
        with pytest.raises(RxNavError) as exc:
            client.rxcui_for_name("ibuprofen")
        assert "502" in str(exc.value)


def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<xml>not json</xml>")

    with _client_with_handler(handler) as client:
        with pytest.raises(RxNavError) as exc:
            client.rxcui_for_name("ibuprofen")
        assert "non-JSON" in str(exc.value)
