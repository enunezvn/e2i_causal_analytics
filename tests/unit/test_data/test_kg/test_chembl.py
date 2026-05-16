"""Unit tests for ChEMBLClient using httpx.MockTransport.

Mirrors the patterns established by ``test_open_targets.py`` and
``test_umls_uts.py``: MockTransport handler, autouse fixture clears
in-process caches, no live HTTP. Tests pin response shapes against the
ChEMBL REST API v34 (https://www.ebi.ac.uk/chembl/api/data/docs).
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.chembl import (
    Activity,
    ChEMBLClient,
    ChEMBLError,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> ChEMBLClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return ChEMBLClient(client=http)


def test_compound_search_returns_chembl_id() -> None:
    """compound_search resolves a drug name to its top ChEMBL molecule ID."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/molecule.json" in request.url.path
        # ChEMBL accepts ?molecule_synonyms__molecule_synonym__iexact=NAME
        # or a generic ?q=NAME — either is acceptable. We assert the name
        # appears somewhere in the URL query string.
        assert "imatinib" in request.url.query.decode("utf-8").lower()
        return httpx.Response(
            200,
            json={
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL941",
                        "pref_name": "IMATINIB",
                    }
                ]
            },
        )

    with _client_with_handler(handler) as client:
        assert client.compound_search("imatinib") == "CHEMBL941"


def test_compound_search_empty_string_skips_network() -> None:
    """Empty name must not trigger an HTTP call."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"molecules": []})

    with _client_with_handler(handler) as client:
        assert client.compound_search("") is None
        assert call_count["n"] == 0


def test_compound_search_returns_none_for_empty_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"molecules": []})

    with _client_with_handler(handler) as client:
        assert client.compound_search("nonsense-drug-xyz") is None


def test_target_search_returns_chembl_id_by_gene_symbol() -> None:
    """target_search resolves a gene symbol → ChEMBL target ID."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/target.json" in request.url.path
        query = request.url.query.decode("utf-8")
        # Must use the live-API-valid denormalized top-level synonym filter,
        # not the nested ``target_components__component_synonyms__component_synonym``
        # path which the live ChEMBL API rejects with HTTP 400 (verified
        # live by primary at iter-1 gate-on-diff).
        assert "target_synonym__iexact" in query
        assert "ABL1" in query
        return httpx.Response(
            200,
            json={
                "targets": [
                    {
                        "target_chembl_id": "CHEMBL1862",
                        "pref_name": "Tyrosine-protein kinase ABL",
                        "target_component_synonyms": [
                            {"component_synonym": "ABL1", "syn_type": "GENE_SYMBOL"}
                        ],
                    }
                ]
            },
        )

    with _client_with_handler(handler) as client:
        assert client.target_search("ABL1") == "CHEMBL1862"


def test_target_search_uses_target_synonym_iexact_filter() -> None:
    """Regression test (#245 iter-1): the cross-walk URL must use
    ``target_synonym__iexact``, NOT the nested
    ``target_components__component_synonyms__component_synonym__iexact``
    that the live ChEMBL API rejects with HTTP 400.

    Pins the exact param key. Reverting the production fix back to the
    nested path must make this test FAIL.
    """
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = request.url.query.decode("utf-8")
        captured["params"] = "&".join(f"{k}={v}" for k, v in request.url.params.multi_items())
        return httpx.Response(
            200,
            json={"targets": [{"target_chembl_id": "CHEMBL1862"}]},
        )

    with _client_with_handler(handler) as client:
        assert client.target_search("ABL1") == "CHEMBL1862"

    # Pin the EXACT filter key the live API accepts.
    assert "target_synonym__iexact" in captured["query"]
    # Verify the value is the gene symbol that was passed.
    assert "ABL1" in captured["query"]
    # Negative pin: the broken nested path must NOT be in the URL.
    assert (
        "target_components__component_synonyms__component_synonym__iexact"
        not in (captured["query"])
    ), (
        "Cross-walk regressed to the live-API-invalid nested filter "
        "path; ChEMBL rejects this with HTTP 400 (see PR #291 iter-1)."
    )


def test_target_search_empty_payload_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"targets": []})

    with _client_with_handler(handler) as client:
        assert client.target_search("NONEXISTENT_GENE") is None


def test_get_bioactivity_returns_activity_records() -> None:
    """get_bioactivity returns IC50/Ki Activity rows for a target ChEMBL ID."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/activity.json" in request.url.path
        query = request.url.query.decode("utf-8")
        assert "CHEMBL1862" in query
        return httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "activity_id": 12345,
                        "molecule_chembl_id": "CHEMBL941",
                        "target_chembl_id": "CHEMBL1862",
                        "standard_type": "IC50",
                        "standard_value": "10.5",
                        "standard_units": "nM",
                        "pchembl_value": "7.98",
                        "document_chembl_id": "CHEMBL1149632",
                        "pubmed_id": "16480739",
                    },
                    {
                        "activity_id": 12346,
                        "molecule_chembl_id": "CHEMBL941",
                        "target_chembl_id": "CHEMBL1862",
                        "standard_type": "Ki",
                        "standard_value": "25.0",
                        "standard_units": "nM",
                        "pchembl_value": "7.60",
                        "document_chembl_id": "CHEMBL1149633",
                        "pubmed_id": None,
                    },
                ]
            },
        )

    with _client_with_handler(handler) as client:
        activities = client.get_bioactivity("CHEMBL1862")
    assert len(activities) == 2
    assert all(isinstance(a, Activity) for a in activities)
    assert activities[0].standard_type == "IC50"
    assert activities[0].standard_value == pytest.approx(10.5)
    assert activities[0].standard_units == "nM"
    assert activities[0].pchembl_value == pytest.approx(7.98)
    assert activities[0].pubmed_id == "16480739"
    assert activities[0].molecule_chembl_id == "CHEMBL941"
    assert activities[0].target_chembl_id == "CHEMBL1862"
    assert activities[1].standard_type == "Ki"
    assert activities[1].pubmed_id is None


def test_get_bioactivity_filters_to_ic50_and_ki() -> None:
    """v1 contract: get_bioactivity returns only IC50 and Ki rows by default.

    ChEMBL activity table carries dozens of standard_types (IC50, Ki, EC50,
    Kd, GI50, AC50, ...). For Layer-2 evidence enrichment we only need
    binding/inhibitory potency canonicals; restrict to IC50 + Ki to keep
    the evidence payload bounded.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "activity_id": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "target_chembl_id": "CHEMBL_T",
                        "standard_type": "IC50",
                        "standard_value": "5.0",
                        "standard_units": "nM",
                    },
                    {
                        "activity_id": 2,
                        "molecule_chembl_id": "CHEMBL1",
                        "target_chembl_id": "CHEMBL_T",
                        "standard_type": "EC50",  # filtered out
                        "standard_value": "100",
                        "standard_units": "nM",
                    },
                    {
                        "activity_id": 3,
                        "molecule_chembl_id": "CHEMBL1",
                        "target_chembl_id": "CHEMBL_T",
                        "standard_type": "Ki",
                        "standard_value": "1.0",
                        "standard_units": "nM",
                    },
                ]
            },
        )

    with _client_with_handler(handler) as client:
        activities = client.get_bioactivity("CHEMBL_T")
    assert {a.standard_type for a in activities} == {"IC50", "Ki"}


def test_get_bioactivity_caches_result() -> None:
    """Repeated bioactivity call for the same target must not re-hit network."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"activities": []})

    with _client_with_handler(handler) as client:
        client.get_bioactivity("CHEMBL_T")
        client.get_bioactivity("CHEMBL_T")
    assert call_count["n"] == 1


def test_http_5xx_raises_chembl_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    with _client_with_handler(handler) as client:
        with pytest.raises(ChEMBLError) as exc:
            client.compound_search("imatinib")
        assert "503" in str(exc.value)


def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    with _client_with_handler(handler) as client:
        with pytest.raises(ChEMBLError):
            client.compound_search("imatinib")


def test_http_429_retries_then_succeeds() -> None:
    """Rate-limit (HTTP 429) is retried with a backoff up to ``max_retries``."""
    call_log: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(len(call_log))
        if len(call_log) <= 2:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(
            200,
            json={"molecules": [{"molecule_chembl_id": "CHEMBL941"}]},
        )

    with _client_with_handler(handler) as client:
        # Use zero-backoff for test speed; only the retry count matters here.
        client._retry_backoff_s = 0.0
        assert client.compound_search("imatinib") == "CHEMBL941"
    # Two 429s + one success = 3 calls.
    assert len(call_log) == 3


def test_http_429_raises_after_max_retries() -> None:
    """After ``max_retries`` 429s the client gives up with ChEMBLError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    with _client_with_handler(handler) as client:
        client._retry_backoff_s = 0.0
        with pytest.raises(ChEMBLError) as exc:
            client.compound_search("imatinib")
        assert "429" in str(exc.value)


def test_cache_namespace_uses_chembl_subdir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Cache namespace must live under ``chembl/`` parallel to
    ``opentargets/`` and ``umls/``. The current v1 backing store is
    in-process LRU; the path-namespace assertion locks in the contract
    for the future disk-backed variant.
    """
    from src.data.kg.chembl import CACHE_NAMESPACE

    assert CACHE_NAMESPACE == "chembl"


def test_open_targets_target_id_to_chembl_target_id() -> None:
    """Cross-walk: given a gene symbol (which Open Targets surfaces via
    ``target.approvedSymbol``), resolve to the ChEMBL target_chembl_id.

    The bridge function is name-decoupled from Open Targets: it accepts a
    gene symbol string. ``KnowledgeGraphQuerier`` is responsible for
    extracting the symbol from the Open Targets payload before calling.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/target.json" in request.url.path
        return httpx.Response(
            200,
            json={
                "targets": [
                    {
                        "target_chembl_id": "CHEMBL1862",
                        "pref_name": "ABL kinase",
                    }
                ]
            },
        )

    with _client_with_handler(handler) as client:
        assert client.open_targets_target_to_chembl("ABL1") == "CHEMBL1862"


def test_open_targets_target_id_to_chembl_target_id_none_on_empty() -> None:
    """No target → return None, not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"targets": []})

    with _client_with_handler(handler) as client:
        assert client.open_targets_target_to_chembl("UNKNOWN_GENE") is None


def test_open_targets_target_id_to_chembl_target_id_handles_none_input() -> None:
    """None / empty input → return None without HTTP."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"targets": []})

    with _client_with_handler(handler) as client:
        assert client.open_targets_target_to_chembl(None) is None
        assert client.open_targets_target_to_chembl("") is None
        assert call_count["n"] == 0


def test_activity_dataclass_is_frozen() -> None:
    """Activity is a frozen dataclass — callers cannot mutate cached records."""
    a = Activity(
        activity_id=1,
        molecule_chembl_id="CHEMBL1",
        target_chembl_id="CHEMBL_T",
        standard_type="IC50",
        standard_value=5.0,
        standard_units="nM",
        pchembl_value=8.3,
        pubmed_id="12345678",
    )
    with pytest.raises(Exception):  # FrozenInstanceError subclass
        a.standard_value = 10.0  # type: ignore[misc]
