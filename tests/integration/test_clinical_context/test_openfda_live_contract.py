"""Live contract tests for the openFDA drug-label API (#1612).

openFDA is prioritised (#1612 AC5) because it is the only one of the four
production APIs feeding a *gate* rather than prose: ``LabelCriteriaProvider`` ->
``label_gate`` consumes openFDA-derived indicated populations to demote
off-label opportunities in ``gap_analyzer/nodes/prioritizer.py`` and
``heterogeneous_optimizer/nodes/policy_learner.py``. Silent degradation there
changes prioritisation output with no test signal at all.

Two layers per API, so an upstream change is caught wherever it lands:

1. **Wire shape** — assert the nested JSON paths our parser actually walks.
2. **Parsed contract** — assert the client returns the typed shape callers read.

Measured live 2026-08-14 (all assertions below were confirmed against the real
endpoint before being written).
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

from src.services.clinical_context.clients import OPENFDA_BASE, _OpenFDAClient
from tests.integration.test_clinical_context._live_gate import requires_network

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_network]

# Generic names are what production actually passes (brand_map drug_name):
# Kisqali->ribociclib, Fabhalta->iptacopan, Remibrutinib->remibrutinib.
_GENERIC = "ribociclib"
# Brand names resolve ONLY via openfda.brand_name; the generic-name search 404s.
_BRAND = "KISQALI"


def _api_key_params() -> dict[str, str]:
    key = os.environ.get("OPENFDA_API_KEY")
    return {"api_key": key} if key else {}


# --------------------------------------------------------------------------- wire


def test_openfda_wire_shape_generic_name() -> None:
    """The nested paths ``_pick_best`` / ``approved_indications`` walk must exist."""
    response = httpx.get(
        f"{OPENFDA_BASE}/label.json",
        params={"search": f'openfda.generic_name:"{_GENERIC}"', "limit": 5, **_api_key_params()},
        timeout=30.0,
    )
    assert response.status_code == 200, f"openFDA generic search HTTP {response.status_code}"
    payload = response.json()

    results = payload.get("results")
    assert isinstance(results, list) and results, "openFDA payload lost its 'results' list"

    record = results[0]
    # _pick_best reads openfda.generic_name as a LIST.
    openfda_block = record.get("openfda")
    assert isinstance(openfda_block, dict), "record lost its 'openfda' block"
    assert isinstance(openfda_block.get("generic_name"), list), (
        "openfda.generic_name is no longer a list — _pick_best would stop matching"
    )
    # approved_indications reads indications_and_usage as a LIST of str.
    assert any(isinstance(r.get("indications_and_usage"), list) for r in results), (
        "no record carries indications_and_usage as a list"
    )


def test_openfda_signals_no_match_with_http_404_not_empty_results() -> None:
    """Pin the 404-on-no-match semantics the retry logic must handle.

    openFDA returns HTTP 404 with ``{"error": {"code": "NOT_FOUND"}}`` for a
    zero-result search rather than 200-with-empty-``results``. ``fetch_label``'s
    brand_name retry depends on distinguishing that from a transport error, so
    this behaviour is part of our contract with the API. If openFDA ever
    switches to 200-with-empty, this test goes red and the retry logic should be
    revisited.
    """
    response = httpx.get(
        f"{OPENFDA_BASE}/label.json",
        params={"search": f'openfda.generic_name:"{_BRAND}"', "limit": 5, **_api_key_params()},
        timeout=30.0,
    )
    # Assert the SEMANTIC contract, not one spelling of it (codex review MED,
    # #1612). The client treats both forms as "no match", so a switch from
    # 404-NOT_FOUND to 200-with-empty-results is not a functional break and
    # must not redden the nightly lane. A third behaviour — some other 404, or
    # a 200 that actually returns results for a brand under generic_name —
    # would break the retry contract and still fails here.
    if response.status_code == 404:
        assert response.json().get("error", {}).get("code") == "NOT_FOUND", (
            "openFDA returned a 404 that is not its NOT_FOUND no-match signal; "
            "_fetch_by_field would treat this as a transport failure"
        )
    else:
        assert response.status_code == 200, (
            f"openFDA zero-result search returned HTTP {response.status_code}; "
            "expected 404 NOT_FOUND or 200 with empty results"
        )
        assert not (response.json().get("results") or []), (
            "openFDA now RESOLVES a brand name under openfda.generic_name; "
            "the generic-then-brand retry order should be revisited"
        )


# ------------------------------------------------------------------------ parsed


def test_fetch_label_by_generic_name_returns_indications() -> None:
    """The production path: brand_map passes the GENERIC name."""
    client = _OpenFDAClient()
    label = client.fetch_label(_GENERIC)
    assert label is not None, f"openFDA returned no label for generic {_GENERIC!r}"

    indications = _OpenFDAClient.approved_indications(label)
    assert indications, "approved_indications extracted nothing from a live label"
    assert all(isinstance(i, str) and i.strip() for i in indications)
    # The label gate matches disease tokens against this text; assert it is real
    # prose rather than an empty/placeholder string.
    assert any(len(i) > 40 for i in indications), "indication text implausibly short"


def test_fetch_label_falls_back_to_brand_name_when_generic_404s() -> None:
    """RED-first regression for the unreachable brand_name retry (#1612).

    ``fetch_label``'s docstring promises: "Searches by ``openfda.generic_name``
    first, with a single retry using ``openfda.brand_name`` when the generic
    search returns empty results." openFDA signals "no match" with HTTP 404, and
    ``_fetch_by_field`` maps 404 -> ``None``, which ``fetch_label`` treats as
    "404 or exception - do not retry" and returns immediately. The documented
    retry is therefore unreachable for every brand-name-only drug.

    Harm is latent rather than live today (production passes generic names via
    ``brand_map.drug_name``), but the code does not do what it says, and any
    caller passing a brand name silently gets ``None`` from a gate input.
    """
    client = _OpenFDAClient()
    label = client.fetch_label(_BRAND)
    assert label is not None, (
        f"fetch_label({_BRAND!r}) returned None despite openfda.brand_name:"
        f'"{_BRAND}" having live results — the documented brand_name retry is unreachable'
    )
    indications = _OpenFDAClient.approved_indications(label)
    assert indications, "brand-name fallback produced a label with no indications"


def test_boxed_warning_shape_when_present() -> None:
    """``boxed_warning`` must stay a list-of-str on the wire (iptacopan has one)."""
    client = _OpenFDAClient()
    label = client.fetch_label("iptacopan")
    assert label is not None, "openFDA returned no label for iptacopan"
    warning = _OpenFDAClient.boxed_warning(label)
    assert isinstance(warning, str) and warning.strip(), (
        "iptacopan's label carries a boxed warning upstream; extraction returned "
        f"{warning!r} — the boxed_warning field shape may have changed"
    )


def test_label_text_supports_gate_token_matching() -> None:
    """The label gate regex-matches clinical tokens against live label text.

    ``LabelCriteriaProvider._FIELD_TOKENS`` evidences criteria by searching the
    label for tokens like ``\\badult``. If the label text stopped carrying them,
    every criterion would silently fall to config-unconfirmed and the gate would
    return indeterminate — visible only as thinner output.
    """
    client = _OpenFDAClient()
    label = client.fetch_label(_GENERIC)
    assert label is not None
    text = " ".join(_OpenFDAClient.approved_indications(label)).lower()
    assert re.search(r"\badult", text), (
        "live ribociclib label no longer contains an 'adult' token; "
        "LabelCriteriaProvider age criteria would stop being label_evidenced"
    )
