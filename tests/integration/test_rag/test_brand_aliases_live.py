"""Live RxNav: the three real ingredient<->brand pairs (measured 2026-09-22)."""

from __future__ import annotations

import socket

import pytest

from src.rag import brand_aliases
from src.rag.brand_aliases import rxnav_brand_aliases


def _network_available() -> bool:
    try:
        socket.create_connection(("rxnav.nlm.nih.gov", 443), timeout=3).close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _network_available(), reason="No outbound network to RxNav."),
]


def test_real_rxnav_resolves_the_three_pairs(monkeypatch):
    monkeypatch.setenv("RXNAV_BRAND_ALIASES", "1")
    brand_aliases.reset_cache()
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"])
    assert "ribociclib" in out["Kisqali"]
    assert "iptacopan" in out["Fabhalta"]
    assert "rhapsido" in out["Remibrutinib"]
