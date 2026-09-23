"""Live RxNav: the three real ingredient<->brand pairs (measured 2026-09-22).

``rxnav_brand_aliases`` degrades by design: an RxNav outage is logged and the
round returns what it gathered. So an outage must be asserted on EXPLICITLY,
with the logged reason in the failure message. Indexing the result directly
turned the 2026-09-23 nightly's connect failure into a bare ``KeyError:
'Kisqali'`` (#2267), which reads like a code defect and which the nightly
classifier (scripts/ci/classify_slow_tests_failure.py) rightly refuses to call
an upstream outage.
"""

from __future__ import annotations

import logging
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


def test_real_rxnav_resolves_the_three_pairs(monkeypatch, caplog):
    monkeypatch.setenv("RXNAV_BRAND_ALIASES", "1")
    brand_aliases.reset_cache()
    with caplog.at_level(logging.WARNING, logger=brand_aliases.logger.name):
        out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"])
    degraded = [r.getMessage() for r in caplog.records if r.name == brand_aliases.logger.name]
    assert not degraded, f"RxNav brand-alias round did not complete: {degraded}"
    assert "ribociclib" in out.get("Kisqali", []), out
    assert "iptacopan" in out.get("Fabhalta", []), out
    assert "rhapsido" in out.get("Remibrutinib", []), out
