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

import pytest

from src.rag import brand_aliases
from src.rag.brand_aliases import rxnav_brand_aliases
from tests.integration.test_clinical_context._live_gate import requires_network

# The gate probes an unrelated host, so an RxNav outage goes RED (#1612); the
# old preflight to the RxNav host itself turned one into a silent skip.
pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_network]


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
