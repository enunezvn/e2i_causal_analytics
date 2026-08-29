"""Shared network gate for the live clinical-context contract tests (#1612).

These tests hit real third-party endpoints. They are ``slow``-marked so the
PR-blocking ``integration-tests`` lane (``-m "not slow"``) deselects them and
``slow-tests.yml`` Job A (``pytest tests/ -m slow``) runs them on the 05:00 UTC
schedule — the routing issue #1612 AC3 asks for, using the marker contract
already guarded by ``tests/integration/test_slow_marker_discipline.py``.

The gate probes reachability ONCE per session rather than per test, so a run
costs one extra request rather than one per case. All four APIs are zero-auth;
``OPENFDA_API_KEY`` only raises openFDA's rate limit (1k/day/IP -> 120k/day), so
the gate is network-based, not key-based (#1612 AC2).
"""

from __future__ import annotations

import functools

import httpx
import pytest

# Probing a host we do not otherwise assert against keeps the gate independent
# of the contracts under test: if ChEMBL is down but the internet is up, the
# ChEMBL test SHOULD go red (that is the degradation signal #1612 asks for),
# rather than silently skipping.
_GATE_URL = "https://connectivitycheck.gstatic.com/generate_204"
_GATE_TIMEOUT_S = 8.0


@functools.lru_cache(maxsize=1)
def network_available() -> bool:
    """True when this host has outbound HTTPS. Cached for the session."""
    try:
        response = httpx.get(_GATE_URL, timeout=_GATE_TIMEOUT_S, follow_redirects=True)
    except Exception:  # noqa: BLE001 - any transport failure means "no network"
        return False
    return response.status_code < 500


requires_network = pytest.mark.skipif(
    not network_available(),
    reason="No outbound network; skipping live third-party contract tests (#1612).",
)
