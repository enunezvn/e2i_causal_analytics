"""Issue #1747 — faithful (real docker-Supabase) proof that the drift_monitor
dispatcher resolver grounds ``features_to_monitor`` in the REAL feature store
at dispatch time.

Load-bearing, NO mocks. Measured substrate (2026-08-20): feature_values is
100% synthetic-tagged; with synthetic included, the default 7d window has ZERO
features with >= 30 samples in both drift windows while 30d has 15. So:

* the ``drift_qualifying_features`` RPC (migration 131) must return 0 rows for
  7d and >= 1 for 30d under synthetic opt-in, and 0 in real-mode;
* the resolver under synthetic opt-in must bind real feature names + a
  supportable window (never the unsupportable 7d default);
* a real-mode resolve must fail closed with ``NeedsStructuredInput`` semantics
  — never the raw ``Failed to build DriftMonitorInput`` crash (the live 2/2
  failure this issue pins).

Gated behind ``E2I_DB_INTEGRATION=1``; run with ``-n0``. Read-only: no rows
are created, so there is nothing to clean up.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput

_RUN = os.environ.get("E2I_DB_INTEGRATION") == "1"

if _RUN:
    from dotenv import load_dotenv

    load_dotenv()

pytestmark = pytest.mark.skipif(
    not _RUN, reason="set E2I_DB_INTEGRATION=1 to run faithful real-DB tests"
)


def _payload(query: str = "Run drift detection on our models") -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": {"user_id": "u-1747"},
        "session_id": "sess-1747-realdb",
        "parsed_query": {"intent": "drift_detection", "entities": []},
    }


def _dispatch(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_name": "drift_monitor",
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 30000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


class TestDriftQualifyingFeaturesRpc:
    """Migration 131 function against the live store — the numbers below are
    structural (a window either supports the min-samples contract or not),
    not pinned row counts."""

    def test_rpc_7d_synthetic_has_no_qualifying_features(self) -> None:
        names = disp._probe_drift_substrate(7, True)  # type: ignore[attr-defined]
        assert names == [], (
            "the 7d default window measured ZERO both-window qualifying "
            f"features — got {names!r}; if this now qualifies, the substrate "
            "grew and the resolver sweep will simply bind 7d (fine) — update "
            "this pin only with fresh measurements"
        )

    def test_rpc_30d_synthetic_has_qualifying_features(self) -> None:
        names = disp._probe_drift_substrate(30, True)  # type: ignore[attr-defined]
        assert len(names) >= 1, "30d measured 15 qualifying features; got none"

    def test_rpc_real_mode_is_empty(self) -> None:
        # The store is 100% synthetic-tagged — real-mode must see nothing.
        assert disp._probe_drift_substrate(30, False) == []  # type: ignore[attr-defined]


class TestResolverAgainstRealSubstrate:
    def test_synthetic_opt_in_binds_real_features_and_window(self) -> None:
        payload = _payload()
        payload["user_context"]["include_synthetic"] = True
        out = disp._resolve_drift_monitor_input(payload, _dispatch())  # type: ignore[attr-defined]
        assert isinstance(out, dict), f"expected bound inputs, got {out!r}"
        assert out["features_to_monitor"], "must bind at least one real feature"
        assert out["include_synthetic"] is True
        assert out["time_window"] != "7d", (
            "7d measured unsupportable — the sweep must have picked a larger "
            "window the data actually supports"
        )
        # every bound name must exist in the features registry
        from src.repositories import get_supabase_client

        client = get_supabase_client()
        rows = (
            client.table("features")
            .select("name")
            .in_("name", out["features_to_monitor"])
            .execute()
        )
        found = {r["name"] for r in (rows.data or [])}
        assert found == set(out["features_to_monitor"])

    def test_real_mode_fails_closed_honestly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The droplet .env may carry the showcase E2I_INCLUDE_SYNTHETIC flag
        # (read fresh per call) — force a true real-mode deployment here.
        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        out = disp._resolve_drift_monitor_input(_payload(), _dispatch())  # type: ignore[attr-defined]
        assert isinstance(out, NeedsStructuredInput)
        assert out.missing == ("features_to_monitor",)
        assert out.user_action
