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

    def test_rpc_counts_match_connector_window_semantics(self) -> None:
        """Codex iter-1 MED: the RPC's window predicates must mirror the
        connector's ``.gte(start).lte(end)`` closed intervals exactly
        (current [now-N, now], baseline [now-2N, now-N]) — the original SQL
        left the current window unbounded above (counting future-dated rows
        the connector would never fetch) and excluded the baseline's now-N
        edge.

        INVARIANT pin, not a red/green discriminator on this store: parity
        only diverges when boundary-instant or future-dated rows exist, and
        the live store has none — the discriminating before/after evidence
        is the transactional psql experiment (future-dated row counted by the
        old SQL, excluded by the new; recorded in the #1747 PR). This pin
        holds the parity contract against future regressions.
        """
        from datetime import UTC, datetime, timedelta

        from src.repositories import get_supabase_client

        client = get_supabase_client()
        window_days = 30
        rows = client.rpc(
            "drift_qualifying_features",
            {"p_window_days": window_days, "p_min_samples": 30, "p_include_synthetic": True},
        ).execute()
        assert rows.data, "30d/synthetic measured 15 qualifying features; got none"

        now = datetime.now(UTC)
        current_start = now - timedelta(days=window_days)
        baseline_start = now - timedelta(days=2 * window_days)

        def _connector_count(feature_id: str, start: datetime, end: datetime) -> int:
            resp = (
                client.table("feature_values")
                .select("id", count="exact", head=True)
                .eq("feature_id", feature_id)
                .gte("event_timestamp", start.isoformat())
                .lte("event_timestamp", end.isoformat())
                .execute()
            )
            assert resp.count is not None
            return int(resp.count)

        for row in rows.data[:3]:
            feature = (
                client.table("features").select("id").eq("name", row["feature_name"]).execute()
            )
            assert feature.data, f"RPC returned unregistered feature {row['feature_name']!r}"
            feature_id = feature.data[0]["id"]
            assert _connector_count(feature_id, current_start, now) == row["current_n"], (
                f"{row['feature_name']}: RPC current_n diverges from the "
                "connector's gte/lte window count"
            )
            assert (
                _connector_count(feature_id, baseline_start, current_start) == row["baseline_n"]
            ), (
                f"{row['feature_name']}: RPC baseline_n diverges from the "
                "connector's gte/lte window count"
            )


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
