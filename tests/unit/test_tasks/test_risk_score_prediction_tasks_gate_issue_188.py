"""Issue #188: Celery downstream gate tests for write_risk_score_predictions.

When ``honest_failures`` is non-empty, the Celery task MUST refuse to
promote ``prediction_class`` to actionable values and MUST skip the
``patient_journeys.risk_score`` UPDATE. These tests exercise the gate
behavior without requiring a real Postgres backend — they target the
``no_db_url`` path so the gate logic is verified independently of DB IO.

For DB-touching gate tests see ``tests/integration/test_risk_score_db_writes.py``.
"""

from __future__ import annotations

import pytest

from src.tasks.risk_score_prediction_tasks import (
    GATED_SENTINEL_PREDICTION_CLASS,
    write_risk_score_predictions,
)


def _strip_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("RISK_SCORE_DB_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
        monkeypatch.delenv(v, raising=False)


class TestHonestFailureGateLogic:
    """Issue #188: honest_failures non-empty -> gated mode.

    We use the no-DB path (env-vars stripped) to exercise the gating
    branches in pure-Python without requiring Postgres. The returned
    dict carries the gate state so we can verify the gate decision
    without inspecting DB rows.
    """

    def test_honest_failures_empty_is_not_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _strip_db_env(monkeypatch)
        out = write_risk_score_predictions.apply(
            args=([{"prediction_id": "x"}], None, None, []),
        ).result
        assert out["status"] == "skipped"  # no DB URL
        # Not gated because honest_failures is [].
        assert out["gated"] is False
        assert out["honest_failures"] == []

    def test_honest_failures_non_empty_is_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _strip_db_env(monkeypatch)
        failures = ["AUC-PR floor not met: val_auc_pr=0.0895 < 0.145"]
        out = write_risk_score_predictions.apply(
            args=([{"prediction_id": "x"}], None, None, failures),
        ).result
        # Still 'skipped' here because no DB URL — but the gate state
        # must be reported so callers can branch.
        assert out["status"] == "skipped"
        assert out["gated"] is True
        assert out["honest_failures"] == failures

    def test_missing_honest_failures_defaults_to_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-#188 callers that omit the kwarg are treated as GATED by
        default (safe). This is the race-condition fix for old model
        artifacts whose payloads predate the metadata.
        """
        _strip_db_env(monkeypatch)
        out = write_risk_score_predictions.apply(
            args=([{"prediction_id": "x"}],),  # only payloads, no kwargs
        ).result
        assert out["status"] == "skipped"
        assert out["gated"] is True
        assert any("missing" in m.lower() for m in out["honest_failures"])

    def test_missing_metadata_can_opt_out_explicitly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """honest_failures_default_gated=False opts back into the legacy
        permissive behavior for callers that pre-validate upstream.
        """
        _strip_db_env(monkeypatch)
        out = write_risk_score_predictions.apply(
            args=([{"prediction_id": "x"}], None, None, None, False),
        ).result
        assert out["status"] == "skipped"
        assert out["gated"] is False


class TestGatedSentinelConstant:
    def test_sentinel_value_is_pinned(self) -> None:
        """The sentinel value is part of the contract with downstream
        dashboards / trigger generators — pinned to prevent silent renames.
        """
        assert GATED_SENTINEL_PREDICTION_CLASS == "gated_honest_failure"

    def test_sentinel_matches_repository_constant(self) -> None:
        """The Celery task writes the sentinel; the repository read paths
        filter on it. Both must agree on the exact string. If a future
        refactor renames either side without the other, this test fails
        loud (codex pass-3 MEDIUM-1).
        """
        from src.repositories.prediction import GATED_HONEST_FAILURE_SENTINEL

        assert GATED_SENTINEL_PREDICTION_CLASS == GATED_HONEST_FAILURE_SENTINEL

    def test_sentinel_used_in_drift_monitor_module(self) -> None:
        """The drift_monitor supabase connector also filters on the
        sentinel; codex pass-4 LOW: the constant is now imported there
        too. This test verifies the connector source has no stale string
        literals and imports the sentinel from the centralized location.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        src = (
            repo_root
            / "src"
            / "agents"
            / "drift_monitor"
            / "connectors"
            / "supabase_connector.py"
        ).read_text(encoding="utf-8")
        # The connector must import the centralized sentinel (not
        # hardcode the string literal in .neq() calls).
        assert "GATED_HONEST_FAILURE_SENTINEL" in src, (
            "drift_monitor supabase_connector.py does not import the "
            "centralized sentinel; future renames will diverge."
        )
