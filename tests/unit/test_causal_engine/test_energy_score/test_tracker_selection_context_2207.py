"""#2207: EnergyScoreMLflowTracker can actually land rows on the live schema.

Census finding (2026-09-22): ``estimator_evaluations`` (causal/011, 220df4cec) had a
writer — ``EnergyScoreMLflowTracker._log_to_database`` — never instantiated outside its
module (0 rows). Reading the writer against the live table showed it could not have
succeeded even if called:

* ``experiment_id`` was ``NOT NULL REFERENCES ml_experiments(id)`` while the tracker
  passes an MLflow experiment id (an int string, not a uuid) or a ``uuid4()`` that no
  ml_experiments row carries — every insert would have failed. Migration causal/012
  makes it nullable and adds the query-time context; the tracker must write NULL
  there, never a fabricated id.
* the DSN came from ``DATABASE_URL`` only, which the api/worker containers do not set;
  they set ``SUPABASE_DB_URL`` (measured 2026-09-22).
* the live energy-score path is the causal_impact estimation node — a query-time path
  with no MLflow run; the tracker needs a way to record a selection WITHOUT opening
  an MLflow run (``mlflow.start_run`` against an unreachable server retries for
  minutes, on the chat request path).
"""

from __future__ import annotations

import os
import uuid
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from src.causal_engine.energy_score.estimator_selector import (
    EstimatorResult,
    EstimatorType,
    SelectionResult,
    SelectionStrategy,
)
from src.causal_engine.energy_score.mlflow_tracker import EnergyScoreMLflowTracker


@pytest.fixture()
def selection_result() -> SelectionResult:
    ok = EstimatorResult(
        estimator_type=EstimatorType.CAUSAL_FOREST,
        success=True,
        ate=0.12,
        ate_std=0.03,
        ate_ci_lower=0.06,
        ate_ci_upper=0.18,
        estimation_time_ms=400.0,
    )
    failed = EstimatorResult(
        estimator_type=EstimatorType.LINEAR_DML,
        success=False,
        error_message="singular matrix",
        error_type="LinAlgError",
        estimation_time_ms=50.0,
    )
    return SelectionResult(
        selected=ok,
        selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
        all_results=[ok, failed],
        selection_reason="only survivor",
        total_time_ms=450.0,
        energy_scores={"causal_forest": 0.42},
        energy_score_gap=0.0,
    )


class _FakeCursor:
    def __init__(self, sink: List[tuple]):
        self._sink = sink

    def execute(self, sql: str, params: Any = None):
        self._sink.append((sql, params))

    def close(self):
        pass


class _FakeConn:
    def __init__(self, sink: List[tuple]):
        self._sink = sink
        self.committed = False

    def cursor(self):
        return _FakeCursor(self._sink)

    def commit(self):
        self.committed = True

    def close(self):
        pass


@pytest.fixture()
def fake_pg():
    sink: List[tuple] = []
    conns: List[_FakeConn] = []

    def _connect(dsn, **kwargs):
        c = _FakeConn(sink)
        c.dsn = dsn  # type: ignore[attr-defined]
        c.kwargs = kwargs  # type: ignore[attr-defined]
        conns.append(c)
        return c

    with patch("psycopg2.connect", side_effect=_connect):
        yield {"sink": sink, "conns": conns}


def _columns(sql: str) -> List[str]:
    head = sql.split("INSERT INTO estimator_evaluations", 1)[1].split("VALUES", 1)[0]
    return [c.strip() for c in head.strip().strip("()").replace("\n", " ").split(",")]


def _param(sql: str, params: tuple, column: str):
    return dict(zip(_columns(sql), params, strict=True))[column]


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_evaluations_writes_one_row_per_estimator_with_query_context(
    fake_pg, selection_result
):
    tracker = EnergyScoreMLflowTracker(
        enable_db_logging=True, enable_mlflow=False, db_connection_string="postgresql://x"
    )
    run_id = tracker.record_evaluations(
        selection_result,
        query_id="q-1",
        session_id="s-1",
        treatment="hcp_engagement",
        outcome="conversion",
        brand="Kisqali",
        region="Northeast",
        data_source="synthetic",
    )

    rows = fake_pg["sink"]
    assert len(rows) == 2
    assert fake_pg["conns"][0].committed
    uuid.UUID(run_id)  # a real uuid that groups this selection's rows
    for sql, params in rows:
        assert _param(sql, params, "experiment_id") is None  # never a fabricated FK
        assert _param(sql, params, "selection_run_id") == run_id
        assert _param(sql, params, "query_id") == "q-1"
        assert _param(sql, params, "session_id") == "s-1"
        assert _param(sql, params, "treatment_variable") == "hcp_engagement"
        assert _param(sql, params, "outcome_variable") == "conversion"
        assert _param(sql, params, "brand") == "Kisqali"
        assert _param(sql, params, "region") == "Northeast"
        assert _param(sql, params, "data_source") == "synthetic"
        assert _param(sql, params, "mlflow_run_id") is None
    ok_sql, ok_params = rows[0]
    assert _param(ok_sql, ok_params, "estimator_type") == "causal_forest"
    assert _param(ok_sql, ok_params, "success") is True
    assert _param(ok_sql, ok_params, "was_selected") is True
    assert _param(ok_sql, ok_params, "ate") == 0.12
    bad_sql, bad_params = rows[1]
    assert _param(bad_sql, bad_params, "estimator_type") == "linear_dml"
    assert _param(bad_sql, bad_params, "success") is False
    assert _param(bad_sql, bad_params, "was_selected") is False
    assert _param(bad_sql, bad_params, "energy_score") is None
    assert _param(bad_sql, bad_params, "error_message") == "singular matrix"


@pytest.mark.unit
def test_log_selection_result_under_a_run_writes_null_experiment_and_the_mlflow_run(
    fake_pg, selection_result
):
    """An MLflow experiment id is not an ml_experiments uuid: it goes to NULL, and the
    run id is kept in mlflow_run_id. The no-context fallback used to invent a uuid4."""
    tracker = EnergyScoreMLflowTracker(
        enable_db_logging=True, enable_mlflow=False, db_connection_string="postgresql://x"
    )
    with tracker.start_selection_run("unit", brand="Remibrutinib"):
        tracker.log_selection_result(selection_result)
    rows = fake_pg["sink"]
    assert len(rows) == 2
    for sql, params in rows:
        assert _param(sql, params, "experiment_id") is None
        assert _param(sql, params, "brand") == "Remibrutinib"
    # and with no context at all — still no fabricated experiment id
    fake_pg["sink"].clear()
    tracker.log_selection_result(selection_result)
    assert all(_param(s, p, "experiment_id") is None for s, p in fake_pg["sink"])


@pytest.mark.unit
def test_dsn_falls_back_to_supabase_db_url():
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SUPABASE_DB_URL")}
    env["SUPABASE_DB_URL"] = "postgresql://postgres:pw@supabase-db:5432/postgres"
    with patch.dict(os.environ, env, clear=True):
        tracker = EnergyScoreMLflowTracker(enable_mlflow=False)
    assert tracker.db_connection_string == env["SUPABASE_DB_URL"]
    env["DATABASE_URL"] = "postgresql://explicit"
    with patch.dict(os.environ, env, clear=True):
        tracker = EnergyScoreMLflowTracker(enable_mlflow=False)
    assert tracker.db_connection_string == "postgresql://explicit"


@pytest.mark.unit
def test_enable_mlflow_false_never_touches_mlflow(selection_result, fake_pg):
    tracker = EnergyScoreMLflowTracker(
        enable_db_logging=True, enable_mlflow=False, db_connection_string="postgresql://x"
    )
    assert tracker._mlflow_available is False
    with patch(
        "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._log_to_mlflow"
    ) as m:
        with tracker.start_selection_run("unit"):
            tracker.log_selection_result(selection_result)
    m.assert_not_called()


@pytest.mark.unit
def test_db_failure_never_raises(selection_result):
    tracker = EnergyScoreMLflowTracker(
        enable_db_logging=True, enable_mlflow=False, db_connection_string="postgresql://x"
    )
    with patch("psycopg2.connect", side_effect=RuntimeError("db down")):
        assert tracker.record_evaluations(selection_result, query_id="q") is None


@pytest.mark.unit
def test_no_dsn_records_nothing_and_returns_none(selection_result):
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SUPABASE_DB_URL")}
    with patch.dict(os.environ, env, clear=True):
        tracker = EnergyScoreMLflowTracker(enable_db_logging=True, enable_mlflow=False)
    with patch("psycopg2.connect", new=MagicMock(side_effect=AssertionError("no connect"))):
        assert tracker.record_evaluations(selection_result, query_id="q") is None


@pytest.mark.unit
def test_migration_012_is_what_the_writer_now_depends_on():
    """The columns record_evaluations writes must exist in the migration that ships with
    it; the FK must have become nullable; the comparison view must be re-keyed."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    sql = (
        repo / "database" / "causal" / "012_estimator_evaluations_selection_context.sql"
    ).read_text()
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    assert "ALTER COLUMN experiment_id DROP NOT NULL" in body
    for col in (
        "selection_run_id",
        "query_id",
        "session_id",
        "mlflow_run_id",
        "treatment_variable",
        "outcome_variable",
        "brand",
        "region",
        "data_source",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in body, col
    assert "PARTITION BY selection_run_id" in body
    assert "PARTITION BY experiment_id" not in body
    # the writer's INSERT names exactly those columns
    import inspect

    src = inspect.getsource(EnergyScoreMLflowTracker._log_to_database)
    for col in ("selection_run_id", "query_id", "session_id", "data_source", "mlflow_run_id"):
        assert col in src, col
