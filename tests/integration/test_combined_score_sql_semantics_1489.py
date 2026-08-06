"""Migration 034 executed against real PostgreSQL (#1489 deferral 4).

Text assertions cannot tell you what plpgsql DOES. These tests load the real
migration into a THROWAWAY database and execute it, then assert the SQL agrees
with ``src/agents/feedback_learner/ragas_scoring.py`` — the Python source of
truth — on the bundle shapes #1485/#1488 made normal.

Nothing here touches the live schema: each run creates its own uniquely-named
database, loads migration 022's prerequisites plus 034, and drops it again.

Run with::

    E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
        tests/integration/test_combined_score_sql_semantics_1489.py

Measured on PostgreSQL 15.8 BEFORE 034 (these are the RED values the migration
fixes; every one of them is reproduced as a regression assertion below):

    partial + perfect bundle, rubric 5.0   0.78   (python: 1.0)
    complete bundle, rubric NULL           0.40   (python: None)
    empty bundle, rubric 5.0               0.60   (python: None)
    determine_improvement_priority(NULL)   'critical'
    determine_improvement_type(NULL, .75)  'workflow'
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, List, Optional

import pytest

from src.agents.feedback_learner.ragas_scoring import RagasBundle, combined_score

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_DOCKER = shutil.which("docker") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_DOCKER),
        reason=(
            "real-PostgreSQL semantics test; set E2I_DB_INTEGRATION=1 with the "
            "supabase-db container reachable via docker"
        ),
    ),
]

REPO_ROOT = Path(__file__).parent.parent.parent
DATABASE_ML = REPO_ROOT / "database" / "ml"
CONTAINER = os.environ.get("E2I_SUPABASE_DB_CONTAINER", "supabase-db")

# The prerequisites migration 022's helpers assume already exist. Kept minimal
# and local so the test never depends on the live schema.
_PREREQUISITE_DDL = """
CREATE TYPE improvement_type AS ENUM ('retrieval','prompt','workflow','none');
CREATE TYPE improvement_priority AS ENUM ('critical','high','medium','low','none');
CREATE TABLE learning_signals (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_value FLOAT,
    signal_details JSONB DEFAULT '{}'::jsonb,
    ragas_scores JSONB DEFAULT '{}'::jsonb,
    ragas_weighted FLOAT,
    rubric_scores JSONB DEFAULT '{}'::jsonb,
    rubric_total FLOAT,
    rubric_weighted FLOAT,
    combined_score FLOAT,
    improvement_type improvement_type,
    improvement_priority improvement_priority,
    processed_at TIMESTAMPTZ
);
"""


def _psql(database: str, sql: str, *, tuples_only: bool = True) -> str:
    """Run SQL in the supabase-db container. Raises on any SQL error."""
    args = [
        "docker",
        "exec",
        "-i",
        CONTAINER,
        "psql",
        "-U",
        "postgres",
        "-d",
        database,
        "-v",
        "ON_ERROR_STOP=1",
    ]
    if tuples_only:
        args += ["-tA"]
    else:
        args += ["-q"]
    result = subprocess.run(args, input=sql, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(f"psql failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def _extract_022_helpers() -> str:
    """Migration 022's SECTION 5 helper functions, verbatim.

    Sliced out rather than restated so the fixture cannot drift from the
    migration it is standing in for.
    """
    sql = (DATABASE_ML / "022_self_improvement_tables.sql").read_text()
    start = sql.index("CREATE OR REPLACE FUNCTION calculate_combined_score")
    end = sql.index("COMMENT ON FUNCTION update_learning_signal_evaluation")
    return sql[start:end]


@pytest.fixture(scope="module")
def scratch_db():
    """A throwaway database carrying 022's helpers and then migration 034."""
    name = f"lane1489_{uuid.uuid4().hex[:12]}"
    _psql("postgres", f"CREATE DATABASE {name};", tuples_only=False)
    try:
        _psql(name, _PREREQUISITE_DDL, tuples_only=False)
        _psql(name, _extract_022_helpers(), tuples_only=False)
        _psql(
            name,
            (DATABASE_ML / "034_combined_score_partial_bundles.sql").read_text(),
            tuples_only=False,
        )
        yield name
    finally:
        _psql("postgres", f"DROP DATABASE IF EXISTS {name} WITH (FORCE);", tuples_only=False)


def _sql_combined(
    db: str, scores: Optional[dict], rubric_total: Optional[float]
) -> Optional[float]:
    scores_literal = "NULL::jsonb" if scores is None else f"'{json.dumps(scores)}'::jsonb"
    rubric_literal = "NULL::float" if rubric_total is None else f"{rubric_total}::float"
    out = _psql(db, f"SELECT calculate_combined_score({scores_literal}, {rubric_literal});")
    return None if out == "" else float(out)


def _python_combined(scores: dict, rubric_total: Optional[float]) -> Optional[float]:
    bundle = RagasBundle(scores=scores)
    return combined_score(bundle.weighted, rubric_total)


class TestSqlAgreesWithPythonSourceOfTruth:
    """The whole point of the fix: one blend, two implementations, same number."""

    @pytest.mark.parametrize(
        "scores,rubric_total",
        [
            # Complete bundle — 022 already agreed here (weights sum to 1).
            (
                {
                    "faithfulness": 0.80,
                    "answer_relevancy": 0.40,
                    "context_precision": 0.90,
                    "context_recall": 0.70,
                    "answer_correctness": 0.60,
                },
                4.0,
            ),
            # The #1485 real-pipeline shape: only two metrics are ever reported.
            ({"faithfulness": 1.0, "answer_relevancy": 1.0}, 5.0),
            ({"faithfulness": 0.524, "answer_relevancy": 0.179}, 4.0),
            # A single measured metric.
            ({"faithfulness": 0.9}, 3.0),
            # #1488 shape: the judge tried and failed, so the key is null-valued
            # rather than absent. Both must read it as unmeasured.
            ({"faithfulness": None, "answer_relevancy": 0.9}, 4.0),
        ],
    )
    def test_combined_score_matches(self, scratch_db, scores, rubric_total):
        assert _sql_combined(scratch_db, scores, rubric_total) == pytest.approx(
            _python_combined(scores, rubric_total), abs=1e-4
        )


class TestRefusesToPublishAHalfThatDoesNotExist:
    def test_ragas_only_row_is_null_not_forty_percent_of_nothing(self, scratch_db):
        """022 returned 0.40 here — 40% of a perfect RAGAS half blended with a
        rubric that was never measured."""
        perfect = dict.fromkeys(
            (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
            ),
            1.0,
        )
        assert _sql_combined(scratch_db, perfect, None) is None
        assert _python_combined(perfect, None) is None

    def test_rubric_only_row_is_null(self, scratch_db):
        """022 returned 0.60 for an EMPTY bundle with a perfect rubric."""
        assert _sql_combined(scratch_db, {}, 5.0) is None

    def test_all_unmeasured_bundle_is_null(self, scratch_db):
        assert _sql_combined(scratch_db, {"faithfulness": None}, 5.0) is None


class TestRoutingHelpersAreNullSafe:
    def test_priority_of_null_is_null_not_critical(self, scratch_db):
        """022 fell through to ELSE and returned 'critical' — the worst verdict
        in the vocabulary, fabricated from an absent measurement."""
        assert _psql(scratch_db, "SELECT determine_improvement_priority(NULL);") == ""

    def test_type_of_null_is_null_not_workflow(self, scratch_db):
        assert _psql(scratch_db, "SELECT determine_improvement_type(NULL, 0.75);") == ""
        assert _psql(scratch_db, "SELECT determine_improvement_type(0.9, NULL);") == ""

    def test_finite_inputs_still_route_as_022_did(self, scratch_db):
        """The fix must not move the routing boundaries for measured rows."""
        rows = _psql(
            scratch_db,
            "SELECT determine_improvement_priority(0.90), determine_improvement_priority(0.75), "
            "determine_improvement_priority(0.60), determine_improvement_priority(0.45), "
            "determine_improvement_priority(0.20);",
        )
        assert rows.split("|") == ["none", "low", "medium", "high", "critical"]


class TestPartialBundleNoLongerMisroutes:
    def test_perfect_partial_row_is_not_blamed_on_retrieval(self, scratch_db):
        """Measured on 022: a bundle whose every judged metric was 1.0 scored
        0.45 after COALESCE, landing under the 0.7 retrieval threshold, so the
        row was routed to 'retrieval' — tune k, chunks, RRF weights — for a
        retrieval that had served it perfectly."""
        signal_id = str(uuid.uuid4())
        _psql(
            scratch_db,
            f"INSERT INTO learning_signals (signal_id) VALUES ('{signal_id}');",
            tuples_only=False,
        )
        _psql(
            scratch_db,
            "SELECT update_learning_signal_evaluation("
            f"'{signal_id}'::uuid, '{{\"faithfulness\":1.0,\"answer_relevancy\":1.0}}'::jsonb, "
            "'{}'::jsonb, 5.0);",
        )
        row = _psql(
            scratch_db,
            "SELECT improvement_type, improvement_priority, ragas_weighted, combined_score "
            f"FROM learning_signals WHERE signal_id = '{signal_id}';",
        ).split("|")
        assert row[0] == "none", f"still misrouted: {row}"
        assert row[1] == "none"
        assert float(row[2]) == pytest.approx(1.0, abs=1e-6)
        assert float(row[3]) == pytest.approx(1.0, abs=1e-4)


class TestNoStaleCompanionComponents:
    def test_signal_value_and_coverage_follow_the_update(self, scratch_db):
        """Measured on 022: after updating a row the #1487 writer had inserted,
        signal_value still read 4.0 while rubric_total read 2.0, and
        ragas_coverage.measured still named two metrics after the stored bundle
        had been replaced by a one-metric one."""
        signal_id = str(uuid.uuid4())
        _psql(
            scratch_db,
            "INSERT INTO learning_signals (signal_id, signal_value, signal_details, "
            "ragas_scores, ragas_weighted, rubric_total, combined_score) VALUES ("
            f"'{signal_id}', 4.0, "
            '\'{"ragas_coverage": {"measured": ["faithfulness", "answer_relevancy"], '
            '"evaluation_model": "gpt-4o"}}\'::jsonb, '
            '\'{"faithfulness":0.524,"answer_relevancy":0.179}\'::jsonb, 0.3707, 4.0, 0.5983);',
            tuples_only=False,
        )
        _psql(
            scratch_db,
            "SELECT update_learning_signal_evaluation("
            f"'{signal_id}'::uuid, '{{\"faithfulness\":0.90}}'::jsonb, "
            '\'{"causal_validity":{"score":2}}\'::jsonb, 2.0);',
        )
        signal_value, rubric_total, coverage = _psql(
            scratch_db,
            "SELECT signal_value, rubric_total, signal_details->'ragas_coverage' "
            f"FROM learning_signals WHERE signal_id = '{signal_id}';",
        ).split("|")

        assert float(signal_value) == float(rubric_total) == 2.0, (
            "signal_value must not keep the rubric score the update replaced"
        )
        measured: List[Any] = json.loads(coverage)["measured"]
        assert measured == ["faithfulness"], (
            f"ragas_coverage still describes a bundle the row no longer holds: {measured}"
        )
