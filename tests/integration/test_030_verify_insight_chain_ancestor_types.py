"""Migration 030 — verify_insight_chain ancestor-type coverage (L21 / #702).

Migration 021 wired ``verify_insight_chain`` invalidation branches for
``causal_path``, ``trigger``, and (self-check only) ``executive_insight`` — but
NOT ``ml_prediction``, even though the SAME migration added
``ml_predictions.invalidated_at`` and the cascade/invalidator treats
``ml_prediction`` and ``executive_insight`` as first-class invalidatable
artifacts. The result was a FALSE-VALID verdict: an insight derived from an
invalidated ``ml_prediction`` (or an invalidated ``executive_insight`` ancestor)
was reported valid, so the read-path verifier served stale data.

Migration 030 ``CREATE OR REPLACE``s the function to add:
  * self-check branch for ``ml_prediction``
  * ancestor-walk branches for ``ml_prediction`` AND ``executive_insight``

Static-content checks run anywhere (CI-runnable). The functional red/green test
needs ``TEST_POSTGRES_URL`` + local ``psql`` and a database carrying the 021 +
core schema; it skips otherwise. The DECISIVE evidence for this fix is the
faithful ``BEGIN…ROLLBACK`` run on the prod droplet (documented in the PR):
under the OLD function an invalidated ``ml_prediction`` ancestor returned
``is_valid=t``; under the NEW function it returns ``is_valid=f``.

Skip is via ``pytest.mark.skipif`` (NOT a self-declared "deferred") — per
feedback_verification_step_evidence_gate.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "database"
    / "memory"
    / "030_verify_insight_chain_ancestor_types.sql"
)


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


def _content() -> str:
    assert MIGRATION_PATH.exists(), f"Migration file not found at {MIGRATION_PATH}"
    return MIGRATION_PATH.read_text()


# ---------------------------------------------------------------------------
# Static-content checks (run anywhere)
# ---------------------------------------------------------------------------


def test_migration_replaces_the_function() -> None:
    """030 must CREATE OR REPLACE the verify_insight_chain function."""
    content = _content().lower()
    assert "create or replace function verify_insight_chain" in content, (
        "030 must CREATE OR REPLACE verify_insight_chain (function-body-only change)."
    )


def test_self_check_covers_ml_prediction() -> None:
    """The self-check (insight queried directly) must handle ml_prediction.

    Falsifiable: if the ml_prediction self-check branch is dropped, calling
    verify_insight_chain('<id>', 'ml_prediction') for an invalidated prediction
    would wrongly return valid.
    """
    content = _content()
    assert "ml_prediction already invalidated" in content, (
        "Missing the self-check branch for an invalidated ml_prediction "
        "(verify_insight_chain('<id>','ml_prediction'))."
    )


def test_ancestor_walk_covers_ml_prediction() -> None:
    """The ancestor walk must flag an invalidated ml_prediction ANCESTOR.

    This is the core L21 bug: an executive_insight derived_from an invalidated
    ml_prediction must verify as stale.
    """
    content = _content()
    assert "ancestor ml_prediction invalidated" in content, (
        "Missing the ancestor-walk branch for an invalidated ml_prediction."
    )


def test_ancestor_walk_covers_executive_insight() -> None:
    """The ancestor walk must flag an invalidated executive_insight ANCESTOR.

    021's self-check handled executive_insight but the ancestor walk did not, so
    an insight consolidated_from an invalidated executive_insight verified valid.
    """
    content = _content()
    assert "ancestor executive_insight invalidated" in content, (
        "Missing the ancestor-walk branch for an invalidated executive_insight."
    )


def test_all_invalidatable_ancestor_types_are_handled() -> None:
    """Completeness guard: the ancestor walk must probe every invalidatable
    artifact type the cascade/invalidator knows about.

    The walk sees source_type values; today the writers emit causal_path /
    episodic_memory, but ml_prediction / executive_insight / trigger are all
    registered as invalidatable and MUST be handled so a future edge writer
    can't silently reintroduce the false-valid gap.
    """
    content = _content()
    for artifact_type in (
        "causal_path",
        "episodic_memory",
        "trigger",
        "ml_prediction",
        "executive_insight",
    ):
        assert f"'{artifact_type}'" in content, (
            f"verify_insight_chain does not reference artifact type {artifact_type!r}"
        )


def test_migration_is_transactional_and_idempotent() -> None:
    """Wrapped in BEGIN/COMMIT and re-runnable (CREATE OR REPLACE)."""
    content = _content()
    assert "BEGIN;" in content and "COMMIT;" in content, (
        "Migration must be wrapped in an explicit BEGIN; … COMMIT; block."
    )
    # CREATE OR REPLACE FUNCTION is inherently re-runnable; assert we did NOT use
    # a bare CREATE FUNCTION (which would error on re-apply).
    assert "create function verify_insight_chain" not in content.lower(), (
        "Use CREATE OR REPLACE FUNCTION (idempotent), not a bare CREATE FUNCTION."
    )


# ---------------------------------------------------------------------------
# Functional red/green (needs TEST_POSTGRES_URL + psql + 021/core schema)
# ---------------------------------------------------------------------------

_SCENARIO_SQL = """
BEGIN;
-- S1: an executive_insight derived_from an INVALIDATED ml_prediction.
INSERT INTO ml_predictions (prediction_id, prediction_timestamp, patient_id,
                            model_type, invalidated_at, invalidation_reason)
VALUES ('T030_PRED_BAD', now(), 'PAT_T030_1', 'risk_score', now(), 'superseded (test 030)');
INSERT INTO executive_insights (insight_id, title, narrative, brand, kpi)
VALUES ('cccccccc-0000-0000-0000-000000000001', 'T030 S1', 'd', 'T030Brand', 'T030S1');
INSERT INTO insight_edges (source_type, source_id, target_type, target_id, edge_type, brand)
VALUES ('ml_prediction', 'T030_PRED_BAD', 'executive_insight',
        'cccccccc-0000-0000-0000-000000000001', 'derived_from', 'T030Brand');
SELECT 'BAD_RESULT:' || CASE WHEN is_valid THEN 'VALID' ELSE 'STALE' END
  FROM verify_insight_chain('cccccccc-0000-0000-0000-000000000001', 'executive_insight');

-- S4: control — the SAME shape but a VALID (not invalidated) prediction.
INSERT INTO ml_predictions (prediction_id, prediction_timestamp, patient_id, model_type)
VALUES ('T030_PRED_OK', now(), 'PAT_T030_2', 'risk_score');
INSERT INTO executive_insights (insight_id, title, narrative, brand, kpi)
VALUES ('cccccccc-0000-0000-0000-000000000002', 'T030 S4', 'd', 'T030Brand', 'T030S4');
INSERT INTO insight_edges (source_type, source_id, target_type, target_id, edge_type, brand)
VALUES ('ml_prediction', 'T030_PRED_OK', 'executive_insight',
        'cccccccc-0000-0000-0000-000000000002', 'derived_from', 'T030Brand');
SELECT 'OK_RESULT:' || CASE WHEN is_valid THEN 'VALID' ELSE 'STALE' END
  FROM verify_insight_chain('cccccccc-0000-0000-0000-000000000002', 'executive_insight');
ROLLBACK;
"""


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL") or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql + a DB carrying the 021/core schema",
)
def test_functional_invalidated_ml_prediction_ancestor_is_flagged() -> None:
    """Apply 030, then prove the verdict flips on an invalidated ml_prediction
    ancestor while a valid one stays valid.

    Red/green: under the OLD (021) function the invalidated case returns
    ``is_valid=t`` (the bug); after 030 it returns ``f``. The control (valid
    prediction) returns ``t`` in both, guarding against over-flagging.

    All seeding happens inside a BEGIN…ROLLBACK so the target DB is untouched.
    """
    url = os.environ["TEST_POSTGRES_URL"]

    apply_result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
        capture_output=True,
        text=True,
    )
    assert apply_result.returncode == 0, (
        f"030 apply failed:\nstderr={apply_result.stderr}\nstdout={apply_result.stdout}"
    )

    run = subprocess.run(
        ["psql", url, "-t", "-A", "-v", "ON_ERROR_STOP=1", "-c", _SCENARIO_SQL],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"scenario run failed:\nstderr={run.stderr}\nstdout={run.stdout}"
    out = run.stdout
    assert "BAD_RESULT:STALE" in out, (
        f"invalidated ml_prediction ancestor must verify as STALE (is_valid=false); got:\n{out}"
    )
    assert "OK_RESULT:VALID" in out, (
        f"valid ml_prediction ancestor must stay VALID (no over-flagging); got:\n{out}"
    )
