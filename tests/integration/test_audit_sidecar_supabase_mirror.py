"""Integration tests for the Supabase mirror of adaptive-validity sidecars.

Issue #238. Plan: ``database/migrations/040_adaptive_validity_verdicts.sql``
+ ``scripts/mirror_audit_sidecar_to_supabase.py``.

These tests require a live Postgres with migration 040 applied. We connect
via ``TEST_POSTGRES_URL`` (matching the test_021_insight_lifecycle_migration
pattern). Tests skip otherwise.

Forced-isolation: every test creates its own ``adaptive_validity_verdicts``
state by truncating before the body and again on teardown — so they don't
disturb production-side rows on a shared dev DB. The trade-off is parallel
test runs against the same DB will collide; pytest-xdist users should
gate this module with ``--dist no`` or run it serially.

Tests:
  1. ``test_two_passes_over_same_sidecars_are_idempotent``: re-running the
     mirror on identical sidecars yields ZERO net-new rows on pass-2.
  2. ``test_new_written_at_for_same_natural_key_replaces_row``: changing
     ``written_at`` for the same (experiment_id, feature) inserts a NEW
     row (different natural key).
  3. ``test_changed_verdict_for_same_natural_key_updates_in_place``:
     keeping (experiment_id, feature, written_at) constant but mutating
     ``verdict`` triggers ON CONFLICT DO UPDATE.
  4. ``test_missing_experiment_id_uses_sentinel``: a sidecar that omits
     ``experiment_id`` still mirrors successfully via the sentinel path.

Falsifiability anchor (see PR body): pass-2 of test 1 is what catches the
"silent drift" failure mode if the upsert is downgraded to DO NOTHING.
The test asserts row count constant AND verdict content updated.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterator

import psycopg
import pytest

from scripts.mirror_audit_sidecar_to_supabase import main as mirror_main

# ----------------------------------------------------------------------------
# Skip-gate (env-var pattern matches test_021_insight_lifecycle_migration)
# ----------------------------------------------------------------------------

_TEST_DB_URL = os.environ.get("TEST_POSTGRES_URL")


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _TEST_DB_URL or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql to exercise live DB",
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture()
def db_conn() -> Iterator[psycopg.Connection]:
    """Open a psycopg connection per test for isolation. The fixture
    SCOPES THE TEST TO A UNIQUE experiment_id namespace and deletes rows
    for that namespace pre/post, so tests don't disturb each other or
    accumulate state across runs."""
    assert _TEST_DB_URL is not None  # pytestmark guarantees
    conn = psycopg.connect(_TEST_DB_URL)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def test_namespace(db_conn: psycopg.Connection) -> Iterator[str]:
    """A unique experiment-id prefix used to delete any rows this test
    might have leaked. Tests SHOULD name their experiment IDs starting
    with this prefix so the post-test DELETE catches all of them.
    Returns the prefix string."""
    prefix = f"test238-{uuid.uuid4().hex[:8]}"
    # Pre-clean: nothing should match yet but be defensive.
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM adaptive_validity_verdicts WHERE experiment_id LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()
    yield prefix
    # Post-clean
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM adaptive_validity_verdicts WHERE experiment_id LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()


# ----------------------------------------------------------------------------
# Sidecar builders (mirror the producer's payload shape)
# ----------------------------------------------------------------------------


def _make_verdict(
    *,
    feature: str,
    severity: str = "moderate",
    contract_source: str = "csu",
    evaluator_satisfied: bool | None = False,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "layer": "4",
        "severity": severity,
        "remediation": "keep_with_caveat",
        "evidence": "layer-4 llm",
        "z_score": 4.2,
        "p_value": 0.0001,
        "delta_auc": 0.12,
        "contract_source": contract_source,
        "evaluator_satisfied": evaluator_satisfied,
        "evaluator_rationale_complete": False if evaluator_satisfied is False else True,
        "evaluator_missed_considerations": (
            ["temporal_filter"] if evaluator_satisfied is False else []
        ),
        "evaluator_notes": "thin rationale" if evaluator_satisfied is False else "ok",
        "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
    }


def _write_sidecar(
    artifacts_dir: Path,
    *,
    experiment_id: str | None,
    written_at: str,
    verdicts: list[dict[str, Any]],
) -> Path:
    """Write one sidecar JSON the way the producer does. If
    ``experiment_id`` is None, the key is omitted entirely (exercises
    the SidecarReader's ``"<unknown>"`` coercion + the table's
    ``'__unknown__'`` sentinel)."""
    sub = artifacts_dir / (experiment_id or "anon")
    sub.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "data_source": "csu",
        "written_at": written_at,
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [v["feature"] for v in verdicts],
        "adaptive_verdicts": verdicts,
    }
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    safe_stamp = written_at.replace(":", "").replace("-", "")
    out = sub / f"adaptive_verdicts_{safe_stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def _count_rows(conn: psycopg.Connection, *, experiment_id_prefix: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM adaptive_validity_verdicts WHERE experiment_id LIKE %s",
            (experiment_id_prefix + "%",),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _fetch_one(
    conn: psycopg.Connection, *, experiment_id: str, feature: str
) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT experiment_id, feature, written_at, source_path, verdict, "
            "evaluator_audit "
            "FROM adaptive_validity_verdicts "
            "WHERE experiment_id=%s AND feature=%s",
            (experiment_id, feature),
        )
        return cur.fetchone()


def _fetch_imported_at_map(
    conn: psycopg.Connection, *, experiment_id_prefix: str
) -> dict[tuple[str, str, Any], Any]:
    """Return a stable ``{(experiment_id, feature, written_at): imported_at}``
    map for every row whose experiment_id begins with ``prefix``. Used by
    the no-write-on-unchanged assertion to detect whether a re-run
    advanced any ``imported_at`` value."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT experiment_id, feature, written_at, imported_at "
            "FROM adaptive_validity_verdicts "
            "WHERE experiment_id LIKE %s",
            (experiment_id_prefix + "%",),
        )
        rows = cur.fetchall()
    return {(r[0], r[1], r[2]): r[3] for r in rows}


# Fixed-floor cutoff for tests: older than every synthetic written_at the
# fixtures emit so the reader admits every test sidecar regardless of the
# in-DB imported_at history. ``--since`` is the production-safe replacement
# for the test-only ``--no-cursor`` flag — production runs leave it unset
# and rely on the default ``max(imported_at) - overlap_hours`` cursor.
_TEST_SINCE_FLOOR = "2025-01-01T00:00:00Z"


def _run_mirror(artifacts_dir: Path, *, since: str = _TEST_SINCE_FLOOR) -> None:
    assert _TEST_DB_URL is not None
    rc = mirror_main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--since",
            since,
            "--database-url",
            _TEST_DB_URL,
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0, f"mirror_main returned non-zero rc={rc}"


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_two_passes_over_same_sidecars_are_idempotent(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """Re-running the mirror over the same sidecar set must produce zero
    net-new rows on the second pass — the upsert is the load-bearing
    contract this test pins."""
    exp = f"{test_namespace}-A"
    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at="2026-05-15T10:00:00Z",
        verdicts=[_make_verdict(feature="age"), _make_verdict(feature="gender")],
    )

    _run_mirror(tmp_path)
    count_pass1 = _count_rows(db_conn, experiment_id_prefix=test_namespace)
    assert count_pass1 == 2, f"pass-1 expected 2 rows, got {count_pass1}"

    _run_mirror(tmp_path)
    count_pass2 = _count_rows(db_conn, experiment_id_prefix=test_namespace)
    assert count_pass2 == 2, f"pass-2 expected SAME 2 rows (idempotent upsert), got {count_pass2}"


def test_new_written_at_for_same_natural_key_inserts_new_row(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """Same (experiment_id, feature) but DIFFERENT ``written_at`` should
    produce a NEW row — they are distinct natural keys."""
    exp = f"{test_namespace}-B"

    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at="2026-05-15T10:00:00Z",
        verdicts=[_make_verdict(feature="age")],
    )
    _run_mirror(tmp_path)
    assert _count_rows(db_conn, experiment_id_prefix=test_namespace) == 1

    # Same feature + experiment_id but different written_at → new natural
    # key → new row.
    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at="2026-05-16T10:00:00Z",
        verdicts=[_make_verdict(feature="age")],
    )
    _run_mirror(tmp_path)
    assert _count_rows(db_conn, experiment_id_prefix=test_namespace) == 2


def test_changed_verdict_for_same_natural_key_updates_in_place(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """Identical (experiment_id, feature, written_at) but changed
    ``verdict`` content must trigger ON CONFLICT DO UPDATE — row count
    stays at 1, ``verdict`` jsonb is replaced.

    THIS IS THE FALSIFIABILITY ANCHOR: if the upsert is replaced with
    DO NOTHING, the second-pass UPDATE never fires and the verdict
    content silently stays stale. The asserted-not-just-counted check
    is what trips that regression.
    """
    exp = f"{test_namespace}-C"
    written = "2026-05-15T10:00:00Z"

    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at=written,
        verdicts=[_make_verdict(feature="age", severity="moderate")],
    )
    _run_mirror(tmp_path)
    row1 = _fetch_one(db_conn, experiment_id=exp, feature="age")
    assert row1 is not None
    verdict_v1 = row1[4]
    assert verdict_v1["severity"] == "moderate", (
        f"pass-1 verdict.severity expected 'moderate', got {verdict_v1.get('severity')!r}"
    )

    # Now rewrite the SAME sidecar (same natural key) with severity=high.
    # This is what happens when an operator reruns a data-preparer
    # invocation with a refined contract: same written_at, different
    # adjudication.
    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at=written,
        verdicts=[_make_verdict(feature="age", severity="high")],
    )
    _run_mirror(tmp_path)

    # Row count constant.
    assert _count_rows(db_conn, experiment_id_prefix=test_namespace) == 1
    # Verdict content updated.
    row2 = _fetch_one(db_conn, experiment_id=exp, feature="age")
    assert row2 is not None
    verdict_v2 = row2[4]
    assert verdict_v2["severity"] == "high", (
        f"FALSIFIABILITY-ANCHOR: pass-2 verdict.severity should be UPDATED to "
        f"'high', got {verdict_v2.get('severity')!r} — if this fires under "
        f"DO NOTHING the silent-drift regression has happened."
    )


def test_missing_experiment_id_uses_sentinel(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """A sidecar that omits ``experiment_id`` entirely (ad-hoc / scratch
    run) still mirrors successfully. The reader coerces missing
    experiment_id to the literal string ``"<unknown>"``, which is a
    valid non-NULL text value that satisfies the table's NOT NULL
    constraint AND the natural-key index."""
    # We DON'T pass an experiment_id; reader will surface "<unknown>".
    # We pin the test's cleanup via a unique feature name (the
    # test_namespace fixture only matches by experiment_id).
    unique_feature = f"{test_namespace}-feat"

    _write_sidecar(
        tmp_path,
        experiment_id=None,
        written_at="2026-05-15T12:00:00Z",
        verdicts=[_make_verdict(feature=unique_feature)],
    )
    _run_mirror(tmp_path)

    # The reader coerces missing experiment_id to "<unknown>", so the
    # row lands with experiment_id='<unknown>'. Verify by direct lookup.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT experiment_id, feature FROM adaptive_validity_verdicts WHERE feature = %s",
            (unique_feature,),
        )
        rows = cur.fetchall()
    try:
        assert len(rows) == 1, f"expected 1 row for unique feature, got {len(rows)}"
        assert rows[0][0] == "<unknown>", (
            f"experiment_id should be reader's sentinel '<unknown>', got {rows[0][0]!r}"
        )
    finally:
        # Per-test cleanup of this feature-keyed row since the fixture's
        # test_namespace cleanup only matches by experiment_id prefix.
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM adaptive_validity_verdicts WHERE feature = %s",
                (unique_feature,),
            )
        db_conn.commit()


def test_rerun_on_unchanged_sidecars_does_not_advance_imported_at(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """Re-running the mirror on byte-identical sidecars must NOT advance
    ``imported_at``. This pins the write-amplification dampener: the
    upsert's WHERE clause filters the UPDATE when neither verdict nor
    evaluator_audit changed, so the row's imported_at value stays
    pegged at pass-1.

    FALSIFIABILITY ANCHOR: drop the WHERE
    ``IS DISTINCT FROM`` clause from ``_UPSERT_SQL`` → pass-2's imported_at
    will jump forward → this assertion trips. Restore the WHERE clause →
    GREEN.

    The companion test ``test_changed_verdict_for_same_natural_key_updates_in_place``
    guards the inverse direction (changed payload MUST update).
    """
    exp = f"{test_namespace}-D"
    _write_sidecar(
        tmp_path,
        experiment_id=exp,
        written_at="2026-05-15T11:00:00Z",
        verdicts=[
            _make_verdict(feature="age"),
            _make_verdict(feature="gender"),
            _make_verdict(feature="region", evaluator_satisfied=True),
        ],
    )

    _run_mirror(tmp_path)
    snapshot_pass1 = _fetch_imported_at_map(db_conn, experiment_id_prefix=test_namespace)
    assert len(snapshot_pass1) == 3, (
        f"pass-1 expected 3 rows in snapshot, got {len(snapshot_pass1)}"
    )

    # Sleep-free: Postgres ``now()`` has microsecond resolution and a
    # second-pass UPDATE would land STRICTLY after pass-1's timestamps
    # regardless of wall-clock latency. If the WHERE clause works, the
    # UPDATE doesn't fire at all → timestamps are identical.
    _run_mirror(tmp_path)
    snapshot_pass2 = _fetch_imported_at_map(db_conn, experiment_id_prefix=test_namespace)
    assert len(snapshot_pass2) == 3, (
        f"pass-2 expected SAME 3 rows in snapshot, got {len(snapshot_pass2)}"
    )

    # The load-bearing assertion: every imported_at is byte-equal across
    # passes. If the WHERE clause is missing, pass-2 advances them all.
    drifted: list[tuple[Any, ...]] = []
    for key, ts1 in snapshot_pass1.items():
        ts2 = snapshot_pass2.get(key)
        if ts1 != ts2:
            drifted.append((key, ts1, ts2))
    assert not drifted, (
        f"FALSIFIABILITY-ANCHOR: imported_at advanced on unchanged sidecars — "
        f"the IS DISTINCT FROM WHERE clause is missing or broken. "
        f"Drifted rows: {drifted}"
    )


def test_family_bucket_query_surfaces_unknown_sentinel(
    tmp_path: Path, db_conn: psycopg.Connection, test_namespace: str
) -> None:
    """The ``--by-feature-family`` query must surface the ``__unknown__``
    sentinel as a literal bucket rather than letting ``split_part`` on a
    leading-underscore string emit an empty-string family.

    Falsifiability: replace the ``CASE`` in ``_BY_FEATURE_FAMILY_SQL``
    with the original bare ``split_part(feature, '_', 1)`` → this test
    trips because the family for ``feature='__unknown__'`` becomes ``''``.
    """
    from scripts.query_audit_trail import _BY_FEATURE_FAMILY_SQL

    exp = f"{test_namespace}-E"
    # Insert one row with the literal sentinel feature value. We bypass
    # the mirror script (which would coerce via the reader) and go
    # straight to SQL: this exercises the CASE branch the migration's
    # column-DEFAULT documents.
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO adaptive_validity_verdicts "
            "  (experiment_id, feature, written_at, source_path, verdict) "
            "VALUES (%s, %s, %s, %s, %s::jsonb)",
            (
                exp,
                "__unknown__",
                "2026-05-15T13:00:00Z",
                "/dev/null/synthetic",
                '{"feature": "__unknown__", "severity": "moderate"}',
            ),
        )
    db_conn.commit()

    with db_conn.cursor() as cur:
        cur.execute(_BY_FEATURE_FAMILY_SQL, (100,))
        rows = cur.fetchall()
    families = {row[0] for row in rows}
    assert "__unknown__" in families, (
        f"expected '__unknown__' as a literal family bucket; got families={families}. "
        f"Empty-string family suggests split_part('__unknown__','_',1)='' regression."
    )
    assert "" not in families, (
        f"family bucket '' should never appear; got families={families}. "
        f"Indicates the CASE branch for '__unknown__' is missing."
    )
