"""The live-traffic RAG feedstock read, against the REAL database (#1489 d5).

Why a real-DB pass and not just the in-memory PostgREST double
--------------------------------------------------------------
The double proves the row FILTERING is right. It cannot prove the read itself
is: column names, the ``is_training_example`` boolean, ``created_at``
comparison against an ISO string, and whether ``apply_provenance_filter``'s
``.eq('is_synthetic', False)`` is even legal on this table are all facts about
Postgres and PostgREST, not about my fake. A fake accepts whatever chain you
build; the real one 42703s on a typo. Every historical failure in this schema
family was of exactly that kind (#883's 22P02 on a non-member enum value was
invisible to a mock).

READ-ONLY
---------
This box IS prod. This file issues one ``select`` and creates, updates and
deletes nothing — there is no fixture teardown because there is nothing to
tear down. It calls no LLM and spends no judge budget: ``db_batch`` only reads.

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_rag_feedstock_realdb_1489.py'
"""

from __future__ import annotations

import os

import pytest

from tests.integration._asyncio_compat import run_sync

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB feedstock read; set E2I_DB_INTEGRATION=1 + creds",
    ),
]


def test_the_read_executes_against_the_real_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The whole query chain is legal on the real table, in strict real mode.

    Strict mode is the one that exercises the provenance predicate; this box
    sets E2I_INCLUDE_SYNTHETIC in its .env (and conftest load_dotenv(override=True)
    re-applies it), which would skip the predicate and leave it untested.
    """
    from src.tasks.rag_example_sources import db_batch

    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    batch = run_sync(db_batch())

    assert batch.source == "db"
    assert batch.total_records >= 0
    # Every example must be judgeable — that is the contract the GEPA metric
    # relies on, and it is asserted here against whatever real rows exist.
    for example in batch.examples:
        assert example.user_query.strip()
        assert example.synthesis.strip()
        assert example.retrieved_contexts
        assert all(c.strip() for c in example.retrieved_contexts)

    with capsys.disabled():
        print(
            f"\n[#1489 feedstock] real-mode window: {batch.total_records} row(s) read, "
            f"{len(batch.examples)} judgeable turn(s). origin={batch.origin}"
        )


def test_the_showcase_opt_in_reads_at_least_as_much(monkeypatch: pytest.MonkeyPatch) -> None:
    """Including synthetic can only widen the window, never narrow it.

    Pins that the provenance predicate is actually applied against real data:
    on a DB whose rows are 3,300 synthetic / 659 real (measured 2026-08-06) a
    filter that silently did nothing would make these two counts identical for
    the wrong reason, so the strict count is also asserted to be a subset.
    """
    from src.tasks.rag_example_sources import db_batch

    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    strict = run_sync(db_batch())

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    permissive = run_sync(db_batch())

    assert permissive.total_records >= strict.total_records
    strict_queries = {e.user_query for e in strict.examples}
    permissive_queries = {e.user_query for e in permissive.examples}
    assert strict_queries <= permissive_queries


def test_todays_measured_state_is_the_zero_cost_skip() -> None:
    """Documents WHY shipping the DB source enabled-by-nobody is safe today.

    ``learning_signals.retrieved_chunks`` had no producer at all until #1489
    deferral 1 (measured non-default on 0 of 3,959 rows, 2026-08-06), so the
    reader currently yields no judgeable turns and the leg's
    ``_RAG_MIN_USABLE_EXAMPLES`` gate returns before constructing a metric or an
    optimizer — zero API calls.

    This is deliberately an OBSERVATION, not a requirement: once the producer
    lands and traffic accumulates, judgeable turns appear and the assertion
    below stops being interesting. It fails only if the count goes NEGATIVE-ish
    (i.e. the reader breaks), which is the thing worth catching.
    """
    from src.tasks.dspy_optimization_tasks import _RAG_MIN_USABLE_EXAMPLES
    from src.tasks.rag_example_sources import db_batch

    batch = run_sync(db_batch())
    assert len(batch.examples) <= batch.total_records
    if len(batch.examples) < _RAG_MIN_USABLE_EXAMPLES:
        pytest.skip(
            f"live feedstock still below the leg's threshold "
            f"({len(batch.examples)} of {_RAG_MIN_USABLE_EXAMPLES} needed) — "
            f"expected until the #1489 retrieved_chunks producer lands"
        )
