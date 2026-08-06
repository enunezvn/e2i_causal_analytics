"""One example source for the #1486 RAG leg, two backings (#1489 deferral 5).

Before this, the leg had exactly one feedstock: a JSON file named by
``DSPY_RAG_RECORDS_PATH`` and produced by a MANUAL ``scripts/replay_golden_set.py``
run. So the "nightly" GEPA cycle could never consume fresh records unattended —
its steady state was a permanent skip. Live traffic already writes the same
judgeable triple to ``learning_signals`` (``training_input`` = the user query,
``training_output`` = the answer, ``retrieved_chunks`` = the evidence), so the
DB is the unattended source the file source could never be.

The two are unified here behind ``load_rag_examples`` so both emit the IDENTICAL
``dspy.Example`` shape — the property that makes them interchangeable to GEPA,
and the one a "unification" that only shared a name would not deliver.

DB tests use a faithful in-memory PostgREST table (honours
select/eq/gte/order/limit/execute) so the REAL query construction and row
filtering run, per the ``test_business_metric_region_paged_931`` idiom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------
# Faithful PostgREST double
# --------------------------------------------------------------------------
class _FakeTable:
    """In-memory PostgREST-like query. Records the chain for assertions."""

    def __init__(self, rows: List[Dict[str, Any]], log: Dict[str, Any]):
        self._rows = rows
        self._log = log
        self._filters: List[tuple] = []
        self._limit: int | None = None
        self._order: tuple | None = None

    def select(self, cols: str):
        self._log["select"] = cols
        return self

    def eq(self, col: str, val: Any):
        self._filters.append(("eq", col, val))
        self._log.setdefault("eq", []).append((col, val))
        return self

    def gte(self, col: str, val: Any):
        self._filters.append(("gte", col, val))
        self._log.setdefault("gte", []).append((col, val))
        return self

    def neq(self, col: str, val: Any):
        self._filters.append(("neq", col, val))
        self._log.setdefault("neq", []).append((col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self._order = (col, desc)
        self._log["order"] = (col, desc)
        return self

    def limit(self, n: int):
        self._limit = n
        self._log["limit"] = n
        return self

    # Writers must never be reachable from a reader.
    def insert(self, *a, **k):  # pragma: no cover - guarded by test
        raise AssertionError("the feedstock reader must not write")

    def update(self, *a, **k):  # pragma: no cover - guarded by test
        raise AssertionError("the feedstock reader must not write")

    def delete(self, *a, **k):  # pragma: no cover - guarded by test
        raise AssertionError("the feedstock reader must not write")

    async def execute(self):
        rows = list(self._rows)
        for kind, col, val in self._filters:
            if kind == "eq":
                rows = [r for r in rows if r.get(col) == val]
            elif kind == "neq":
                # PostgREST compares the JSONB value; "[]" is the empty-array
                # literal, and a SQL NULL never satisfies <> (so it is excluded).
                empty = json.loads(val) if isinstance(val, str) else val
                rows = [r for r in rows if r.get(col) is not None and r.get(col) != empty]
            else:
                rows = [r for r in rows if str(r.get(col, "")) >= str(val)]
        if self._order:
            col, desc = self._order
            rows.sort(key=lambda r: str(r.get(col) or ""), reverse=desc)
        if self._limit is not None:
            rows = rows[: self._limit]
        res = MagicMock()
        res.data = rows
        return res


class _FakeClient:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.log: Dict[str, Any] = {}

    def table(self, name: str):
        self.log["table"] = name
        return _FakeTable(self._rows, self.log)


def _row(
    query: str = "How did Kisqali TRx trend?",
    answer: str = "Kisqali TRx rose 4% over the quarter.",
    chunks: Any = None,
    *,
    is_synthetic: bool = False,
    created_at: str = "2026-08-05T12:00:00+00:00",
) -> Dict[str, Any]:
    if chunks is None:
        chunks = [{"content": "Kisqali TRx by week, Northeast, 2026-Q2."}]
    return {
        "training_input": query,
        "training_output": answer,
        "retrieved_chunks": chunks,
        "is_synthetic": is_synthetic,
        "is_training_example": True,
        "created_at": created_at,
    }


def _records_file(tmp_path: Path, records: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"records": records}))
    return path


def _record(
    query: str = "How did Kisqali TRx trend?",
    answer: str = "Kisqali TRx rose 4% over the quarter.",
    contexts: Any = None,
    intent: str | None = "kpi_query",
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "query": query,
        "response_text": answer,
        "contexts": ["Kisqali TRx by week, Northeast, 2026-Q2."] if contexts is None else contexts,
    }
    if intent is not None:
        rec["detected_intent"] = intent
    return rec


# --------------------------------------------------------------------------
# The unification property
# --------------------------------------------------------------------------
class TestBothSourcesEmitOneShape:
    def test_file_and_db_produce_identical_example_shapes(self, tmp_path: Path) -> None:
        """The whole point of the seam: GEPA cannot tell the two apart.

        If these shapes diverge, the "unified" provider is a name only — the
        optimizer would see different input keys depending on which source fed
        it, and the prompt would be tuned against a different signature.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch, records_batch

        query, answer, ctx = "q", "a", "evidence one"
        # No detected_intent on either side: the DB rows carry none (the writers
        # persist query/answer/chunks, not the routing label), so pinning full
        # equality requires comparing like with like. The intent divergence when
        # a record DOES carry one is asserted separately below.
        file_batch = records_batch(
            str(_records_file(tmp_path, [_record(query, answer, [ctx], intent=None)]))
        )
        db = asyncio.run(db_batch(client=_FakeClient([_row(query, answer, [{"content": ctx}])])))

        assert len(file_batch.examples) == 1
        assert len(db.examples) == 1
        f, d = file_batch.examples[0], db.examples[0]
        assert f.toDict().keys() == d.toDict().keys()
        assert f.inputs().toDict() == d.inputs().toDict()
        assert f.toDict() == d.toDict()

    def test_intent_is_the_one_field_the_db_cannot_supply(self, tmp_path: Path) -> None:
        """Records keep their routing label; DB turns honestly say UNKNOWN.

        The writers persist query/answer/chunks, not the detected intent, so
        inventing one for a DB turn would put a fabricated routing label into
        the optimizer's input distribution. "UNKNOWN" is the same value the file
        source already uses for a record that carries none.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch, records_batch

        with_intent = records_batch(str(_records_file(tmp_path, [_record(intent="kpi_query")])))
        without = records_batch(str(_records_file(tmp_path, [_record(intent=None)])))
        db = asyncio.run(db_batch(client=_FakeClient([_row()])))

        assert with_intent.examples[0].intent == "kpi_query"
        assert without.examples[0].intent == "UNKNOWN"
        assert db.examples[0].intent == "UNKNOWN"

    def test_db_examples_carry_the_metric_and_signature_fields(self) -> None:
        """retrieved_contexts (metric) and evidence_board (signature) both set."""
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        batch = asyncio.run(db_batch(client=_FakeClient([_row()])))
        ex = batch.examples[0]
        assert ex.retrieved_contexts == ["Kisqali TRx by week, Northeast, 2026-Q2."]
        assert json.loads(ex.evidence_board) == ex.retrieved_contexts
        assert set(ex.inputs().toDict()) == {
            "user_query",
            "investigation_goal",
            "evidence_board",
            "intent",
        }


# --------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------
class TestUnreadableRecordsFiles:
    """A file that cannot be parsed is an unavailable SOURCE, not a crash.

    ``RagExampleSourceUnavailable`` is the contract for "could not read it at
    all", and the leg turns it into a logged skip naming the remedy. A raw
    ``JSONDecodeError`` escaping instead gets caught by the beat's blanket
    guard and reported as a leg FAILURE with a parser traceback for a reason —
    same non-abort outcome, but an operator reading it learns nothing about
    which file to regenerate.
    """

    def test_corrupt_json_is_an_unavailable_source(self, tmp_path: Path) -> None:
        from src.tasks.rag_example_sources import RagExampleSourceUnavailable, records_batch

        path = tmp_path / "records.json"
        path.write_text("{not json")
        with pytest.raises(RagExampleSourceUnavailable) as exc:
            records_batch(str(path))
        assert str(path) in str(exc.value)
        assert "replay_golden_set" in str(exc.value), "the remedy must name the regenerator"

    def test_the_leg_reports_a_corrupt_file_as_a_skip_with_a_remedy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        import src.tasks.dspy_optimization_tasks as task_module
        from src.tasks import rag_example_sources as mod

        monkeypatch.chdir(tmp_path)
        path = tmp_path / "records.json"
        path.write_text('{"records": [')
        monkeypatch.setenv(mod.RAG_RECORDS_PATH_ENV, str(path))

        with caplog.at_level(logging.INFO, logger="src.tasks.dspy_optimization_tasks"):
            result = asyncio_run(task_module.run_rag_prompt_optimization())

        assert result["status"] == "skipped", result
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "replay_golden_set" in blob, blob

    def test_valid_json_that_is_not_records_reads_as_zero_not_a_crash(self, tmp_path: Path) -> None:
        """A bare scalar or a wrong-typed ``records`` key is honestly empty."""
        from src.tasks.rag_example_sources import records_batch

        for body in ("42", '"hello"', "null", "{}", '{"records": 7}'):
            path = tmp_path / "r.json"
            path.write_text(body)
            batch = records_batch(str(path))
            assert batch.examples == (), body
            assert batch.total_records == 0, body


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


class TestSourcePrecedence:
    def test_explicit_file_path_wins_over_the_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit beats ambient: a path is an operator's reproducible choice."""
        import asyncio

        from src.tasks import rag_example_sources as mod

        path = _records_file(tmp_path, [_record("from-file", "a", ["c"])])
        monkeypatch.setenv(mod.RAG_RECORDS_PATH_ENV, str(path))
        monkeypatch.setenv(mod.RAG_DB_FEEDSTOCK_ENV, "true")

        batch = asyncio.run(mod.load_rag_examples(client=_FakeClient([_row("from-db")])))
        assert batch.source == mod.SOURCE_FILE
        assert [e.user_query for e in batch.examples] == ["from-file"]

    def test_db_is_used_when_no_path_and_the_flag_is_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from src.tasks import rag_example_sources as mod

        monkeypatch.delenv(mod.RAG_RECORDS_PATH_ENV, raising=False)
        monkeypatch.setenv(mod.RAG_DB_FEEDSTOCK_ENV, "true")

        batch = asyncio.run(mod.load_rag_examples(client=_FakeClient([_row("from-db")])))
        assert batch.source == mod.SOURCE_DB
        assert [e.user_query for e in batch.examples] == ["from-db"]

    def test_db_stays_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default OFF preserves the measured zero-cost steady state exactly.

        Turning the DB source on makes the nightly leg start spending its judge
        budget the moment live traffic supplies enough turns. That is the
        intent, but it is an operator's cost decision, not a merge side effect.
        """
        import asyncio

        from src.tasks import rag_example_sources as mod

        monkeypatch.delenv(mod.RAG_RECORDS_PATH_ENV, raising=False)
        monkeypatch.delenv(mod.RAG_DB_FEEDSTOCK_ENV, raising=False)

        with pytest.raises(mod.RagExampleSourceUnavailable) as exc:
            asyncio.run(mod.load_rag_examples(client=_FakeClient([_row()])))
        # The skip must name BOTH ways in, or the operator learns only one.
        assert mod.RAG_RECORDS_PATH_ENV in str(exc.value)
        assert mod.RAG_DB_FEEDSTOCK_ENV in str(exc.value)

    def test_flag_parsing_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'false' must not enable the source (the bool('false') is True trap)."""
        import asyncio

        from src.tasks import rag_example_sources as mod

        monkeypatch.delenv(mod.RAG_RECORDS_PATH_ENV, raising=False)
        for value in ("false", "0", "no", "", "maybe"):
            monkeypatch.setenv(mod.RAG_DB_FEEDSTOCK_ENV, value)
            with pytest.raises(mod.RagExampleSourceUnavailable):
                asyncio.run(mod.load_rag_examples(client=_FakeClient([_row()])))


# --------------------------------------------------------------------------
# The DB source
# --------------------------------------------------------------------------
class TestDbSource:
    def test_rows_without_evidence_are_dropped(self) -> None:
        """A no-context turn cannot be judged; keeping it fabricates a 0.0.

        Same reason load_rag_examples_from_records drops them: the RAGAS metric
        REFUSES an unjudgeable example and GEPA silently converts the refusal to
        failure_score 0.0.

        The empty and NULL cases are removed server-side by the ``neq`` narrowing
        (SQL NULL never satisfies ``<>``), so ``total_records`` — the CANDIDATE
        window — is 1. The shapes the server cannot judge (a non-empty array of
        contentless chunks) are covered by the shape test below.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        rows = [_row("kept"), _row("dropped", chunks=[]), _row("also-dropped", chunks=None)]
        rows[2]["retrieved_chunks"] = None
        batch = asyncio.run(db_batch(client=_FakeClient(rows)))
        assert [e.user_query for e in batch.examples] == ["kept"]
        assert batch.total_records == 1

    def test_rows_without_a_query_or_answer_are_dropped(self) -> None:
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        rows = [
            _row("kept"),
            _row("", "has answer"),
            _row("has query", ""),
            _row("   ", "   "),
        ]
        batch = asyncio.run(db_batch(client=_FakeClient(rows)))
        assert [e.user_query for e in batch.examples] == ["kept"]

    def test_chunk_dicts_and_bare_strings_both_read(self) -> None:
        """Tolerant on read, by evidence, not by hedging.

        The #1489 producer writes ``{"content": ...}`` dicts
        (src/rag/retrieved_chunks.py) but migration 022 put no shape constraint
        on the JSONB column, and the replay path's own contexts are bare
        strings. A reader that understood only one shape would silently drop
        every row written by the other producer.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        rows = [
            _row("dicts", chunks=[{"content": "c1"}, {"content": "c2", "source": "kg"}]),
            _row("strings", chunks=["c3"]),
            _row("mixed", chunks=[{"content": "c4"}, "c5"]),
        ]
        batch = asyncio.run(db_batch(client=_FakeClient(rows)))
        got = {e.user_query: e.retrieved_contexts for e in batch.examples}
        assert got == {"dicts": ["c1", "c2"], "strings": ["c3"], "mixed": ["c4", "c5"]}

    def test_empty_and_shapeless_chunks_do_not_become_blank_contexts(self) -> None:
        """A chunk with no text is not evidence; a blank context would be judged."""
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        rows = [
            _row("dropped", chunks=[{"source": "kg"}, {"content": "   "}, 17, None]),
            _row("kept", chunks=[{"content": "real"}, {"content": ""}]),
        ]
        batch = asyncio.run(db_batch(client=_FakeClient(rows)))
        got = {e.user_query: e.retrieved_contexts for e in batch.examples}
        assert got == {"kept": ["real"]}

    def test_synthetic_rows_are_excluded_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Training the served prompt on showcase fixtures is the harm here.

        Measured on the live DB 2026-08-06: 3,300 of 3,959 learning_signals rows
        are is_synthetic, and ALL 3,300 carry a query. Without the provenance
        predicate the feedstock is overwhelmingly generated text.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        client = _FakeClient([_row("real"), _row("fake", is_synthetic=True)])
        batch = asyncio.run(db_batch(client=client))
        assert [e.user_query for e in batch.examples] == ["real"]
        assert ("is_synthetic", False) in client.log.get("eq", [])

    def test_showcase_deployment_opt_in_includes_synthetic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same reversible switch every other read-path chokepoint honours."""
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
        client = _FakeClient([_row("real"), _row("fake", is_synthetic=True)])
        batch = asyncio.run(db_batch(client=client))
        assert sorted(e.user_query for e in batch.examples) == ["fake", "real"]
        assert ("is_synthetic", False) not in client.log.get("eq", [])

    def test_the_read_is_bounded_and_ordered_newest_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounded AND newest-first, both asserted on the OUTPUT.

        Ordering is not cosmetic: the window is capped at ``DB_ROW_CAP``, so an
        unordered read would keep an arbitrary 500 candidate rows instead of the
        most recent ones, and the prompt would be tuned on whatever Postgres
        happened to return. Rows are fed oldest-first here so a missing
        ``.order(..., desc=True)`` leaves them in that order and fails — an
        earlier version of this test asserted only the logged chain and passed
        with the ordering deleted.
        """
        import asyncio

        from src.tasks import rag_example_sources as mod

        client = _FakeClient(
            [_row(f"q{i}", created_at=f"2026-08-0{i}T00:00:00+00:00") for i in (1, 2, 3)]
        )
        batch = asyncio.run(mod.db_batch(client=client))
        assert client.log["table"] == mod.FEEDSTOCK_TABLE
        assert client.log["limit"] == mod.DB_ROW_CAP
        assert client.log.get("gte"), "the read must be bounded by a created_at cutoff"
        assert client.log.get("order") == ("created_at", True), client.log.get("order")
        assert [e.user_query for e in batch.examples] == ["q3", "q2", "q1"]

    def test_chunkless_rows_are_narrowed_out_before_the_row_cap(self) -> None:
        """Measured 2026-08-06: the window came back 499 of 500 rows, ALL of them
        rating/thumbs/implicit_positive signals that never carry evidence.

        Filtering those in Python only would mean that, once the producer lands,
        judgeable turns sit just past the boundary and the leg reports a
        confident shortfall it did not actually measure. So the narrowing has to
        happen server-side, before the cap.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        client = _FakeClient([_row("kept"), _row("chunkless", chunks=[])])
        batch = asyncio.run(db_batch(client=client))
        assert ("retrieved_chunks", "[]") in client.log.get("neq", [])
        # total_records counts the WINDOW, so the narrowing must be visible there
        # and not just in the usable count.
        assert batch.total_records == 1
        assert [e.user_query for e in batch.examples] == ["kept"]

    def test_evidence_bearing_rows_with_no_text_are_diagnosed_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The measured cross-lane dead-end must not present as a quiet zero.

        Measured 2026-08-06 against the live DB and the parallel producer
        branch: the ONLY signal that will carry ``retrieved_chunks`` is the
        cognitive Reflector's ``agent`` signal, whose dict uses the keys
        ``query``/``response`` — but ``SignalCollector`` reads
        ``signal.get("input")``/``("output")``, so ``training_input`` and
        ``training_output`` land EMPTY (133 of 356 agent rows, and all 133
        summarizer + 133 investigator rows, are empty today).

        So after both lanes merge this reader can see candidate rows and still
        produce nothing. Refusing them is correct — the Reflector's ``query`` is
        a synthetic descriptor ("Intent: X, Evidence: N items"), not the user's
        question, so reading it would feed GEPA a fabricated prompt. But a
        silent zero would look like "no traffic yet" instead of "the writer is
        not persisting the text", so the reason has to be in the log.
        """
        import asyncio
        import logging

        from src.tasks.rag_example_sources import db_batch

        rows = [_row("", "", [{"content": "real evidence"}]) for _ in range(3)]
        with caplog.at_level(logging.WARNING, logger="src.tasks.rag_example_sources"):
            batch = asyncio.run(db_batch(client=_FakeClient(rows)))

        assert batch.examples == ()
        assert batch.total_records == 3
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "training_input" in blob and "training_output" in blob, blob
        assert "3" in blob, blob

    def test_the_diagnosis_names_the_real_reason_not_a_guessed_one(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A confident wrong diagnosis is worse than a vague one.

        A row can be unusable because its TEXT is blank or because its
        EVIDENCE could not be read (a chunk dict with no ``content`` key passes
        the server-side non-empty-array narrowing but yields no passage). Those
        need different fixes in different files, so the warning must count what
        actually happened rather than assert the first cause.
        """
        import asyncio
        import logging

        from src.tasks.rag_example_sources import db_batch

        rows = [
            _row("", "", [{"content": "evidence"}]),
            _row("q", "a", [{"text": "wrong key"}]),
            _row("q2", "a2", [{"source": "kg"}]),
        ]
        with caplog.at_level(logging.WARNING, logger="src.tasks.rag_example_sources"):
            batch = asyncio.run(db_batch(client=_FakeClient(rows)))

        assert batch.examples == ()
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "1" in blob and "2" in blob, (
            f"expected a 1-blank-text / 2-bad-evidence split: {blob}"
        )
        assert "evidence" in blob.lower(), blob

    def test_partial_loss_is_still_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """One usable row must not silence the diagnosis for the other forty.

        Firing only when NOTHING is usable hides the common real case: the
        feedstock is quietly a fraction of the traffic that carried evidence.
        """
        import asyncio
        import logging

        from src.tasks.rag_example_sources import db_batch

        rows = [_row("good", "answer", [{"content": "ctx"}])] + [
            _row("", "", [{"content": "ctx"}]) for _ in range(4)
        ]
        with caplog.at_level(logging.WARNING, logger="src.tasks.rag_example_sources"):
            batch = asyncio.run(db_batch(client=_FakeClient(rows)))

        assert [e.user_query for e in batch.examples] == ["good"]
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "4" in blob, f"the 4 dropped rows must be reported: {blob}"

    def test_every_row_is_either_a_turn_or_exactly_one_recorded_drop(self) -> None:
        """The accounting invariant the diagnosis rests on.

        If a row could vanish without being counted, or be counted twice, the
        breakdown in the warning would be arithmetic that does not add up — and
        an operator would size the problem wrongly. Swept over the product of
        blank/whitespace/None text and ten evidence shapes, plus non-object
        rows.
        """
        import itertools

        from src.tasks.rag_example_sources import _turns_from_rows

        chunk_shapes: List[Any] = [
            None,
            [],
            "notalist",
            [{"content": "c"}],
            [{"content": "  "}],
            [{"text": "c"}],
            ["bare"],
            [{"content": "c"}, 17],
            [None],
            [{"source": "kg"}, {"content": "ok"}],
        ]
        texts: List[Any] = ["q", "", "   ", None]
        rows: List[Any] = [
            {"training_input": q, "training_output": a, "retrieved_chunks": ch}
            for q, a, ch in itertools.product(texts, texts, chunk_shapes)
        ]
        rows += ["not a dict", None, 42, {"training_input": "q", "training_output": "a"}]

        turns, drops = _turns_from_rows(rows)

        assert len(turns) + sum(drops.values()) == len(rows)
        assert all(
            t.query.strip()
            and t.answer.strip()
            and t.contexts
            and all(c.strip() for c in t.contexts)
            for t in turns
        ), "a turn escaped with blank content"
        assert set(drops) <= {
            "blank training_input/training_output",
            "no readable evidence text in retrieved_chunks",
            "row is not an object",
        }, drops

    def test_no_diagnosis_when_there_were_no_candidates_at_all(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Control: an empty window is the ordinary state, not a defect."""
        import asyncio
        import logging

        from src.tasks.rag_example_sources import db_batch

        with caplog.at_level(logging.WARNING, logger="src.tasks.rag_example_sources"):
            batch = asyncio.run(db_batch(client=_FakeClient([])))

        assert batch.examples == ()
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "training_input" not in blob, f"warned about an empty window: {blob}"

    def test_lookback_window_is_env_tunable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        from src.tasks import rag_example_sources as mod

        client = _FakeClient([_row()])
        monkeypatch.setenv(mod.RAG_DB_LOOKBACK_DAYS_ENV, "7")
        asyncio.run(mod.db_batch(client=client))
        seven = client.log["gte"][0][1]

        client2 = _FakeClient([_row()])
        monkeypatch.setenv(mod.RAG_DB_LOOKBACK_DAYS_ENV, "90")
        asyncio.run(mod.db_batch(client=client2))
        assert client2.log["gte"][0][1] < seven

    def test_no_client_is_an_unavailable_source_not_an_empty_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty batch would read as 'measured, and there is nothing'.

        That is a different fact from 'could not look', and only the second one
        should leave the records un-fingerprinted for a retry.
        """
        import asyncio

        import src.memory.services.factories as factories
        from src.tasks import rag_example_sources as mod

        monkeypatch.setattr(factories, "get_supabase_client", lambda: None, raising=True)
        with pytest.raises(mod.RagExampleSourceUnavailable):
            asyncio.run(mod.db_batch(client=None))


# --------------------------------------------------------------------------
# Fingerprint continuity + dedup
# --------------------------------------------------------------------------
class TestFingerprints:
    def test_file_fingerprint_is_byte_identical_to_the_shipped_one(self, tmp_path: Path) -> None:
        """Deployed dedup state must survive this refactor.

        ``.trigger_state.json`` on the prod volume holds digests computed by the
        shipped ``_rag_records_fingerprint``. A different digest for identical
        records would re-spend the whole judge budget on the next beat.
        """
        from src.tasks.dspy_optimization_tasks import _rag_records_fingerprint
        from src.tasks.rag_example_sources import records_batch

        path = _records_file(tmp_path, [_record()])
        assert records_batch(str(path)).fingerprint(40) == _rag_records_fingerprint(str(path), 40)

    def test_fingerprint_covers_the_budget_for_both_sources(self, tmp_path: Path) -> None:
        import asyncio

        from src.tasks.rag_example_sources import db_batch, records_batch

        file_batch = records_batch(str(_records_file(tmp_path, [_record()])))
        assert file_batch.fingerprint(40) != file_batch.fingerprint(60)

        db = asyncio.run(db_batch(client=_FakeClient([_row()])))
        assert db.fingerprint(40) != db.fingerprint(60)

    def test_db_fingerprint_is_stable_across_row_order(self) -> None:
        """Ties in created_at must not look like new feedstock.

        A reorder that changed the digest would re-run the whole compile and
        re-spend the budget on identical turns.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        rows = [_row("a"), _row("b"), _row("c")]
        one = asyncio.run(db_batch(client=_FakeClient(rows)))
        two = asyncio.run(db_batch(client=_FakeClient(list(reversed(rows)))))
        assert one.fingerprint(40) == two.fingerprint(40)

    def test_db_fingerprint_moves_when_content_changes(self) -> None:
        import asyncio

        from src.tasks.rag_example_sources import db_batch

        one = asyncio.run(db_batch(client=_FakeClient([_row("a")])))
        two = asyncio.run(db_batch(client=_FakeClient([_row("a"), _row("b")])))
        assert one.fingerprint(40) != two.fingerprint(40)

    def test_sources_do_not_collide_on_identical_content(self, tmp_path: Path) -> None:
        """Same turns from a different source is a different measurement.

        Two assertions, because they cover different things. The digests differ
        today for a STRUCTURAL reason — file material is raw JSON bytes, DB
        material is a concatenation of hex digests, and valid JSON can never be
        pure hex — so the inequality alone would still hold with the ``db:``
        namespace removed (verified by mutation: dropping the prefix left this
        test green). The namespace is defence against a future change to either
        material's format, so it is asserted directly rather than inferred.
        """
        import asyncio

        from src.tasks.rag_example_sources import db_batch, records_batch

        q, a, c = "q", "a", "ctx"
        file_batch = records_batch(str(_records_file(tmp_path, [_record(q, a, [c])])))
        db = asyncio.run(db_batch(client=_FakeClient([_row(q, a, [{"content": c}])])))
        assert file_batch.fingerprint(40) != db.fingerprint(40)
        assert db.fingerprint_material.startswith(b"db:"), (
            "the DB digest must be namespaced by source, so the two materials "
            "cannot collide even if their formats converge later"
        )


# --------------------------------------------------------------------------
# Backwards compatibility with the shipped #1486 surface
# --------------------------------------------------------------------------
class _FakeOptimizer:
    """Stands in for a compiled GEPA run, without spending a judge call."""

    def __init__(self, instructions: str, metric: Any) -> None:
        self._instructions = instructions
        self._metric = metric

    def compile(self, module: Any, trainset: Any = None, valset: Any = None) -> Any:
        import dspy

        from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature

        return dspy.ChainOfThought(EvidenceSynthesisSignature.with_instructions(self._instructions))


def _arm_db_leg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: List[Dict[str, Any]],
    *,
    instructions: str = "AN IMPROVED PROMPT",
) -> _FakeClient:
    """Arm the leg on the DB feedstock with the judge/optimizer stubbed.

    The client is injected by patching the real factory the source resolves
    through, so ``_resolve_client`` runs for real rather than being bypassed.
    """
    import src.memory.services.factories as factories
    import src.tasks.dspy_optimization_tasks as task_module
    from src.tasks import rag_example_sources as mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(mod.RAG_RECORDS_PATH_ENV, raising=False)
    monkeypatch.setenv(mod.RAG_DB_FEEDSTOCK_ENV, "true")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)

    client = _FakeClient(rows)
    monkeypatch.setattr(factories, "get_supabase_client", lambda: client, raising=True)

    class _Metric:
        _degraded_examples = 0

        def __call__(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
            raise AssertionError("the stubbed optimizer must not score examples")

        @property
        def degraded_examples(self) -> int:
            return self._degraded_examples

    metric = _Metric()
    monkeypatch.setattr(
        "src.optimization.gepa.metrics.get_metric_for_agent", lambda *a, **k: metric, raising=True
    )
    monkeypatch.setattr(
        "src.optimization.gepa.create_gepa_optimizer",
        lambda **k: _FakeOptimizer(instructions, metric),
        raising=True,
    )
    monkeypatch.setattr(
        "src.optimization.dspy_lm.ensure_dspy_configured", lambda *a, **k: True, raising=True
    )
    # Imported for the chdir-relative _STATE_PATH it owns; referenced so the
    # import is not mistaken for an unused one.
    assert task_module.RAG_RECORDS_PATH_ENV == mod.RAG_RECORDS_PATH_ENV
    return client


class TestTheLegActuallyUsesTheDbSource:
    """#1486 was filed because a complete implementation had zero callers.

    A feedstock the nightly leg cannot reach would be the same defect wearing a
    new name, so these drive ``run_rag_prompt_optimization`` itself.
    """

    @pytest.mark.asyncio
    async def test_live_traffic_rows_drive_a_real_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.tasks.dspy_optimization_tasks as task_module
        from src.rag.cognitive_rag_dspy import OPTIMIZED_SYNTHESIS_AGENT_NAME

        rows = [_row(f"q{i}", f"answer {i}", [{"content": f"ctx{i}"}]) for i in range(8)]
        _arm_db_leg(tmp_path, monkeypatch, rows)

        result = await task_module.run_rag_prompt_optimization()

        assert result["status"] == "completed", result
        assert result["source"] == "db"
        assert result["examples"] == 8
        artifacts = tmp_path / "optimized_modules" / OPTIMIZED_SYNTHESIS_AGENT_NAME
        assert len(list(artifacts.glob("gepa_*.json"))) == 1

    @pytest.mark.asyncio
    async def test_the_saved_artifact_records_which_feedstock_tuned_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A golden-set prompt and a live-traffic prompt are not interchangeable."""
        import src.tasks.dspy_optimization_tasks as task_module
        from src.rag.cognitive_rag_dspy import OPTIMIZED_SYNTHESIS_AGENT_NAME

        rows = [_row(f"q{i}", f"answer {i}", [{"content": f"ctx{i}"}]) for i in range(8)]
        _arm_db_leg(tmp_path, monkeypatch, rows)
        await task_module.run_rag_prompt_optimization()

        artifacts = tmp_path / "optimized_modules" / OPTIMIZED_SYNTHESIS_AGENT_NAME
        blob = json.loads(next(iter(artifacts.glob("gepa_*.json"))).read_text())
        metadata = blob["metadata"]
        assert metadata["source"] == "db"
        assert metadata["source_records"].startswith("learning_signals")
        assert metadata["examples"] == 8

    @pytest.mark.asyncio
    async def test_an_unchanged_db_window_does_not_re_spend_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that makes an UNATTENDED nightly cycle affordable.

        Without it the beat re-compiles against identical turns every 24h and
        spends the whole DSPY_RAG_MAX_METRIC_CALLS judge budget each time.
        """
        import src.tasks.dspy_optimization_tasks as task_module

        rows = [_row(f"q{i}", f"answer {i}", [{"content": f"ctx{i}"}]) for i in range(8)]
        _arm_db_leg(tmp_path, monkeypatch, rows)

        first = await task_module.run_rag_prompt_optimization()
        assert first["status"] == "completed", first

        second = await task_module.run_rag_prompt_optimization()
        assert second["status"] == "skipped"
        assert "already" in second["reason"].lower()

    @pytest.mark.asyncio
    async def test_a_new_live_turn_reopens_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control for the dedup above: fresh traffic must NOT be deduped away."""
        import src.memory.services.factories as factories
        import src.tasks.dspy_optimization_tasks as task_module

        rows = [_row(f"q{i}", f"answer {i}", [{"content": f"ctx{i}"}]) for i in range(8)]
        _arm_db_leg(tmp_path, monkeypatch, rows)
        assert (await task_module.run_rag_prompt_optimization())["status"] == "completed"

        grown = rows + [_row("brand new turn", "brand new answer", [{"content": "new ctx"}])]
        monkeypatch.setattr(
            factories, "get_supabase_client", lambda: _FakeClient(grown), raising=True
        )
        again = await task_module.run_rag_prompt_optimization()
        assert again["status"] == "completed", again
        assert again["examples"] == 9

    @pytest.mark.asyncio
    async def test_too_few_live_turns_skips_before_any_judge_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Today's real state: 0 rows carry evidence, so this IS the live path.

        Measured on the live DB 2026-08-06: retrieved_chunks is non-default on 0
        of 3,959 rows, so until the #1489 producer lands every beat lands here —
        at zero API cost, which is what makes shipping the source safe now.
        """
        import src.tasks.dspy_optimization_tasks as task_module

        rows = [_row(f"q{i}", f"answer {i}", chunks=[]) for i in range(40)]
        _arm_db_leg(tmp_path, monkeypatch, rows)

        result = await task_module.run_rag_prompt_optimization()
        assert result["status"] == "skipped"
        assert result["usable_examples"] == 0
        # 0 CANDIDATES, not 0 of 40: the has-evidence narrowing runs server-side,
        # so these rows never enter the window. "no turns carry evidence yet" is
        # the honest diagnosis, and it is the one an operator needs today.
        assert result["total_records"] == 0
        assert result["source"] == "db"
        # ...but a bare "0 record(s)" reads as "no traffic at all", which is a
        # DIFFERENT and wrong diagnosis — there were 40 recent training rows,
        # none carrying evidence. The reported reason and payload must say which.
        assert "candidate" in result["reason"], result["reason"]
        assert "learning_signals" in result["origin"], result
        assert not (tmp_path / "optimized_modules").exists() or not list(
            (tmp_path / "optimized_modules").glob("*/gepa_*.json")
        )

    @pytest.mark.asyncio
    async def test_an_explicit_records_file_still_wins_at_the_leg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.tasks.dspy_optimization_tasks as task_module
        from src.tasks import rag_example_sources as mod

        rows = [_row(f"db{i}", f"answer {i}", [{"content": f"ctx{i}"}]) for i in range(8)]
        _arm_db_leg(tmp_path, monkeypatch, rows)
        records = [_record(f"file{i}", f"answer {i}", [f"ctx{i}"]) for i in range(8)]
        monkeypatch.setenv(mod.RAG_RECORDS_PATH_ENV, str(_records_file(tmp_path, records)))

        result = await task_module.run_rag_prompt_optimization()
        assert result["status"] == "completed", result
        assert result["source"] == "file"


class TestShippedSurfaceStillWorks:
    def test_the_old_import_path_still_resolves(self, tmp_path: Path) -> None:
        """#1486's tests and any operator tooling import from the task module."""
        from src.tasks import rag_example_sources as mod
        from src.tasks.dspy_optimization_tasks import (
            RAG_RECORDS_PATH_ENV,
            load_rag_examples_from_records,
        )

        assert RAG_RECORDS_PATH_ENV == mod.RAG_RECORDS_PATH_ENV
        path = _records_file(tmp_path, [_record()])
        assert len(load_rag_examples_from_records(str(path))) == 1
