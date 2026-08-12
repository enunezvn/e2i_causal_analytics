"""Offline guard: the corpus indexer never embeds synthetic business_metrics
(Shard 07 R12).

``_fetch_brand_rows`` (and the brand-discovery select in ``index_business_metrics``)
read the live ``business_metrics`` fact table. Shard 02 stamps synthetic rows with
``is_synthetic=true``; the prod RAG corpus must default-exclude them so the chatbot
never surfaces synthetic KPI prose. These tests assert the ``.eq('is_synthetic',
False)`` predicate is appended to the source query, using a fluent fake that records
every ``.eq`` call (no DB, no mock of the unit under test).

Env isolation (#1495): ``apply_provenance_filter`` is deliberately gated on
``E2I_INCLUDE_SYNTHETIC`` (WS-SYNTH showcase instances include synthetic KPI prose
so the corpus isn't empty), and that var IS set on showcase/dev hosts (this repo's
``.env`` plus the find_dotenv walk-up class, PR #1414). The real-mode tests below
therefore pin real mode explicitly via an autouse ``delenv``, and a companion test
pins the showcase branch (filter skipped) so both sides of the gate stay covered.
"""

import asyncio
import uuid as _uuid
from typing import Any

import pytest

from src.rag import corpus_ingestion as ci


@pytest.fixture(autouse=True)
def _pin_real_mode_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin real mode for every test in this module regardless of host env.

    Without this, any host exporting ``E2I_INCLUDE_SYNTHETIC`` (showcase/dev
    boxes) makes production legitimately skip the filter and the four
    real-mode tests fail for an environmental — not functional — reason.
    Showcase-mode tests re-set the var explicitly with ``monkeypatch.setenv``
    (the shared per-test monkeypatch applies the delenv first, then the
    test-body setenv, so both compose deterministically).
    """
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


class _RecordingQuery:
    """Fluent supabase-py query stub that records every ``.eq`` call."""

    def __init__(self, calls: list[tuple[Any, ...]]):
        self._calls = calls

    def select(self, *_a: Any, **_k: Any) -> "_RecordingQuery":
        return self

    def eq(self, *a: Any) -> "_RecordingQuery":
        self._calls.append(a)
        return self

    @property
    def not_(self) -> "_RecordingQuery":
        return self

    def is_(self, *_a: Any) -> "_RecordingQuery":
        return self

    def order(self, *_a: Any, **_k: Any) -> "_RecordingQuery":
        return self

    def limit(self, *_a: Any) -> "_RecordingQuery":
        return self

    def range(self, *_a: Any) -> "_RecordingQuery":
        return self

    def execute(self) -> Any:
        class _R:
            data: list[Any] = []

        return _R()


class _RecordingClient:
    def __init__(self, calls: list[tuple[Any, ...]]):
        self._calls = calls

    def table(self, _name: str) -> _RecordingQuery:
        return _RecordingQuery(self._calls)


def test_fetch_brand_rows_excludes_synthetic_business_metrics() -> None:
    calls: list[tuple[Any, ...]] = []
    ci._fetch_brand_rows(_RecordingClient(calls), "Kisqali", 50, latest_per_combo=False)
    assert ("is_synthetic", False) in calls


def test_fetch_brand_rows_excludes_synthetic_latest_per_combo() -> None:
    calls: list[tuple[Any, ...]] = []
    ci._fetch_brand_rows(_RecordingClient(calls), "Kisqali", 50, latest_per_combo=True)
    assert ("is_synthetic", False) in calls


def test_index_business_metrics_brand_discovery_excludes_synthetic() -> None:
    import asyncio

    calls: list[tuple[Any, ...]] = []
    # brands=None triggers the brand-discovery select; empty data -> early return
    # (no _existing_corpus_descriptions / insert path), so this isolates the
    # discovery query predicate.
    asyncio.run(ci.index_business_metrics(supabase_client=_RecordingClient(calls)))
    assert ("is_synthetic", False) in calls


def test_existing_corpus_descriptions_excludes_synthetic() -> None:
    """Shard 07 R15: the dedup read must not let a synthetic episodic
    description suppress ingesting a real business_metrics row."""
    calls: list[tuple[Any, ...]] = []
    ci._existing_corpus_descriptions(_RecordingClient(calls), "corpus_ingestion")
    assert ("is_synthetic", False) in calls
    # the agent_name filter must still be present (real reader, not vacuous).
    assert ("agent_name", "corpus_ingestion") in calls


def test_fetch_brand_rows_showcase_mode_skips_synthetic_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterization of the OTHER side of the gate (#1495, WS-SYNTH).

    On a showcase instance (``E2I_INCLUDE_SYNTHETIC=true``)
    ``apply_provenance_filter`` must SKIP the ``.eq('is_synthetic', False)``
    predicate so synthetic KPI prose can populate the corpus. The brand
    predicate must still be present — proof the reader actually ran and the
    absence assertion is not vacuous.
    """
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    calls: list[tuple[Any, ...]] = []
    ci._fetch_brand_rows(_RecordingClient(calls), "Kisqali", 50, latest_per_combo=False)
    assert ("brand", "Kisqali") in calls
    assert ("is_synthetic", False) not in calls


# =============================================================================
# #1552 — period-grain labeling + stale-prose reconciliation
#
# 2026-08-11 eval Q6.5: the chat answer rendered "Northeast | Jun/Jul 2026 |
# 252,475" next to "Northeast | Aug 2026 | 48,655" and called the gap "an
# unexplained scale discontinuity". Measured root cause (psql against the live
# supabase-db, 2026-08-12):
#   * business_metrics is UNIFORMLY monthly-grain (every metric_date is a
#     month start — measured: DISTINCT day-offset = 0). There is no 2-month
#     bucket and no MTD bucket upstream.
#   * The "Jun/Jul" pseudo-bucket came from a STALE episodic corpus row
#     ("...on 2026-06-01: value 252475.15", ingested 2026-06-19 from a
#     pre-freeze substrate state) sitting next to the valid Jul row with
#     byte-identical values; the current fact table has Jun = 211473.23.
#     Corpus dedup is insert-only, so the stale row was never reconciled.
#   * The prose carried NO grain label ("on 2026-06-01" reads as a day), so
#     the synthesizer could not know each figure was a monthly bucket.
#
# Fixture values below are captured VERBATIM from the live DB (psql
# supabase-db, 2026-08-12): business_metrics Kisqali/TRx northeast rows for
# 2026-06-01 / 2026-07-01 / 2026-08-01 and the three episodic_memories
# corpus_ingestion descriptions for the same combo.
# =============================================================================

# --- VERBATIM fact rows (business_metrics, Kisqali TRx northeast) ------------
_FACT_JUN = {
    "metric_name": "trx",
    "brand": "Kisqali",
    "region": "northeast",
    "metric_date": "2026-06-01",
    "value": 211473.23,
    "target": 255161.56,
    "achievement_rate": 0.829,
    "year_over_year_change": 0.254,
    "roi": 3.40,
}
_FACT_JUL = {
    "metric_name": "trx",
    "brand": "Kisqali",
    "region": "northeast",
    "metric_date": "2026-07-01",
    "value": 252475.15,
    "target": 269839.25,
    "achievement_rate": 0.936,
    "year_over_year_change": 0.283,
    "roi": 1.86,
}
_FACT_AUG = {
    "metric_name": "trx",
    "brand": "Kisqali",
    "region": "northeast",
    "metric_date": "2026-08-01",
    "value": 48654.99,
    "target": 65800.04,
    "achievement_rate": 0.739,
    "year_over_year_change": 0.220,
    "roi": 3.22,
}

# --- VERBATIM episodic corpus descriptions (agent_name='corpus_ingestion') ---
# Stale: attributes the JUL values to JUN (pre-freeze substrate state). Does
# NOT match any current fact row.
_PROSE_STALE_JUN = (
    "trx for Kisqali in the northeast on 2026-06-01: value 252475.15, "
    "target 269839.25, achievement 93.6%, year-over-year +28.3%, ROI 1.86."
)
# Legacy-template but VALUE-VALID: matches the current Jul/Aug fact rows.
_PROSE_LEGACY_JUL = (
    "trx for Kisqali in the northeast on 2026-07-01: value 252475.15, "
    "target 269839.25, achievement 93.6%, year-over-year +28.3%, ROI 1.86."
)
_PROSE_LEGACY_AUG = (
    "trx for Kisqali in the northeast on 2026-08-01: value 48654.99, "
    "target 65800.04, achievement 73.9%, year-over-year +22.0%, ROI 3.22."
)


def test_render_month_start_row_labels_calendar_month_grain() -> None:
    """A month-start row (the measured universal shape of business_metrics)
    must be rendered with an EXPLICIT calendar-month grain label, not the
    ambiguous 'on <date>' form that let the synthesizer read 2-month and MTD
    buckets into a uniformly-monthly substrate (6.5)."""
    prose = ci.render_business_metric(_FACT_AUG)
    assert "calendar month 2026-08" in prose
    assert "monthly grain" in prose
    assert "August 2026" in prose
    assert " on 2026-08-01:" not in prose
    # Values stay verbatim (F3 anti-mocking).
    assert "value 48654.99" in prose
    assert "target 65800.04" in prose
    assert "achievement 73.9%" in prose
    assert "year-over-year +22.0%" in prose
    assert "ROI 3.22" in prose


def test_render_non_month_start_date_keeps_honest_on_date_form() -> None:
    """A row NOT on the month-start lattice must NOT claim monthly grain —
    it keeps the honest legacy 'on <date>' form."""
    row = dict(_FACT_AUG, metric_date="2026-08-10")
    prose = ci.render_business_metric(row)
    assert "on 2026-08-10" in prose
    assert "calendar month" not in prose
    assert "monthly grain" not in prose


def test_render_accepts_date_object_and_iso_string_equivalently() -> None:
    """supabase-py returns metric_date as an ISO string; generators/tests may
    pass datetime.date. Both must produce identical grain-labeled prose."""
    from datetime import date

    prose_str = ci.render_business_metric(_FACT_AUG)
    prose_date = ci.render_business_metric(dict(_FACT_AUG, metric_date=date(2026, 8, 1)))
    assert prose_str == prose_date


class _FakeReconcileQuery:
    """Functional fluent supabase-py fake over an in-memory store.

    Applies ``eq`` filters only for columns present on the stored rows (the
    provenance predicate targets a column the fixtures deliberately omit —
    its presence is pinned by the recording tests above). Supports the
    paginated reads, the ``delete().in_()`` path, and records deletions.
    """

    def __init__(self, table: str, store: dict[str, Any]):
        self._table = table
        self._store = store
        self._filters: list[tuple[str, Any]] = []
        self._in: tuple[str, list[Any]] | None = None
        self._range: tuple[int, int] | None = None
        self._limit: int | None = None
        self._delete = False
        self._order: list[tuple[str, bool]] = []

    def select(self, *_a: Any, **_k: Any) -> "_FakeReconcileQuery":
        return self

    def delete(self) -> "_FakeReconcileQuery":
        self._delete = True
        return self

    def eq(self, col: str, val: Any) -> "_FakeReconcileQuery":
        self._filters.append((col, val))
        return self

    def in_(self, col: str, vals: list[Any]) -> "_FakeReconcileQuery":
        self._in = (col, list(vals))
        return self

    @property
    def not_(self) -> "_FakeReconcileQuery":
        return self

    def is_(self, *_a: Any) -> "_FakeReconcileQuery":
        return self

    def order(self, col: str, *, desc: bool = False) -> "_FakeReconcileQuery":
        self._order.append((col, desc))
        return self

    def limit(self, n: int) -> "_FakeReconcileQuery":
        self._limit = n
        return self

    def range(self, lo: int, hi: int) -> "_FakeReconcileQuery":
        self._range = (lo, hi)
        return self

    def _rows(self) -> list[dict[str, Any]]:
        rows = list(self._store.get(self._table, []))
        for col, val in self._filters:
            rows = [r for r in rows if col not in r or r[col] == val]
        if self._in is not None:
            col, vals = self._in
            rows = [r for r in rows if r.get(col) in vals]
        # Honor .order() faithfully — _latest_per_combo depends on the real
        # query's metric_date-DESC ordering (latest snapshot first).
        for col, desc in reversed(self._order):
            rows.sort(key=lambda r: str(r.get(col, "")), reverse=desc)
        return rows

    def execute(self) -> Any:
        class _R:
            data: list[dict[str, Any]] = []

        r = _R()
        if self._delete:
            doomed = self._rows()
            self._store.setdefault("_deleted", []).extend(row["memory_id"] for row in doomed)
            self._store[self._table] = [
                row for row in self._store.get(self._table, []) if row not in doomed
            ]
            r.data = doomed
            return r
        rows = self._rows()
        if self._range is not None:
            lo, hi = self._range
            rows = rows[lo : hi + 1]
        elif self._limit is not None:
            rows = rows[: self._limit]
        r.data = rows
        return r


class _FakeReconcileClient:
    def __init__(self, store: dict[str, Any]):
        self.store = store

    def table(self, name: str) -> _FakeReconcileQuery:
        return _FakeReconcileQuery(name, self.store)


def _reconcile_store() -> dict[str, Any]:
    """In-memory DB mirroring the measured live state for one combo."""
    return {
        "business_metrics": [dict(_FACT_JUN), dict(_FACT_JUL), dict(_FACT_AUG)],
        "episodic_memories": [
            {
                "memory_id": str(_uuid.uuid4()),
                "description": _PROSE_STALE_JUN,
                "agent_name": "corpus_ingestion",
                "brand": "kisqali",
            },
            {
                "memory_id": str(_uuid.uuid4()),
                "description": _PROSE_LEGACY_JUL,
                "agent_name": "corpus_ingestion",
                "brand": "kisqali",
            },
            {
                "memory_id": str(_uuid.uuid4()),
                "description": _PROSE_LEGACY_AUG,
                "agent_name": "corpus_ingestion",
                "brand": "kisqali",
            },
        ],
    }


def _run_index(
    monkeypatch: pytest.MonkeyPatch,
    store: dict[str, Any],
    **kwargs: Any,
) -> list[str]:
    """Run index_business_metrics against the fake store, recording inserts."""
    inserted_texts: list[str] = []

    async def _fake_insert(*, memory: Any, text_to_embed: str, session_id: Any = None) -> str:
        inserted_texts.append(text_to_embed)
        return str(_uuid.uuid4())

    monkeypatch.setattr(ci, "insert_episodic_memory_with_text", _fake_insert)
    store["_inserted_texts"] = inserted_texts
    return asyncio.run(
        ci.index_business_metrics(supabase_client=_FakeReconcileClient(store), **kwargs)
    )


def test_reconcile_deletes_stale_prose_and_migrates_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full-coverage sync must (a) DELETE the stale row that misattributes
    Jul's values to Jun — the direct cause of 6.5's invented 'Jun/Jul 2026'
    bucket — and (b) migrate value-valid legacy-template prose to the
    grain-labeled template instead of leaving unlabeled rows behind."""
    store = _reconcile_store()
    stale_id = store["episodic_memories"][0]["memory_id"]
    legacy_jul_id = store["episodic_memories"][1]["memory_id"]
    legacy_aug_id = store["episodic_memories"][2]["memory_id"]

    _run_index(monkeypatch, store, latest_per_combo=True)

    deleted = store.get("_deleted", [])
    # (a) stale mis-dated prose removed
    assert stale_id in deleted
    # (b) legacy-template (unlabeled) prose superseded, not accumulated
    assert legacy_jul_id in deleted
    assert legacy_aug_id in deleted

    inserted = store["_inserted_texts"]
    # Migrated + latest snapshots re-indexed under the grain-labeled template:
    # Jul (migrated), Aug (latest per combo + migrated, deduped to one).
    assert any("calendar month 2026-07" in t and "value 252475.15" in t for t in inserted)
    assert any("calendar month 2026-08" in t and "value 48654.99" in t for t in inserted)
    # The stale Jun VALUES are NOT re-indexed under any June label.
    assert not any("2026-06" in t and "252475.15" in t for t in inserted)


def test_reconcile_keeps_current_template_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prose already in the current grain-labeled template and matching a
    current fact row must be KEPT (idempotent re-sync: no delete, no
    re-insert)."""
    store = _reconcile_store()
    # Replace the corpus with the current-template rendering of the Jul row.
    current_jul = ci.render_business_metric(_FACT_JUL)
    keep_id = str(_uuid.uuid4())
    store["episodic_memories"] = [
        {
            "memory_id": keep_id,
            "description": current_jul,
            "agent_name": "corpus_ingestion",
            "brand": "kisqali",
        }
    ]

    _run_index(monkeypatch, store, latest_per_combo=True)

    assert keep_id not in store.get("_deleted", [])
    inserted = store["_inserted_texts"]
    # Jul already indexed -> only the missing Aug snapshot is inserted.
    assert not any("calendar month 2026-07" in t for t in inserted)
    assert any("calendar month 2026-08" in t for t in inserted)


def test_reconcile_skipped_on_bounded_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded scan (latest_per_combo=False, limit_per_brand) sees only a
    SLICE of the fact table — reconciling against it would delete valid
    prose. The bounded path must never delete."""
    store = _reconcile_store()
    _run_index(monkeypatch, store, latest_per_combo=False, limit_per_brand=1)
    assert store.get("_deleted", []) == []
