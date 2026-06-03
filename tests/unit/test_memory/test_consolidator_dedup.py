"""Unit tests for episodic deduplication in Consolidator (issue #388).

These tests pin the contract for ``Consolidator.deduplicate_episodic`` and
the pure helper ``_compute_dedup_signature``. The strategy is
exact-match dedup — collapse rows that share ``(brand, event_type,
event_subtype, causal_path_id)`` (or the description-hash fallback when
``causal_path_id IS NULL``) into a single canonical row with a
``dedup_counter`` recording the merged count. Brand boundary is
preserved by including ``brand`` in every signature variant.

Out of scope here: semantic-embedding dedup, fuzzy-key dedup, cross-brand
dedup. See issue #388 §Out of scope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.lifecycle.consolidator import (
    Consolidator,
    _compute_dedup_signature,
)

# ---------------------------------------------------------------------------
# Fake supabase that supports the dedup query surface (select, update, delete,
# is_, in_, neq). Reuses the shape of test_consolidator.py::FakeSupabase but
# extended for the dedup path.
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._select_cols: Optional[str] = None
        self._select_count_mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._is_null_cols: List[str] = []
        self._is_not_null_cols: List[str] = []
        self._gte: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}
        self._mode = None  # 'select' | 'update' | 'delete'
        self._in_filters: Dict[str, List[Any]] = {}
        self._range: Optional[tuple] = None  # (start, end) inclusive, PostgREST-style

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        self._select_cols = cols
        self._select_count_mode = count
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def delete(self) -> "_FakeQuery":
        self._mode = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def is_(self, col: str, val: Any) -> "_FakeQuery":
        # supabase-py treats is_("col", "null") AND is_("col", None) as
        # "col IS NULL"; is_("col", "not.null") AND is_("col", "not null") AND
        # NotNullSentinel as "col IS NOT NULL". We accept the common shapes.
        if val == "null" or val is None:
            self._is_null_cols.append(col)
        elif isinstance(val, str) and val.lower().replace(" ", "").replace(".", "") == "notnull":
            self._is_not_null_cols.append(col)
        else:
            self._is_null_cols.append(col)
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._in_filters[col] = list(vals)
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        self._range = (start, end)
        return self

    def _match(self) -> List[Dict[str, Any]]:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or 0) >= threshold]
        for col in self._is_null_cols:
            rows = [r for r in rows if r.get(col) is None]
        for col in self._is_not_null_cols:
            rows = [r for r in rows if r.get(col) is not None]
        for col, vals in self._in_filters.items():
            rows = [r for r in rows if r.get(col) in vals]
        return rows

    def execute(self) -> MagicMock:
        rows = self._match()
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        elif self._mode == "delete":
            # Remove all matching rows in place.
            keep = [r for r in self.store.rows[self.table_name] if r not in rows]
            self.store.rows[self.table_name] = keep
            rows = []
        mock = MagicMock()
        mock.data = rows
        mock.count = len(rows) if self._select_count_mode == "exact" else None
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "causal_paths": [],
            "episodic_memories": [],
            "procedural_memories": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(fake_supabase: FakeSupabase):
    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=fake_supabase,
    ):
        yield


# ---------------------------------------------------------------------------
# _compute_dedup_signature contract
# ---------------------------------------------------------------------------


def test_compute_dedup_signature_primary_includes_causal_path() -> None:
    """When causal_path_id is set, signature must include it (primary key)."""
    row = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-abc",
        "description": "any description here",
        "agent_name": "estimator",
    }
    sig = _compute_dedup_signature(row)
    assert sig is not None
    assert sig.startswith("v1:primary:")


def test_compute_dedup_signature_falls_back_when_causal_path_null() -> None:
    """When causal_path_id is None, signature switches to the fallback variant."""
    row = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": None,
        "description": "the same prose",
        "agent_name": "estimator",
    }
    sig = _compute_dedup_signature(row)
    assert sig is not None
    assert sig.startswith("v1:fallback:")


def test_compute_dedup_signature_returns_none_when_required_fields_missing() -> None:
    """Missing brand / event_type / event_subtype means no safe dedup key."""
    row = {
        "brand": None,
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-abc",
    }
    assert _compute_dedup_signature(row) is None


def test_compute_dedup_signature_deterministic_for_identical_input() -> None:
    """Two rows with identical key fields produce identical signatures."""
    row_a = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-abc",
        "description": "d",
        "agent_name": "a",
    }
    row_b = dict(row_a)
    assert _compute_dedup_signature(row_a) == _compute_dedup_signature(row_b)


def test_compute_dedup_signature_distinct_across_brand() -> None:
    """Brand difference must change the signature (brand-boundary defense in
    depth — also enforced by the DB partial-unique-index)."""
    row_a = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-abc",
        "description": "d",
        "agent_name": "a",
    }
    row_b = dict(row_a, brand="Fabhalta")
    assert _compute_dedup_signature(row_a) != _compute_dedup_signature(row_b)


# ---------------------------------------------------------------------------
# deduplicate_episodic — required test cases from the issue body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_collapses_identical_episodic_rows(
    fake_supabase: FakeSupabase,
) -> None:
    """3 rows with identical (brand, event_type, event_subtype,
    causal_path_id) collapse to 1 row with ``dedup_counter == 3``."""
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-abc",
                "agent_name": "estimator",
                "description": "irrelevant when causal_path_id is set",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 1, f"expected 1 row after dedup, got {len(rows)}"
    assert rows[0]["dedup_counter"] == 3


@pytest.mark.asyncio
async def test_dedup_preserves_distinct_episodic_rows(
    fake_supabase: FakeSupabase,
) -> None:
    """3 rows that differ in ``event_subtype`` are all preserved with
    ``dedup_counter == 1`` (no collapsing)."""
    for sub in ("a", "b", "c"):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m-{sub}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": sub,
                "causal_path_id": "cp-abc",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": "2026-05-20T00:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 3
    assert all(r["dedup_counter"] == 1 for r in rows)


@pytest.mark.asyncio
async def test_dedup_respects_brand_boundary(fake_supabase: FakeSupabase) -> None:
    """2 rows with identical key fields but different brands MUST both
    remain. Brand boundary is sacrosanct."""
    for brand in ("Kisqali", "Fabhalta"):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m-{brand}",
                "brand": brand,
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-abc",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": "2026-05-20T00:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    # Run dedup unscoped so it sweeps all brands; still must NOT collapse
    # across brand.
    await consolidator.deduplicate_episodic(brand=None, region=None)
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 2
    brands = sorted(r["brand"] for r in rows)
    assert brands == ["Fabhalta", "Kisqali"]


@pytest.mark.asyncio
async def test_dedup_handles_null_causal_path_via_fallback_key(
    fake_supabase: FakeSupabase,
) -> None:
    """2 rows with ``causal_path_id IS NULL`` but identical fallback-key
    fields (brand, event_type, event_subtype, agent_name, description)
    collapse to 1 row."""
    for i in range(2):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": None,
                "agent_name": "estimator",
                "description": "identical prose for fallback dedup",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 1, f"expected fallback collapse, got {len(rows)} rows"
    assert rows[0]["dedup_counter"] == 2


@pytest.mark.asyncio
async def test_promotion_threshold_respects_deduplicated_counts(
    fake_supabase: FakeSupabase,
) -> None:
    """5 episodic rows deduped to 1 row with ``dedup_counter == 5`` MUST
    still trigger semantic promotion via the ``SUM(dedup_counter)``
    effective-count surface — proving the promotion threshold honors
    deduplicated counts, not raw row counts.
    """
    # Causal path candidate.
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp-merged",
            "brand": "Kisqali",
            "validation_status": "confirmed",
            "confirmation_count": 1,
            "consolidated_at": None,
        }
    )
    # 5 identical-key episodic rows citing cp-merged. After dedup, 1 row with
    # dedup_counter=5 remains. Semantic promotion threshold = 3, so 5 >= 3
    # must still fire.
    for i in range(5):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-merged",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    result = await consolidator.run(brand="Kisqali")
    # After run() the consolidator should have (a) deduped to 1 row with
    # counter=5 then (b) promoted the causal path because effective_count
    # (= SUM(dedup_counter) = 5) >= 3.
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 1, "expected dedup to collapse 5 → 1"
    assert rows[0]["dedup_counter"] == 5
    assert result.promoted_to_semantic == 1
    cp = fake_supabase.rows["causal_paths"][0]
    assert cp["consolidated_at"] is not None
    # Honors the effective (deduplicated-count-aware) confirmation count.
    assert cp["confirmation_count"] == 5


# ---------------------------------------------------------------------------
# iter-1 follow-ups: late-insert merge contract (H1) + per-group atomicity (M1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_collapses_late_inserted_duplicate_after_canonical_stamped(
    fake_supabase: FakeSupabase,
) -> None:
    """A duplicate that arrives AFTER the canonical has been stamped MUST
    be merged into the existing canonical (not silently stamped — that
    would fail the partial-unique-index) and DELETED.

    Iter-0 bug: the singleton path stamped a new row with sig_X without
    checking whether sig_X was already on a canonical row. In a real DB
    the second UPDATE would hit a UniqueViolation; in this fake-DB
    harness it would silently produce two rows with the same signature
    (because the fake has no unique-index enforcement), inflating the
    promotion count via SUM(dedup_counter).

    Iter-1 fix (Option B): pre-check by SELECT (brand, sig) before
    stamping. If a canonical already exists for that (brand, sig),
    increment ITS counter by the new row's counter and DELETE the new
    row instead of stamping it.
    """
    # Phase 1: seed 3 identical rows.
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-late-merge",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )
    consolidator = Consolidator()
    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)
    # Sanity check phase-1 result.
    rows_after_run1 = fake_supabase.rows["episodic_memories"]
    assert len(rows_after_run1) == 1
    canonical_row = rows_after_run1[0]
    assert canonical_row["dedup_counter"] == 3
    assert canonical_row["dedup_signature"] is not None
    canonical_sig = canonical_row["dedup_signature"]

    # Phase 2: insert a 4th identical row simulating a late arrival.
    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": "m-late",
            "brand": "Kisqali",
            "region": "northeast",
            "event_type": "ANALYSIS_COMPLETED",
            "event_subtype": "ate_estimation",
            "causal_path_id": "cp-late-merge",
            "agent_name": "estimator",
            "description": "d",
            "occurred_at": "2026-05-20T05:00:00Z",
            "dedup_signature": None,
            "dedup_counter": 1,
        }
    )

    # Phase 3: re-run dedup. The late row should be merged into the
    # existing canonical, NOT stamped as a duplicate-canonical with the
    # same sig (which would fail the DB partial-unique-index).
    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)
    rows_after_run2 = fake_supabase.rows["episodic_memories"]
    assert len(rows_after_run2) == 1, (
        f"expected exactly 1 row (canonical merged with late), "
        f"got {len(rows_after_run2)} — late row was not merged correctly"
    )
    assert rows_after_run2[0]["dedup_counter"] == 4
    # Only the original canonical's signature remains.
    assert rows_after_run2[0]["dedup_signature"] == canonical_sig
    # The late row was DELETED, not stamped — only one row carries sig.
    sigs = [r["dedup_signature"] for r in rows_after_run2]
    assert sigs.count(canonical_sig) == 1


@pytest.mark.asyncio
async def test_dedup_atomicity_on_delete_failure(
    fake_supabase: FakeSupabase,
) -> None:
    """If the duplicate-DELETE step fails mid-group, the canonical's
    counter MUST NOT be left bumped — otherwise promotion's
    SUM(dedup_counter) double-counts (the canonical bumped + the
    surviving duplicate's old counter).

    Iter-0 bug: counter increment + delete were not atomic; a delete
    failure stranded an inconsistent state.

    Iter-1 fix: wrap each group's (stamp + delete) in a transactional
    boundary. On any per-group failure, ROLL BACK so the group's
    counter/deletes stay in sync. Record the error in
    ``ConsolidationResult.errors``.
    """
    # Seed 3 identical rows.
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-atomic",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )

    # Patch the consolidator's delete-step entry-point so it raises.
    # We patch the FakeSupabase's table().delete() chain by wrapping the
    # `delete` method on _FakeQuery to raise.
    consolidator = Consolidator()

    original_table = fake_supabase.table
    delete_calls = {"n": 0}

    def boom_table(name: str) -> _FakeQuery:
        q = original_table(name)
        original_delete = q.delete

        def failing_delete() -> _FakeQuery:
            if name == "episodic_memories":
                delete_calls["n"] += 1
                # Raise on the FIRST delete attempt for episodic memory rows.
                raise RuntimeError("simulated delete failure")
            return original_delete()

        q.delete = failing_delete  # type: ignore[method-assign]
        return q

    fake_supabase.table = boom_table  # type: ignore[method-assign]

    result = await consolidator.deduplicate_episodic(brand="Kisqali", region=None)

    # Rollback contract: 3 rows still present, canonical counter
    # UNCHANGED, no row carries a stamped signature.
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 3, f"expected all 3 rows preserved on delete failure, got {len(rows)}"
    assert all(r["dedup_counter"] == 1 for r in rows), (
        "canonical counter was bumped despite delete failure — non-atomic collapse"
    )
    assert all(r["dedup_signature"] is None for r in rows), (
        "signature was stamped despite delete failure — non-atomic collapse"
    )
    # The error must surface on ConsolidationResult.
    assert any("delete" in e.lower() for e in result.errors), (
        f"per-group failure must surface in result.errors; got {result.errors!r}"
    )


# ---------------------------------------------------------------------------
# iter-2 follow-ups: brand-skip on unreverted error (new-H1),
# same-pass concurrent-winner recovery (new-M1),
# post-statement failure atomicity (L1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promote_to_semantic_skips_brand_with_unreverted_dedup_error(
    fake_supabase: FakeSupabase,
) -> None:
    """If a group's compensating revert ITSELF fails, the canonical is
    left bumped while duplicates survive. Promotion's SUM(dedup_counter)
    would then double-count. Iter-2 fix: promotion must SKIP that brand.

    Contract:
        * ``ConsolidationResult.brands_with_dedup_errors`` is a Set[str]
          (typed, not implied via parsing ``errors``). Brand is added
          when BOTH the original mutation failed AND the compensating
          revert failed (revertable failures stay clean).
        * ``_promote_to_semantic`` short-circuits before its SUM query
          for any brand in that set, AND records a skip-message in
          ``errors``.
    """
    # Seed a causal_path that WOULD be promotable if SUM(dedup_counter)
    # were trusted.
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp-unrev",
            "brand": "Kisqali",
            "validation_status": "confirmed",
            "confirmation_count": 1,
            "consolidated_at": None,
        }
    )
    # Seed 3 identical-key rows under the causal path.
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-unrev",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )

    # Patch the FakeSupabase so BOTH the delete AND the compensating
    # revert UPDATE raise. The revert UPDATE is identifiable as one
    # that sets ``dedup_signature`` to None (i.e. NULLing the
    # stamp) — that's how the production code spells the revert.
    consolidator = Consolidator()
    original_table = fake_supabase.table

    def boom_table(name: str) -> _FakeQuery:
        q = original_table(name)
        original_update = q.update

        def failing_delete() -> _FakeQuery:
            raise RuntimeError("simulated delete failure")

        def maybe_failing_update(payload: Dict[str, Any]) -> _FakeQuery:
            # The compensating revert sets dedup_signature back to None.
            # That's how we identify it (vs the initial stamp UPDATE which
            # sets dedup_signature to a v1:... string). Raise on revert.
            if "dedup_signature" in payload and payload.get("dedup_signature") is None:
                raise RuntimeError("simulated revert failure (unrevertable)")
            return original_update(payload)

        if name == "episodic_memories":
            q.delete = failing_delete  # type: ignore[method-assign]
            q.update = maybe_failing_update  # type: ignore[method-assign]
        return q

    fake_supabase.table = boom_table  # type: ignore[method-assign]

    result = await consolidator.run(brand="Kisqali")

    # Brand-skip contract: Kisqali must be in brands_with_dedup_errors.
    assert "Kisqali" in result.brands_with_dedup_errors, (
        f"Kisqali should be marked as unreverted; got {result.brands_with_dedup_errors!r}"
    )
    # And a skip-message must surface in errors.
    assert any("skip-promotion" in e.lower() for e in result.errors), (
        f"promotion-skip message must surface in errors; got {result.errors!r}"
    )
    # And, critically, _promote_to_semantic must NOT have stamped the
    # causal path with the (over-counted) effective count.
    cp = fake_supabase.rows["causal_paths"][0]
    assert cp["consolidated_at"] is None, (
        "promotion fired despite unreverted dedup error — double-count risk"
    )
    assert result.promoted_to_semantic == 0


@pytest.mark.asyncio
async def test_dedup_recovers_same_pass_when_concurrent_winner_stamped_first(
    fake_supabase: FakeSupabase,
) -> None:
    """A concurrent consolidator pass can win the race to stamp a
    canonical between our SELECT (no canonical found) and our UPDATE
    (stamp). The partial-unique-index rejects our stamp with
    UniqueViolation. Iter-2 fix: re-query the canonical and merge into
    it in the SAME ``deduplicate_episodic`` call — don't leave the
    loser unstamped for the next run.
    """
    # Pre-seed a "concurrent-winner" canonical row that ALREADY has the
    # signature stamped. The signature must match what
    # _compute_dedup_signature would yield for the loser row below.
    loser_row_data = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-race",
        "agent_name": "estimator",
        "description": "d",
    }
    expected_sig = _compute_dedup_signature(loser_row_data)
    assert expected_sig is not None

    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": "m-winner",
            "brand": "Kisqali",
            "region": "northeast",
            **loser_row_data,
            "occurred_at": "2026-05-20T00:00:00Z",
            "dedup_signature": expected_sig,
            "dedup_counter": 1,
        }
    )
    # The "loser" row arrives — still unstamped. The candidate filter
    # in deduplicate_episodic only picks up signature-IS-NULL rows.
    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": "m-loser",
            "brand": "Kisqali",
            "region": "northeast",
            **loser_row_data,
            "occurred_at": "2026-05-20T01:00:00Z",
            "dedup_signature": None,
            "dedup_counter": 1,
        }
    )

    # Patch the FakeSupabase so the FIRST canonical-lookup returns
    # empty (forcing the fresh-stamp path), then the UPDATE stamp
    # raises a unique-violation-shaped error. The handler must
    # re-query, find the pre-existing canonical, and merge.
    consolidator = Consolidator()
    original_table = fake_supabase.table
    lookup_calls = {"n": 0}
    stamp_attempts = {"n": 0}

    class _UniqueViolationStub(Exception):
        """Stand-in for psycopg.errors.UniqueViolation that the
        consolidator catches by class-name shape; the production code
        recognizes any exception with 'unique' in its message and
        re-queries."""

        pass

    def boom_table(name: str) -> _FakeQuery:
        q = original_table(name)
        original_select = q.select
        original_update = q.update

        def select_skipping_winner(cols: str, count: Optional[str] = None) -> _FakeQuery:
            # The canonical-lookup call uses the EXACT column string
            # "memory_id, brand, dedup_signature, dedup_counter, occurred_at"
            # (no event_type/event_subtype/causal_path_id) — distinguish
            # it from the main candidate-rows select which includes
            # those. The first canonical-lookup must return empty so the
            # fresh-stamp path runs; the second (re-query after stamp's
            # UniqueViolation) returns the winner normally.
            res = original_select(cols, count)
            is_canonical_lookup = (
                "event_type" not in cols and "memory_id" in cols and "dedup_signature" in cols
            )
            if is_canonical_lookup and lookup_calls["n"] == 0:
                original_execute = res.execute
                lookup_calls["n"] += 1

                def empty_execute() -> Any:
                    real = original_execute()
                    real.data = []
                    return real

                res.execute = empty_execute  # type: ignore[method-assign]
            elif is_canonical_lookup:
                lookup_calls["n"] += 1
            return res

        def failing_first_stamp(payload: Dict[str, Any]) -> _FakeQuery:
            # The stamp UPDATE sets dedup_signature to a v1:... value.
            if (
                payload.get("dedup_signature")
                and str(payload["dedup_signature"]).startswith("v1:")
                and stamp_attempts["n"] == 0
            ):
                stamp_attempts["n"] += 1
                # Mimic a UniqueViolation surfaced on the stamp.
                raise _UniqueViolationStub("duplicate key value violates unique constraint")
            return original_update(payload)

        if name == "episodic_memories":
            q.select = select_skipping_winner  # type: ignore[method-assign]
            q.update = failing_first_stamp  # type: ignore[method-assign]
        return q

    fake_supabase.table = boom_table  # type: ignore[method-assign]

    result = await consolidator.deduplicate_episodic(brand="Kisqali", region=None)

    # Same-pass recovery: the loser must be MERGED into the winner.
    # End state: 1 row (the winner), counter = 1 (winner) + 1 (loser) = 2,
    # signature unchanged.
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 1, (
        f"expected loser merged into winner in same pass, got {len(rows)} rows: {rows!r}"
    )
    assert rows[0]["memory_id"] == "m-winner", "winner canonical must survive"
    assert rows[0]["dedup_counter"] == 2, (
        f"loser must be merged into winner's counter; got {rows[0]['dedup_counter']}"
    )
    # No unrevertable errors recorded (the stamp failure was recovered).
    assert "Kisqali" not in result.brands_with_dedup_errors


@pytest.mark.asyncio
async def test_dedup_atomicity_on_post_statement_failure(
    fake_supabase: FakeSupabase,
) -> None:
    """Real-DB partial-failure: the DELETE statement REACHES the DB and
    mutates the underlying store, then the response fails (network drop,
    connection reset). Iter-2 L1: the revert must fire and the fake's
    post-mutation state must be restored back to pre-mutation.

    This is the deeper-than-iter-1 atomicity test — iter-1's atomicity
    test patched .delete() to raise BEFORE mutating; this one mutates
    THEN raises on the response."""
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "region": "northeast",
                "event_type": "ANALYSIS_COMPLETED",
                "event_subtype": "ate_estimation",
                "causal_path_id": "cp-postfail",
                "agent_name": "estimator",
                "description": "d",
                "occurred_at": f"2026-05-20T0{i}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )

    consolidator = Consolidator()
    original_table = fake_supabase.table
    delete_executes = {"n": 0}

    def boom_table(name: str) -> _FakeQuery:
        q = original_table(name)
        original_delete = q.delete

        def post_statement_delete() -> _FakeQuery:
            # Run the real delete chain (so the fake mutates), but wrap
            # execute() to raise AFTER the mutation has been applied.
            res = original_delete()
            original_execute = res.execute

            def execute_then_raise() -> Any:
                # Apply the real mutation first, then raise on the response.
                if name == "episodic_memories":
                    delete_executes["n"] += 1
                    original_execute()  # mutate the underlying store
                    raise RuntimeError("simulated post-statement failure")
                return original_execute()

            res.execute = execute_then_raise  # type: ignore[method-assign]
            return res

        if name == "episodic_memories":
            q.delete = post_statement_delete  # type: ignore[method-assign]
        return q

    fake_supabase.table = boom_table  # type: ignore[method-assign]

    result = await consolidator.deduplicate_episodic(brand="Kisqali", region=None)

    # The DELETE statement reached the store and removed the
    # duplicates BEFORE the .execute() raised. The compensating
    # revert MUST then have nulled the canonical's signature so
    # subsequent runs can retry the group cleanly.
    rows = fake_supabase.rows["episodic_memories"]
    # The canonical (m0) survived; the duplicates (m1, m2) were
    # deleted by the post-statement-success-then-fail. The revert
    # cannot bring them back — the contract is "canonical reverted
    # to pre-stamp state so the next run can re-examine". So we
    # expect 1 row remaining (the canonical), with dedup_signature
    # = None (reverted) and dedup_counter = 1 (pre-stamp).
    canonical_rows = [r for r in rows if r["dedup_signature"] is None]
    assert len(canonical_rows) >= 1, (
        f"reverted canonical must exist with signature=None; rows={rows!r}"
    )
    # And the error must surface in result.errors (delete + revert recorded).
    assert any("delete" in e.lower() for e in result.errors), (
        f"post-statement failure must surface; got {result.errors!r}"
    )
    # The canonical's counter must NOT have been left bumped to merged
    # value (= 3); it should be back at pre-stamp value (= 1).
    for r in canonical_rows:
        assert r.get("dedup_counter") == 1, (
            f"canonical counter must revert to pre-stamp value=1, got {r.get('dedup_counter')}"
        )


# ---------------------------------------------------------------------------
# iter-3 follow-ups: multi-row recovery counter correctness (new-NEW-H1)
# + _is_unique_violation false-positive rejection (new-NEW-M2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_unique_violation_multi_row_counter_correctness(
    fake_supabase: FakeSupabase,
) -> None:
    """Iter-3 new-NEW-H1: when a multi-row group hits UniqueViolation on
    its stamp UPDATE, the recovery path must add EXACTLY ``SUM(group
    counters)`` to the existing canonical — not the canonical's own
    counter PLUS the merged_counter (which would double-count).

    Iter-2 bug: ``_stamp_dedup_signature`` was called with
    ``counter=merged_counter`` (e.g. 3 for a 3-row group of singletons).
    The IntegrityError handler passed that 3 to
    ``_recover_unique_violation`` as ``loser_counter``, which then
    landed in the incoming list as the loser's own counter. Plus the
    siblings each contributed their own counter (1 each). Total:
    ``existing(1) + loser(3) + R2(1) + R3(1) = 6``. Expected
    ``existing(1) + sum(group) = 1 + 3 = 4``.

    Iter-3 fix: drop the redundant ``loser_counter`` param and read
    each row's OWN counter from its dict in the incoming list.
    """
    # Pre-seed the "winner" canonical with counter=1 + matching sig.
    winner_row_fields = {
        "brand": "Kisqali",
        "event_type": "ANALYSIS_COMPLETED",
        "event_subtype": "ate_estimation",
        "causal_path_id": "cp-multi",
        "agent_name": "estimator",
        "description": "d",
    }
    expected_sig = _compute_dedup_signature(winner_row_fields)
    assert expected_sig is not None
    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": "m-winner",
            "region": "northeast",
            **winner_row_fields,
            "occurred_at": "2026-05-20T00:00:00Z",
            "dedup_signature": expected_sig,
            "dedup_counter": 1,
        }
    )
    # Seed 3 unstamped duplicates of the same key — loser group R1/R2/R3.
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m-loser{i}",
                "region": "northeast",
                **winner_row_fields,
                "occurred_at": f"2026-05-20T0{i + 1}:00:00Z",
                "dedup_signature": None,
                "dedup_counter": 1,
            }
        )

    # Patch select to make the FIRST canonical-lookup empty (forcing the
    # fresh-stamp path), then patch update to raise UniqueViolation on
    # the first stamp attempt (the 3-row group's stamp). Subsequent
    # canonical-lookups + update calls go through normally so the
    # recovery path can re-query and merge.
    consolidator = Consolidator()
    original_table = fake_supabase.table
    lookup_calls = {"n": 0}
    stamp_attempts = {"n": 0}

    class _UniqueViolationStub(Exception):
        pass

    def boom_table(name: str) -> _FakeQuery:
        q = original_table(name)
        original_select = q.select
        original_update = q.update

        def select_skipping_winner(cols: str, count: Optional[str] = None) -> _FakeQuery:
            res = original_select(cols, count)
            is_canonical_lookup = (
                "event_type" not in cols and "memory_id" in cols and "dedup_signature" in cols
            )
            if is_canonical_lookup and lookup_calls["n"] == 0:
                original_execute = res.execute
                lookup_calls["n"] += 1

                def empty_execute() -> Any:
                    real = original_execute()
                    real.data = []
                    return real

                res.execute = empty_execute  # type: ignore[method-assign]
            elif is_canonical_lookup:
                lookup_calls["n"] += 1
            return res

        def first_stamp_violates(payload: Dict[str, Any]) -> _FakeQuery:
            if (
                payload.get("dedup_signature")
                and str(payload["dedup_signature"]).startswith("v1:")
                and stamp_attempts["n"] == 0
            ):
                stamp_attempts["n"] += 1
                raise _UniqueViolationStub("duplicate key value violates unique constraint")
            return original_update(payload)

        if name == "episodic_memories":
            q.select = select_skipping_winner  # type: ignore[method-assign]
            q.update = first_stamp_violates  # type: ignore[method-assign]
        return q

    fake_supabase.table = boom_table  # type: ignore[method-assign]

    await consolidator.deduplicate_episodic(brand="Kisqali", region=None)

    # End state: 1 row (the winner), counter = 1 (existing) + 3 (the
    # three losers, each contributing 1) = 4. NOT 6.
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) == 1, f"expected all losers merged into winner, got {len(rows)} rows: {rows!r}"
    assert rows[0]["memory_id"] == "m-winner"
    assert rows[0]["dedup_counter"] == 4, (
        f"counter must be 1+3=4 (existing + sum(group)), got {rows[0]['dedup_counter']} "
        f"— inflation bug if 6, off-by-something if other"
    )
    assert rows[0]["dedup_signature"] == expected_sig


def test_is_unique_violation_rejects_non_constraint_exceptions_with_unique_in_message() -> None:
    """Iter-3 new-NEW-M2: ``_is_unique_violation`` must NOT return True
    for non-DB-constraint exceptions that happen to mention "unique" in
    their class name or message. The iter-2 matcher was too broad — any
    custom exception with "unique" anywhere was routed through the
    recovery path, suppressing semantic promotion via the brand-error
    mark.

    Iter-3 tightens the matcher to require BOTH class-name signal
    ("UniqueViolation" substring) AND message signal ("unique" + a
    DB-constraint token like "constraint" or "index").
    """
    from src.memory.lifecycle.consolidator import Consolidator

    # Class name carries "Unique" but not "Violation" → REJECT.
    class UniqueIDError(Exception):
        pass

    assert not Consolidator._is_unique_violation(UniqueIDError("something")), (
        "class with 'Unique' but no 'Violation' must NOT be treated as UniqueViolation"
    )

    # Bare Exception with "unique" in message → REJECT (no class match).
    assert not Consolidator._is_unique_violation(Exception("unique connection error")), (
        "bare Exception with 'unique' in message but no class-name match must be REJECTED"
    )

    # Class name has "Violation" but message lacks "unique" / "constraint" / "index" → REJECT.
    class CustomViolation(Exception):
        pass

    assert not Consolidator._is_unique_violation(CustomViolation("not really")), (
        "Violation class without unique/constraint/index in message must NOT match"
    )

    # Even broader negative: class has 'unique' but no 'Violation' →
    # REJECT. (e.g., a non-DB unique-id-generator error)
    class UniqueGenError(Exception):
        pass

    assert not Consolidator._is_unique_violation(UniqueGenError("collision in id")), (
        "class with 'unique' but no 'Violation' must NOT match"
    )

    # POSITIVE: the test stub used in iter-2 tests must STILL match.
    class _UniqueViolationStub(Exception):
        pass

    assert Consolidator._is_unique_violation(
        _UniqueViolationStub("duplicate key value violates unique constraint")
    ), "real-shape UniqueViolation (class + message both signal) must match"

    # POSITIVE: the alternative message wording with 'unique index' also matches.
    assert Consolidator._is_unique_violation(
        _UniqueViolationStub("duplicate key value violates unique index uix_episodic")
    ), "'unique index' phrasing must also match (real Postgres uses either wording)"
