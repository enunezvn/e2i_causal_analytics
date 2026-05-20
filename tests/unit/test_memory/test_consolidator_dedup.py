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
