"""Unit tests for the shared RRF dedup canonicalization helper."""

from src.rag.fusion_utils import dedup_key


def test_dedup_key_identical_content_distinct_ids_collide():
    """Identical content under different source_ids must produce the SAME key."""
    k1 = dedup_key("Causal analysis ATE=0.413", "mem_a")
    k2 = dedup_key("Causal analysis ATE=0.413", "mem_b")
    assert k1 == k2


def test_dedup_key_whitespace_and_case_insensitive():
    """Trivial formatting differences must not defeat content dedup."""
    assert dedup_key("  TRx Trend  ", "id1") == dedup_key("trx trend", "id2")


def test_dedup_key_empty_content_falls_back_to_source_id():
    """Empty/blank content must NOT collapse unrelated rows; fall back to source_id.

    Both '' and whitespace-only content normalize to empty and fall back to the
    SAME source_id key when the source_id matches -- that is correct: a row with no
    usable content is identified solely by its source_id.
    """
    assert dedup_key("", "id1") != dedup_key("", "id2")
    assert dedup_key("   ", "id1") != dedup_key("", "id2")
    # blank-vs-empty under the SAME source_id intentionally collapse (same fallback key)
    assert dedup_key("   ", "id3") == dedup_key("", "id3")
