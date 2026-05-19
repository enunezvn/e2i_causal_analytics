"""Tests for the labeled-query-set loader.

The loader is responsible for:
- Parsing the JSONL file (skipping ``#`` comments).
- Validating each row against the schema documented in
  ``tests/benchmarks/data/CURATION.md``.
- Rejecting duplicate ``query_id`` values.
- Returning a typed ``LabeledQuery`` per row.

Falsifiability anchors per test are documented inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from tests.benchmarks._loader import LabeledQuery, load_queries

_DATA_FILE = Path(__file__).resolve().parent / "data" / "retrieval_queries.jsonl"


class TestLoadQueries:
    """Loader contract over the canonical shipped file."""

    def test_loads_at_least_30_queries(self) -> None:
        """Issue #377 DoD floor: ≥30 labeled queries."""
        queries = load_queries(_DATA_FILE)
        # Cite issue #377 verbatim: "Aim for ≥30 queries".
        assert len(queries) >= 30, f"Issue #377 DoD requires >=30 queries, got {len(queries)}"

    def test_returns_labeled_query_objects(self) -> None:
        """Each row should parse to a ``LabeledQuery`` with required fields."""
        queries: List[LabeledQuery] = load_queries(_DATA_FILE)
        assert all(isinstance(q, LabeledQuery) for q in queries)
        sample = queries[0]
        assert sample.query_id
        assert sample.query_text
        assert isinstance(sample.relevant_doc_ids, list)
        assert sample.category in {
            "brand-scoped",
            "kpi-scoped",
            "entity-grounded",
            "mixed-source",
        }
        assert sample.tier3_consumer in {
            "drift_monitor",
            "experiment_designer",
            "experiment_monitor",
            "health_score",
        }
        assert isinstance(sample.filters, dict)
        assert isinstance(sample.max_staleness, float)

    def test_query_ids_unique(self) -> None:
        """Duplicate ``query_id`` values would corrupt per-query aggregation."""
        queries = load_queries(_DATA_FILE)
        ids = [q.query_id for q in queries]
        assert len(ids) == len(set(ids)), "Duplicate query_id values found in shipped query-set"

    def test_covers_all_four_categories(self) -> None:
        """Issue #377 scope §A requires all four categories present."""
        queries = load_queries(_DATA_FILE)
        cats = {q.category for q in queries}
        assert cats == {
            "brand-scoped",
            "kpi-scoped",
            "entity-grounded",
            "mixed-source",
        }, f"Expected all four categories, got {cats}"

    def test_covers_all_tier3_consumers(self) -> None:
        """Plan §Recommended-sequencing item 1 wires four named Tier 3 agents."""
        queries = load_queries(_DATA_FILE)
        consumers = {q.tier3_consumer for q in queries}
        assert consumers == {
            "drift_monitor",
            "experiment_designer",
            "experiment_monitor",
            "health_score",
        }, f"Expected all four Tier 3 consumers, got {consumers}"


class TestLoaderRejectsDuplicates:
    """Synthetic test of the loader's duplicate-rejection guard.

    Falsifiability anchor: write a temp JSONL file with two rows sharing
    the same ``query_id``; loader must raise.
    """

    def test_duplicate_query_id_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "dupes.jsonl"
        bad_file.write_text(
            '{"query_id":"X","query_text":"a","relevant_doc_ids":[],'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
            '{"query_id":"X","query_text":"b","relevant_doc_ids":[],'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_queries(bad_file)


class TestLoaderSkipsComments:
    """Loader must skip ``#``-prefixed lines (header / section markers)."""

    def test_comment_lines_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "with_comments.jsonl"
        f.write_text(
            "# header comment\n"
            "# section comment\n"
            '{"query_id":"A","query_text":"q","relevant_doc_ids":["d1"],'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
            "# trailing comment\n"
        )
        queries = load_queries(f)
        assert len(queries) == 1
        assert queries[0].query_id == "A"


class TestLoaderBlankLines:
    """Blank lines (whitespace-only) MUST be skipped silently."""

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "with_blanks.jsonl"
        f.write_text(
            "\n"
            "   \n"
            '{"query_id":"A","query_text":"q","relevant_doc_ids":["d1"],'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
            "\n"
        )
        queries = load_queries(f)
        assert len(queries) == 1


class TestLoaderRejectsMalformed:
    """Loader rejects rows missing required fields with informative errors."""

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.jsonl"
        # Missing relevant_doc_ids
        f.write_text(
            '{"query_id":"A","query_text":"q",'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
        )
        with pytest.raises((KeyError, ValueError)):
            load_queries(f)

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad_cat.jsonl"
        f.write_text(
            '{"query_id":"A","query_text":"q","relevant_doc_ids":["d1"],'
            '"category":"nonsense","expected_sources":["triggers"],'
            '"tier3_consumer":"drift_monitor","filters":{},'
            '"max_staleness":0.0}\n'
        )
        with pytest.raises(ValueError, match="category"):
            load_queries(f)

    def test_unknown_tier3_consumer_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad_consumer.jsonl"
        f.write_text(
            '{"query_id":"A","query_text":"q","relevant_doc_ids":["d1"],'
            '"category":"brand-scoped","expected_sources":["triggers"],'
            '"tier3_consumer":"nonsense","filters":{},'
            '"max_staleness":0.0}\n'
        )
        with pytest.raises(ValueError, match="tier3_consumer"):
            load_queries(f)
