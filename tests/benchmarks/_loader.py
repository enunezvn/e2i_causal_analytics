"""Labeled-query-set loader for the retrieval-quality benchmark.

Parses ``tests/benchmarks/data/retrieval_queries.jsonl`` (or any compatible
file) into a list of ``LabeledQuery`` dataclasses for the harness.

Schema documented in ``tests/benchmarks/data/CURATION.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Constants — these match the Tier 3 wire-in landed by PR #374 (issue #373)
# and the curated query categories from issue #377 scope §A.
_VALID_CATEGORIES = frozenset({"brand-scoped", "kpi-scoped", "entity-grounded", "mixed-source"})
_VALID_TIER3_CONSUMERS = frozenset(
    {"drift_monitor", "experiment_designer", "experiment_monitor", "health_score"}
)
_REQUIRED_FIELDS = (
    "query_id",
    "query_text",
    "relevant_doc_ids",
    "category",
    "expected_sources",
    "tier3_consumer",
    "filters",
    "max_staleness",
)


@dataclass(frozen=True)
class LabeledQuery:
    """A single labeled query / expected-relevant-doc pair.

    Fields mirror the JSONL schema documented in CURATION.md. Frozen so
    accidental mutation in the harness (e.g., normalising filters in
    place) raises rather than silently corrupting the shared list.
    """

    query_id: str
    query_text: str
    relevant_doc_ids: List[str]
    category: str
    expected_sources: List[str]
    tier3_consumer: str
    filters: Dict[str, Any]
    max_staleness: float
    relevance_grades: Dict[str, int] = field(default_factory=dict)
    notes: Optional[str] = None


def _validate_row(row: Dict[str, Any], line_no: int) -> None:
    """Validate a single parsed JSON row; raise on schema violation.

    Errors are deliberately specific so a malformed query-set is debuggable
    without printing the whole file.
    """
    for field_name in _REQUIRED_FIELDS:
        if field_name not in row:
            raise KeyError(f"line {line_no}: missing required field {field_name!r}")
    if row["category"] not in _VALID_CATEGORIES:
        raise ValueError(
            f"line {line_no}: unknown category {row['category']!r}; "
            f"expected one of {sorted(_VALID_CATEGORIES)}"
        )
    if row["tier3_consumer"] not in _VALID_TIER3_CONSUMERS:
        raise ValueError(
            f"line {line_no}: unknown tier3_consumer "
            f"{row['tier3_consumer']!r}; expected one of "
            f"{sorted(_VALID_TIER3_CONSUMERS)}"
        )

    # Type guards. ``bool`` must be excluded from numeric checks because
    # ``isinstance(True, int)`` is True (per PR #374 codex-finding pattern).
    if isinstance(row["max_staleness"], bool) or not isinstance(row["max_staleness"], (int, float)):
        raise ValueError(
            f"line {line_no}: max_staleness must be numeric, got "
            f"{type(row['max_staleness']).__name__}"
        )

    if not isinstance(row["relevant_doc_ids"], list):
        raise ValueError(
            f"line {line_no}: relevant_doc_ids must be list, got "
            f"{type(row['relevant_doc_ids']).__name__}"
        )
    if not isinstance(row["expected_sources"], list):
        raise ValueError(
            f"line {line_no}: expected_sources must be list, got "
            f"{type(row['expected_sources']).__name__}"
        )
    if not isinstance(row["filters"], dict):
        raise ValueError(
            f"line {line_no}: filters must be dict, got {type(row['filters']).__name__}"
        )


def load_queries(path: Union[str, Path]) -> List[LabeledQuery]:
    """Parse a JSONL labeled-query-set file into ``LabeledQuery`` objects.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of ``LabeledQuery`` in file order.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if a row is missing a required field.
        ValueError: if a row's category / tier3_consumer / types fail
            validation, OR if duplicate ``query_id`` values appear.
        json.JSONDecodeError: if a non-comment line is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"query-set file not found: {file_path}")

    queries: List[LabeledQuery] = []
    seen_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            # Skip blank lines and comment lines (header / section markers).
            if not stripped or stripped.startswith("#"):
                continue

            row = json.loads(stripped)
            _validate_row(row, line_no)

            qid = row["query_id"]
            if qid in seen_ids:
                raise ValueError(
                    f"line {line_no}: duplicate query_id {qid!r}; "
                    f"every row must have a unique query_id"
                )
            seen_ids.add(qid)

            queries.append(
                LabeledQuery(
                    query_id=qid,
                    query_text=row["query_text"],
                    relevant_doc_ids=list(row["relevant_doc_ids"]),
                    category=row["category"],
                    expected_sources=list(row["expected_sources"]),
                    tier3_consumer=row["tier3_consumer"],
                    filters=dict(row["filters"]),
                    max_staleness=float(row["max_staleness"]),
                    relevance_grades=dict(row.get("relevance_grades", {})),
                    notes=row.get("notes"),
                )
            )

    return queries
