"""Cross-fold metric aggregation for repeated_k10 evaluation mode.

Public API:
- :class:`AggregateStat` — frozen per-metric aggregate (mean ± std + percentile CI + BCa CI).
- :func:`aggregate_fold_metrics` — across-fold aggregator consumed by
  ``_run_repeated_splits`` (shard 21 §D).

See :mod:`.fold_aggregator` for the implementation + design rationale.
"""

from .fold_aggregator import (
    AggregateStat,
    aggregate_fold_metrics,
    flatten_fold_record,
)

__all__ = ["AggregateStat", "aggregate_fold_metrics", "flatten_fold_record"]
