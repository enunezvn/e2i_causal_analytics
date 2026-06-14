"""MetricRecorder — idempotent delete-then-insert wrapper around PerformanceMetricRepository.

Design notes
------------
* ``_resolve_model_id`` is imported at module level so tests can monkeypatch
  it via ``src.mlops.gold_standard_eval.recorder._resolve_model_id``.
* ``split_version`` in P1 is carried ONLY via the delete-scope filter
  (``delete_metrics(model_id, source, split_version)``).  ``record_metrics``
  does not expose a metadata kwarg, so embedding split_version into each row's
  metadata would require extending PerformanceMetricRecord — deferred to a
  later task.  The idempotency guarantee (re-run clears exactly the prior rows
  for this source+split) is fully preserved: the delete scope matches the insert
  scope because both are keyed by (model_id, source).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

# Import the async resolver at module level so it can be monkeypatched in tests.
from src.repositories.drift_monitoring import _resolve_model_id

if TYPE_CHECKING:
    from src.repositories.drift_monitoring import PerformanceMetricRepository

logger = logging.getLogger(__name__)


class MetricRecorder:
    """Idempotent recorder: delete prior rows for a run, then bulk-insert.

    Usage::

        recorder = MetricRecorder(repo)
        await recorder.record_run(
            model_version="propensity_v2.1.0",
            points=[
                (datetime(2026, 1, 1, tzinfo=timezone.utc), {"auc_roc": 0.82}, 500),
                (datetime(2026, 2, 1, tzinfo=timezone.utc), {"auc_roc": 0.83}, 490),
            ],
            source="backtest_wf",
            split_version="e2i_pilot_v3",
        )

    The delete step fires once with the fully-resolved ``model_id`` to clear any
    rows from a prior run (the ``ml_performance_metrics`` table has no unique
    key, so without the delete step re-runs accumulate duplicate rows).
    """

    def __init__(self, repo: "PerformanceMetricRepository") -> None:
        self.repo = repo

    async def record_run(
        self,
        model_version: str,
        points: list[tuple[datetime, dict[str, float], int]],
        *,
        source: str,
        split_version: str | None = None,
    ) -> None:
        """Delete prior rows for this (model_version, source, split_version), then insert.

        Args:
            model_version: Model handle or uuid string passed to the repository.
            points: Sequence of ``(measured_at_month, metrics_dict, sample_size)``
                tuples.  Each maps to one ``record_metrics`` call with
                ``measured_at=measured_at_month``.
            source: Tag for the metric origin (``'backtest_wf'`` or ``'holdout'``).
            split_version: Optional cohort/split label.  Used to scope the
                delete filter so only rows from this split are cleared.  NOT
                embedded in individual metric rows in P1 (see module docstring).
                Must be ``None`` until row-level metadata storage is implemented
                (P2); passing a non-None value would cause the delete filter to
                match no rows while the insert still fires, breaking idempotency.
        """
        if split_version is not None:
            raise NotImplementedError(
                "split_version isolation requires writing it to row metadata (P2); pass None"
            )

        # Step 1 — resolve model handle → uuid ONCE
        model_id = await _resolve_model_id(self.repo.client, model_version)

        # Step 2 — delete prior rows (idempotency); fires BEFORE any insert
        deleted = await self.repo.delete_metrics(model_id, source, split_version)
        logger.info(
            "MetricRecorder: deleted %d prior row(s) for model_id=%r source=%r split_version=%r",
            deleted,
            model_id,
            source,
            split_version,
        )

        # Step 3 — insert one record_metrics call per point
        for measured_at_month, metrics, sample_size in points:
            await self.repo.record_metrics(
                model_version,
                metrics,
                sample_size,
                measured_at_month,  # window_start = month boundary
                measured_at_month,  # window_end   = same (month-grain; caller refines if needed)
                measured_at=measured_at_month,
                source=source,
            )
