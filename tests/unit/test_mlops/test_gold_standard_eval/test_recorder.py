"""Unit tests for MetricRecorder — orchestration only (fake repo, no real DB).

Asserts:
  - delete called ONCE, BEFORE any record call
  - delete receives the resolved model_id, correct source, correct split_version
  - one record call per point, with the right measured_at and source
"""

import datetime as dt

import pytest

from src.mlops.gold_standard_eval.recorder import MetricRecorder


class FakeRepo:
    """Boundary double that captures call order and args."""

    def __init__(self) -> None:
        self.client = "fake_client"
        self.calls: list[tuple] = []

    async def delete_metrics(
        self,
        model_id: str,
        source: str,
        split_version: str | None = None,
    ) -> int:
        self.calls.append(("delete", model_id, source, split_version))
        return 0

    async def record_metrics(
        self,
        model_version: str,
        metrics: dict[str, float],
        sample_size: int,
        window_start: dt.datetime,
        window_end: dt.datetime,
        *,
        measured_at: dt.datetime | None = None,
        source: str | None = None,
    ) -> list:
        self.calls.append(("record", measured_at, tuple(sorted(metrics.items())), source))
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_mid(client, model_version):  # noqa: D401
    """Stand-in for _resolve_model_id that immediately returns a fixed uuid."""
    return "mid"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_deletes_once_before_inserts(monkeypatch):
    """delete_metrics must be called ONCE, BEFORE any record_metrics calls."""
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)

    m1 = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    m2 = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)

    await rec.record_run(
        "mv",
        [(m1, {"auc_roc": 0.8}, 100), (m2, {"auc_roc": 0.81}, 90)],
        source="backtest_wf",
        split_version="e2i_pilot_v3",
    )

    kinds = [c[0] for c in repo.calls]
    assert kinds == ["delete", "record", "record"], (
        f"Expected [delete, record, record], got {kinds}"
    )

    # delete called with resolved model_id, correct source, correct split_version
    assert repo.calls[0] == ("delete", "mid", "backtest_wf", "e2i_pilot_v3")

    # first insert: measured_at = m1, source correct
    assert repo.calls[1][1] == m1
    assert repo.calls[1][3] == "backtest_wf"

    # second insert: measured_at = m2, source correct
    assert repo.calls[2][1] == m2
    assert repo.calls[2][3] == "backtest_wf"


@pytest.mark.asyncio
async def test_record_run_no_points_still_deletes(monkeypatch):
    """Even with an empty points list, delete must still be called (idempotency)."""
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)

    await rec.record_run("mv", [], source="holdout", split_version=None)

    kinds = [c[0] for c in repo.calls]
    assert kinds == ["delete"], f"Expected [delete], got {kinds}"
    assert repo.calls[0] == ("delete", "mid", "holdout", None)


@pytest.mark.asyncio
async def test_record_run_passes_metrics_through(monkeypatch):
    """Metric dict contents must be passed unchanged to record_metrics."""
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)

    m1 = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    metrics = {"auc_roc": 0.75, "brier": 0.18}

    await rec.record_run("mv", [(m1, metrics, 200)], source="backtest_wf")

    # record call tuple: ("record", measured_at, sorted_items_tuple, source)
    _, measured_at, items_tuple, src = repo.calls[1]
    assert measured_at == m1
    assert dict(items_tuple) == metrics
    assert src == "backtest_wf"
