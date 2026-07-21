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
        cis: dict[str, tuple[float, float]] | None = None,
    ) -> list:
        self.calls.append(("record", measured_at, tuple(sorted(metrics.items())), source, cis))
        return []

    async def record_curve(
        self,
        model_version: str,
        kind: str,
        value: float,
        payload: dict,
        sample_size: int,
        window_start: dt.datetime,
        window_end: dt.datetime,
        *,
        measured_at: dt.datetime | None = None,
        source: str | None = None,
    ):
        self.calls.append(("curve", kind, value, dict(payload), source, measured_at))
        return None


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
        split_version=None,
    )

    kinds = [c[0] for c in repo.calls]
    assert kinds == ["delete", "record", "record"], (
        f"Expected [delete, record, record], got {kinds}"
    )

    # delete called with resolved model_id, correct source, split_version=None
    assert repo.calls[0] == ("delete", "mid", "backtest_wf", None)

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

    # record call tuple: ("record", measured_at, sorted_items_tuple, source, cis)
    _, measured_at, items_tuple, src, _cis = repo.calls[1]
    assert measured_at == m1
    assert dict(items_tuple) == metrics
    assert src == "backtest_wf"


@pytest.mark.asyncio
async def test_record_run_passes_cis_to_record_metrics(monkeypatch):
    """B2: a bootstrap CI keyed by metric name flows through to record_metrics
    so the repository can write it into the row's ci_lower/ci_upper columns."""
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)
    m1 = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)

    await rec.record_run(
        "mv",
        [(m1, {"calibration_slope": 1.4455, "auc_roc": 0.66}, 415)],
        source="holdout",
        cis={"calibration_slope": (1.22, 1.67)},
    )

    # record call tuple: ("record", measured_at, sorted_items_tuple, source, cis)
    assert repo.calls[1][4] == {"calibration_slope": (1.22, 1.67)}


@pytest.mark.asyncio
async def test_record_run_cis_defaults_to_none(monkeypatch):
    """Callers that omit cis (e.g. the walk-forward trend) record no CI."""
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)
    m1 = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)

    await rec.record_run("mv", [(m1, {"auc_roc": 0.8}, 100)], source="backtest_wf")

    assert repo.calls[1][4] is None


@pytest.mark.asyncio
async def test_record_run_split_version_not_none_raises(monkeypatch):
    """Passing a non-None split_version must raise NotImplementedError (P2 guard).

    split_version is filtered in delete_metrics but never written to row
    metadata, so a non-None value breaks idempotency (delete matches nothing
    while insert still fires). The fail-closed guard prevents accidental misuse
    until the P2 row-metadata extension lands.
    """
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)

    m1 = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)

    with pytest.raises(NotImplementedError, match="split_version isolation"):
        await rec.record_run(
            "mv",
            [(m1, {"auc_roc": 0.8}, 100)],
            source="backtest_wf",
            split_version="x",
        )


@pytest.mark.asyncio
async def test_record_curves_deletes_disjoint_source_then_inserts(monkeypatch):
    """record_curves delete-then-inserts under 'holdout_curve' (NOT 'holdout').

    The disjoint source keeps it from clobbering the scalar holdout rows that
    record_run writes under source='holdout'.
    """
    import src.mlops.gold_standard_eval.recorder as R

    monkeypatch.setattr(R, "_resolve_model_id", _async_mid)

    repo = FakeRepo()
    rec = MetricRecorder(repo)
    ts = dt.datetime(2026, 6, 10, tzinfo=dt.timezone.utc)

    await rec.record_curves(
        "mv",
        [
            ("confusion_matrix", 0.68, {"tn": 1, "fp": 2, "fn": 3, "tp": 4}),
            ("roc_curve", 0.67, {"points": [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}]}),
        ],
        measured_at=ts,
        sample_size=500,
    )

    kinds = [c[0] for c in repo.calls]
    assert kinds == ["delete", "curve", "curve"], f"got {kinds}"
    # Disjoint source so the scalar holdout rows are not deleted.
    assert repo.calls[0] == ("delete", "mid", "holdout_curve", None)
    # First curve: confusion payload + source + measured_at carried through.
    assert repo.calls[1][1] == "confusion_matrix"
    assert repo.calls[1][3] == {"tn": 1, "fp": 2, "fn": 3, "tp": 4}
    assert repo.calls[1][4] == "holdout_curve"
    assert repo.calls[1][5] == ts
    # Second curve: roc.
    assert repo.calls[2][1] == "roc_curve"
    assert repo.calls[2][3]["points"][0]["tpr"] == 0.0


@pytest.mark.asyncio
async def test_record_curves_unresolved_model_raises(monkeypatch):
    """An unresolved handle must fail closed (no NULL-model_id rows)."""
    import src.mlops.gold_standard_eval.recorder as R

    async def _none(client, model_version):
        return None

    monkeypatch.setattr(R, "_resolve_model_id", _none)

    repo = FakeRepo()
    rec = MetricRecorder(repo)

    with pytest.raises(ValueError, match="did not"):
        await rec.record_curves(
            "mv",
            [("confusion_matrix", 0.5, {})],
            measured_at=dt.datetime(2026, 6, 10, tzinfo=dt.timezone.utc),
            sample_size=10,
        )
    # Fail-closed BEFORE any delete/insert.
    assert repo.calls == []
