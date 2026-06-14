"""Unit tests for WalkForwardRunner — expanding-origin, strictly out-of-sample.

Pure (no I/O): the runner takes an injected raw frame plus injectable
``fit_fn`` / ``predict_fn`` seams so a trivial deterministic model can be wired
in.  The three behaviours under test are the runner's whole contract:

  (a) one result per QUALIFYING month;
  (b) STRICT out-of-sample — month-M's own rows are NEVER in month-M's training
      set (captured via an instrumented ``fit_fn``);
  (c) months with ``n_eval < n_min`` or ``train_n < min_train_n`` are SKIPPED
      (absent from results) and recorded in a ``.skipped`` log.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.walk_forward import WalkForwardRunner


def _monotone_frame() -> pd.DataFrame:
    """Frame spanning 5 months; label perfectly separable by ``disease_severity``.

    Per-month row counts are deliberately uneven so the guards can be exercised:
      2026-01 -> 1 row  (too thin: train empty AND eval below n_min)
      2026-02 -> 4 rows
      2026-03 -> 4 rows
      2026-04 -> 4 rows
      2026-05 -> 1 row  (eval below n_min -> skipped on the eval guard)

    A trivial model that scores ``disease_severity`` directly separates the
    label (sev >= 0.5 -> 1) within every month, so AUC is well-defined.
    """
    rows: list[dict] = []
    months = {
        "2026-01-15": 1,
        "2026-02-15": 4,
        "2026-03-15": 4,
        "2026-04-15": 4,
        "2026-05-15": 1,
    }
    pid = 0
    for date_str, n in months.items():
        ts = pd.Timestamp(date_str, tz="UTC")
        for i in range(n):
            # Alternate label so each multi-row month has both classes.
            label = i % 2
            sev = 0.9 if label == 1 else 0.1
            rows.append(
                {
                    "patient_id": f"scvpt_{pid}",
                    "journey_start_date": ts,
                    "data_split": "train",  # MUST be ignored by walk-forward
                    "treatment_initiated": label,
                    "disease_severity": sev,
                    "academic_hcp": label,
                    "geographic_region": "west" if label == 1 else "south",
                }
            )
            pid += 1
    return pd.DataFrame(rows)


class _CapturingFit:
    """Injectable fit_fn that records the set of training journey_start_date
    months it was handed, so the test can prove month-M's own rows never leak
    into month-M's training set."""

    def __init__(self) -> None:
        # eval_month (Timestamp) -> set of training months (Timestamp) seen
        self.train_months_by_eval: dict[pd.Timestamp, set] = {}
        # eval_month -> set of training patient_ids seen
        self.train_pids_by_eval: dict[pd.Timestamp, set] = {}
        self._pending_eval_month: pd.Timestamp | None = None

    def __call__(self, train_df: pd.DataFrame):
        # The runner sets the eval month right before each fit so we can key by it.
        em = self._pending_eval_month
        months = set(
            pd.to_datetime(train_df["journey_start_date"], utc=True)
            .dt.tz_localize(None)
            .dt.to_period("M")
        )
        self.train_months_by_eval[em] = months
        self.train_pids_by_eval[em] = set(train_df["patient_id"])
        # Trivial deterministic "model": just the column name to score on.
        return ("sev_model", None)


def _predict(model, fb, eval_df: pd.DataFrame) -> np.ndarray:
    """Trivial deterministic predict: score == disease_severity (separable)."""
    return eval_df["disease_severity"].to_numpy(dtype=float)


def test_walk_forward_one_result_per_qualifying_month_and_strict_oos():
    frame = _monotone_frame()
    cap = _CapturingFit()

    runner = WalkForwardRunner(
        INITIATION,
        fit_fn=cap,
        predict_fn=_predict,
        min_train_n=2,
        n_min=2,
        # Hook so the capturing fit can key training data by the eval month.
        on_month=lambda em: setattr(cap, "_pending_eval_month", em),
    )
    results = runner.run(frame)

    # (a) one result per QUALIFYING month.
    # 2026-01: train empty + eval n=1 -> skipped.
    # 2026-02: train = Jan (1 row) < min_train_n=2 -> skipped (train guard).
    # 2026-03: train = Jan+Feb (5 rows) ok, eval n=4 ok -> EMIT.
    # 2026-04: train = Jan+Feb+Mar (9 rows) ok, eval n=4 ok -> EMIT.
    # 2026-05: eval n=1 < n_min=2 -> skipped (eval guard).
    # The runner emits the CANONICAL first-instant-of-month key (not the raw
    # day-of-month a row happened to carry) so the metric timestamp is the clean
    # month boundary MetricRecorder.record_run stores as window_start/window_end.
    emitted_months = [pd.Timestamp(m) for (m, _metrics, _n) in results]
    assert emitted_months == [
        pd.Timestamp("2026-03-01", tz="UTC"),
        pd.Timestamp("2026-04-01", tz="UTC"),
    ], f"unexpected emitted months: {emitted_months}"

    # Each emitted point is (datetime, metrics_dict, n_eval) consumable by
    # MetricRecorder.record_run.
    for month, metrics, n_eval in results:
        assert isinstance(month, dt.datetime)
        assert isinstance(metrics, dict)
        assert "auc_roc" in metrics
        assert isinstance(n_eval, int) and n_eval > 0

    # n_eval matches the month's row count (4 in both qualifying months).
    assert results[0][2] == 4
    assert results[1][2] == 4

    # (b) STRICT out-of-sample: for every emitted eval month, that month is
    # NEVER among the training months, and none of the month's own patient_ids
    # appear in its training set.
    for month, _metrics, _n in results:
        eval_period = pd.Timestamp(month).tz_localize(None).to_period("M")
        train_months = cap.train_months_by_eval[pd.Timestamp(month)]
        assert eval_period not in train_months, (
            f"LEAKAGE: eval month {eval_period} present in its own training set "
            f"{sorted(str(p) for p in train_months)}"
        )
        # Strictly earlier: every training month < eval month.
        assert all(p < eval_period for p in train_months), (
            f"LEAKAGE: a training month is not strictly < eval month {eval_period}: "
            f"{sorted(str(p) for p in train_months)}"
        )
        # No row identity leak either.
        own_pids = set(
            frame.loc[
                pd.to_datetime(frame["journey_start_date"], utc=True)
                .dt.tz_localize(None)
                .dt.to_period("M")
                == eval_period,
                "patient_id",
            ]
        )
        assert not (own_pids & cap.train_pids_by_eval[pd.Timestamp(month)]), (
            "LEAKAGE: an eval-month patient_id appeared in its own training set"
        )


def test_walk_forward_skips_are_logged_not_emitted():
    frame = _monotone_frame()
    cap = _CapturingFit()
    runner = WalkForwardRunner(
        INITIATION,
        fit_fn=cap,
        predict_fn=_predict,
        min_train_n=2,
        n_min=2,
        on_month=lambda em: setattr(cap, "_pending_eval_month", em),
    )
    results = runner.run(frame)

    emitted = {pd.Timestamp(m) for (m, _x, _y) in results}
    skipped_months = {pd.Timestamp(s.month) for s in runner.skipped}

    # The three non-qualifying months are recorded as skips, not emitted.
    # Skip months use the same canonical first-of-month key the runner emits.
    jan = pd.Timestamp("2026-01-01", tz="UTC")
    feb = pd.Timestamp("2026-02-01", tz="UTC")
    may = pd.Timestamp("2026-05-01", tz="UTC")
    for m in (jan, feb, may):
        assert m in skipped_months, f"{m} should be in .skipped"
        assert m not in emitted, f"{m} should NOT be emitted"

    # Skip reasons are recorded (train-guard vs eval-guard).
    reason_by_month = {pd.Timestamp(s.month): s.reason for s in runner.skipped}
    assert "train" in reason_by_month[feb].lower(), reason_by_month[feb]
    # May has 1 eval row -> eval guard.
    assert "eval" in reason_by_month[may].lower(), reason_by_month[may]

    # Skips carry the counts that triggered them.
    by_month = {pd.Timestamp(s.month): s for s in runner.skipped}
    assert by_month[may].n_eval == 1
    assert by_month[feb].train_n == 1  # only Jan precedes Feb


def test_walk_forward_default_window_mode_is_expanding():
    """The committed default mode is the one the experiment selected.

    This test pins the *constructor default* so a future edit can't silently
    flip the production window mode; the experiment log records WHY.
    """
    runner = WalkForwardRunner(INITIATION, fit_fn=_CapturingFit(), predict_fn=_predict)
    assert runner.window_mode == "expanding", (
        "The experiment selected 'expanding' as the default window mode; "
        "a different default would change production behaviour without a new experiment."
    )
