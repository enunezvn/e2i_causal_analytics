"""TimesFM 2.5 on the forecast worker: batching, degradation and real inference (#2115).

Two separable things are under test.

1. THE DISPATCH CONTRACT — what the platform does when the worker is there, when it is
   not, and when it is too slow. This is the half that decides whether a chat answer
   appears at all, so it is exercised through the real Celery task object (eager mode
   runs the real task body; it is not a stand-in for one).
2. THE MODEL ITSELF — a real TimesFM forward pass over the real PROD series. It needs
   transformers>=5.10 AND the 200M weights, which is the prod api image and NOT a CI
   runner, so it skips with a reason rather than pretending. The faithful environment
   for this half is the container; see docs/demos/results for the lane's run of it.
"""

from __future__ import annotations

import os

import pytest

from src.kpi.forecast import backtest as bt
from src.kpi.forecast import models as fm
from src.kpi.forecast import timesfm as tf
from tests.unit.test_kpi.conftest_forecast_fixture import series_values

# Importing the Celery app autodiscovers every task module (dowhy, torch); the real
# TimesFM cases below load 200M weights. Explicit budget, not the 30 s default.
pytestmark = pytest.mark.timeout(600)


# ---------------------------------------------------------------- batching, the point
def test_the_whole_backtest_is_one_worker_round_trip_not_one_per_origin():
    """24 origins must cost ONE dispatch: the 200M weights load once per call, not 24x.

    Measured 2026-09-20 in the prod api image: 0.6 s to load the weights, 0.37 s per
    forecast. Per-origin dispatch would pay the load 24 times and put ~15 s of pure
    overhead into every chat answer.
    """
    calls = []

    def fake_transport(contexts, horizon):
        calls.append((len(contexts), horizon))
        return [[float(c[-1])] * horizon for c in contexts]

    y = series_values("Kisqali")
    score = bt.score_model(
        predict_many=tf.batched_predictor(fake_transport),
        y=y,
        horizon=6,
        origins=24,
        name="timesfm_2_5",
    )
    assert len(calls) == 1, f"expected one round trip, got {len(calls)}"
    assert calls[0] == (24, 6)
    assert score.n_origins == 24


def test_a_batched_predictor_keeps_its_results_aligned_to_their_origins():
    """Result k must be the forecast for context k — a shuffle would score the wrong model."""
    contexts_seen = []

    def transport(contexts, horizon):
        contexts_seen.extend(list(c)[-1] for c in contexts)
        return [[float(list(c)[-1])] * horizon for c in contexts]

    y = [float(i) for i in range(60)]
    predict_many = tf.batched_predictor(transport)
    out = predict_many([y[:30], y[:40], y[:50]], 2)
    assert contexts_seen == [29.0, 39.0, 49.0]
    assert out == [[29.0, 29.0], [39.0, 39.0], [49.0, 49.0]]


def test_one_bad_entry_in_a_batch_is_a_failed_origin_not_a_failed_forecast():
    """The worker returning a null for one context must not lose the other 23."""

    def transport(contexts, horizon):
        return [None if i == 0 else [1.0] * horizon for i, _ in enumerate(contexts)]

    score = bt.score_model(
        predict_many=tf.batched_predictor(transport),
        y=[100.0] * 60,
        horizon=6,
        origins=5,
        name="timesfm_2_5",
    )
    assert score.n_origins == 4 and score.n_failed_origins == 1


# ------------------------------------------------------------------------ degradation
def test_an_unreachable_worker_raises_the_typed_unavailable_error():
    """The service catches exactly this to drop TimesFM and serve Holt-Winters."""

    def transport(contexts, horizon):
        raise OSError("connection refused")

    with pytest.raises(tf.ForecastWorkerUnavailable):
        tf.batched_predictor(transport)([[1.0] * 30], 6)


def test_the_availability_probe_is_bounded_however_the_broker_behaves():
    """The probe is what bounds the chat turn, so the probe is what gets timed.

    MEASURED 2026-09-20: ``celery_app.send_task`` does NOT respect the
    ``.get(timeout=...)`` that follows it. ``on_task_call`` starts the Redis result
    consumer, and with no broker that enters kombu's ``retry_over_time`` loop and sleeps
    a second at a time indefinitely; a local run of the service suite hung there until
    pytest killed it. Asserting ``WORKER_TIMEOUT_SECONDS`` would have stayed green
    through exactly that bug -- the constant was always right, the dispatch just ignored
    it -- so what is asserted here is wall-clock behaviour against whatever broker this
    process actually has.
    """
    import time

    started = time.monotonic()
    ready, why = tf.worker_available()
    elapsed = time.monotonic() - started
    assert isinstance(ready, bool) and why, "the probe must always say what it found"
    assert elapsed < 4 * tf.PROBE_TIMEOUT_SECONDS, (
        f"the probe took {elapsed:.1f}s; it runs on the common path and bounds the turn"
    )


def test_a_dispatch_refuses_immediately_when_no_worker_consumes_the_queue():
    """With no worker the dispatch must raise from the PROBE, never reach send_task.

    Skipped rather than faked if a forecast worker happens to be running in this
    environment -- the live path is certified separately in
    docs/demos/results/2026-09-20_lane2115_forecast/.
    """
    import time

    ready, why = tf.worker_available()
    if ready:
        pytest.skip(f"a forecast worker is live here ({why}); the absent-worker path needs none")

    started = time.monotonic()
    with pytest.raises(tf.ForecastWorkerUnavailable) as exc:
        tf.dispatch_batch([[float(i) for i in range(40)]], 6)
    elapsed = time.monotonic() - started
    assert "not available" in str(exc.value)
    assert elapsed < 4 * tf.PROBE_TIMEOUT_SECONDS, (
        f"dispatch took {elapsed:.1f}s with no worker; it must fail over to Holt-Winters fast"
    )
    assert 0 < tf.WORKER_TIMEOUT_SECONDS <= 120, "a chat turn cannot wait longer than this"


def test_the_queue_and_task_name_are_the_ones_the_worker_actually_consumes():
    from src.workers.celery_app import celery_app

    assert tf.FORECAST_QUEUE == "forecast"
    assert tf.TIMESFM_TASK_NAME == "src.tasks.forecast_timesfm_batch"
    queues = {q.name for q in celery_app.conf.task_queues}
    assert tf.FORECAST_QUEUE in queues, "the queue must exist or every dispatch dead-letters"
    routed = celery_app.conf.task_routes.get(tf.TIMESFM_TASK_NAME)
    assert routed == {"queue": tf.FORECAST_QUEUE}


def test_the_task_is_registered_by_importing_the_package_a_worker_imports():
    """The capability, not a proxy for it.

    Importing ``src.tasks.forecast_tasks`` directly would prove only that the
    decorator runs -- which it always does. What a worker actually does is import
    ``src.tasks``, and ``autodiscover_tasks`` does NOT reach into it (it looks for a
    ``tasks`` submodule of each listed package), so the module has to be imported from
    ``src/tasks/__init__.py`` by hand. Assert the registration that the worker's own
    import path produces, or a dispatch dead-letters in production while this test
    stays green.
    """
    import src.tasks  # noqa: F401  -- exactly what the worker imports
    from src.workers.celery_app import celery_app

    assert tf.TIMESFM_TASK_NAME in celery_app.tasks
    assert celery_app.tasks[tf.TIMESFM_TASK_NAME].queue == tf.FORECAST_QUEUE


# ---------------------------------------------------------------- context length guard
def test_the_context_length_and_model_are_the_ones_the_compose_service_sets():
    """A service that sets an env var the code ignores is a knob that does nothing.

    Both are read from the environment, so the values declared on worker_forecast are
    the values the worker actually uses.
    """
    import importlib

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv("E2I_FORECAST_CONTEXT_LEN", "128")
        monkey.setenv("E2I_TIMESFM_MODEL", "google/timesfm-2.0-500m-pytorch")
        reloaded = importlib.reload(tf)
        assert reloaded.FORECAST_CONTEXT_LEN == 128
        assert reloaded.TIMESFM_MODEL_ID == "google/timesfm-2.0-500m-pytorch"
        assert len(reloaded.trim_context([float(i) for i in range(500)])) == 128
    finally:
        monkey.undo()
        importlib.reload(tf)
    assert tf.FORECAST_CONTEXT_LEN == 256, "the default must survive the reload"


def test_the_context_is_truncated_to_the_length_that_fits_the_worker_memory_budget():
    """The model default context_length is 16384 and OOMs a 3 GB worker.

    Measured 2026-09-20 in the prod api image: at forecast_context_len=256 the peak
    RSS is 1394 MB, which is what the worker is sized for.
    """
    assert tf.FORECAST_CONTEXT_LEN == 256
    long_series = [float(i) for i in range(5000)]
    trimmed = tf.trim_context(long_series)
    assert len(trimmed) == 256
    assert trimmed[-1] == 4999.0, "truncation must keep the NEWEST months"


def test_a_short_series_is_not_padded_to_the_context_length():
    assert len(tf.trim_context([1.0, 2.0, 3.0])) == 3


# ------------------------------------------------------------- the real model, for real
def _timesfm_ready() -> str:
    try:
        from transformers import TimesFm2_5ModelForPrediction  # noqa: F401
    except Exception:
        return "transformers does not ship TimesFm2_5ModelForPrediction (needs >=5.10)"
    if not tf.weights_cached() and os.getenv("E2I_TIMESFM_ALLOW_DOWNLOAD") != "1":
        return "TimesFM 2.5 weights are not in the local HF cache"
    return ""


@pytest.mark.timeout(600)
@pytest.mark.skipif(bool(_timesfm_ready()), reason=_timesfm_ready() or "ready")
def test_timesfm_really_forecasts_the_live_series_and_beats_naive():
    """A real forward pass, real weights, the real PROD series — no stand-in anywhere."""
    y = series_values("Kisqali")
    forecasts = tf.forecast_batch([y[:-6]], horizon=6)
    assert len(forecasts) == 1 and len(forecasts[0]) == 6
    assert all(v > 0 for v in forecasts[0])

    score = bt.score_model(
        predict_many=tf.batched_predictor(tf.forecast_batch),
        y=y,
        horizon=6,
        origins=bt.DEFAULT_ORIGINS,
        name="timesfm_2_5",
    )
    naive = bt.score_model(
        lambda a, h: [a[-1]] * h, y, horizon=6, origins=bt.DEFAULT_ORIGINS, name="naive"
    )
    assert score.n_failed_origins == 0
    # Measured 2026-09-20 in the prod api image: 7.58% monthly MAPE against naive 10.42%.
    assert score.monthly_mape < naive.monthly_mape
    assert score.monthly_mape < 10.0


@pytest.mark.timeout(600)
@pytest.mark.skipif(bool(_timesfm_ready()), reason=_timesfm_ready() or "ready")
def test_the_celery_task_body_returns_exactly_what_the_model_returns():
    """The worker must not transform the forecast on its way back.

    The task is a thin shell over ``forecast_batch`` so that the code running in the
    worker IS the code these tests exercise. This runs both for real, on real weights,
    and compares them -- a shell that rounded, reordered or truncated would show up
    here. It deliberately calls the task BODY rather than dispatching, so it needs no
    broker; the full broker -> worker -> weights chain is certified live in
    docs/demos/results/2026-09-20_lane2115_forecast/.
    """
    from src.tasks.forecast_tasks import forecast_timesfm_batch

    y = series_values("Kisqali")
    direct = tf.forecast_batch([y[:-6], y[:-12]], horizon=6)
    through_task = forecast_timesfm_batch.run([y[:-6], y[:-12]], 6)
    assert through_task == pytest.approx(direct, rel=1e-9)


def test_the_catalogue_entry_routes_through_the_worker_rather_than_running_in_process():
    """A TimesFM that quietly ran in-process would put a 1.4 GB spike inside e2i_api.

    No weights needed: this is about which code path the catalogue entry takes.
    """
    import inspect

    from src.kpi.forecast import models as models_module

    assert fm.get_model("timesfm_2_5").requires_worker is True
    source = inspect.getsource(models_module._timesfm_predict)
    assert "predict_via_worker" in source
    assert "forecast_batch" not in source, "the catalogue must not call the model directly"
