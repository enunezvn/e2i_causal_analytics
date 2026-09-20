"""TimesFM 2.5 on the dedicated ``forecast`` Celery worker (#2115, Lane B).

Owner decision 2026-09-15: TimesFM 2.5 (``google/timesfm-2.5-200m-transformers``,
Apache-2.0) runs on its OWN worker built from the api image, concurrency 1, with
``HF_HOME`` on a named volume — never inside ``e2i_api`` and not in ``e2i_bentoml``.

The reasons are measured, not stylistic. In the prod api image, capped at 3 GB and 2
CPU (2026-09-20): the weights load in 0.6 s warm and one forecast takes 0.37 s, but
peak RSS is 1394 MB. ``e2i_api`` serves every chat turn on a box whose swap is already
half used; a 1.4 GB spike inside it is how the app tier dies. ``e2i_bentoml`` has no
torch and a 512 MB limit, and a failed start there rolls back the app tier.

TWO THINGS MATTER MOST IN THIS MODULE.

* **Batching.** A rolling-origin backtest asks for 24 forecasts. Dispatching 24 tasks
  would pay the weight load 24 times and put ~15 s of pure overhead into a chat turn.
  One task takes every origin's context and returns every forecast, aligned.
* **Degradation.** The worker is optional by design. Everything that can go wrong with
  it — not deployed, busy, slow, broken — surfaces as ``ForecastWorkerUnavailable``,
  which the service catches to drop TimesFM and serve Holt-Winters alone, saying so.
  A forecast must never depend on a worker the box may not have headroom to run.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-transformers"
TIMESFM_TASK_NAME = "src.tasks.forecast_timesfm_batch"
FORECAST_QUEUE = "forecast"

#: The model's own default is 16384, which OOMs a 3 GB worker. 256 months is ~21 years
#: — longer than any canonical series (165 months on 2026-09-20) — and peaks at
#: 1394 MB, which is what the worker is sized for.
FORECAST_CONTEXT_LEN = 256

#: A chat turn cannot wait longer than this. Measured cost of the real work is ~9 s for
#: 24 origins plus a 0.6 s warm load, so 90 s is a generous ceiling that still fails the
#: turn over to Holt-Winters rather than hanging it.
WORKER_TIMEOUT_SECONDS = int(os.getenv("E2I_FORECAST_WORKER_TIMEOUT", "90"))

#: How long the AVAILABILITY probe may take. It runs on the common path — most
#: requests happen with no forecast worker deployed — so it has to be cheap.
PROBE_TIMEOUT_SECONDS = float(os.getenv("E2I_FORECAST_PROBE_TIMEOUT", "2.0"))


class ForecastWorkerUnavailable(RuntimeError):
    """TimesFM could not be reached or did not answer in time. Holt-Winters still can."""


def trim_context(y: Sequence[float]) -> List[float]:
    """Keep the NEWEST ``FORECAST_CONTEXT_LEN`` observations (an old tail is not context)."""
    values = [float(v) for v in y]
    return values[-FORECAST_CONTEXT_LEN:]


def batched_predictor(transport: Any):
    """Adapt a batch transport to the ``predict_many`` shape ``backtest.score_model`` takes.

    ``transport(contexts, horizon)`` is either the Celery dispatch or, inside the
    worker, the local forward pass — same signature, so the worker and the caller run
    the SAME code path over the same contexts.
    """

    def predict_many(
        contexts: Sequence[Sequence[float]], horizon: int
    ) -> List[Optional[Sequence[float]]]:
        trimmed = [trim_context(c) for c in contexts]
        try:
            raw = transport(trimmed, horizon)
        except ForecastWorkerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — every transport failure is one outcome
            raise ForecastWorkerUnavailable(f"TimesFM dispatch failed: {exc}") from exc
        results = list(raw)
        if len(results) != len(trimmed):
            raise ForecastWorkerUnavailable(
                f"TimesFM returned {len(results)} forecasts for {len(trimmed)} contexts"
            )
        return [None if r is None else list(r) for r in results]

    return predict_many


# --------------------------------------------------------------------------- transport
def worker_available(timeout: float = PROBE_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Is a worker actually CONSUMING the forecast queue right now?

    This probe is not politeness, it is the thing that keeps a chat turn from hanging.
    MEASURED 2026-09-20: ``celery_app.send_task`` with the Redis result backend does
    NOT respect the ``.get(timeout=...)`` that follows it — ``on_task_call`` starts the
    result consumer, and with no broker it enters kombu's ``retry_over_time`` loop and
    sleeps a second at a time indefinitely. A local run of the service suite hung there
    until the pytest timeout killed it. So the dispatch is gated on a bounded probe
    instead of on a timeout the dispatch ignores.

    It asks the CAPABILITY (a worker consuming this queue), not a proxy for it: a live
    broker with no forecast worker, and a running worker consuming other queues, both
    answer False — and each would otherwise have parked the batch on a queue nothing
    drains until ``.get`` gave up.
    """
    try:
        from src.workers.celery_app import celery_app
    except Exception as exc:  # noqa: BLE001
        return False, f"Celery is not importable: {exc}"
    try:
        inspector = celery_app.control.inspect(timeout=timeout)
        active = inspector.active_queues() or {}
    except Exception as exc:  # noqa: BLE001
        return False, f"the broker did not answer in {timeout}s: {exc}"
    for worker, queues in active.items():
        for queue in queues or ():
            if queue.get("name") == FORECAST_QUEUE:
                return True, f"{worker} is consuming {FORECAST_QUEUE}"
    if not active:
        return False, "no Celery worker answered the broker"
    return False, f"no worker is consuming the {FORECAST_QUEUE!r} queue"


def dispatch_batch(
    contexts: Sequence[Sequence[float]], horizon: int
) -> List[Optional[List[float]]]:
    """Send one batch to the forecast worker and wait for it.

    Every failure mode collapses to ``ForecastWorkerUnavailable`` on purpose: the
    caller's only decision is whether TimesFM took part in this request.
    """
    ready, why = worker_available()
    if not ready:
        raise ForecastWorkerUnavailable(f"the TimesFM forecast worker is not available: {why}")
    from src.workers.celery_app import celery_app

    try:
        async_result = celery_app.send_task(
            TIMESFM_TASK_NAME,
            args=[[list(map(float, c)) for c in contexts], int(horizon)],
            queue=FORECAST_QUEUE,
            retry=False,
        )
        return list(async_result.get(timeout=WORKER_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001
        raise ForecastWorkerUnavailable(f"TimesFM worker did not answer: {exc}") from exc


def predict_via_worker(y: Sequence[float], horizon: int) -> List[float]:
    """The single-series entry the model catalogue uses."""
    out = batched_predictor(dispatch_batch)([y], horizon)[0]
    if out is None:
        raise ForecastWorkerUnavailable("TimesFM returned no forecast for the series")
    return [float(v) for v in out]


# ------------------------------------------------------------- the model, in the worker
_MODEL: Any = None


def weights_cached() -> bool:
    """True when the 200M weights are already on disk, so a load costs no network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    try:
        hit = try_to_load_from_cache(TIMESFM_MODEL_ID, "model.safetensors")
    except Exception:  # noqa: BLE001
        return False
    return isinstance(hit, str)


def load_model() -> Any:
    """Load TimesFM once per worker process and keep it (concurrency 1, so one copy)."""
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import TimesFm2_5ModelForPrediction

        # The worker is capped at 2 CPUs; letting torch spawn more threads than that
        # makes it slower, not faster, and oversubscribes a box that has no room.
        torch.set_num_threads(int(os.getenv("E2I_FORECAST_TORCH_THREADS", "2")))
        model = TimesFm2_5ModelForPrediction.from_pretrained(TIMESFM_MODEL_ID, dtype=torch.float32)
        model.eval()
        _MODEL = model
        logger.info("TimesFM 2.5 loaded (%s)", TIMESFM_MODEL_ID)
    return _MODEL


def forecast_batch(
    contexts: Sequence[Sequence[float]], horizon: int
) -> List[Optional[List[float]]]:
    """One real forward pass over every context. Runs INSIDE the forecast worker.

    A context that the model cannot serve yields ``None`` for that entry alone, so one
    bad origin costs one origin.
    """
    import torch

    model = load_model()
    past_values = [torch.tensor(trim_context(c), dtype=torch.float32) for c in contexts]
    with torch.no_grad():
        output = model(past_values=past_values, forecast_context_len=FORECAST_CONTEXT_LEN)
    out: List[Optional[List[float]]] = []
    for row in output.mean_predictions:
        values = [float(v) for v in row[:horizon]]
        out.append(values if len(values) == horizon else None)
    return out
