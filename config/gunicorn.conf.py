"""Gunicorn server configuration for the E2I Causal Analytics API.

Priority 3 of the API memory-reduction effort: optional ``--preload`` plus
``gc.freeze()`` to trim per-worker baseline RSS via copy-on-write sharing of
import-time objects between the master and forked workers.

#1560 repurposes the same preload capability as the fix for the measured
/chat/stream stream-tear root cause: without preload, EVERY worker re-imports
the whole app on the event-loop thread at boot (py-spy 2026-08-12: MainThread
174.8/175s inside importlib during ``gunicorn util.import_app``), uvicorn's
``callback_notify`` heartbeat never fires, and the arbiter SIGABRTs the worker
mid-stream at first-notify+120s. Each murder recycles a worker to cold, so the
next heavy turn pays the same import storm — self-perpetuating. With preload
the master imports once pre-fork and murdered/recycled workers re-fork WARM.

ENABLEMENT
----------
``preload_app`` is gated on the ``GUNICORN_PRELOAD`` env var (in-code default
False). Since #1560 the deployed stack enables it via compose ``x-common-env``
(``GUNICORN_PRELOAD: ${GUNICORN_PRELOAD:-true}``); setting
``GUNICORN_PRELOAD=false`` in the host ``.env`` is the rollback kill switch.
The original PSS A/B gate from PR #569 was superseded by the #1560 measured
root cause: preload is now motivated by worker-boot loop starvation, not
memory alone (a local PSS A/B is recorded on the #1560 PR).

This file is wired in via ``gunicorn --config /app/config/gunicorn.conf.py``.
The ``--workers`` CLI flag remains the single source of truth for the worker
count; this file deliberately does NOT set ``workers`` to avoid drift.

Why hooks are needed when preload is on
---------------------------------------
With ``--preload`` the app is imported ONCE in the master, then workers are
``fork``-ed. POSIX ``fork`` does not copy threads or re-run module import, so
any thread/socket created at import time is dead in the children. Three
import-time fork hazards are re-initialized per worker in ``post_fork``:

1. SHAP realtime explainer's module-global ``ThreadPoolExecutor``.
2. OpenTelemetry's ``BatchSpanProcessor`` background export thread.
3. Sentry's SDK transport (re-init is cheap and ensures a clean per-worker
   transport after fork).

``gc.freeze()`` is called in ``when_ready`` (master, after app import, before
workers fork): it moves all currently-tracked objects into the permanent
generation so the cyclic GC stops traversing them. That prevents GC from
dirtying the shared CoW pages of long-lived import-time objects in each worker.
"""

from __future__ import annotations

import gc
import os


def _flag_enabled(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# Dark in-code default; the deployed stack flips it on via compose (#1560).
preload_app = _flag_enabled("GUNICORN_PRELOAD")

# NOTE: `workers` is intentionally NOT set here. The `--workers` CLI flag owns
# the worker count (currently 2, a memory-ceiling guard). Setting it here would
# create drift between the config and the compose/Dockerfile invocations.

# --- #1560 master-side pre-fork warm of heavy LAZY leaves --------------------
#
# The app's own boot import tree (measured 2026-08-13, full `import
# src.api.main` probe) already contains dspy, litellm, transformers,
# sentence_transformers, sklearn, torch, statsmodels, shap, networkx, mlflow
# and dowhy — under preload all of those fork warm. econml and causalml were
# MEASURED absent from the boot tree, and their top-level ``__init__`` is
# near-empty (0.00s to import post-app): the real request-time import cost
# lives in the SUBMODULES named below, which are exactly the seams production
# code imports function-locally on the event loop (heterogeneous_optimizer
# cate_estimator + causal_impact refutation + digital_twin cohort estimator +
# ml_foundation model_trainer + energy_score estimator_selector +
# mlops/optuna_optimizer for econml; causal_engine/uplift for causalml;
# measured post-app-import: econml.dml 0.15s, causalml.inference.tree 2.54s
# on a warm page cache — multiplied heavily by the prod box's memory
# pressure and cold page cache, cf. the container's minutes-long boot for a
# tree this venv imports in ~40s). Importing them here lands the cost in the
# master ONCE, pre-fork and pre-``gc.freeze``, so every worker (including
# murder-respawned ones) inherits them CoW-shared and the request-time
# imports become sys.modules dict hits.
#
# dowhy is listed although currently boot-tree-resident: if a future refactor
# drops it from the boot tree its request seams would wedge again; when
# already imported the entry is a no-op dict hit.
_PRELOAD_WARM_MODULES: tuple[str, ...] = (
    "dowhy",
    "econml.dml",
    "econml.dr",
    "econml.inference",
    "econml.metalearners",
    "econml.orf",
    "econml.sklearn_extensions.linear_model",
    "causalml.inference.tree",
    "causalml.inference.meta",
)


def _warm_heavy_leaf_imports(server) -> None:  # noqa: ANN001 - gunicorn hook arg
    """Import heavy lazy-leaf modules in the master, fail-open per module."""
    import importlib
    import time

    for name in _PRELOAD_WARM_MODULES:
        start = time.perf_counter()
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - warm must never block boot
            try:
                server.log.warning(
                    "[gunicorn.conf] preload warm: import %s FAILED "
                    "(lazy request-time path unchanged): %s",
                    name,
                    exc,
                )
            except Exception:  # pragma: no cover - logging is best-effort
                pass
        else:
            try:
                server.log.info(
                    "[gunicorn.conf] preload warm: import %s ok in %.1fs",
                    name,
                    time.perf_counter() - start,
                )
            except Exception:  # pragma: no cover - logging is best-effort
                pass


def when_ready(server) -> None:  # noqa: ANN001 - gunicorn hook signature
    """Master-process hook: warm heavy lazy leaves, then freeze GC, pre-fork.

    No-op unless preload is enabled. When enabled, the #1560 heavy-leaf warm
    runs FIRST (so the freeze below covers those modules' import-time objects
    too), then ``gc.freeze()`` moves all import-time objects into the
    permanent generation so the cyclic collector won't re-dirty their
    copy-on-write pages in forked workers.
    """
    if not preload_app:
        return

    _warm_heavy_leaf_imports(server)

    # Collect first so only live objects get frozen, then freeze them so GC
    # stops touching (and thus stops un-sharing) their pages in the workers.
    gc.collect()
    gc.freeze()
    try:
        server.log.info(
            "[gunicorn.conf] preload ON: gc.freeze() applied in master "
            "before fork (%d objects frozen)",
            gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1,
        )
    except Exception:  # pragma: no cover - logging is best-effort
        pass


def post_worker_init(worker) -> None:  # noqa: ANN001 - gunicorn hook signature
    """Arm faulthandler so an arbiter SIGABRT murder names the wedged frame.

    #1560 observability — active in BOTH preload modes. The arbiter murders a
    worker whose heartbeat file goes stale for ``--timeout`` seconds (measured
    root cause: synchronous module imports starving the event loop, so
    uvicorn's ``callback_notify`` never runs).

    Why THIS hook and THIS mechanism (both were measured, 2026-08-13):

    * gunicorn's ``worker_abort`` hook NEVER fires under
      ``uvicorn.workers.UvicornWorker``: its ``init_signals`` override resets
      every gunicorn signal handler to SIG_DFL (uvicorn/workers.py, "Reset
      signals so Gunicorn doesn't swallow subprocess return codes"), so the
      murder SIGABRT terminates via the default disposition (the observed
      exit code 134) without ever entering ``handle_abort``.
    * ``faulthandler.register(SIGABRT)`` raises ``RuntimeError: signal 6
      cannot be registered, use enable() instead`` — SIGABRT is one of
      faulthandler's fatal signals. ``faulthandler.enable(all_threads=True)``
      installs a C-level handler that dumps EVERY thread's stack to stderr
      (docker logs, marker line ``Fatal Python error: Aborted``) and then
      re-raises with the default disposition, so the worker still dies with
      134 and the arbiter's murder semantics are unchanged. Being C-level it
      fires even while the interpreter is blocked inside a C call.
    * ``post_worker_init`` runs AFTER uvicorn's signal reset
      (workers/base.py: init_signals -> load_wsgi -> post_worker_init), so
      the handler is not wiped — and before the worker's first ``notify()``,
      which is the earliest moment a murder can land (gunicorn's timeout
      clock only arms at first notify), so every murderable moment is
      covered. Bonus: SIGSEGV/SIGFPE/SIGBUS/SIGILL dumps come along free.
    """
    import faulthandler

    try:
        faulthandler.enable(all_threads=True)
    except Exception:  # pragma: no cover - observability must not kill boot
        try:
            worker.log.warning(
                "[gunicorn.conf] post_worker_init: faulthandler.enable failed",
                exc_info=True,
            )
        except Exception:
            pass
        return
    try:
        worker.log.info(
            "[gunicorn.conf] faulthandler armed for SIGABRT dumps pid=%d (#1560)",
            os.getpid(),
        )
    except Exception:  # pragma: no cover - logging is best-effort
        pass


def post_fork(server, worker) -> None:  # noqa: ANN001 - gunicorn hook signature
    """Worker-process hook: re-initialize import-time fork hazards.

    No-op unless preload is enabled. When enabled, each forked worker gets a
    live SHAP thread pool, a live OpenTelemetry export thread, and a fresh
    Sentry transport. Imports are done lazily INSIDE this function to avoid
    import-order issues at config load time.
    """
    if not preload_app:
        return

    # --- Hazard 1: SHAP realtime explainer ThreadPoolExecutor -------------
    try:
        from src.mlops import shap_explainer_realtime

        shap_explainer_realtime.reset_executor()
    except Exception:  # pragma: no cover - best-effort, must not kill worker
        try:
            worker.log.warning(
                "[gunicorn.conf] post_fork: SHAP executor reset failed",
                exc_info=True,
            )
        except Exception:
            pass

    # --- Hazard 2: OpenTelemetry BatchSpanProcessor export thread ---------
    try:
        from src.api.dependencies import opentelemetry_config

        if opentelemetry_config.OTEL_ENABLED:
            opentelemetry_config.reinitialize_opentelemetry()
    except Exception:  # pragma: no cover
        try:
            worker.log.warning("[gunicorn.conf] post_fork: OTEL re-init failed", exc_info=True)
        except Exception:
            pass

    # --- Hazard 3: Sentry SDK transport ----------------------------------
    # Sentry's init is factored into configure_sentry() in main.py, which itself
    # is gated on SENTRY_DSN (no-op when unset). Re-calling it after fork gives
    # the worker a clean transport rather than one inherited across fork.
    if os.environ.get("SENTRY_DSN"):
        try:
            from src.api.main import configure_sentry

            configure_sentry()
        except Exception:  # pragma: no cover
            try:
                worker.log.warning(
                    "[gunicorn.conf] post_fork: Sentry re-init failed",
                    exc_info=True,
                )
            except Exception:
                pass

    # #1560 dispatcher live-verification marker: proves this worker forked from
    # a preloaded master (app + heavy leaves inherited, no per-worker re-import).
    try:
        worker.log.info(
            "[gunicorn.conf] preload ON: worker pid=%d forked warm (#1560)",
            os.getpid(),
        )
    except Exception:  # pragma: no cover - logging is best-effort
        pass
