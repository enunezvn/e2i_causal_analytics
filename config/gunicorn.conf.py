"""Gunicorn server configuration for the E2I Causal Analytics API.

Priority 3 of the API memory-reduction effort: optional ``--preload`` plus
``gc.freeze()`` to trim per-worker baseline RSS via copy-on-write sharing of
import-time objects between the master and forked workers.

SHIPS DARK BY DEFAULT
---------------------
``preload_app`` is gated on the ``GUNICORN_PRELOAD`` env var and defaults to
False, so with no env change the API behaves EXACTLY as before (each worker
imports the app independently, no shared CoW pages, hooks are inert). Enabling
preload is gated on a later faithful PSS A/B measurement; flip it on with
``GUNICORN_PRELOAD=true`` once that measurement justifies it.

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


# DARK by default: only True when GUNICORN_PRELOAD is explicitly truthy.
preload_app = _flag_enabled("GUNICORN_PRELOAD")

# NOTE: `workers` is intentionally NOT set here. The `--workers` CLI flag owns
# the worker count (currently 2, a memory-ceiling guard). Setting it here would
# create drift between the config and the compose/Dockerfile invocations.


def when_ready(server) -> None:  # noqa: ANN001 - gunicorn hook signature
    """Master-process hook: freeze GC before workers are forked.

    No-op unless preload is enabled. When enabled, ``gc.freeze()`` moves all
    import-time objects into the permanent generation so the cyclic collector
    won't re-dirty their copy-on-write pages in forked workers.
    """
    if not preload_app:
        return

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
