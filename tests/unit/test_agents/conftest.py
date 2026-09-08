"""Agents-lane conftest: bound MLflow's HTTP retry budget (issue #1962).

Why this exists
---------------
``Agents Unit Tests`` is the one unit lane that is *service-provisioned* with a
real MLflow tracking server (``.github/workflows/backend-tests.yml``, job
``agents-tests``), precisely because several agent tests dynamically exercise
UN-mocked ``mlflow.*`` calls — e.g. ``causal_impact``'s
``mlflow_tracker.start_analysis_run`` calls ``mlflow.get_experiment_by_name``
inside a ``try/except``, so the call is *made* whether or not a server answers.

The ``try/except`` bounds the *blast radius* of a dead server. It does not bound
the *time*: the exception only arrives once MLflow's HTTP client has exhausted
its retry budget. MLflow's shipped defaults are
``MLFLOW_HTTP_REQUEST_TIMEOUT=120`` / ``MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`` /
``MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=2``, i.e. **eight** attempts of up to 120s
each plus ~254s of exponential backoff — up to ~20 minutes for a single
metadata call.

Measured, one ``mlflow.get_experiment_by_name`` per row, on MLflow 3.15.1 — the
version ``requirements.txt`` pins, installed in a scratch venv so the numbers are
faithful to CI rather than to this box's older 3.11.1. Both versions were run for
the two unbounded rows and agree (3.11.1: 246.2s and >900s):

=================================  ==================  ==================
tracking URI                        shipped defaults    bounds below
=================================  ==================  ==================
live local server                            0.31s              0.31s
closed local port (ECONNREFUSED)            246.5s               ~2s
unroutable host (packets dropped)     >900s (killed)             32.0s
=================================  ==================  ==================

Why a *bound* and not a mock
----------------------------
This lane deliberately runs against a REAL MLflow (SQLite backend, so the model
registry ``model_deployer`` exercises is supported). Patching the ``mlflow``
surface out from under it would make the very tests the lane exists for vacuous.
These four variables change nothing while the server is healthy — a healthy
localhost metadata call is ~0.3s, 33x under the 10s cap — and only convert the
*unhealthy* case from an unbounded stall into a prompt ``MlflowException`` that
the trackers' existing ``except`` clauses already handle.

Why the lane hangs instead of failing, without this
---------------------------------------------------
Two session-level guards already exist and neither covers this shape:

* ``pyproject``'s ``timeout_method="thread"`` is a per-*test* guarantee, and a
  test that outlives it does not fail — ``pytest_timeout`` calls ``os._exit`` on
  the xdist worker, which under ``--max-worker-restart=0`` ends the whole
  session with no verdict.
* ``tests/stall_watchdog.py`` (#1655) fires on 600s of *silence*. A dead MLflow
  does not produce silence: it produces a steady drip of ~246s tests, so the
  controller keeps emitting reports and the watchdog never arms. The shard just
  runs ~4x its normal wall clock and is killed at the job cap — which reports as
  ``cancelled``, fails ``Backend CI Success``, and **skips** the deploy chain
  rather than failing it.

With the bounds below the same dead server produces a *named failing test*
instead, inside the lane's ``--timeout=180`` per-test budget.

Sizing
------
Worst case per REST call = ``(MAX_RETRIES + 1) * TIMEOUT + backoff``
= ``3 * 10 + (0 + 2)`` = **32s measured**, which is:

* 5.6x under this lane's ``--timeout=180`` per-test budget, so a dead server
  fails the test rather than killing its worker;
* 33x over the measured healthy-call latency, so no headroom is lost;
* still ``MAX_RETRIES=2``, so a transient 500 from the SQLite-backed server
  under ``-n 2`` contention is still ridden out rather than turned into a flake.

Hard assignments, not ``setdefault``: ``tests/conftest.py`` runs
``load_dotenv(override=True)`` (and again at configure time), so a stray value in
a developer ``.env`` or in the CI environment would otherwise silently defeat the
bound. This matches the ``MLFLOW_ALLOW_FILE_STORE`` / ``MLFLOW_UV_AUTO_DETECT``
precedent in ``tests/conftest.py``. MLflow reads all four out of ``os.environ``
at call time and keys its cached ``requests.Session`` on them, so assigning here
takes effect even when ``mlflow`` was imported earlier and has already issued a
request (verified).

Locked by ``tests/unit/test_agents/test_mlflow_http_budget_1962.py``.
"""

from __future__ import annotations

import os

#: Seconds MLflow waits for connect+read on one REST attempt (shipped default 120).
MLFLOW_HTTP_REQUEST_TIMEOUT = "10"
#: Retries after the first attempt (shipped default 7).
MLFLOW_HTTP_REQUEST_MAX_RETRIES = "2"
#: Exponential backoff base in seconds (shipped default 2).
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = "1"
#: Random jitter added to each backoff interval (shipped default 1.0).
MLFLOW_HTTP_REQUEST_BACKOFF_JITTER = "0"

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = MLFLOW_HTTP_REQUEST_TIMEOUT
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = MLFLOW_HTTP_REQUEST_MAX_RETRIES
os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR
os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER"] = MLFLOW_HTTP_REQUEST_BACKOFF_JITTER
