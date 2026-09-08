"""Lock the agents lane's MLflow HTTP retry budget (issue #1962).

``tests/unit/test_agents/conftest.py`` bounds MLflow's HTTP client so that an
unreachable tracking server produces a *named failing test* instead of an
unbounded stall that runs the shard to its job cap. Read that module's docstring
for the measurements and the sizing argument; this file pins the contract:

1. the bounds are actually armed in-session (a stray ``.env`` value or a later
   ``load_dotenv(override=True)`` must not silently defeat them);
2. the worst-case cost of one REST call still fits inside the lane's own
   ``--timeout``, read out of the workflow so the two cannot drift apart;
3. and — the positive control — a REST call against a dead tracking URI really
   does come back promptly. Without the conftest this same call takes 246s
   (measured, ECONNREFUSED, MLflow 3.15.1 defaults), which is past the lane's
   180s per-test budget, and ``timeout_method="thread"`` turns that into an
   ``os._exit`` on the xdist worker rather than a failure.

Test 3 runs the call on a *daemon* thread and joins with a timeout, so a
regression fails this one test in bounded time instead of reproducing the very
hang the fix removes.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import pytest
import yaml

from tests.unit.test_agents import conftest as agents_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-tests.yml"
LANE_JOB = "agents-tests"
_LANE_TIMEOUT = re.compile(r"--timeout[= ](\d+)")

#: Closed port on loopback: connect() gets ECONNREFUSED immediately, so the only
#: thing this can measure is MLflow's retry/backoff budget. Port 1 is reserved
#: (tcpmux) and never bound by this repo's services.
DEAD_TRACKING_URI = "http://127.0.0.1:1"

EXPECTED = {
    "MLFLOW_HTTP_REQUEST_TIMEOUT": agents_conftest.MLFLOW_HTTP_REQUEST_TIMEOUT,
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": agents_conftest.MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": agents_conftest.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER": agents_conftest.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER,
}


def _worst_case_call_seconds() -> float:
    """Upper bound on one MLflow REST call, from the armed bounds.

    ``(retries + 1)`` attempts of ``timeout`` seconds each, plus urllib3's
    exponential backoff between them: ``backoff_factor * 2 ** (n - 1)`` for
    n = 1..retries, with the first interval clamped to 0.
    """
    timeout = float(EXPECTED["MLFLOW_HTTP_REQUEST_TIMEOUT"])
    retries = int(EXPECTED["MLFLOW_HTTP_REQUEST_MAX_RETRIES"])
    backoff = float(EXPECTED["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"])
    jitter = float(EXPECTED["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER"])
    sleeps = sum(backoff * 2 ** (n - 1) + jitter for n in range(1, retries + 1)) - backoff
    return (retries + 1) * timeout + max(sleeps, 0.0)


def _lane_per_test_timeout() -> int:
    """The ``--timeout`` the agents lane passes pytest, read from the workflow."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = (workflow.get("jobs") or {})[LANE_JOB]
    scripts = "\n".join(step.get("run", "") for step in job.get("steps", []))
    matches = _LANE_TIMEOUT.findall(scripts)
    assert matches, f"{WORKFLOW.name}::{LANE_JOB} passes pytest no --timeout"
    return min(int(m) for m in matches)


class TestMlflowHttpBudgetArmed:
    """The bounds are live in this session."""

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_environment_carries_the_bound(self, name: str) -> None:
        assert os.environ.get(name) == EXPECTED[name], (
            f"{name} is {os.environ.get(name)!r}, expected {EXPECTED[name]!r}. "
            "tests/unit/test_agents/conftest.py assigns it unconditionally; a "
            "later load_dotenv(override=True) or a lane env var has defeated it."
        )

    def test_mlflow_itself_reads_the_bound(self) -> None:
        """Pin the values through MLflow's own accessors, not just os.environ."""
        from mlflow.environment_variables import (
            MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
            MLFLOW_HTTP_REQUEST_MAX_RETRIES,
            MLFLOW_HTTP_REQUEST_TIMEOUT,
        )

        assert MLFLOW_HTTP_REQUEST_TIMEOUT.get() == int(EXPECTED["MLFLOW_HTTP_REQUEST_TIMEOUT"])
        assert MLFLOW_HTTP_REQUEST_MAX_RETRIES.get() == int(
            EXPECTED["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
        )
        assert MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.get() == int(
            EXPECTED["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"]
        )


class TestMlflowHttpBudgetFitsTheLane:
    """The bound is small enough to surface as a test failure, not a job cap."""

    def test_worst_case_call_fits_the_lane_per_test_timeout(self) -> None:
        worst = _worst_case_call_seconds()
        lane = _lane_per_test_timeout()
        assert worst < lane / 2, (
            f"one MLflow REST call can cost up to {worst:g}s against a dead "
            f"tracking server, but {LANE_JOB} runs pytest with --timeout={lane}. "
            "A call that outlives the per-test timeout does not fail the test — "
            'timeout_method="thread" os._exit()s the xdist worker and, under '
            "--max-worker-restart=0, ends the session with no verdict. Lower the "
            "bounds in tests/unit/test_agents/conftest.py or raise the lane's "
            "--timeout."
        )

    def test_bound_is_far_above_a_healthy_call(self) -> None:
        """A healthy localhost metadata call measured 0.31s; keep real headroom."""
        assert float(EXPECTED["MLFLOW_HTTP_REQUEST_TIMEOUT"]) >= 5.0


class TestDeadTrackingUriFailsFast:
    """Positive control: force the trigger and show the call comes back."""

    def test_rest_call_against_dead_uri_returns_within_the_bound(self) -> None:
        from mlflow.tracking import MlflowClient

        budget = _worst_case_call_seconds() + 5.0
        outcome: dict[str, object] = {}

        def _call() -> None:
            started = time.monotonic()
            try:
                MlflowClient(tracking_uri=DEAD_TRACKING_URI).get_experiment_by_name(
                    "e2i_causal/causal_impact/issue-1962-probe"
                )
                outcome["raised"] = None
            except Exception as exc:  # noqa: BLE001 — any failure is a pass here
                outcome["raised"] = exc
            outcome["elapsed"] = time.monotonic() - started

        # daemon=True: if the fix regresses, this thread is still blocked in
        # socket retry/backoff when we give up. A daemon thread does not hold
        # interpreter shutdown, so the session reports THIS failure instead of
        # inheriting the stall.
        worker = threading.Thread(target=_call, daemon=True, name="mlflow-1962-probe")
        worker.start()
        worker.join(timeout=budget)

        assert not worker.is_alive(), (
            f"an MLflow REST call to {DEAD_TRACKING_URI} was still retrying after "
            f"{budget:g}s. The agents-lane HTTP bounds are not in effect; with "
            "MLflow's shipped defaults this same call takes 246s (measured) and a "
            "dead tracking server runs the shard to its job cap (issue #1962)."
        )
        assert "elapsed" in outcome
        assert float(outcome["elapsed"]) < budget  # type: ignore[arg-type]
