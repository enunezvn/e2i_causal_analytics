"""#1601: causal_impact's heavy off-loads run on a BOUNDED pool, not the loop default.

``asyncio.to_thread`` dispatches to the event loop's DEFAULT executor
(``min(32, cpu+4)`` = 12 threads on the prod box), which sidesteps every
in-process compute bound the api container has. The causal_impact agent graph
off-loaded four genuinely heavy sync calls that way:

* ``estimation.py`` — the energy-score estimator selection (fits the whole
  econml registry).
* ``refutation.py`` — the DoWhy reconstruction, the 1-sim per-refit
  calibration, and the full refutation suite.

Measured on this box (2026-08-14, real callables, real frame shapes):

===========================================  ========  ==============
call                                             wall    peak RSS delta
===========================================  ========  ==============
``_select_estimator_with_energy_score``        124.9s          177.4 MB
  (37,515x12 — largest observed prod frame)
``_reconstruct_dowhy_artifacts`` (5k subsample) 10.2s            0.3 MB
``refute_estimate(num_simulations=1)``           9.7s            0.5 MB
===========================================  ========  ==============

So estimation is the MEMORY term (177 MB x 12 concurrent ~= 2.1 GiB against a
5G cgroup already carrying ~549 MB/worker of imports) and refutation is a pure
CPU-TIME term. Both belong on a bounded pool; neither belongs on the *shared*
heavy-compute pool, which is a SINGLE thread serving reject-fast API endpoints
and #1598's composer sync tools under a 120s envelope — a ~350s causal run
would starve all of them. Hence a separate, small, bounded agent-compute pool.

These tests pin: WHICH pool runs each call, that the pool is bounded and
distinct from the shared one, that the cooperative ``compute_deadline`` (not a
``wait_for`` envelope) remains the abort mechanism, and that the happy path is
behaviour-preserving.
"""

import asyncio
import contextvars
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes import refutation as _ref_mod
from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.api.dependencies import compute as _compute_mod
from src.causal_engine.errors import RefutationError

AGENT_POOL_PREFIX = "agent-compute"
SHARED_POOL_PREFIX = "heavy-compute"


@pytest.fixture(autouse=True)
def _clean_pools():
    """Start every case from fresh pools so env overrides re-apply."""
    _compute_mod._reset_limiter_cache_for_tests()
    yield
    _compute_mod._reset_limiter_cache_for_tests()


def _frame(n: int = 200, seed: int = 1601) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "accepted": rng.integers(0, 2, n),
            "converted": rng.integers(0, 2, n),
            "confidence_score": rng.uniform(0.5, 1.0, n),
        }
    )


def _estimation_state(frame: pd.DataFrame, **extra):
    state = {
        "causal_graph": {
            "treatment_nodes": ["accepted"],
            "outcome_nodes": ["converted"],
            "nodes": ["accepted", "converted", "confidence_score"],
            "adjustment_sets": [["confidence_score"]],
        },
        "data_cache": {"estimation_data": frame},
        "data_source": "test_frame",
        "parameters": {},
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    state.update(extra)
    return state


def _refutation_state(frame: pd.DataFrame, **extra):
    state = {
        "query_id": "q-1601",
        "brand": "TestBrand",
        "treatment_var": "accepted",
        "outcome_var": "converted",
        "confounders": ["confidence_score"],
        "data_source": "synthetic",
        "estimation_result": {
            "method": "LinearDML",
            "selected_estimator": "linear_dml",
            "ate": 0.05,
            "ate_ci_lower": 0.04,
            "ate_ci_upper": 0.06,
            "effect_size": "small",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": len(frame),
            "covariates_adjusted": ["confidence_score"],
            "heterogeneity_detected": False,
        },
        "estimation_data": frame,
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    state.update(extra)
    return state


def _thread_recorder(store, key, ret):
    """A cheap REAL callable that records which pool thread ran it.

    Legitimate instrumentation: the heavy compute is replaced by a trivial real
    function so the test observes the EXECUTOR, not the estimator maths.
    """

    def _fn(*args, **kwargs):
        store[key] = threading.current_thread().name
        return ret

    return _fn


# --------------------------------------------------------------------------- #
# (a) the off-loads land on the bounded agent pool, not the loop default
# --------------------------------------------------------------------------- #


class TestOffloadsUseTheBoundedAgentPool:
    @pytest.mark.asyncio
    async def test_estimation_selection_runs_on_the_agent_compute_pool(self, monkeypatch):
        seen: dict = {}
        node = EstimationNode.__new__(EstimationNode)
        monkeypatch.setattr(
            node,
            "_select_estimator_with_energy_score",
            _thread_recorder(seen, "thread", ({"method": "LinearDML", "ate": 0.1}, {}, 1.0)),
        )

        result = await node.execute(_estimation_state(_frame()))

        assert result.get("status") == "computing", result.get("error_message")
        assert seen["thread"].startswith(AGENT_POOL_PREFIX), (
            "estimation must off-load onto the BOUNDED agent-compute pool; "
            f"observed thread={seen['thread']!r}"
        )

    @pytest.mark.asyncio
    async def test_estimation_does_not_use_the_loop_default_executor(self, monkeypatch):
        """``asyncio.to_thread`` == the loop's default executor. Pin it unused."""
        calls: list = []
        real_to_thread = asyncio.to_thread

        async def spy_to_thread(func, /, *args, **kwargs):
            calls.append(getattr(func, "__name__", repr(func)))
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)
        node = EstimationNode.__new__(EstimationNode)
        monkeypatch.setattr(
            node,
            "_select_estimator_with_energy_score",
            lambda *a, **k: ({"method": "LinearDML", "ate": 0.1}, {}, 1.0),
        )

        await node.execute(_estimation_state(_frame()))

        assert calls == [], f"estimation still reached the default executor via {calls}"

    @pytest.mark.asyncio
    async def test_refutation_reconstruct_and_suite_run_on_the_agent_compute_pool(
        self, monkeypatch
    ):
        seen: dict = {}
        node = RefutationNode()

        monkeypatch.setattr(
            _ref_mod,
            "_reconstruct_dowhy_artifacts",
            _thread_recorder(seen, "recon", (SimpleNamespace(), object(), object())),
        )

        def spy_suite(**kwargs):
            seen["suite"] = threading.current_thread().name
            raise RefutationError("stop after capture", details={"reason": "test"})

        monkeypatch.setattr(node.runner, "run_all_tests", spy_suite)

        await node.execute(_refutation_state(_frame()))

        assert seen["recon"].startswith(AGENT_POOL_PREFIX), (
            f"DoWhy reconstruction must run on the agent pool; got {seen['recon']!r}"
        )
        assert seen["suite"].startswith(AGENT_POOL_PREFIX), (
            f"the refutation suite must run on the agent pool; got {seen['suite']!r}"
        )

    @pytest.mark.asyncio
    async def test_refutation_calibration_refute_runs_on_the_agent_compute_pool(self, monkeypatch):
        """refutation.py:1240 — the 1-sim calibration is REAL DoWhy compute too.

        The issue cited :1210 and :1257 but not this one; it is the same class
        of call in the same block and is off-loaded identically.
        """
        seen: dict = {}
        clock = {"now": 5000.0}
        monkeypatch.setattr(_ref_mod.time, "monotonic", lambda: clock["now"])

        class _Model:
            def refute_estimate(self, *a, **k):
                seen["calibration"] = threading.current_thread().name
                clock["now"] += 2.0

        def fake_recon(**kwargs):
            clock["now"] += 10.0
            return (_Model(), object(), object())

        node = RefutationNode()
        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(
            node.runner,
            "run_all_tests",
            lambda **k: (_ for _ in ()).throw(
                RefutationError("stop after capture", details={"reason": "test"})
            ),
        )

        # A deadline with ample headroom so the calibration branch is taken.
        await node.execute(_refutation_state(_frame(), compute_deadline=clock["now"] + 10_000.0))

        assert seen["calibration"].startswith(AGENT_POOL_PREFIX), (
            f"the calibration refute must run on the agent pool; got {seen['calibration']!r}"
        )


# --------------------------------------------------------------------------- #
# (b) the pool is bounded, and SEPARATE from the shared heavy-compute pool
# --------------------------------------------------------------------------- #


class TestAgentPoolIsBoundedAndSeparate:
    def test_container_wide_budget_stays_within_the_cpu_quota(self):
        """The invariant that matters is CONTAINER-wide, not per-process.

        The pool is per gunicorn worker process, and the api container runs
        ``--workers 2`` inside ``cpus: '2'`` (docker/docker-compose.yml). So the
        real budget is ``workers x per-process threads``, and it must not exceed
        the CPU quota — otherwise the causal runs simply thrash each other and
        legitimate 223-300s suites start missing their dispatch budgets.

        Pinned as the invariant rather than the literal constant so that
        re-sizing the pool cannot silently break the container-wide guarantee.
        """
        gunicorn_workers = 2  # docker/docker-compose.yml: WORKERS / --workers
        cpu_quota = 2  # docker/docker-compose.yml: deploy.resources.limits.cpus
        per_process = _compute_mod._DEFAULT_AGENT_COMPUTE_WORKERS

        assert per_process >= 1
        assert per_process * gunicorn_workers <= cpu_quota, (
            f"{gunicorn_workers} gunicorn workers x {per_process} agent-compute "
            f"threads = {per_process * gunicorn_workers} concurrent heavy causal "
            f"computations on a {cpu_quota}-CPU container"
        )

    @pytest.mark.asyncio
    async def test_pool_actually_bounds_concurrency_to_its_configured_size(self):
        """Concurrency cap, measured by how many callables overlap in time."""
        workers = _compute_mod._agent_compute_workers_from_env()
        assert workers == _compute_mod._DEFAULT_AGENT_COMPUTE_WORKERS

        live = 0
        peak = 0
        lock = threading.Lock()
        release = threading.Event()

        def occupy():
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            release.wait(timeout=5)
            with lock:
                live -= 1

        tasks = [
            asyncio.create_task(_compute_mod.run_in_agent_compute_executor(occupy))
            for _ in range(6)
        ]
        await asyncio.sleep(0.4)
        observed_peak = peak
        release.set()
        await asyncio.gather(*tasks)

        assert observed_peak <= workers, (
            f"agent pool ran {observed_peak} callables concurrently; cap is {workers}"
        )
        assert observed_peak == workers, "the pool should actually use its full budget"

    @pytest.mark.asyncio
    async def test_agent_pool_is_not_the_shared_heavy_compute_pool(self):
        """The regression guard for #1598's composer tools and #1590's SHAP.

        The shared pool is ONE thread serving reject-fast API endpoints and
        composer sync tools under a 120s envelope. A causal run holds its thread
        for ~350s measured. If the two shared an executor, every composer sync
        step and every SHAP rank_drivers call would queue behind a causal
        analysis and blow their envelopes. They must be distinct executors.
        """
        agent_thread: dict = {}
        shared_thread: dict = {}

        await _compute_mod.run_in_agent_compute_executor(
            lambda: agent_thread.setdefault("name", threading.current_thread().name)
        )
        await _compute_mod.run_in_bounded_executor(
            lambda: shared_thread.setdefault("name", threading.current_thread().name)
        )

        assert agent_thread["name"].startswith(AGENT_POOL_PREFIX)
        assert shared_thread["name"].startswith(SHARED_POOL_PREFIX)
        assert _compute_mod._get_agent_compute_executor(2) is not _compute_mod._get_executor(1)

    @pytest.mark.asyncio
    async def test_a_saturated_agent_pool_does_not_block_the_shared_pool(self):
        """Concretely: a long causal run must not stall composer/SHAP work."""
        release = threading.Event()
        busy = [
            asyncio.create_task(
                _compute_mod.run_in_agent_compute_executor(lambda: release.wait(timeout=5))
            )
            for _ in range(4)  # 2 running + 2 queued => agent pool fully saturated
        ]
        await asyncio.sleep(0.2)

        # The shared pool must still answer promptly while the agent pool is full.
        shared = await asyncio.wait_for(
            _compute_mod.run_in_bounded_executor(lambda: "composer-step-ok"), timeout=2.0
        )
        assert shared == "composer-step-ok"

        release.set()
        await asyncio.gather(*busy)

    def test_pool_size_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("AGENT_COMPUTE_EXECUTOR_WORKERS", "5")
        assert _compute_mod._agent_compute_workers_from_env() == 5
        monkeypatch.setenv("AGENT_COMPUTE_EXECUTOR_WORKERS", "0")
        assert _compute_mod._agent_compute_workers_from_env() == 1, "must clamp to >= 1"
        monkeypatch.setenv("AGENT_COMPUTE_EXECUTOR_WORKERS", "not-an-int")
        assert (
            _compute_mod._agent_compute_workers_from_env()
            == _compute_mod._DEFAULT_AGENT_COMPUTE_WORKERS
        ), "a malformed override must fall back to the default, not crash"


# --------------------------------------------------------------------------- #
# (c) the cooperative deadline stays the abort mechanism (NOT a wait_for)
# --------------------------------------------------------------------------- #


class TestCooperativeDeadlineNotAnEnvelope:
    @pytest.mark.asyncio
    async def test_estimation_refuses_to_start_past_an_expired_deadline(self, monkeypatch):
        """Don't start ~125s of doomed compute that would hold a bounded slot.

        Mirrors the pre-existing refutation guard (refutation.py:1161). Once the
        thread occupies a BOUNDED pool, starting work for an already-expired turn
        denies the slot to a live turn.
        """
        started: list = []
        node = EstimationNode.__new__(EstimationNode)
        monkeypatch.setattr(
            node,
            "_select_estimator_with_energy_score",
            lambda *a, **k: (started.append(1), ({"method": "LinearDML"}, {}, 1.0))[1],
        )

        state = _estimation_state(_frame(), compute_deadline=0.0)  # long past
        result = await node.execute(state)

        assert started == [], "estimation must not start once the budget is exhausted"
        assert result.get("status") == "failed"
        assert "budget" in (result.get("error_message") or "").lower()

    @pytest.mark.asyncio
    async def test_budget_is_rechecked_on_the_worker_thread_after_queueing(self, monkeypatch):
        """codex round-1 HIGH: a BOUNDED pool queues, so a pre-submit check goes stale.

        A call can pass the deadline check on the event loop, then wait in the
        queue behind another agent-compute task until its budget is spent, and
        start anyway — holding a scarce slot for a turn that is already lost,
        with the caller's ``wait_for`` possibly already gone. The budget must be
        re-checked ON the worker thread, at the moment the callable starts.
        """
        import time as _time

        started: list = []
        node = EstimationNode.__new__(EstimationNode)

        def _sel(*a, **k):
            started.append(1)
            return ({"method": "LinearDML", "ate": 0.1}, {}, 1.0)

        monkeypatch.setattr(node, "_select_estimator_with_energy_score", _sel)

        # Occupy every pool thread so the estimation below must queue.
        release = threading.Event()
        occupants = [
            asyncio.create_task(
                _compute_mod.run_in_agent_compute_executor(lambda: release.wait(timeout=5))
            )
            for _ in range(_compute_mod._agent_compute_workers_from_env())
        ]
        await asyncio.sleep(0.1)  # let them all claim their threads

        # Headroom NOW (so the pre-submit check passes) but gone before the
        # queued call can be picked up.
        state = _estimation_state(_frame(), compute_deadline=_time.monotonic() + 0.3)
        exec_task = asyncio.create_task(node.execute(state))

        await asyncio.sleep(0.8)  # deadline lapses while the call sits in the queue
        release.set()
        await asyncio.gather(*occupants)
        result = await exec_task

        assert started == [], (
            "estimation started after its budget lapsed in the pool queue — "
            "it would hold a bounded slot for a turn that is already lost"
        )
        assert result.get("status") == "failed"
        assert "budget" in (result.get("error_message") or "").lower()

    @pytest.mark.asyncio
    async def test_refutation_budget_is_rechecked_after_queueing(self, monkeypatch):
        """Same queueing gap on the refutation side, with its own error shape.

        The reconstruction is the first pooled call, so a budget that lapses in
        the queue must stop it before it starts and surface the structured
        fail-closed refutation error — tagged as the QUEUED case, not the
        pre-flight one.
        """
        import time as _time

        started: list = []
        node = RefutationNode()

        monkeypatch.setattr(
            _ref_mod,
            "_reconstruct_dowhy_artifacts",
            lambda **k: (started.append(1), (SimpleNamespace(), object(), object()))[1],
        )

        release = threading.Event()
        occupants = [
            asyncio.create_task(
                _compute_mod.run_in_agent_compute_executor(lambda: release.wait(timeout=5))
            )
            for _ in range(_compute_mod._agent_compute_workers_from_env())
        ]
        await asyncio.sleep(0.1)

        state = _refutation_state(_frame(), compute_deadline=_time.monotonic() + 0.3)
        exec_task = asyncio.create_task(node.execute(state))

        await asyncio.sleep(0.8)
        release.set()
        await asyncio.gather(*occupants)
        result = await exec_task

        assert started == [], "reconstruction started after its budget lapsed in the queue"
        assert result.get("status") == "failed"
        assert (
            result.get("refutation_error_details", {}).get("reason")
            == "time_budget_exceeded_queued_refutation"
        ), result.get("refutation_error_details")

    @pytest.mark.asyncio
    async def test_estimation_runs_when_the_deadline_has_headroom(self, monkeypatch):
        """The guard must not fire on a healthy turn (no false positives)."""
        import time as _time

        started: list = []
        node = EstimationNode.__new__(EstimationNode)

        def _sel(*a, **k):
            started.append(1)
            return ({"method": "LinearDML", "ate": 0.1}, {}, 1.0)

        monkeypatch.setattr(node, "_select_estimator_with_energy_score", _sel)

        state = _estimation_state(_frame(), compute_deadline=_time.monotonic() + 10_000.0)
        result = await node.execute(state)

        assert started == [1]
        assert result.get("status") == "computing", result.get("error_message")

    @pytest.mark.asyncio
    async def test_a_long_suite_inside_its_budget_is_never_aborted(self, monkeypatch):
        """No ``wait_for`` envelope may sit between the node and the suite.

        The agent-path suite is DESIGNED to run ~223s inside a 240s cooperative
        budget (router.py:55-68); #1598's 120s composer envelope would kill it.
        The cooperative deadline — which lets the thread RETURN instead of being
        abandoned — is the only abort mechanism here.
        """
        node = RefutationNode()

        monkeypatch.setattr(
            _ref_mod,
            "_reconstruct_dowhy_artifacts",
            lambda **k: (SimpleNamespace(), object(), object()),
        )

        def slow_suite(**kwargs):
            # The suite must be allowed to RUN TO COMPLETION. Raising a sentinel
            # after the sleep proves the node waited for this call rather than
            # abandoning it behind a timeout (which would surface as a
            # TimeoutError / envelope message instead of the sentinel).
            threading.Event().wait(0.5)
            raise RefutationError("suite-ran-to-completion", details={"reason": "test"})

        monkeypatch.setattr(node.runner, "run_all_tests", slow_suite)

        result = await node.execute(_refutation_state(_frame()))

        assert "suite-ran-to-completion" in (result.get("error_message") or ""), (
            "the suite was aborted instead of being allowed to finish: "
            f"{result.get('error_message')!r}"
        )


# --------------------------------------------------------------------------- #
# (d) behaviour preservation
# --------------------------------------------------------------------------- #


class TestBehaviourPreserved:
    @pytest.mark.asyncio
    async def test_offloaded_result_is_identical_to_the_direct_call(self, monkeypatch):
        payload = (
            {"method": "LinearDML", "ate": 0.1234, "energy_score": 0.5},
            {"selected": "linear_dml"},
            42.0,
        )
        node = EstimationNode.__new__(EstimationNode)
        monkeypatch.setattr(node, "_select_estimator_with_energy_score", lambda *a, **k: payload)

        result = await node.execute(_estimation_state(_frame()))

        assert result["estimation_result"] == payload[0]
        assert result["estimator_selection_result"] == payload[1]
        assert result["energy_score_latency_ms"] == payload[2]
        assert result["best_energy_score"] == 0.5

    @pytest.mark.asyncio
    async def test_tool_raised_exception_propagates_as_a_structured_failure(self, monkeypatch):
        node = EstimationNode.__new__(EstimationNode)

        def boom(*a, **k):
            raise RuntimeError("estimator exploded")

        monkeypatch.setattr(node, "_select_estimator_with_energy_score", boom)

        result = await node.execute(_estimation_state(_frame()))

        # F-006 contract: an exception from the offloaded callable must surface
        # as the structured fail-closed EstimationError — never a silent-wrong
        # estimate. The off-load change must not alter that wrapping.
        assert result.get("status") == "failed"
        assert "refusing silent fallback" in (result.get("error_message") or "")
        assert result.get("estimation_error_details", {}).get("selection_strategy") == "best_energy"

    @pytest.mark.asyncio
    async def test_offload_copies_contextvars_like_to_thread_did(self):
        """``asyncio.to_thread`` copies the context; ``run_in_executor`` does not.

        The swap must not silently drop contextvar propagation, so the helper
        mirrors ``to_thread``'s ``contextvars.copy_context()``.
        """
        var: contextvars.ContextVar[str] = contextvars.ContextVar("probe_1601")
        var.set("carried")

        got = await _compute_mod.run_in_agent_compute_executor(lambda: var.get("MISSING"))

        assert got == "carried"


# --------------------------------------------------------------------------- #
# (e) the orphan hole the bounded pool makes worse
# --------------------------------------------------------------------------- #


class TestDispatcherSetsCooperativeDeadline:
    def test_explicit_analyst_spec_still_gets_a_compute_deadline(self):
        """dispatcher path (1) returned the spec verbatim with NO deadline.

        With ``compute_deadline`` unset the refutation node runs the FULL
        105-sim suite (~728s) against a 300s dispatch timeout — the graph is torn
        down and the uncancellable thread is orphaned. That already wasted a CPU
        core; now it would also hold a bounded agent-pool slot, so the hole has
        to close.
        """
        import time as _time

        from src.agents.orchestrator.nodes.dispatcher import _resolve_causal_impact_input

        dispatch = {
            "agent_name": "causal_impact",
            "priority": "critical",
            "timeout_ms": 300000,
            "parameters": {
                "treatment_var": "accepted",
                "outcome_var": "converted",
                "confounders": ["confidence_score"],
            },
        }
        before = _time.monotonic()
        resolved = _resolve_causal_impact_input({"query": "why did conversion move?"}, dispatch)

        assert resolved["treatment_var"] == "accepted"  # passthrough intact
        deadline = resolved.get("compute_deadline")
        assert deadline is not None, "explicit-spec dispatch must still bound refutation"
        # 300s * 0.8 fraction, allowing for clock drift during the call.
        assert before + 235 <= deadline <= before + 245

    def test_no_timeout_means_no_fabricated_deadline(self):
        from src.agents.orchestrator.nodes.dispatcher import _resolve_causal_impact_input

        dispatch = {
            "agent_name": "causal_impact",
            "priority": "critical",
            "timeout_ms": 0,
            "parameters": {
                "treatment_var": "accepted",
                "outcome_var": "converted",
                "confounders": ["confidence_score"],
            },
        }
        resolved = _resolve_causal_impact_input({"query": "q"}, dispatch)
        assert "compute_deadline" not in resolved
