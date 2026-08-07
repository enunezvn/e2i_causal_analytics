"""#1515: the chatbot DSPy optimization queue drainer.

Contract under test (src/tasks/chatbot_optimization_tasks.py):

- a celery-beat task polls the 035 ``chatbot_optimization_requests`` table via
  ``get_next_optimization_request``, CLAIMS the row with a compare-and-set
  UPDATE (status pending -> processing; 035's get_next has no
  FOR UPDATE SKIP LOCKED and its status updater has no prior-status guard, so
  the claim must be the guard), executes via the REAL executor seam
  (``ChatbotOptimizer.optimize_module`` — the path #1507 fixed), and closes out
  via ``update_optimization_request_status``;
- the whole cycle is behind an opt-in fail-closed cost gate
  (``CHATBOT_OPT_DRAIN_ENABLED``, #1513 precedent) because the executor runs
  LLM-expensive GEPA;
- the producer (``submit_signals_for_optimization``) is routed here: it runs
  once per cycle when the queue is idle, so signals -> requests -> execution is
  a closed loop;
- bookkeeping: orphaned 'processing' rows (worker died mid-run) are re-pended,
  and stale pending rows are cancelled via
  ``cancel_stale_optimization_requests``.

Unit tests substitute two boundaries, both explicitly:
- the DB, with FakeQueueDB (_fake_supabase_queue.py); real-DB fidelity runs in
  tests/integration/test_chatbot_optimization_queue_db.py;
- the LLM-expensive executor, by monkeypatching ``_execute_request`` (the
  drainer-side seam) or ``ChatbotOptimizer.optimize_module``. One test below
  pins that ``_execute_request`` itself calls the REAL optimizer seam, so the
  substitution cannot silently detach the drainer from production code.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.tasks.chatbot_optimization_tasks as drain_mod
from tests.unit.test_tasks._fake_supabase_queue import FakeQueueDB, make_request_row

FACTORY = "src.memory.services.factories.get_async_supabase_service_client"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from the shipped default: gate unset (fail-closed).

    A dummy service key is set because the drainer refuses to run without one
    (codex iter-1 HIGH: an anon-key client would silently no-op the CAS claim
    under 035's RLS); the DB itself is always a fake here, so the value is
    never used as a credential.
    """
    for var in (
        drain_mod.DRAIN_ENABLED_ENV,
        drain_mod.DRAIN_MAX_PER_CYCLE_ENV,
        drain_mod.STALE_HOURS_ENV,
        drain_mod.ZOMBIE_HOURS_ENV,
        drain_mod.MIN_SIGNALS_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key-not-a-credential")
    yield


def _enable(monkeypatch, **env):
    monkeypatch.setenv(drain_mod.DRAIN_ENABLED_ENV, "1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _ok_executor(best_score: float = 0.85):
    return AsyncMock(return_value={"success": True, "best_score": best_score})


# =============================================================================
# Cost gate (fail-closed, #1513 precedent)
# =============================================================================


class TestCostGate:
    @pytest.mark.asyncio
    async def test_unset_gate_skips_without_touching_the_db(self):
        factory = AsyncMock(side_effect=AssertionError("DB client built despite closed gate"))
        with patch(FACTORY, new=factory):
            result = await drain_mod._drain_cycle()
        assert result["status"] == "skipped"
        assert drain_mod.DRAIN_ENABLED_ENV in result["reason"]
        factory.assert_not_awaited()

    @pytest.mark.parametrize("raw", ["0", "false", "no", "banana", "", "  "])
    def test_gate_parses_fail_closed(self, monkeypatch, raw):
        monkeypatch.setenv(drain_mod.DRAIN_ENABLED_ENV, raw)
        assert drain_mod._drain_enabled() is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "TRUE", " Yes "])
    def test_gate_truthy_values_enable(self, monkeypatch, raw):
        monkeypatch.setenv(drain_mod.DRAIN_ENABLED_ENV, raw)
        assert drain_mod._drain_enabled() is True

    @pytest.mark.asyncio
    async def test_force_bypasses_gate_only(self, monkeypatch):
        db = FakeQueueDB(rows=[make_request_row("req_forced")])
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", _ok_executor()):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle(force=True)
        assert result["status"] == "completed"
        assert db.row("req_forced")["status"] == "completed"


# =============================================================================
# Drain cycle: peek -> claim -> execute -> close out
# =============================================================================


class TestDrainCycle:
    @pytest.mark.asyncio
    async def test_happy_path_completes_request(self, monkeypatch):
        _enable(monkeypatch)
        db = FakeQueueDB(
            rows=[make_request_row("req_a", "agent_router", budget="medium", min_reward=0.6)]
        )
        executor = _ok_executor(0.91)
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        # Polled via the 035 function...
        assert any(fn == "get_next_optimization_request" for fn, _ in db.rpc_calls)
        # ...executed through the seam with the request's own parameters...
        (row_arg,) = executor.await_args.args
        assert row_arg["module_name"] == "agent_router"
        assert row_arg["budget"] == "medium"
        assert row_arg["min_reward"] == 0.6
        # ...and closed out via the 035 function.
        close_outs = [p for fn, p in db.rpc_calls if fn == "update_optimization_request_status"]
        assert close_outs and close_outs[-1]["p_status"] == "completed"
        assert close_outs[-1]["p_optimized_score"] == 0.91

        row = db.row("req_a")
        assert row["status"] == "completed"
        assert row["optimized_score"] == 0.91
        assert result["status"] == "completed"
        assert result["executed"] == [
            {
                "request_id": "req_a",
                "module_name": "agent_router",
                "status": "completed",
                "close_out": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_claim_is_a_guarded_update(self, monkeypatch):
        """The pending->processing transition must be a conditional UPDATE
        (eq status='pending'), NOT the unguarded 035 status RPC."""
        _enable(monkeypatch)
        db = FakeQueueDB(rows=[make_request_row("req_a")])
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", _ok_executor()):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    await drain_mod._drain_cycle()

        claim_ops = [
            op
            for op in db.table_ops
            if op["mode"] == "update" and op["payload"].get("status") == "processing"
        ]
        assert claim_ops, "no compare-and-set claim UPDATE issued"
        claim = claim_ops[0]
        assert ("eq", "status", "pending") in claim["filters"]
        assert ("eq", "request_id", "req_a") in claim["filters"]
        assert claim["payload"].get("started_at"), "claim must stamp started_at"
        # And the RPC updater must NOT have been used for the claim.
        assert not any(
            p.get("p_status") == "processing"
            for fn, p in db.rpc_calls
            if fn == "update_optimization_request_status"
        )

    @pytest.mark.asyncio
    async def test_lost_claim_skips_execution_and_repeeks(self, monkeypatch):
        """A competing claimer between peek and claim: no execution for the
        lost row; the next peek serves the next pending row."""
        _enable(monkeypatch, **{drain_mod.DRAIN_MAX_PER_CYCLE_ENV: "2"})
        now = datetime.now(timezone.utc)
        db = FakeQueueDB(
            rows=[
                make_request_row("req_lost", priority=3, created_at=now.isoformat()),
                make_request_row(
                    "req_next",
                    "agent_router",
                    created_at=(now + timedelta(seconds=1)).isoformat(),
                ),
            ],
            race_first_peek=True,
        )
        executor = _ok_executor()
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        executed_ids = [call.args[0]["request_id"] for call in executor.await_args_list]
        assert executed_ids == ["req_next"], (
            "lost claim must not execute; drainer re-peeks the next pending row"
        )
        assert db.row("req_lost")["status"] == "processing"  # the other claimer's
        assert db.row("req_next")["status"] == "completed"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed_executor_result_closes_out_failed(self, monkeypatch):
        _enable(monkeypatch)
        db = FakeQueueDB(rows=[make_request_row("req_a")])
        failing = AsyncMock(
            return_value={"success": False, "error": "Insufficient training signals: 3 (need 10+)"}
        )
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", failing):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        row = db.row("req_a")
        assert row["status"] == "failed"
        assert "Insufficient training signals" in row["error_message"]
        assert result["executed"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_executor_exception_closes_out_failed_and_cycle_survives(self, monkeypatch):
        _enable(monkeypatch, **{drain_mod.DRAIN_MAX_PER_CYCLE_ENV: "2"})
        now = datetime.now(timezone.utc)
        db = FakeQueueDB(
            rows=[
                make_request_row("req_boom", priority=3, created_at=now.isoformat()),
                make_request_row(
                    "req_ok",
                    "agent_router",
                    created_at=(now + timedelta(seconds=1)).isoformat(),
                ),
            ]
        )

        async def _boom_then_ok(row):
            if row["request_id"] == "req_boom":
                raise RuntimeError("GEPA exploded")
            return {"success": True, "best_score": 0.7}

        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", _boom_then_ok):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        assert db.row("req_boom")["status"] == "failed"
        assert "GEPA exploded" in db.row("req_boom")["error_message"]
        assert db.row("req_ok")["status"] == "completed"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_max_per_cycle_bounds_llm_spend(self, monkeypatch):
        _enable(monkeypatch)  # default max per cycle == 1
        now = datetime.now(timezone.utc)
        db = FakeQueueDB(
            rows=[
                make_request_row(f"req_{i}", created_at=(now + timedelta(seconds=i)).isoformat())
                for i in range(3)
            ]
        )
        executor = _ok_executor()
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    await drain_mod._drain_cycle()

        assert executor.await_count == 1
        statuses = sorted(r["status"] for r in db.rows)
        assert statuses == ["completed", "pending", "pending"]

    @pytest.mark.asyncio
    async def test_empty_queue_executes_nothing(self, monkeypatch):
        _enable(monkeypatch)
        db = FakeQueueDB(rows=[])
        executor = _ok_executor()
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()
        executor.assert_not_awaited()
        assert result["status"] == "completed"
        assert result["executed"] == []


# =============================================================================
# Bookkeeping: zombie recovery + stale cancellation
# =============================================================================


class TestBookkeeping:
    @pytest.mark.asyncio
    async def test_orphaned_processing_rows_are_repended(self, monkeypatch):
        """A worker that died mid-GEPA leaves status='processing'; the next
        cycle must return it to the queue (worker-restart durability)."""
        _enable(monkeypatch, **{drain_mod.DRAIN_MAX_PER_CYCLE_ENV: "0"})
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        fresh = datetime.now(timezone.utc).isoformat()
        db = FakeQueueDB(
            rows=[
                make_request_row("req_zombie", status="processing", started_at=old),
                make_request_row("req_live", status="processing", started_at=fresh),
            ]
        )
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                result = await drain_mod._drain_cycle()

        assert db.row("req_zombie")["status"] == "pending"
        assert db.row("req_live")["status"] == "processing", (
            "a recent processing row is a live GEPA run, not a zombie"
        )
        assert result["zombies_recovered"] == 1

    @pytest.mark.asyncio
    async def test_stale_pending_rows_cancelled_via_035_function(self, monkeypatch):
        _enable(
            monkeypatch,
            **{drain_mod.DRAIN_MAX_PER_CYCLE_ENV: "0", drain_mod.STALE_HOURS_ENV: "24"},
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
        db = FakeQueueDB(rows=[make_request_row("req_old", created_at=old)])
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                result = await drain_mod._drain_cycle()

        assert ("cancel_stale_optimization_requests", {"p_max_age_hours": 24}) in db.rpc_calls
        assert db.row("req_old")["status"] == "cancelled"
        assert result["stale_cancelled"] == 1

    @pytest.mark.asyncio
    async def test_stale_default_is_a_week_not_the_sql_default(self, monkeypatch):
        """The SQL default (24h) would cancel queued work faster than a
        1-execution/cycle drainer can serve a 4-module burst; the drainer must
        pass its own, longer default explicitly."""
        _enable(monkeypatch, **{drain_mod.DRAIN_MAX_PER_CYCLE_ENV: "0"})
        db = FakeQueueDB(rows=[])
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                await drain_mod._drain_cycle()
        stale_calls = [p for fn, p in db.rpc_calls if fn == "cancel_stale_optimization_requests"]
        assert stale_calls == [{"p_max_age_hours": 168}]


# =============================================================================
# Producer routing (#1515 acceptance: submit_signals_for_optimization routed)
# =============================================================================


class TestProducerRouting:
    @pytest.mark.asyncio
    async def test_producer_runs_when_queue_is_idle(self, monkeypatch):
        _enable(monkeypatch, **{drain_mod.MIN_SIGNALS_ENV: "75"})
        db = FakeQueueDB(rows=[])
        submit = AsyncMock(return_value={"intent_classifier": "insufficient_signals:0"})
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch("src.api.routes.chatbot_dspy.submit_signals_for_optimization", submit):
                with patch.object(drain_mod, "_execute_request", _ok_executor()):
                    result = await drain_mod._drain_cycle()
        submit.assert_awaited_once_with(min_signals=75)
        assert result["produced"] == {"intent_classifier": "insufficient_signals:0"}

    @pytest.mark.asyncio
    async def test_producer_skipped_while_requests_in_flight(self, monkeypatch):
        """No duplicate enqueueing: while pending/processing rows exist the
        producer holds off (drain first, produce when idle)."""
        _enable(monkeypatch)
        db = FakeQueueDB(rows=[make_request_row("req_a")])
        submit = AsyncMock()
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch("src.api.routes.chatbot_dspy.submit_signals_for_optimization", submit):
                with patch.object(drain_mod, "_execute_request", _ok_executor()):
                    await drain_mod._drain_cycle()
        submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_producer_failure_does_not_abort_the_drain(self, monkeypatch):
        _enable(monkeypatch)
        db = FakeQueueDB(rows=[make_request_row("req_a")])
        submit = AsyncMock(side_effect=RuntimeError("signals table on fire"))
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch("src.api.routes.chatbot_dspy.submit_signals_for_optimization", submit):
                with patch.object(drain_mod, "_execute_request", _ok_executor()):
                    result = await drain_mod._drain_cycle()
        assert db.row("req_a")["status"] == "completed"
        assert result["status"] == "completed"


# =============================================================================
# The executor seam is the REAL one
# =============================================================================


class TestExecutorSeam:
    @pytest.mark.asyncio
    async def test_execute_request_calls_the_real_optimizer_seam(self):
        """_execute_request must call ChatbotOptimizer.optimize_module — the
        production executor whose save step #1507 fixed. This pins the seam so
        the unit-level _execute_request substitution above cannot hide a
        detached drainer."""
        optimizer = MagicMock()
        optimizer.optimize_module = AsyncMock(return_value={"success": True, "best_score": 0.9})
        row = {
            "request_id": "req_a",
            "module_name": "query_rewriter",
            "budget": "medium",
            "min_reward": 0.65,
        }
        with patch("src.api.routes.chatbot_dspy.get_chatbot_optimizer", return_value=optimizer):
            result = await drain_mod._execute_request(row)
        optimizer.optimize_module.assert_awaited_once_with(
            "query_rewriter", budget="medium", min_reward=0.65
        )
        assert result == {"success": True, "best_score": 0.9}


# =============================================================================
# Celery wiring
# =============================================================================


class TestCeleryWiring:
    def test_task_registered_and_scheduled(self):
        from src.workers.celery_app import celery_app

        assert "src.tasks.drain_chatbot_optimization_queue" in celery_app.tasks
        entry = celery_app.conf.beat_schedule.get("chatbot-optimization-drain")
        assert entry is not None, "beat entry 'chatbot-optimization-drain' missing"
        assert entry["task"] == "src.tasks.drain_chatbot_optimization_queue"
        assert entry["options"]["queue"] == "analytics"
        # codex iter-2 LOW: a manual `celery call` (force=True) without an
        # explicit queue must not land the GEPA executor on worker_light.
        assert celery_app.conf.task_routes["src.tasks.drain_chatbot_optimization_queue"] == {
            "queue": "analytics"
        }

    def test_task_returns_failed_dict_instead_of_raising(self, monkeypatch):
        _enable(monkeypatch)
        with patch.object(drain_mod, "_drain_cycle", side_effect=RuntimeError("loop broke")):
            result = drain_mod.drain_chatbot_optimization_queue.apply(args=()).get(propagate=False)
        assert result["status"] == "failed"
        assert "loop broke" in result["reason"]

    @pytest.mark.asyncio
    async def test_no_client_is_a_failed_cycle_not_a_crash(self, monkeypatch):
        _enable(monkeypatch)
        with patch(FACTORY, new=AsyncMock(side_effect=RuntimeError("no env"))):
            result = await drain_mod._drain_cycle()
        assert result["status"] == "failed"
        assert "database" in result["reason"].lower()


# =============================================================================
# Silent-degradation guards (codex iter-1)
# =============================================================================


class TestSilentDegradationGuards:
    @pytest.mark.asyncio
    async def test_missing_service_key_fails_the_cycle_loudly(self, monkeypatch):
        """codex iter-1 HIGH: the client factory silently falls back to the
        ANON key, and 035's RLS grants queue writes to service_role ONLY — an
        anon client would no-op the CAS claim and every cycle would report
        'completed' with nothing executed. No service key => a LOUD failed
        cycle before any client is even built."""
        _enable(monkeypatch)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        factory = AsyncMock(side_effect=AssertionError("client built without a service key"))
        with patch(FACTORY, new=factory):
            result = await drain_mod._drain_cycle()
        assert result["status"] == "failed"
        assert "service" in result["reason"].lower()
        factory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_out_failure_is_retried_then_reported_not_swallowed(self, monkeypatch):
        """codex iter-2 HIGH: after a SUCCESSFUL GEPA run, an unguarded
        close-out that raises would abort the cycle with the row stuck in
        'processing' — zombie recovery would later re-pend it and RE-EXECUTE
        work whose optimized module was already saved. The close-out must be
        retried in-process (never re-running GEPA), and a final failure must
        be reported, not swallowed into an aborted cycle."""
        _enable(monkeypatch)
        monkeypatch.setattr(drain_mod, "_CLOSE_OUT_RETRY_DELAY_S", 0.0)
        db = FakeQueueDB(rows=[make_request_row("req_a")], fail_status_rpc_times=99)
        executor = _ok_executor()
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        # The cycle survives and reports the truth.
        assert result["status"] == "completed"
        (entry,) = result["executed"]
        assert entry["status"] == "completed"
        assert entry["close_out"] is False
        # Bounded retries, all against the RPC — and GEPA ran exactly ONCE.
        attempts = [1 for fn, _ in db.rpc_calls if fn == "update_optimization_request_status"]
        assert len(attempts) == drain_mod._CLOSE_OUT_ATTEMPTS
        assert executor.await_count == 1
        # The row stays 'processing'; zombie recovery is the documented
        # at-least-once backstop for this window.
        assert db.row("req_a")["status"] == "processing"

    @pytest.mark.asyncio
    async def test_close_out_transient_failure_recovers_on_retry(self, monkeypatch):
        _enable(monkeypatch)
        monkeypatch.setattr(drain_mod, "_CLOSE_OUT_RETRY_DELAY_S", 0.0)
        db = FakeQueueDB(rows=[make_request_row("req_a")], fail_status_rpc_times=1)
        executor = _ok_executor(0.8)
        with patch(FACTORY, new=AsyncMock(return_value=db)):
            with patch.object(drain_mod, "_execute_request", executor):
                with patch.object(drain_mod, "_produce_requests", AsyncMock(return_value=None)):
                    result = await drain_mod._drain_cycle()

        assert db.row("req_a")["status"] == "completed"
        (entry,) = result["executed"]
        assert entry["close_out"] is True
        assert executor.await_count == 1

    def test_zombie_floor_exceeds_celery_hard_time_limit(self, monkeypatch):
        """codex iter-1 MEDIUM: re-pending a STILL-RUNNING GEPA job would allow
        double execution. On this deployment that cannot happen because celery
        kills every task at task_time_limit=7200 (2h) — but only while the
        zombie cutoff stays ABOVE that limit. Pin the floor (3h > 2h hard
        limit) so an operator setting a tiny value cannot re-open the race."""
        monkeypatch.setenv(drain_mod.ZOMBIE_HOURS_ENV, "1")
        assert drain_mod._zombie_hours() == 3

        from src.workers.celery_app import celery_app

        hard_limit_hours = celery_app.conf.task_time_limit / 3600
        assert drain_mod._ZOMBIE_HOURS_FLOOR > hard_limit_hours, (
            "zombie floor must exceed the celery hard time limit, or a live "
            "run could be re-pended and double-executed"
        )
