"""Record every composition for the learning loop (spec §5.3–§5.5, ml/041).

Two parts:

- **The serializer** (``to_record`` and its helpers) is the only path from composer models to the
  recording RPCs, and it sends structure only. Sub-questions and steps are positional; intents are
  normalized to the decomposer's vocabulary; input maps keep only the tool's declared parameter
  names (others are counted); a string value is a public column name only if the database catalog
  lists it, else its length; a ``$step`` reference becomes a step number, with its field only if
  the producer's registered output model declares it; outputs are the output model's field names
  (others counted); error text is never sent. Steps of tools the live registry does not know are
  not sent at all, and their names are not kept.
- **``CompositionRecorder``** turns the composer's phase boundaries into seeded, idempotent RPC
  writes on one background chain per composition. Every method is a synchronous enqueue: nothing
  on the user's path awaits the database, and nothing raises into ``compose()``. Each write has a
  timeout and one retry, then it is logged, counted in ``e2i_composer_record_failures_total{rpc}`` and
  dropped; the finish snapshot restores every phase field and re-sends every step the recorder was
  given, so one lost write loses nothing the finish can carry. A heartbeat keeps
  ``last_activity_at`` fresh while the composition runs. ``drain()`` lets the API lifespan flush
  in-flight records at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import UUID

import numpy as np

from src.tool_registry.registry import get_registry

from .models.composition_models import (
    DecompositionResult,
    ExecutionPlan,
    ExecutionTrace,
    StepResult,
)
from .registry_sync import RegistrySync, default_registry_sync
from .rpc_port import RpcPort

logger = logging.getLogger(__name__)

# The decomposer's declared intent vocabulary (decomposer.py prompt); anything else is OTHER.
NORMALIZED_INTENTS = frozenset(
    {"CAUSAL", "COMPARATIVE", "PREDICTIVE", "DESCRIPTIVE", "EXPERIMENTAL"}
)

WRITE_TIMEOUT_S = 5.0
RETRY_DELAY_S = 1.0
# v_active_compositions calls a run abandoned after 5 minutes without a write: five heartbeats.
HEARTBEAT_S = 60.0
# No composition legitimately runs this long; a recorder whose owner never finished stops
# keeping its episode alive, so the episode reads abandoned instead of live forever.
HEARTBEAT_MAX_S = 3 * 3600.0

# The seed fields an episode records (spec §5.3); nothing else in a caller's seed is sent.
ENTRY_POINTS = frozenset({"chat_tool", "orchestrator_agent", "direct"})

#: Recording is an explicit per-process opt-in. The API containers set it (docker-compose
#: ``x-common-env``); a composer run in tests, scripts or benchmarks records nothing, so their
#: compositions never reach the reliability readers the planner uses.
LEARNING_LOOP_ENV = "TOOL_COMPOSER_LEARNING_LOOP_ENABLED"
# Failures worth one retry: the transport or a timeout. Anything else (a rejected payload, a
# database error) fails the same way again, so it is counted at once.
_TRANSPORT_ERROR_NAMES = frozenset(
    {"OperationalError", "InterfaceError", "TransportError", "NetworkError", "TimeoutException"}
)


# =============================================================================
# Serializer
# =============================================================================


def _registered(tool_name: str) -> bool:
    return get_registry().get(tool_name) is not None


def _declared_inputs(tool_name: str) -> Set[str]:
    registered = get_registry().get(tool_name)
    if registered is None:
        return set()
    return {parameter.name for parameter in registered.schema.input_parameters}


def _output_fields(tool_name: str) -> List[str]:
    registered = get_registry().get(tool_name)
    if registered is None or registered.pydantic_output_model is None:
        return []
    return list(registered.pydantic_output_model.model_fields)


def _step_numbers(plan: ExecutionPlan) -> Dict[str, int]:
    numbers: Dict[str, int] = {}
    for number, step in enumerate(plan.steps):
        numbers.setdefault(step.step_id, number)
    return numbers


def _reference(
    value: str,
    step_numbers: Mapping[str, int],
    fields_of_step: Optional[Callable[[int], List[str]]],
) -> Dict[str, Any]:
    source, *path = value[1:].split(".")
    number = step_numbers.get(source)
    if number is None:  # $context.<key>, or an unknown source
        return {"type": "ref", "step": None, "field": None}
    field = path[0] if path else None
    known = fields_of_step(number) if fields_of_step is not None else []
    return {"type": "ref", "step": number, "field": field if field in known else None}


def structure_value(
    value: Any,
    *,
    allowlist: Optional[frozenset[str]],
    step_numbers: Optional[Mapping[str, int]] = None,
    fields_of_step: Optional[Callable[[int], List[str]]] = None,
) -> Any:
    """One parameter value as structure: settings are kept, anything data-like is reduced.

    Total: a value that cannot even be measured is reduced to ``{"type": "str", "len": None}``.
    """
    try:
        return _structure_value(
            value, allowlist=allowlist, step_numbers=step_numbers, fields_of_step=fields_of_step
        )
    except Exception:  # noqa: BLE001 - an unmeasurable value is sent as nothing
        return {"type": "str", "len": None}


def _structure_value(
    value: Any,
    *,
    allowlist: Optional[frozenset[str]],
    step_numbers: Optional[Mapping[str, int]],
    fields_of_step: Optional[Callable[[int], List[str]]],
) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        if value.startswith("$"):
            return _reference(value, step_numbers or {}, fields_of_step)
        if allowlist is not None and value in allowlist:
            return {"type": "column", "name": value}
        return {"type": "str", "len": len(value)}
    if hasattr(value, "columns") and hasattr(value, "shape"):
        rows, columns = value.shape[0], value.shape[1]
        return {"type": "frame", "rows": int(rows), "columns": int(columns)}
    if isinstance(value, Mapping):
        return {"type": "dict", "len": len(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"type": "list", "len": len(value)}
    return {"type": "str", "len": None}


def _input_params(
    mapping: Mapping[str, Any],
    tool_name: str,
    *,
    allowlist: Optional[frozenset[str]],
    step_numbers: Mapping[str, int],
    fields_of_step: Callable[[int], List[str]],
) -> Dict[str, Any]:
    declared = _declared_inputs(tool_name)
    params: Dict[str, Any] = {}
    undeclared = 0
    for key, value in mapping.items():
        if key in declared:
            params[key] = structure_value(
                value, allowlist=allowlist, step_numbers=step_numbers, fields_of_step=fields_of_step
            )
        else:
            undeclared += 1
    if undeclared:
        params["undeclared_params"] = undeclared
    return params


def sub_questions_record(decomposition: DecompositionResult) -> List[Dict[str, Any]]:
    records = []
    for index, sub_question in enumerate(decomposition.sub_questions):
        intent = str(sub_question.intent or "").strip().upper()
        records.append(
            {"index": index, "intent": intent if intent in NORMALIZED_INTENTS else "OTHER"}
        )
    return records


def plan_record(plan: ExecutionPlan, *, allowlist: Optional[frozenset[str]]) -> Dict[str, Any]:
    numbers = _step_numbers(plan)

    def fields_of_step(number: int) -> List[str]:
        return _output_fields(plan.steps[number].tool_name)

    steps = [
        {
            "step_number": number,
            "tool_name": step.tool_name if _registered(step.tool_name) else None,
            "depends_on_steps": [numbers[d] for d in step.depends_on_steps if d in numbers],
            "input_params": _input_params(
                step.input_mapping,
                step.tool_name,
                allowlist=allowlist,
                step_numbers=numbers,
                fields_of_step=fields_of_step,
            ),
        }
        for number, step in enumerate(plan.steps)
    ]
    try:
        repaired = plan.execution_order_repaired
    except ValueError:  # an invalid graph: execution raises too; there is no order to report
        repaired = None
    return {"steps": steps, "execution_order_repaired": repaired}


def groups_record(plan: ExecutionPlan) -> List[List[int]]:
    """The executed waves, as step numbers."""
    numbers = _step_numbers(plan)
    try:
        order = plan.get_execution_order()
    except ValueError:
        return []
    return [[numbers[step_id] for step_id in group if step_id in numbers] for group in order]


def step_record(
    step_number: int,
    result: StepResult,
    plan: ExecutionPlan,
    *,
    allowlist: Optional[frozenset[str]],
) -> Dict[str, Any]:
    numbers = _step_numbers(plan)
    number = numbers.get(result.step_id, step_number)
    step = plan.steps[number] if 0 <= number < len(plan.steps) else None
    sub_question_index = {sq.id: i for i, sq in enumerate(plan.decomposition.sub_questions)}

    def fields_of_step(n: int) -> List[str]:
        return _output_fields(plan.steps[n].tool_name)

    output = result.output.result if isinstance(result.output.result, Mapping) else {}
    fields = _output_fields(result.tool_name)
    return {
        "step_number": number,
        "tool_name": result.tool_name,
        "input_params": _input_params(
            step.input_mapping if step is not None else {},
            result.tool_name,
            allowlist=allowlist,
            step_numbers=numbers,
            fields_of_step=fields_of_step,
        ),
        "output_keys": {
            "keys": [f for f in fields if f in output],
            "other_keys": sum(1 for key in output if key not in fields),
        },
        "depends_on_steps": (
            [numbers[d] for d in step.depends_on_steps if d in numbers] if step is not None else []
        ),
        "serves_sub_question": (
            str(sub_question_index[result.sub_question_id])
            if result.sub_question_id in sub_question_index
            else None
        ),
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        # The executor fills duration_ms only on its success and cache paths; a failed step's
        # latency is still its wall time.
        "latency_ms": result.duration_ms
        or max(int((result.completed_at - result.started_at).total_seconds() * 1000), 0),
        "outcome_class": result.outcome_class,
        "attempts": result.attempts,
        "cache_hit": result.cache_hit,
        "error_type": result.error_type,
    }


def to_record(
    *,
    decomposition: DecompositionResult,
    plan: ExecutionPlan,
    trace: ExecutionTrace,
    allowlist: Optional[frozenset[str]],
) -> Dict[str, Any]:
    """Everything a composition record sends about its models, as structure."""
    numbers = _step_numbers(plan)
    return {
        "sub_questions": sub_questions_record(decomposition),
        "tool_plan": plan_record(plan, allowlist=allowlist),
        "parallelizable_groups": groups_record(plan),
        "steps": [
            step_record(numbers.get(result.step_id, index), result, plan, allowlist=allowlist)
            for index, result in enumerate(trace.step_results)
            if _registered(result.tool_name)
        ],
    }


# =============================================================================
# Recorder
# =============================================================================


def _text(value: Any, max_len: int) -> Optional[str]:
    return value[:max_len] if isinstance(value, str) else None


def seed_record(seed: Mapping[str, Any], composition_id: str) -> Dict[str, Any]:
    """The episode identity every RPC carries, and nothing else from the caller's seed."""
    if not isinstance(seed, Mapping):
        seed = {}
    audit = seed.get("audit_workflow_id")
    try:
        audit_id: Optional[str] = str(UUID(str(audit))) if audit is not None else None
    except (TypeError, ValueError):
        audit_id = None
    entry_point = seed.get("entry_point")
    return {
        "composition_id": _text(seed.get("composition_id"), 100) or composition_id,
        "query_text": _text(seed.get("query_text"), 1000) or "",
        "session_id": _text(seed.get("session_id"), 100),
        "user_id": _text(seed.get("user_id"), 100),
        "entry_point": (
            entry_point if isinstance(entry_point, str) and entry_point in ENTRY_POINTS else None
        ),
        "brand": _text(seed.get("brand"), 100),
        "region": _text(seed.get("region"), 100),
        "audit_workflow_id": audit_id,
        "is_synthetic": seed.get("is_synthetic") is True,
    }


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return any(cls.__name__ in _TRANSPORT_ERROR_NAMES for cls in type(exc).__mro__)


_pending: Set[asyncio.Task] = set()
_heartbeats: Set[asyncio.Task] = set()


def _track(task: asyncio.Task, into: Set[asyncio.Task]) -> None:
    into.add(task)
    task.add_done_callback(into.discard)


def pending_count() -> int:
    return sum(1 for task in _pending if not task.done())


async def drain(timeout: float = 5.0, *, cancel_heartbeats: bool = False) -> int:
    """Wait up to ``timeout`` seconds for in-flight recording writes; return how many remain.

    ``cancel_heartbeats=True`` (shutdown) first stops every heartbeat of this event loop: the
    worker is going away, so its unfinished compositions will read abandoned, which they are.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    if cancel_heartbeats:
        beats = [t for t in _heartbeats if not t.done() and t.get_loop() is loop]
        for beat in beats:
            beat.cancel()
        if beats:
            await asyncio.wait(beats, timeout=max(timeout, 0.1))
    while True:
        tasks = [t for t in _pending if not t.done() and t.get_loop() is loop]
        if not tasks:
            return 0
        remaining = deadline - loop.time()
        if remaining <= 0:
            return len(tasks)
        await asyncio.wait(tasks, timeout=remaining)


def _count_failure(rpc: str) -> None:
    try:
        from src.api.routes.metrics import inc_composer_record_failure

        inc_composer_record_failure(rpc)
    except Exception:  # noqa: BLE001 - metrics are optional
        logger.debug("composer recording failure metric unavailable", exc_info=True)


_OMITTED = object()


def learning_loop_enabled() -> bool:
    """Whether this process records compositions; read on every call (truthy: 1 / true / yes)."""
    return os.getenv(LEARNING_LOOP_ENV, "").strip().lower() in ("1", "true", "yes")


class NullRecorder:
    """The recorder where the learning loop is off: the composer's calls do nothing."""

    def __init__(self, composition_id: str) -> None:
        self.composition_id = composition_id

    def start(self) -> None:
        return None

    def decomposed(self, decomposition: DecompositionResult, *, latency_ms: float) -> None:
        return None

    def planned(
        self, plan: ExecutionPlan, *, latency_ms: float, plan_source: Optional[str]
    ) -> None:
        return None

    def step(self, step_number: int, result: StepResult) -> None:
        return None

    def executed(self, *, latency_ms: float) -> None:
        return None

    def finish(self, **_fields: Any) -> None:
        return None

    def cancelled(self, phase: str) -> None:
        return None


class GuardedRecorder:
    """Wraps a recorder so one that raises can never fail a composition.

    ``CompositionRecorder`` already guards its own writes; this covers whatever recorder the
    composer is handed, a factory's included. It catches ``Exception`` only, so a cancellation
    passing through ``cancelled()`` still reaches the caller.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.composition_id = getattr(inner, "composition_id", "")

    def _call(self, name: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(self._inner, name)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recording never fails a composition
            logger.warning(
                "composer recording %s failed for %s (%s)",
                name,
                self.composition_id,
                type(exc).__name__,
            )

    def start(self) -> None:
        self._call("start")

    def decomposed(self, decomposition: DecompositionResult, *, latency_ms: float) -> None:
        self._call("decomposed", decomposition, latency_ms=latency_ms)

    def planned(
        self, plan: ExecutionPlan, *, latency_ms: float, plan_source: Optional[str]
    ) -> None:
        self._call("planned", plan, latency_ms=latency_ms, plan_source=plan_source)

    def step(self, step_number: int, result: StepResult) -> None:
        self._call("step", step_number, result)

    def executed(self, *, latency_ms: float) -> None:
        self._call("executed", latency_ms=latency_ms)

    def finish(self, **fields: Any) -> None:
        self._call("finish", **fields)

    def cancelled(self, phase: str) -> None:
        self._call("cancelled", phase)


class CompositionRecorder:
    """Seeded, idempotent, fail-open recording of one composition."""

    def __init__(
        self,
        composition_id: str,
        seed: Mapping[str, Any],
        *,
        port: Optional[RpcPort] = None,
        sync: Optional[RegistrySync] = None,
        heartbeat_s: float = HEARTBEAT_S,
        heartbeat_max_s: float = HEARTBEAT_MAX_S,
        write_timeout_s: float = WRITE_TIMEOUT_S,
        retry_delay_s: float = RETRY_DELAY_S,
        sync_timeout_s: Optional[float] = None,
    ):
        self.composition_id = composition_id
        try:
            self._seed = seed_record(seed, composition_id)
        except Exception:  # noqa: BLE001 - a seed that cannot be read records the id alone
            self._seed = seed_record({}, composition_id)
        self._sync = sync or default_registry_sync()
        self._port = port
        self._heartbeat_s = heartbeat_s
        self._heartbeat_max_s = heartbeat_max_s
        self._write_timeout_s = write_timeout_s
        self._retry_delay_s = retry_delay_s
        # Building the sync payload imports the tool registrations on its first use.
        self._sync_timeout_s = sync_timeout_s if sync_timeout_s is not None else 6 * write_timeout_s
        self._tail: Optional[asyncio.Task] = None
        self._heartbeat: Optional[asyncio.Task] = None
        self._decomposition: Optional[DecompositionResult] = None
        self._plan: Optional[ExecutionPlan] = None
        self._plan_source: Optional[str] = None
        self._latencies: Dict[str, Optional[float]] = {}
        self._steps: Dict[int, StepResult] = {}
        self._finished = False

    @property
    def port(self) -> RpcPort:
        return self._port or self._sync.port

    @property
    def heartbeat_stopped(self) -> bool:
        return self._heartbeat is None or self._heartbeat.done()

    # -- the composer's phase boundaries (synchronous enqueues) ---------------------------------

    def start(self) -> None:
        self._guard(self._start)

    def decomposed(self, decomposition: DecompositionResult, *, latency_ms: float) -> None:
        def record() -> None:
            self._decomposition = decomposition
            self._latencies["decompose_latency_ms"] = latency_ms
            self._phase(
                "PLANNING",
                {
                    "sub_questions": lambda allowlist: sub_questions_record(decomposition),
                    "decompose_latency_ms": latency_ms,
                },
            )

        self._guard(record)

    def planned(
        self, plan: ExecutionPlan, *, latency_ms: float, plan_source: Optional[str]
    ) -> None:
        def record() -> None:
            self._plan = plan
            self._plan_source = plan_source
            self._latencies["plan_latency_ms"] = latency_ms
            self._phase(
                "EXECUTING",
                {
                    "tool_plan": lambda allowlist: plan_record(plan, allowlist=allowlist),
                    "plan_source": plan_source,
                    "plan_latency_ms": latency_ms,
                },
            )

        self._guard(record)

    def step(self, step_number: int, result: StepResult) -> None:
        """``PlanExecutor.execute(on_step_result=...)``: retain the result, enqueue its write."""

        def record() -> None:
            self._steps[step_number] = result
            self._enqueue(lambda: self._write_steps([step_number]))

        self._guard(record)

    def executed(self, *, latency_ms: float) -> None:
        def record() -> None:
            self._latencies["execute_latency_ms"] = latency_ms
            plan = self._plan
            self._phase(
                "SYNTHESIZING",
                {
                    "parallelizable_groups": lambda allowlist: (
                        groups_record(plan) if plan is not None else []
                    ),
                    "execute_latency_ms": latency_ms,
                },
            )

        self._guard(record)

    def finish(
        self,
        *,
        status: str,
        outcome: str,
        total_latency_ms: Optional[float] = None,
        failed_phase: Optional[str] = None,
        error_type: Optional[str] = None,
        synthesize_latency_ms: Optional[float] = None,
        tools_executed: Optional[int] = None,
        tools_succeeded: Optional[int] = None,
    ) -> None:
        """Terminal write: re-send every retained step, then the complete snapshot. Once only."""

        def record() -> None:
            if self._finished:
                return
            self._finished = True
            self._stop_heartbeat()
            if synthesize_latency_ms is not None:
                self._latencies["synthesize_latency_ms"] = synthesize_latency_ms
            if self._steps:
                numbers = sorted(self._steps)
                self._enqueue(lambda: self._write_steps(numbers))
            fields: Dict[str, Any] = {
                "status": status,
                "outcome": outcome,
                "failed_phase": failed_phase,
                "error_type": error_type,
                "total_latency_ms": total_latency_ms,
                "tools_executed": tools_executed,
                "tools_succeeded": tools_succeeded,
                **self._latencies,
            }
            decomposition, plan, plan_source = self._decomposition, self._plan, self._plan_source
            if decomposition is not None:
                fields["sub_questions"] = lambda allowlist: sub_questions_record(decomposition)
            if plan is not None:
                fields["tool_plan"] = lambda allowlist: plan_record(plan, allowlist=allowlist)
                fields["parallelizable_groups"] = lambda allowlist: groups_record(plan)
                fields["plan_source"] = plan_source

            async def write() -> None:
                final = await self._build("composer_record_finish", fields)
                await self._write(
                    "composer_record_finish", {"p_seed": self._seed, "p_final": final}
                )

            self._enqueue(write)

        self._guard(record)

    def cancelled(self, phase: str) -> None:
        """The composition was cancelled while in ``phase``; the caller re-raises."""
        self.finish(
            status="FAILED", outcome="cancelled", failed_phase=phase, error_type="CancelledError"
        )

    # -- internals ------------------------------------------------------------------------------

    def _guard(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - recording never fails a composition
            logger.warning(
                "composer recording skipped a write for %s (%s)",
                self.composition_id,
                type(exc).__name__,
            )

    def _start(self) -> None:
        self._enqueue(lambda: self._write("composer_record_start", {"p_seed": self._seed}))
        if self._heartbeat is None:
            self._heartbeat = asyncio.get_running_loop().create_task(self._heartbeat_loop())
            _track(self._heartbeat, _heartbeats)
            # The heartbeat also ends with the task that started it (the composition): a caller
            # that exits without finish, by returning, raising or being cancelled, leaves no
            # heartbeat keeping its episode alive.
            owner = asyncio.current_task()
            if owner is not None:
                owner.add_done_callback(lambda _task: self._stop_heartbeat())

    def _stop_heartbeat(self) -> None:
        if self._heartbeat is not None and not self._heartbeat.done():
            self._heartbeat.cancel()

    async def _heartbeat_loop(self) -> None:
        # Not on the chain: a slow write must not delay liveness. It ends at finish (cancelled),
        # when the database says the episode is already terminal, or at the lifetime cap.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._heartbeat_max_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(self._heartbeat_s, remaining))
            if loop.time() >= deadline:
                return
            alive = await self._write("composer_record_heartbeat", {"p_seed": self._seed})
            if alive is False:
                return

    def _enqueue(self, write: Callable[[], Awaitable[Any]]) -> None:
        previous = self._tail

        async def chained() -> None:
            if previous is not None:
                await asyncio.wait({previous})  # order only; never inherit its outcome
            try:
                await write()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - recording never fails a composition
                logger.warning(
                    "composer recording write failed for %s (%s)",
                    self.composition_id,
                    type(exc).__name__,
                )

        task = asyncio.get_running_loop().create_task(chained())
        self._tail = task
        _track(task, _pending)

    async def _build(self, rpc: str, fields: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate a payload's structural parts; a part that fails is omitted and counted."""
        allowlist = await self._allowlist()
        built: Dict[str, Any] = {}
        for key, value in fields.items():
            if callable(value):
                try:
                    value = value(allowlist)
                except Exception as exc:  # noqa: BLE001 - the rest of the payload still records
                    logger.warning(
                        "composer recording could not serialize %s of %s for %s (%s)",
                        key,
                        rpc,
                        self.composition_id,
                        type(exc).__name__,
                    )
                    _count_failure(rpc)
                    value = _OMITTED
            if value is not _OMITTED:
                built[key] = value
        return built

    def _phase(self, status: str, fields: Mapping[str, Any]) -> None:
        async def write() -> None:
            patch = await self._build("composer_record_phase", fields)
            await self._write(
                "composer_record_phase",
                {"p_seed": self._seed, "p_status": status, "p_patch": patch},
            )

        self._enqueue(write)

    async def _allowlist(self) -> Optional[frozenset[str]]:
        try:
            return await self._sync.column_allowlist(timeout=self._write_timeout_s)
        except Exception:  # noqa: BLE001 - failed: no names are kept this time
            return None

    async def _write_steps(self, numbers: List[int]) -> None:
        plan = self._plan
        if plan is None:
            return
        allowlist = await self._allowlist()
        payload = []
        for n in sorted(set(numbers)):
            result = self._steps.get(n)
            if result is None or not _registered(result.tool_name):
                continue
            try:
                payload.append(step_record(n, result, plan, allowlist=allowlist))
            except Exception as exc:  # noqa: BLE001 - the other steps still record
                logger.warning(
                    "composer recording could not serialize step %s for %s (%s)",
                    n,
                    self.composition_id,
                    type(exc).__name__,
                )
                _count_failure("composer_record_steps")
        if not payload:
            return
        params = {"p_seed": self._seed, "p_steps": payload}
        receipt = await self._write("composer_record_steps", params)
        if not isinstance(receipt, Mapping):
            return
        if receipt.get("unknown_tools") or receipt.get("schema_mismatch_tools"):
            # The DB registry is behind the running code (the startup sync has not landed).
            try:
                await self._sync.sync_once(timeout=self._sync_timeout_s)
            except Exception:  # noqa: BLE001 - a failed sync leaves the steps unrecorded
                logger.warning(
                    "composer recording: registry sync did not complete for %s", self.composition_id
                )
            # Re-send whenever the registry is now synced, by this recorder or another one.
            if self._sync.synced and receipt.get("unknown_tools"):
                await self._write("composer_record_steps", params)

    async def _write(self, rpc: str, params: Dict[str, Any]) -> Any:
        started = time.perf_counter()
        for attempt in (1, 2):
            try:
                result = await asyncio.wait_for(
                    self.port.call(rpc, params), timeout=self._write_timeout_s
                )
                logger.debug(
                    "composer recording %s for %s took %.1f ms",
                    rpc,
                    self.composition_id,
                    (time.perf_counter() - started) * 1000,
                )
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - counted and dropped below
                if attempt == 1 and _retryable(exc):
                    await asyncio.sleep(self._retry_delay_s)
                    continue
                logger.warning(
                    "composer recording write %s failed for %s%s (%s); the composition is unaffected",
                    rpc,
                    self.composition_id,
                    " after one retry" if attempt == 2 else "",
                    type(exc).__name__,
                )
                _count_failure(rpc)
                return None
        return None
