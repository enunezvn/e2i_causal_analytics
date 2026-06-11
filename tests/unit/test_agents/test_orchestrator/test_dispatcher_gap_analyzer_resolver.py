"""Issue #874: gap_analyzer dispatch-dead on the chat path.

Every live chat dispatch of gap_analyzer failed in ~7ms with the agent's raw
``ValueError("Missing required field: metrics")`` — the orchestrator's generic
payload carries no ``metrics``/``segments``/``brand``, and gap_analyzer had no
``INPUT_RESOLVERS`` entry (the #839/F12-F14 family). It fires as
heterogeneous_optimizer's ``fallback_agent`` and on direct performance_gap-intent
queries, so it was dead via chat unconditionally.

The fix mirrors ``_resolve_heterogeneous_optimizer_input``:

* explicit analyst-supplied inputs in ``dispatch.parameters`` pass through;
* otherwise the resolver DERIVES ``metrics``/``segments``/``brand`` from the REAL
  ``business_metrics`` substrate (the table the production connector reads
  post-#856), honoring the ``include_synthetic`` opt-in channels (#872 plumb);
* when the substrate genuinely has no rows → ``NeedsStructuredInput`` fail-closed
  with an actionable reason — never a raw field-validation crash, never
  fabricated inputs.

The substrate probe is monkeypatched here (unit scope, no DB); the faithful
real-DB proof lives in
``tests/integration/test_dispatcher_gap_analyzer_substrate_realdb.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput


def _state(query: str, *, entities=None, user_context=None, params=None) -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-874",
        "parsed_query": {"intent": "performance_gap", "entities": entities or []},
        "dispatch_plan": [
            {
                "agent_name": "gap_analyzer",
                "priority": "high",
                "parameters": params or {},
                "timeout_ms": 15000,
                "fallback_agent": None,
                "execution_mode": "parallel",
            }
        ],
        "parallel_groups": [["gap_analyzer"]],
    }


def _dispatch(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_name": "gap_analyzer",
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 15000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _probe_stub(
    metrics: List[str], brand: Optional[str], regions: List[str]
) -> Tuple[List[str], Optional[str], List[str]]:
    return metrics, brand, regions


# ---------------------------------------------------------------------------
# Registry shape (RED on main: gap_analyzer has no resolver entry)
# ---------------------------------------------------------------------------


def test_gap_analyzer_registered_in_input_resolvers() -> None:
    assert "gap_analyzer" in disp.INPUT_RESOLVERS, "gap_analyzer missing from INPUT_RESOLVERS"


def test_gap_analyzer_in_fail_closed_on_failed_status() -> None:
    """gap_analyzer's output contract sets status='failed' on workflow/connector
    failures (agent.py _build_error_output; gap_detector error path) — exactly the
    contract the domain-failure guard exists for. A failed gap run must never be
    laundered into a successful dispatch."""
    assert "gap_analyzer" in disp._FAIL_CLOSED_ON_FAILED_STATUS


# ---------------------------------------------------------------------------
# Bare chat dispatch: NEVER the raw 7ms field-validation crash (RED on main)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_chat_dispatch_is_not_raw_missing_field_error(monkeypatch) -> None:
    """A bare chat dispatch (no structured params) with NO substrate must fail
    closed with NeedsStructuredInput semantics — a clear, actionable error — and
    NOT the agent's raw ``Missing required field: metrics`` ValueError."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    # No substrate rows (clean real-mode DB) — the resolver must fail closed
    # BEFORE the agent method, so no DB/LLM is touched.
    monkeypatch.setattr(
        disp, "_probe_gap_substrate", lambda *a, **k: _probe_stub([], None, []), raising=False
    )

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    node = DispatcherNode(agent_registry={"gap_analyzer": agent})
    out = await node.execute(
        _state("where are our biggest gaps?", user_context={"brand": "Kisqali"})
    )

    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "missing required field" not in err, f"raw field-validation crash leaked: {err}"
    assert "metrics" in err and "brand" in err, f"error must name the missing inputs: {err}"
    assert "fabricat" in err, f"error must state nothing was fabricated: {err}"


# ---------------------------------------------------------------------------
# Substrate-derived binding
# ---------------------------------------------------------------------------


def test_resolver_builds_inputs_from_substrate(monkeypatch) -> None:
    """Brand named in user_context + substrate rows exist → the resolver binds the
    REAL metric names and the real segment dimension; nothing invented."""
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: _probe_stub(
            ["conversion_rate", "trx"], "Kisqali", ["midwest", "northeast"]
        ),
        raising=False,
    )

    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps for kisqali",
            "session_id": "s1",
            "user_context": {"brand": "kisqali"},
            "parsed_query": {"entities": []},
        },
        _dispatch(),
    )
    assert isinstance(resolved, dict)
    assert resolved["metrics"] == ["conversion_rate", "trx"]
    assert resolved["segments"] == ["region"]
    # Canonical (data) spelling wins over the chat casing.
    assert resolved["brand"] == "Kisqali"
    # Default = real mode.
    assert resolved["include_synthetic"] is False


def test_resolver_forwards_include_synthetic_opt_in(monkeypatch) -> None:
    """The #872 opt-in channels (filters / user_context) must reach BOTH the
    substrate probe AND the agent input (per-run connector opt-in)."""
    seen: Dict[str, Any] = {}

    def _capture(brand: str, include_synthetic: bool):
        seen["brand"] = brand
        seen["include_synthetic"] = include_synthetic
        return _probe_stub(["trx"], "Kisqali", ["west"])

    monkeypatch.setattr(disp, "_probe_gap_substrate", _capture, raising=False)
    base = {
        "query": "gaps",
        "session_id": "s1",
        "user_context": {"brand": "Kisqali"},
        "parsed_query": {"entities": []},
    }

    # filters channel
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {**base, "filters": {"include_synthetic": True}}, _dispatch()
    )
    assert isinstance(resolved, dict)
    assert seen["include_synthetic"] is True
    assert resolved["include_synthetic"] is True

    # user_context channel (the only caller-stash field the live chat path threads)
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {**base, "user_context": {"brand": "Kisqali", "include_synthetic": True}}, _dispatch()
    )
    assert isinstance(resolved, dict)
    assert seen["include_synthetic"] is True
    assert resolved["include_synthetic"] is True

    # default: real mode
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"]({**base}, _dispatch())
    assert isinstance(resolved, dict)
    assert seen["include_synthetic"] is False
    assert resolved["include_synthetic"] is False


def test_resolver_merges_router_config_overrides(monkeypatch) -> None:
    """Config-only parameters (gap_type/time_period/...) supplied WITHOUT the full
    required trio still apply on top of the substrate-derived inputs."""
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: _probe_stub(["trx"], "Kisqali", ["west"]),
        raising=False,
    )
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps",
            "session_id": "s1",
            "user_context": {"brand": "Kisqali"},
            "parsed_query": {"entities": []},
        },
        _dispatch({"gap_type": "all", "time_period": "2012-01-01_2026-12-31"}),
    )
    assert isinstance(resolved, dict)
    assert resolved["gap_type"] == "all"
    assert resolved["time_period"] == "2012-01-01_2026-12-31"
    assert resolved["metrics"] == ["trx"]


# ---------------------------------------------------------------------------
# Fail-closed paths
# ---------------------------------------------------------------------------


def test_resolver_no_brand_fails_closed(monkeypatch) -> None:
    """No brand in parsed_query/user_context → fail closed naming the gap inputs;
    the probe must not even run (nothing to scope a substrate read to)."""
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")),
        raising=False,
    )
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {"query": "where are the gaps?", "session_id": "s1", "user_context": {}},
        _dispatch(),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert "brand" in resolved.missing
    assert "brand" in resolved.reason.lower()
    assert resolved.rest_endpoint  # actionable: POST /api/gaps/analyze


def test_resolver_empty_substrate_fails_closed(monkeypatch) -> None:
    """Brand named but the (provenance-filtered) substrate has NO rows → fail
    closed with a reason naming the substrate — exactly the clean-DB real-mode
    case where only an explicit include_synthetic opt-in may read synthetic."""
    monkeypatch.setattr(
        disp, "_probe_gap_substrate", lambda *a, **k: _probe_stub([], None, []), raising=False
    )
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps for Kisqali",
            "session_id": "s1",
            "user_context": {"brand": "Kisqali"},
            "parsed_query": {"entities": []},
        },
        _dispatch(),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert "business_metrics" in resolved.reason


def test_resolver_probe_exception_fails_closed(monkeypatch) -> None:
    """An operational probe failure (DB unreachable) fails closed — never an
    unhandled raise, never fabricated inputs."""

    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(disp, "_probe_gap_substrate", _boom, raising=False)
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps for Kisqali",
            "session_id": "s1",
            "user_context": {"brand": "Kisqali"},
            "parsed_query": {"entities": []},
        },
        _dispatch(),
    )
    assert isinstance(resolved, NeedsStructuredInput)


# ---------------------------------------------------------------------------
# Explicit analyst-supplied params win (no substrate probe)
# ---------------------------------------------------------------------------


def test_resolver_explicit_params_win(monkeypatch) -> None:
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not probe substrate")),
        raising=False,
    )
    params = {
        "metrics": ["trx", "market_share"],
        "segments": ["region"],
        "brand": "Fabhalta",
        "gap_type": "vs_target",
        "include_synthetic": True,
    }
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {"query": "gaps", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch(params),
    )
    assert isinstance(resolved, dict)
    assert resolved["metrics"] == ["trx", "market_share"]
    assert resolved["segments"] == ["region"]
    assert resolved["brand"] == "Fabhalta"
    assert resolved["gap_type"] == "vs_target"
    assert resolved["include_synthetic"] is True


def test_resolver_explicit_params_honor_channel_opt_in(monkeypatch) -> None:
    """codex #874 R1 HIGH: an explicit trio + the opt-in arriving ONLY via a
    channel (parameters.filters / agent_input.filters / user_context) must still
    reach the agent — otherwise the run silently analyzes the wrong provenance
    mode (real-mode empty on a clean DB) while looking successful."""
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not probe substrate")),
        raising=False,
    )
    trio = {"metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"}

    # parameters.filters channel
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {"query": "gaps", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch({**trio, "filters": {"include_synthetic": True}}),
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is True

    # user_context channel
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps",
            "session_id": "s1",
            "user_context": {"include_synthetic": True},
            "parsed_query": {},
        },
        _dispatch(dict(trio)),
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is True

    # an EXPLICIT parameters.include_synthetic=False beats the ambient channels
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": "gaps",
            "session_id": "s1",
            "user_context": {"include_synthetic": True},
            "parsed_query": {},
        },
        _dispatch({**trio, "include_synthetic": False}),
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is False

    # default: real mode, explicitly set on the output (never left implicit)
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {"query": "gaps", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch(dict(trio)),
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is False


def test_resolver_partial_params_brand_seeds_derivation(monkeypatch) -> None:
    """codex #874 R1 MED: parameters={'brand': ...} WITHOUT metrics/segments must
    seed the substrate derivation (router-supplied brand wins over chat extraction),
    not fail closed as 'no brand named'."""
    seen: Dict[str, Any] = {}

    def _capture(brand: str, include_synthetic: bool):
        seen["brand"] = brand
        return _probe_stub(["trx"], "Kisqali", ["west"])

    monkeypatch.setattr(disp, "_probe_gap_substrate", _capture, raising=False)
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {"query": "gaps", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch({"brand": "kisqali", "gap_type": "all"}),
    )
    assert isinstance(resolved, dict)
    assert seen["brand"] == "kisqali"
    assert resolved["brand"] == "Kisqali"  # canonical data spelling
    assert resolved["gap_type"] == "all"


# ---------------------------------------------------------------------------
# Substrate probe: distinct discovery pages to slice exhaustion (codex R1/R2)
# ---------------------------------------------------------------------------


def _fake_probe_client(pages: List[List[Dict[str, str]]], cap: int, ranges: List[tuple]):
    """A chainable fake PostgREST client serving ``pages`` for range scans and a
    one-row presence response for ``.limit(1)`` brand probes."""

    class _Query:
        def __init__(self) -> None:
            self._mode = "scan"
            self._page = 0

        def eq(self, *args, **kwargs):
            return self

        def order(self, *args, **kwargs):
            return self

        def limit(self, n):
            self._mode = "presence"
            return self

        def range(self, start, end):
            ranges.append((start, end))
            self._page = start // cap
            return self

        def execute(self):
            result = MagicMock()
            if self._mode == "presence":
                result.data = [{"brand": "Kisqali"}]
            else:
                result.data = pages[self._page] if self._page < len(pages) else []
            return result

    class _Table:
        def select(self, cols):
            return _Query()

    class _Client:
        def table(self, name):
            assert name == "business_metrics"
            return _Table()

    return _Client()


def test_probe_gap_substrate_pages_to_exhaustion(monkeypatch) -> None:
    """The distinct metric/region scan must page until the brand slice is
    EXHAUSTED (no unsound 'saturation' early stop — a repeat page proves nothing
    about later pages, codex R2) and must collect distincts from EVERY page."""
    import src.repositories

    cap = 3
    monkeypatch.setattr(disp, "_GAP_PROBE_ROW_CAP", cap)

    pages = [
        [{"metric_name": "trx", "region": "west"}] * cap,  # page 0: full
        [{"metric_name": "trx", "region": "west"}] * cap,  # page 1: full, repeats
        [{"metric_name": "nrx", "region": "east"}] * 2,  # page 2: short -> exhausted
    ]
    ranges: List[tuple] = []
    monkeypatch.setattr(
        src.repositories, "get_supabase_client", lambda: _fake_probe_client(pages, cap, ranges)
    )

    metrics, canonical, regions = disp._probe_gap_substrate("Kisqali", True)
    assert canonical == "Kisqali"
    # Distinct values from ALL pages — including the late-arriving nrx/east.
    assert metrics == ["nrx", "trx"]
    assert regions == ["east", "west"]
    assert ranges == [(0, cap - 1), (cap, 2 * cap - 1), (2 * cap, 3 * cap - 1)]


def test_probe_gap_substrate_respects_page_bound(monkeypatch) -> None:
    """An unexhausted slice stops at _GAP_PROBE_MAX_PAGES (bounded dispatch
    latency) — warns, returns what it found, never fails closed on MORE data."""
    import src.repositories

    cap = 2
    monkeypatch.setattr(disp, "_GAP_PROBE_ROW_CAP", cap)
    monkeypatch.setattr(disp, "_GAP_PROBE_MAX_PAGES", 2)

    pages = [
        [{"metric_name": "trx", "region": "west"}] * cap,
        [{"metric_name": "nrx", "region": "east"}] * cap,
        [{"metric_name": "market_share", "region": "south"}] * cap,  # beyond the bound
    ]
    ranges: List[tuple] = []
    monkeypatch.setattr(
        src.repositories, "get_supabase_client", lambda: _fake_probe_client(pages, cap, ranges)
    )

    metrics, canonical, regions = disp._probe_gap_substrate("Kisqali", True)
    assert canonical == "Kisqali"
    assert metrics == ["nrx", "trx"]  # bound hit: page 2 not read, result still real
    assert regions == ["east", "west"]
    assert len(ranges) == 2


# ---------------------------------------------------------------------------
# Strict provenance-flag parsing (codex R2): ambiguity stays real-mode
# ---------------------------------------------------------------------------


def test_provenance_flag_coercion_is_strict() -> None:
    assert disp._coerce_provenance_flag(True) is True
    assert disp._coerce_provenance_flag("true") is True
    assert disp._coerce_provenance_flag("1") is True
    assert disp._coerce_provenance_flag("yes") is True
    # bool("false") is True — the naive coercion this guards against.
    assert disp._coerce_provenance_flag("false") is False
    assert disp._coerce_provenance_flag("0") is False
    assert disp._coerce_provenance_flag(False) is False
    assert disp._coerce_provenance_flag(None) is False
    assert disp._coerce_provenance_flag(1) is False  # non-bool, non-str: fail closed
    assert disp._coerce_provenance_flag({"opt": True}) is False


def test_resolver_string_false_param_does_not_opt_in(monkeypatch) -> None:
    """An explicit ``include_synthetic='false'`` (string, e.g. from a JSON router
    payload) must opt OUT — and a ``None`` param defers to the ambient channel."""
    monkeypatch.setattr(
        disp,
        "_probe_gap_substrate",
        lambda *a, **k: _probe_stub(["trx"], "Kisqali", ["west"]),
        raising=False,
    )
    trio = {"metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"}
    ambient = {
        "query": "gaps",
        "session_id": "s1",
        "user_context": {"include_synthetic": True},
        "parsed_query": {},
    }

    # String "false" with an ambient True channel: explicit opt-out wins.
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        ambient, _dispatch({**trio, "include_synthetic": "false"})
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is False

    # None means "unset": the ambient channel governs.
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        ambient, _dispatch({**trio, "include_synthetic": None})
    )
    assert isinstance(resolved, dict)
    assert resolved["include_synthetic"] is True


# ---------------------------------------------------------------------------
# Domain-failure guard: status='failed' output → success=False (RED on main)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gap_analyzer_failed_status_fails_dispatch_closed() -> None:
    """A gap run that executes but reports status='failed' (e.g. a fail-closed
    connector raise recorded by gap_detector) must yield success=False, not a
    laundered 'successful' empty analysis."""

    async def fake_run(input_data):  # noqa: ANN001
        return {
            "status": "failed",
            "errors": [{"node": "gap_detector", "error": "Supabase unreachable"}],
            "prioritized_opportunities": [],
        }

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    node = DispatcherNode(agent_registry={"gap_analyzer": agent})
    out = await node.execute(
        _state(
            "gaps",
            params={"metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"},
        )
    )
    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "supabase unreachable" in err
    assert "fabricat" in err


@pytest.mark.asyncio
async def test_gap_analyzer_completed_status_still_succeeds() -> None:
    """The guard must not over-fire: a completed run stays a successful dispatch."""

    async def fake_run(input_data):  # noqa: ANN001
        return {
            "status": "completed",
            "executive_summary": "2 real opportunities",
            "prioritized_opportunities": [{"rank": 1}, {"rank": 2}],
        }

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    node = DispatcherNode(agent_registry={"gap_analyzer": agent})
    out = await node.execute(
        _state(
            "gaps",
            params={"metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"},
        )
    )
    res = out["agent_results"][0]
    assert res["success"] is True, res.get("error")
    assert res["result"]["executive_summary"] == "2 real opportunities"
