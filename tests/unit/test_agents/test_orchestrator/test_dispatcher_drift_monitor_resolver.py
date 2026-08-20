"""Issue #1747: drift_monitor dispatch-dead on the chat path.

Every live chat dispatch of drift_monitor failed input coercion before the
agent graph ran::

    Failed to build DriftMonitorInput: 1 validation error for DriftMonitorInput
    features_to_monitor
      Field required [type=missing, input_value={'query': ...}]

``DriftMonitorInput.features_to_monitor`` is required with ``min_length=1`` but
the dispatcher had no ``_resolve_drift_monitor_input`` — unlike gap_analyzer
(#874), heterogeneous_optimizer (#1726), causal_impact and tool_composer. The
only prior mechanism (``_agent_specific_defaults`` deriving features from
``parsed_query.entities`` kpi/feature_name mentions, #260) produces nothing on
a bare chat dispatch — measured live 2/2 (wave-18 follow-up forced probes
drift.f1/drift.f2).

The fix mirrors the #874/#1726 resolvers, grounded in the measured substrate
(2026-08-20 live DB): the feature store is 100% synthetic-tagged, the default
7d window has ZERO features with >= 30 samples in both drift windows, while
30d has 15 — so the resolver must (a) honor the include_synthetic opt-in
channels, (b) probe candidate windows and bind the smallest that the data
actually supports, and (c) fail closed honestly when nothing qualifies. It
must NEVER bind a chat-derived brand filter: only ~21% of recent
feature_values carry a brand entity key, so a brand filter starves the
windows below min-samples.

The substrate probe is monkeypatched here (unit scope, no DB); the faithful
real-DB proof lives in
``tests/integration/test_dispatcher_drift_monitor_substrate_realdb.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput


@pytest.fixture(autouse=True)
def _real_mode_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the strict real-mode default: the showcase E2I_INCLUDE_SYNTHETIC
    flag is read fresh from env on every call (provenance.py) and would flip
    the opt-in ambient-True on a showcase box (.env contamination family,
    #1414)."""
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def _payload(
    query: str = "Run drift detection",
    *,
    entities: Optional[List[Dict[str, Any]]] = None,
    user_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-1747",
        "parsed_query": {"intent": "drift_detection", "entities": entities or []},
    }


def _dispatch(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_name": "drift_monitor",
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 30000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _resolve(payload: Dict[str, Any], dispatch: Dict[str, Any]):
    return disp._resolve_drift_monitor_input(payload, dispatch)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Registry shape (RED on main: drift_monitor has no resolver entry)
# ---------------------------------------------------------------------------


def test_drift_monitor_registered_in_input_resolvers() -> None:
    assert "drift_monitor" in disp.INPUT_RESOLVERS, "drift_monitor missing from INPUT_RESOLVERS"


def test_drift_monitor_in_fail_closed_on_failed_status() -> None:
    # run() sets status="failed" on node errors (agent.py) — a failed drift
    # sweep must never be laundered into a successful dispatch narrative.
    assert "drift_monitor" in disp._FAIL_CLOSED_ON_FAILED_STATUS


# ---------------------------------------------------------------------------
# (1) explicit analyst-supplied features pass through verbatim
# ---------------------------------------------------------------------------


def test_explicit_features_pass_through_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_probe(*a: Any, **k: Any) -> List[str]:  # pragma: no cover - must not fire
        raise AssertionError("substrate probe must not run on the explicit-params path")

    monkeypatch.setattr(disp, "_probe_drift_substrate", _no_probe)
    params = {
        "features_to_monitor": ["trx_total", "hcp_engagement_frequency"],
        "model_id": "model-x",
        "time_window": "14d",
        "brand": "Kisqali",
        "psi_threshold": 0.2,
    }
    out = _resolve(_payload(), _dispatch(params))
    assert isinstance(out, dict)
    assert out["features_to_monitor"] == ["trx_total", "hcp_engagement_frequency"]
    assert out["model_id"] == "model-x"
    assert out["time_window"] == "14d"
    assert out["brand"] == "Kisqali"
    assert out["psi_threshold"] == 0.2


def test_explicit_path_forwards_include_synthetic_opt_in() -> None:
    params = {"features_to_monitor": ["trx_total"], "include_synthetic": True}
    out = _resolve(_payload(), _dispatch(params))
    assert isinstance(out, dict)
    assert out["include_synthetic"] is True


def test_explicit_path_defaults_real_mode() -> None:
    out = _resolve(_payload(), _dispatch({"features_to_monitor": ["trx_total"]}))
    assert isinstance(out, dict)
    assert out["include_synthetic"] is False


# ---------------------------------------------------------------------------
# (2) user-named kpi/feature entities bind verbatim (the user's ask wins;
# per-feature honesty is the agent's job)
# ---------------------------------------------------------------------------


def test_entity_named_features_bind_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_probe(*a: Any, **k: Any) -> List[str]:  # pragma: no cover - must not fire
        raise AssertionError("substrate probe must not run when the user named features")

    monkeypatch.setattr(disp, "_probe_drift_substrate", _no_probe)
    payload = _payload(
        "Is trx_total drifting?",
        entities=[
            {"type": "kpi", "value": "trx_total"},
            {"type": "feature_name", "value": "hcp_engagement_frequency"},
            {"type": "brand", "value": "Kisqali"},
        ],
    )
    out = _resolve(payload, _dispatch())
    assert isinstance(out, dict)
    assert out["features_to_monitor"] == ["trx_total", "hcp_engagement_frequency"]
    # chat-derived brand must NOT become a filter (measured: only ~21% of
    # recent feature_values carry a brand entity key — a brand filter starves
    # the windows below min-samples).
    assert "brand" not in out


# ---------------------------------------------------------------------------
# (3) substrate sweep: smallest qualifying window, capped features, no brand
# ---------------------------------------------------------------------------


def _window_probe(mapping: Dict[int, List[str]]):
    calls: List[int] = []

    def probe(window_days: int, include_synthetic: bool) -> List[str]:
        calls.append(window_days)
        return mapping.get(window_days, [])

    return probe, calls


def test_sweep_picks_smallest_qualifying_window(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, calls = _window_probe({7: [], 14: [], 30: ["f_a", "f_b"], 90: ["f_a", "f_b", "f_c"]})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch())
    assert isinstance(out, dict)
    assert out["features_to_monitor"] == ["f_a", "f_b"]
    assert out["time_window"] == "30d"
    assert calls == [7, 14, 30], "must stop at the first qualifying window"


def test_sweep_caps_bound_features(monkeypatch: pytest.MonkeyPatch) -> None:
    many = [f"f_{i:02d}" for i in range(40)]
    probe, _ = _window_probe({7: many})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch())
    assert isinstance(out, dict)
    cap = disp._DRIFT_MAX_FEATURES  # type: ignore[attr-defined]
    assert len(out["features_to_monitor"]) == cap
    # probe returns names ordered by current-window support; the cap keeps the
    # best-supported prefix.
    assert out["features_to_monitor"] == many[:cap]


def test_sweep_never_binds_chat_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, _ = _window_probe({7: ["f_a"]})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    payload = _payload(
        "Check drift for Kisqali models",
        entities=[{"type": "brand", "value": "Kisqali"}],
    )
    out = _resolve(payload, _dispatch())
    assert isinstance(out, dict)
    assert "brand" not in out


def test_sweep_forwards_include_synthetic_to_probe_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: List[bool] = []

    def probe(window_days: int, include_synthetic: bool) -> List[str]:
        seen.append(include_synthetic)
        return ["f_a"]

    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(
        _payload(user_context={"user_id": "u1", "include_synthetic": True}),
        _dispatch(),
    )
    assert isinstance(out, dict)
    assert out["include_synthetic"] is True
    assert seen == [True]


def test_explicit_time_window_param_restricts_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, calls = _window_probe({14: ["f_a"]})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch({"time_window": "14d"}))
    assert isinstance(out, dict)
    assert calls == [14], "an analyst-chosen window must not be silently widened"
    assert out["time_window"] == "14d"
    assert out["features_to_monitor"] == ["f_a"]


def test_explicit_time_window_that_has_no_data_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, calls = _window_probe({})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch({"time_window": "1d"}))
    assert isinstance(out, NeedsStructuredInput)
    assert calls == [1]
    assert "1d" in out.reason


# ---------------------------------------------------------------------------
# (4) fail closed: honest, actionable, never fabricated
# ---------------------------------------------------------------------------


def test_fail_closed_when_no_window_qualifies(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, calls = _window_probe({})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert out.agent_name == "drift_monitor"
    assert out.missing == ("features_to_monitor",)
    assert out.rest_endpoint == "POST /api/monitoring/drift/detect"
    assert out.user_action, "fail-closed must carry a user-facing invitation (#1451)"
    # every candidate window was probed before giving up
    assert calls == [int(w[:-1]) for w in disp._DRIFT_WINDOW_CANDIDATES]  # type: ignore[attr-defined]


def test_fail_closed_reason_names_provenance_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, _ = _window_probe({})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    real_mode = _resolve(_payload(), _dispatch())
    assert isinstance(real_mode, NeedsStructuredInput)
    assert "real-mode" in real_mode.reason or "synthetic" in real_mode.reason


def test_probe_exception_fails_closed_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(window_days: int, include_synthetic: bool) -> List[str]:
        raise RuntimeError("supabase down")

    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    out = _resolve(_payload(), _dispatch())
    assert isinstance(out, NeedsStructuredInput), "a probe outage must fail closed, not raise"


# ---------------------------------------------------------------------------
# (5) the exact live failure surface: resolver output must coerce into
# DriftMonitorInput (the pre-fix crash was at this seam)
# ---------------------------------------------------------------------------


def test_resolved_output_coerces_into_drift_monitor_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agents.drift_monitor.agent import DriftMonitorInput

    probe, _ = _window_probe({7: ["f_a", "f_b"]})
    monkeypatch.setattr(disp, "_probe_drift_substrate", probe)
    payload = _payload()
    dispatch = _dispatch()
    resolved = _resolve(payload, dispatch)
    assert isinstance(resolved, dict)
    merged = dict(payload)
    merged.update(resolved)
    model = disp._coerce_to_input_model(DriftMonitorInput, merged, dispatch, "drift_monitor")
    assert model.features_to_monitor == ["f_a", "f_b"]
    assert model.time_window == "7d"
