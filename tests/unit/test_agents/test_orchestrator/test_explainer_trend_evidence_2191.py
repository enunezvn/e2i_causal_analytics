"""#2191 point 4: KPI trend asks bind a real series, never a scalar.

The value-lookup regex deliberately routes these gold ``explanation`` asks to
the explainer.  The scalar calculator cannot answer them, though: a trend is a
sequence of observations.  The repository already has exactly that substrate
in ``kpi_history`` (the same monthly series used by the Time-Series page and
``renderKpiTrend``), so the explainer resolver must bind those rows or fail
closed when no rows exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput
from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE


def _dispatch() -> dict[str, Any]:
    return {
        "agent_name": "explainer",
        "priority": "high",
        "parameters": {},
        "timeout_ms": 15_000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _agent_input(
    query: str, *, agent_results: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "query": query,
        "session_id": "sess-2191",
        "user_context": {},
        "parsed_query": {"entities": []},
        "agent_results": agent_results or [],
    }


@dataclass
class _Response:
    data: list[dict[str, Any]]


class _HistoryQuery:
    def __init__(self, rows: list[dict[str, Any]], calls: list[tuple[str, Any]]):
        self._rows = rows
        self._calls = calls

    def select(self, columns: str):
        self._calls.append(("select", columns))
        return self

    def eq(self, column: str, value: Any):
        self._calls.append(("eq", (column, value)))
        return self

    def gte(self, column: str, value: Any):
        self._calls.append(("gte", (column, value)))
        return self

    def lte(self, column: str, value: Any):
        self._calls.append(("lte", (column, value)))
        return self

    def order(self, column: str, *, desc: bool = False):
        self._calls.append(("order", (column, desc)))
        return self

    def limit(self, value: int):
        self._calls.append(("limit", value))
        return self

    def execute(self) -> _Response:
        self._calls.append(("execute", None))
        return _Response(self._rows)


class _HistoryClient:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    def table(self, name: str) -> _HistoryQuery:
        self.calls.append(("table", name))
        return _HistoryQuery(self.rows, self.calls)


def _rows(kpi_id: str) -> list[dict[str, Any]]:
    return [
        {
            "kpi_id": kpi_id,
            "brand": "Remibrutinib" if kpi_id == "WS3-BI-007" else "",
            "region": "",
            "metric_date": "2026-06-01",
            "value": 100.0,
            "status": "informational",
            "source": "business_metrics.value",
            "is_synthetic": False,
        },
        {
            "kpi_id": kpi_id,
            "brand": "Remibrutinib" if kpi_id == "WS3-BI-007" else "",
            "region": "",
            "metric_date": "2026-07-01",
            "value": 112.0,
            "status": "informational",
            "source": "business_metrics.value",
            "is_synthetic": False,
        },
    ]


class _ScalarCalculatorMustNotRun:
    def calculate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a time-series ask must not call the scalar KPI calculator")


@pytest.mark.parametrize(
    ("query", "kpi_id", "brand", "expects_window"),
    [
        (
            "Can you show me the trend of Remibrutinib NBRx over the past 6 months?",
            "WS3-BI-007",
            "Remibrutinib",
            True,
        ),
        ("What is the trajectory of weekly TRx?", "WS3-BI-005", None, False),
        (
            "Show me how Remibrutinib NRx evolved over the past 90 days",
            "WS3-BI-006",
            "Remibrutinib",
            True,
        ),
    ],
)
def test_trend_asks_bind_monthly_history_not_a_scalar(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    kpi_id: str,
    brand: str | None,
    expects_window: bool,
) -> None:
    """The three issue reproductions must reach the existing history substrate."""
    assert KPI_VALUE_LOOKUP_RE.search(query), "entry gate must remain gold-compatible"
    client = _HistoryClient(_rows(kpi_id))
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator", lambda: _ScalarCalculatorMustNotRun()
    )

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, dict), resolved
    evidence = resolved["analysis_results"]
    assert len(evidence) == 1
    payload = evidence[0]
    assert payload["agent"] == "kpi_history"
    assert payload["analysis_type"] == "kpi_trend"
    assert payload["kpi_id"] == kpi_id
    assert payload["brand"] == brand
    assert payload["grain"] == "monthly"
    assert payload["points"] == [
        {"metric_date": "2026-06-01", "value": 100.0, "status": "informational"},
        {"metric_date": "2026-07-01", "value": 112.0, "status": "informational"},
    ]
    assert "value" not in payload, "a series ask must not masquerade as one scalar"
    assert "formatted_value" not in payload
    assert "100" in payload["key_findings"][0]
    assert "112" in payload["key_findings"][0]
    if expects_window:
        assert any(call[0] == "gte" for call in client.calls)
        assert any(call[0] == "lte" for call in client.calls)
    else:
        assert all(call[0] not in {"gte", "lte"} for call in client.calls)
    assert ("table", "kpi_history") in client.calls
    assert ("eq", ("kpi_id", kpi_id)) in client.calls
    assert ("eq", ("brand", brand or "")) in client.calls


def test_weekly_word_is_disclosed_as_monthly_history_grain(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _HistoryClient(_rows("WS3-BI-005"))
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the trajectory of weekly TRx?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    warnings = resolved["analysis_results"][0]["warnings"]
    assert any("monthly" in warning.lower() and "weekly" in warning.lower() for warning in warnings)


@pytest.mark.asyncio
async def test_real_explainer_narrates_the_bound_series(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.explainer import ExplainerAgent

    client = _HistoryClient(_rows("WS3-BI-005"))
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)
    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the TRx trend?"), _dispatch()
    )
    assert isinstance(resolved, dict), resolved

    output = await ExplainerAgent(use_llm=False).explain(**resolved)

    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "monthly history" in narrative
    assert "2026-06-01" in narrative and "2026-07-01" in narrative
    assert "100" in narrative and "112" in narrative


def test_empty_history_fails_closed_without_falling_back_to_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _HistoryClient([])
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator", lambda: _ScalarCalculatorMustNotRun()
    )

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("Show me the trend of Remibrutinib NBRx over the past 6 months"),
        _dispatch(),
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert ("table", "kpi_history") in client.calls


def test_empty_current_trend_does_not_explain_stale_prior_turn_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognizing the current ask makes an empty series authoritative."""
    client = _HistoryClient([])
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)
    stale = [
        {
            "agent_name": "gap_analyzer",
            "success": True,
            "result": {"key_findings": ["unrelated prior-turn result"]},
        }
    ]

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the TRx trend?", agent_results=stale), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert ("table", "kpi_history") in client.calls


@pytest.mark.parametrize(
    "query",
    [
        "What is TRx cost?",
        "By pharmacy, show me the TRx trend.",
        "Show me the trend of TRx and NRx.",
        "Show me the trend of TRx cost.",
        "Show me the trend of TRx drivers.",
    ],
)
def test_non_trend_or_unrepresentable_trend_scope_does_not_bind_history(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    client = _HistoryClient(_rows("WS3-BI-005"))
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: client)

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert all(call[0] != "table" for call in client.calls)
