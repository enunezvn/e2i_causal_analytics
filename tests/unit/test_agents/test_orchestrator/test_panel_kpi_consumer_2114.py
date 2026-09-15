"""The panel vocabulary ANSWERS through the real consumer path, and #1475 stays armed.

Entry-gate first: these phrasings are the ones that pass KPI_VALUE_LOOKUP_RE. The bare
aliases ("NRx panel") are correct for the RESOLVER tests and wrong here — they have no
interrogative lead-in, so they never reach the guards and would assert nothing.

The seam is the REAL one: _kpi_lookup_evidence does `from src.api.routes.kpi import
get_kpi_calculator` INSIDE the function and calls `.calculate(kpi.id, context=context)`
(dispatcher.py). Patch that name. Asserting only "evidence is not None" would let a
resolver-correct/consumer-wrong path pass, and asserting only "evidence is None" on the
refusal cases would let an unrelated calculator exception masquerade as guard enforcement
— the engine call is wrapped in `except Exception: return None` (codex r19-02).
"""

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence
from src.kpi.models import KPIResult, KPIStatus

#: The REAL result model, not a hand-rolled stand-in.
#:
#: The plan's draft used a 4-field dataclass (`value`, `error`, `window_status`,
#: `metadata`). `_kpi_lookup_evidence` reads SEVEN attributes off this object --
#: measured: value, status, error, metadata, window_requested, window_applied,
#: window_status -- so the stub raised `AttributeError: '_Result' object has no
#: attribute 'status'` at dispatcher.py:2369 and all five answer cases failed for a
#: reason that had nothing to do with the panel vocabulary they were written to test.
#:
#: Binding to `KPIResult` instead of listing the seven means the double cannot drift
#: from the contract: a field added to the real model arrives here automatically, and
#: one removed fails loudly. Same lesson as the r6 wrong-layer fake -- a double is only
#: evidence while it is faithful to the thing it stands in for.


class _RecordingCalculator:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def calculate(self, kpi_id: str, context: Dict[str, Any]) -> KPIResult:
        self.calls.append(kpi_id)
        return KPIResult(
            kpi_id=kpi_id,
            value=123.0,
            status=KPIStatus.INFORMATIONAL,
            metadata={"context": {"data_through": "2026-08-31"}},
        )


@pytest.fixture
def calculator(monkeypatch) -> _RecordingCalculator:
    rec = _RecordingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: rec)
    return rec


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("What is NRx panel for Kisqali?", "WS3-BI-012"),
        ("What is the TRx panel for Kisqali?", "WS3-BI-011"),
        ("What is panel NBRx for Kisqali?", "WS3-BI-013"),
        ("What is TRx share panel for Kisqali?", "WS3-BI-014"),
        ("show me the NRx panel", "WS3-BI-012"),
    ],
)
def test_a_panel_lookup_resolves_and_answers(query, expected_id, calculator):
    """Branch 1 of the Step 4 triage: a legitimate panel lookup must RESOLVE AND ANSWER.
    The calculator must RECEIVE the expected id, and the evidence must carry its value."""
    evidence = _kpi_lookup_evidence({"query": query})
    assert calculator.calls == [expected_id], (query, calculator.calls)
    assert evidence, f"{query!r} produced no evidence — it did not answer"
    rendered = " ".join(str(e) for e in evidence)
    assert "123" in rendered, (query, rendered[:200])


@pytest.mark.parametrize(
    "query,why",
    [
        (
            "What is the cost of NRx panel?",
            "governing 'of' head — the KPI is a modifier, not the asked value",
        ),
        ("What are NBRx panel drivers?", "causal right-head — a bare value does not answer it"),
    ],
)
def test_the_1475_guards_are_still_armed_over_panel_phrasings(query, why, calculator):
    """Branch 2: a cost / drivers phrasing must KEEP its refusal, and must be refused BEFORE
    the engine is consulted. Zero calculator calls is the assertion that distinguishes a guard
    refusal from an engine failure swallowed by the fail-closed except."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} reached the engine; the guard did not refuse it"
