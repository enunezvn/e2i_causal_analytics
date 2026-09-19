"""A KPI's OWN registry name must identify it (#2114, r12 CI triage).

The lane added the family aliases "observed rx events" / "rx events" -> WS3-BI-011,
and every panel KPI's registry name BEGINS with that phrase:

    WS3-BI-011  'Observed Rx Events - Patient Panel TRx (TRx Panel)'
    WS3-BI-012  'Observed Rx Events - Patient Panel NRx (NRx Panel)'
    WS3-BI-013  'Observed Rx Events - Patient Panel NBRx (NBRx Panel)'
    WS3-BI-014  'Observed Rx Events - Patient Panel TRx Share (TRx Share Panel)'

So passing 012's own name grounded 012 (via "NRx Panel") AND 011 (via the family
alias), the " - " in the name matched `_KPI_COORDINATOR_RE`, and the multi-KPI veto
refused the ask. **The name coordinated with itself.** Measured before the fix:

    WS3-BI-011 bare -> success=True                      (011 is the alias target)
    WS3-BI-012 bare -> success=False, no 'kpi_id' key
    WS3-BI-013 bare -> success=False, no 'kpi_id' key
    WS3-BI-014 bare -> success=False, no 'kpi_id' key

THREE OF THE FOUR PANEL KPIs COULD NOT BE COMPUTED BY THEIR OWN NAME AT ALL — with
or without an axis. The 12 `KeyError: 'kpi_id'` failures in
test_chatbot_kpi_axis_gate_1911.py were the symptom; the missing key is the refusal
shape, not the defect.

⚠ WHAT THIS FIX RESTORES IS **IDENTITY**, NOT COMPUTATION — a precision the first
report of it blurred. Measured at ce735f312, bare (no brand) and with a brand:

    011  bare success=True   kpi_id=WS3-BI-011   | +brand success=True
    012  bare success=True   kpi_id=WS3-BI-012   | +brand success=True
    013  bare success=False  kpi_id=WS3-BI-013   | +brand success=True
    014  bare success=False  kpi_id=WS3-BI-014   | +brand success=True

All four IDENTIFY in every case, which is the defect this fix closes. Two of the
four do not COMPUTE bare, for an unrelated domain reason ("no brand specified for
new-to-brand prescriptions (NBRx)" / "for TRx share") — well-formed refusals that
carry the kpi_id. The tests below supply a brand, so they prove computation UNDER
THAT CONDITION and say nothing about the bare case.

"All four compute by their own registry name" was therefore ONE WORD WIDER THAN
ITS MEASUREMENT: true of these tests' conditions, wrong if read unconditionally.
Same family as the battery count that said ten beside a table of twelve. Stated
here so the next reader does not pin "all four compute bare" as an expectation.

⭐ THIRD INSTANCE OF ONE GENUS IN ONE DAY: A SUBSTRING MATCH READ AS WHOLE-SPAN
IDENTITY. r12 finding 3 — `brand_from_text('cost kisqali')` matched a brand INSIDE
the span. r12-7 — the resolver matched 'west' inside 'west coast'. Here — the family
alias matches INSIDE the registry name and is read as a second KPI. Three different
resolvers, one mistake. Expect a fourth.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.kpi.models import KPIResult, KPIStatus
from src.kpi.registry import get_registry

_PANEL_IDS = ("WS3-BI-011", "WS3-BI-012", "WS3-BI-013", "WS3-BI-014")


class _CapturingCalculator:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def calculate(self, kpi_id: str, context: Dict[str, Any]) -> KPIResult:
        self.calls.append(kpi_id)
        return KPIResult(
            kpi_id=kpi_id,
            value=1.0,
            status=KPIStatus.INFORMATIONAL,
            metadata={"context": {"data_through": "2026-08-31"}},
        )


@pytest.fixture
def calculator(monkeypatch) -> _CapturingCalculator:
    rec = _CapturingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: rec)
    return rec


@pytest.mark.asyncio
@pytest.mark.parametrize("kpi_id", _PANEL_IDS)
async def test_a_panel_kpi_computes_by_its_own_registry_name(kpi_id, calculator):
    """The registry name is the one string guaranteed to name exactly this KPI."""
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    kpi = get_registry().get(kpi_id)
    assert kpi is not None
    resp = await kpi_calculate_tool.ainvoke({"kpi_name": kpi.name, "brand": "Remibrutinib"})

    assert resp["success"] is True, resp
    assert resp["kpi_id"] == kpi_id, resp
    assert calculator.calls == [kpi_id], calculator.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kpi_name,why",
    [
        (
            "Observed Rx Events - Patient Panel NRx (NRx Panel) and ROI",
            "a real coordination that CONTAINS a registry name must still refuse",
        ),
        (
            "TRx and NRx",
            "the ordinary two-metric ask #1637 exists to catch",
        ),
        (
            "TRx Panel and NRx Panel",
            "two panel KPIs named by alias",
        ),
    ],
)
async def test_a_genuinely_coordinated_ask_still_refuses(kpi_name, why, calculator):
    """THE OTHER DIRECTION, and the one that makes the fix a fix rather than a hole.

    The exactness test compares the WHOLE normalized string against the KPI's whole
    normalized name, so appending " and ROI" makes them unequal and the multi-KPI veto
    applies as before. A fix that greened the panel names by weakening the veto would
    have reopened #1637."""
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name, "brand": "Remibrutinib"})

    assert resp["success"] is False, (why, resp)
    assert calculator.calls == [], (why, calculator.calls)
    assert "one KPI per call" in resp["error"], (why, resp["error"])
