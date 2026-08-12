"""#1538: kpi_calculate_tool speaks the platform region vocabulary and echoes
region provenance truthfully.

Before this change the tool passed ``region`` RAW into the calculator context
(no ``resolve_region_label`` — the #1537 brand-normalization precedent applied
to region) and ``_kpi_result_to_response`` echoed ``"region": region``
UNCONDITIONALLY — the #1534 defect class: a scope echoed that the calculator
may have silently ignored, so the synthesizer captions a global figure as
regional.

Pure unit coverage (no DB): normalization at the tool seam, fail-fast on an
unmappable region (with the known-label list, before any calculator call), and
the provenance-truthful response echo.
"""

from typing import Any

import pytest

from src.kpi.models import KPIResult, KPIStatus
from src.kpi.registry import get_registry


@pytest.mark.unit
def test_kpi_calculate_input_region_mentions_vocabulary():
    """The LLM only passes clean args if the schema teaches the vocabulary."""
    from src.api.routes.chatbot_tools import KpiCalculateInput

    desc = KpiCalculateInput.model_fields["region"].description or ""
    for label in ("northeast", "south", "midwest", "west"):
        assert label in desc


# ---- response echo: provenance-truthful -------------------------------------


@pytest.mark.unit
def test_response_echoes_region_when_applied():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-005")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-005",
        value=249.0,
        status=KPIStatus.UNKNOWN,
        region_requested="northeast",
        region_applied="northeast",
        region_status="applied",
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali", region="northeast")
    assert resp["region"] == "northeast"
    assert resp["region_status"] == "applied"
    assert "region_note" not in resp


@pytest.mark.unit
def test_response_never_echoes_unapplied_region():
    """A calculator without a region variant keeps its global value; the echo
    must say so instead of captioning the figure with the ignored region."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS1-MP-001")  # model-performance: no region variant
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS1-MP-001",
        value=0.87,
        status=KPIStatus.GOOD,
        region_requested="northeast",
        region_applied=None,
        region_status="not_applicable",
    )
    resp = _kpi_result_to_response(kpi, result, brand=None, region="northeast")
    assert resp["region"] is None
    assert resp["region_status"] == "not_applicable"
    note = resp.get("region_note", "")
    assert "northeast" in note and "not" in note.lower()


@pytest.mark.unit
def test_response_region_default_when_none_requested():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-005")
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-005", value=10.0, status=KPIStatus.UNKNOWN)
    resp = _kpi_result_to_response(kpi, result)
    assert resp["region"] is None
    assert resp["region_status"] == "default"
    assert "region_note" not in resp


# ---- tool seam: normalization + fail-fast -----------------------------------


class _CapturingCalc:
    def __init__(self, result: KPIResult):
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def calculate(self, kpi_id, context=None, **kwargs):
        self.calls.append({"kpi_id": kpi_id, "context": context})
        return self._result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_normalizes_region_synonym_into_context(monkeypatch):
    """'North East' must reach the calculator as the enum label 'northeast'
    (resolve_region_label with synonyms — the platform's one vocabulary)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    calc = _CapturingCalc(
        KPIResult(
            kpi_id="WS3-BI-005",
            value=249.0,
            status=KPIStatus.UNKNOWN,
            region_requested="northeast",
            region_applied="northeast",
            region_status="applied",
        )
    )
    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: calc, raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "TRx", "brand": "Kisqali", "region": "North East"}
    )
    assert resp["success"] is True
    assert calc.calls[0]["context"]["region"] == "northeast"
    assert resp["region"] == "northeast"
    assert resp["region_status"] == "applied"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_fails_fast_on_unmappable_region(monkeypatch):
    """An unknown region must never reach the DB: fail with the known-label
    list (the #1501/#1537 enum fail-fast precedent), not a misleading 0/None."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    calc = _CapturingCalc(KPIResult(kpi_id="WS3-BI-005", value=0.0, status=KPIStatus.UNKNOWN))
    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: calc, raising=False)

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx", "region": "EMEA"})
    assert resp["success"] is False
    assert "EMEA" in resp["error"]
    for label in ("midwest", "northeast", "south", "west"):
        assert label in resp["error"]
    assert calc.calls == []  # fail-fast: no calculator/DB touch


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_normalizes_natural_phrasings_into_context(monkeypatch):
    """#1565: 'the Northeast region' / 'West Coast' are fully determinable
    phrasings — they must reach the calculator as the enum label, and the
    #1538 provenance echo must carry the APPLIED label, never the raw arg."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    for supplied, label in [("the Northeast region", "northeast"), ("West Coast", "west")]:
        calc = _CapturingCalc(
            KPIResult(
                kpi_id="WS3-BI-005",
                value=249.0,
                status=KPIStatus.UNKNOWN,
                region_requested=label,
                region_applied=label,
                region_status="applied",
            )
        )
        monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda c=calc: c, raising=False)

        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Kisqali", "region": supplied}
        )
        assert resp["success"] is True, supplied
        assert calc.calls[0]["context"]["region"] == label, supplied
        assert resp["region"] == label, supplied
        assert resp["region_status"] == "applied", supplied


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_clarifies_on_ambiguous_region(monkeypatch):
    """#1565: an ambiguous phrasing ('East Coast' spans two census regions)
    still fails closed pre-DB, but the failure must be a clarify — the hint
    tells the LLM to ask which of the four US census regions is meant —
    instead of a dead-end error."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    calc = _CapturingCalc(KPIResult(kpi_id="WS3-BI-005", value=0.0, status=KPIStatus.UNKNOWN))
    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: calc, raising=False)

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx", "region": "East Coast"})
    assert resp["success"] is False
    assert "East Coast" in resp["error"]
    # Failure-closed pin from #1538 stays: the error names the known labels
    # and the calculator/DB is never touched.
    for lbl in ("midwest", "northeast", "south", "west"):
        assert lbl in resp["error"]
    assert calc.calls == []
    # The #1565 upgrade: a clarify hint naming the census regions, phrased as
    # a question to relay to the user.
    hint = resp.get("hint", "")
    assert "census region" in hint
    for lbl in ("northeast", "south", "midwest", "west"):
        assert lbl in hint
    assert "ask" in hint.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_surfaces_not_applicable_provenance(monkeypatch):
    """When the engine reports the region was NOT applied, the tool response
    must carry that verdict so the synthesizer never mislabels the figure."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    calc = _CapturingCalc(
        KPIResult(
            kpi_id="WS3-BI-004",  # HCP coverage: genuinely global-only
            value=0.87,
            status=KPIStatus.GOOD,
            region_requested="west",
            region_applied=None,
            region_status="not_applicable",
        )
    )
    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: calc, raising=False)

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "HCP coverage", "region": "west"})
    assert resp["success"] is True
    assert resp["region"] is None
    assert resp["region_status"] == "not_applicable"
    assert "west" in resp.get("region_note", "")
