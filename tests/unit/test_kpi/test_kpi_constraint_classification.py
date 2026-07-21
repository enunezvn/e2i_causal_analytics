"""Machine-readable constraint classification on KPI definitions (constraint-aware
insights plan, 2026-07-20) + the WS1-DQ-006 rename.

Three optional per-KPI fields become first-class KPIMetadata attributes so the
insight-narrative builder cites a deterministic classification instead of the LM
re-deciding actionability per generation:

  * actionability: reader_actionable | structurally_constrained | mixed
    (+ actionability_owner, levers)
  * data_plane: claims | crm | platform | mixed (explicit per-KPI in yaml,
    workstream default in code)
  * measurement_caveat: the complete caveat inventory (ownership table in the
    plan; recall's redefinition-transition caveat was authored by the WS2 plan)

WS1-DQ-006 is renamed to its DESCRIPTIVE name — "Geographic Consistency Gap"
(it is a max regional GAP; lower is better) — with a direction field the grid
and build_grounding consume. The anti-descriptive name was the root cause of
both LM and human misreads (10.2% read as "very low consistency").
"""

import pytest

from src.kpi.registry import KPIRegistry

_VALID_ACTIONABILITY = {"reader_actionable", "structurally_constrained", "mixed"}
_VALID_PLANES = {"claims", "crm", "platform", "mixed"}


@pytest.fixture(scope="module")
def registry() -> KPIRegistry:
    return KPIRegistry()  # singleton; loads definitions on first construction


def _kpi(registry, kpi_id):
    kpi = registry.get(kpi_id)
    assert kpi is not None, f"{kpi_id} missing from registry"
    return kpi


def test_ws1_dq_006_renamed_to_gap_with_direction(registry):
    kpi = _kpi(registry, "WS1-DQ-006")
    assert kpi.name == "Geographic Consistency Gap"
    assert kpi.direction == "lower_is_better"
    assert kpi.measurement_caveat and "lower is better" in kpi.measurement_caveat.lower()


@pytest.mark.parametrize(
    "kpi_id,expected_plane",
    [
        ("WS2-TR-001", "claims"),  # precision: outcome from claims Rx
        ("WS2-TR-002", "claims"),  # recall: new starts from claims Rx
        ("WS2-TR-005", "claims"),  # false alert: outcome-failure derived
        ("WS2-TR-007", "claims"),  # lead time: trigger->outcome interval
        ("WS2-TR-003", "crm"),  # action taken: rep behavior in CRM
        ("WS2-TR-004", "crm"),  # acceptance: rep disposition in CRM
        ("WS2-TR-006", "crm"),  # override: rep disposition in CRM
        ("WS2-TR-008", "platform"),  # CFR: trigger-engine change tracking
        ("WS3-BI-005", "claims"),  # TRx
        ("WS3-BI-006", "claims"),  # NRx
        ("WS3-BI-007", "claims"),  # NBRx
        ("WS3-BI-008", "claims"),  # TRx share
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_data_plane_classification(registry, kpi_id, expected_plane):
    assert _kpi(registry, kpi_id).data_plane == expected_plane


def test_ws1_kpis_default_to_platform_plane(registry):
    """No explicit data_plane in yaml -> workstream default (WS1 = platform:
    model metrics and DQ checks are computed on-platform, current as shown)."""
    assert _kpi(registry, "WS1-MP-001").data_plane == "platform"
    assert _kpi(registry, "WS1-DQ-001").data_plane == "platform"


@pytest.mark.parametrize(
    "kpi_id",
    ["WS2-TR-001", "WS2-TR-002", "WS2-TR-005", "WS2-TR-007", "WS1-DQ-006"],
)
def test_caveat_inventory_is_authored(registry, kpi_id):
    kpi = _kpi(registry, kpi_id)
    assert kpi.measurement_caveat and len(kpi.measurement_caveat) > 30, (
        f"{kpi_id} missing its measurement_caveat (ownership table row)"
    )


def test_ws3_commercial_kpis_are_reader_actionable_with_commercial_levers(registry):
    """The v1-review lesson: WS3 volume/share KPIs are genuinely actionable
    commercial signals — they must carry reader_actionable classification with
    commercial levers, never be caveated into structural hedging."""
    for kpi_id in ("WS3-BI-005", "WS3-BI-008"):
        kpi = _kpi(registry, kpi_id)
        assert kpi.actionability == "reader_actionable"
        assert kpi.actionability_owner == "brand_team"
        assert kpi.levers, f"{kpi_id} must name commercial levers"
        assert kpi.measurement_caveat, f"{kpi_id} carries the claims-maturity trend-vs-level caveat"


def test_dq006_is_structurally_constrained_owned_by_data_strategy(registry):
    kpi = _kpi(registry, "WS1-DQ-006")
    assert kpi.actionability == "structurally_constrained"
    assert kpi.actionability_owner == "data_strategy"


def test_all_classifications_use_valid_enums(registry):
    for kpi in registry.get_all():
        if kpi.actionability is not None:
            assert kpi.actionability in _VALID_ACTIONABILITY, kpi.id
        if kpi.data_plane is not None:
            assert kpi.data_plane in _VALID_PLANES, kpi.id
        if kpi.direction is not None:
            assert kpi.direction in {"lower_is_better", "higher_is_better"}, kpi.id
