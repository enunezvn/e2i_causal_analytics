"""The canonical and patient-panel Rx-volume families (canonical TRx lane)."""

import subprocess
import sys


def _vf():
    import src.kpi.volume_family as vf

    return vf


def test_the_map_is_a_bijection_between_disjoint_families():
    vf = _vf()
    assert dict(vf.CANONICAL_TO_PANEL) == {
        "WS3-BI-005": "WS3-BI-011",
        "WS3-BI-006": "WS3-BI-012",
        "WS3-BI-007": "WS3-BI-013",
        "WS3-BI-008": "WS3-BI-014",
    }
    assert {v: k for k, v in vf.CANONICAL_TO_PANEL.items()} == dict(vf.PANEL_TO_CANONICAL)
    assert not vf.CANONICAL_VOLUME_KPI_IDS & vf.PANEL_VOLUME_KPI_IDS
    assert vf.PANEL_RX_COUNT_KPI_IDS == {"WS3-BI-011", "WS3-BI-012", "WS3-BI-013"}


def test_patient_axes_are_the_calculator_context_keys():
    assert _vf().PATIENT_AXES == ("segment", "therapy_line", "biologic", "ige_tier")


def test_the_scale_note_is_exactly_the_measured_sentence():
    # An independent copy, written out here rather than built from the module, so
    # the sentence cannot drift factually and stay green (substring checks let
    # 597 match 1597). The 597 window is the data frontier's
    # (event_date >= max(event_date) - 30, inclusive = 2026-08-14..2026-09-14), not
    # a trailing 30 days from the run day, which reads 572
    # (docs/demos/results/2026-09-15_trx_canonical/scale_note_597_provenance.txt).
    expected = (
        "measured 2026-09-15, Kisqali canonical TRx for 2026-08 was 800,349 "
        "(business_metrics, brand x region x calendar month) against 597 patient-panel "
        "prescription events from 2026-08-14 through 2026-09-14 inclusive "
        "(treatment_events), about 1,300x"
    )
    assert _vf().MEASURED_SCALE_NOTE == expected


def test_one_null_dimension_rule_in_sql_and_in_python():
    """codex r2 HIGH: the headline SQL, the series SQL and the history handler share it."""
    vf = _vf()
    assert vf.DIMENSIONED_ROW_SQL == "brand IS NOT NULL AND region IS NOT NULL"
    assert vf.has_dimensions({"brand": "Kisqali", "region": "west"})
    for row in (
        {"brand": None, "region": "west"},
        {"brand": "Kisqali", "region": None},
        {"brand": "Kisqali"},
        {"brand": "", "region": "west"},
    ):
        assert not vf.has_dimensions(row), row


def test_importing_it_is_cheap():
    code = (
        "import sys; import src.kpi.volume_family; "
        "print([n for n in sorted(sys.modules) if n.startswith('src.services') "
        "or n in ('aiohttp','dspy','langgraph','pandas')])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout


import pytest  # noqa: E402

from src.kpi.volume_family import CANONICAL_TO_PANEL  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    from src.kpi.registry import get_registry

    return get_registry()


def test_every_family_id_is_registered(registry):
    for canonical, panel in CANONICAL_TO_PANEL.items():
        assert registry.get(canonical) is not None, canonical
        assert registry.get(panel) is not None, panel


@pytest.mark.parametrize("kpi_id", sorted(CANONICAL_TO_PANEL))
def test_canonical_volume_reads_business_metrics_monthly(registry, kpi_id):
    kpi = registry.get(kpi_id)
    assert kpi.tables == ["business_metrics"]
    assert kpi.windowable == "clean" and kpi.window == {"column": "metric_date"}
    assert kpi.frequency == "monthly"
    assert "Panel" not in kpi.name
    caveat = (kpi.measurement_caveat or "").lower()
    assert CANONICAL_TO_PANEL[kpi_id].lower() in caveat
    assert "in-progress month" in caveat and "seasonal" in caveat


@pytest.mark.parametrize("kpi_id", sorted(CANONICAL_TO_PANEL.values()))
def test_panel_reads_the_event_ledger(registry, kpi_id):
    kpi = registry.get(kpi_id)
    assert kpi.tables == ["treatment_events"]
    assert kpi.windowable == "clean" and kpi.window == {"column": "event_date"}
    assert kpi.name.startswith("Observed Rx Events - Patient Panel ")
    assert kpi.name.endswith(" Panel)")
    assert "different" in (kpi.measurement_caveat or "").lower()


def test_canonical_share_has_no_event_era_threshold(registry):
    assert registry.get("WS3-BI-008").threshold is None
    panel = registry.get("WS3-BI-014").threshold
    assert panel is not None and panel.target == 0.30


def test_canonical_names_are_unchanged(registry):
    assert registry.get("WS3-BI-005").name == "Total Prescriptions (TRx)"
    assert registry.get("WS3-BI-006").name == "New Prescriptions (NRx)"
    assert registry.get("WS3-BI-007").name == "New-to-Brand Prescriptions (NBRx)"
    assert registry.get("WS3-BI-008").name == "TRx Share"
