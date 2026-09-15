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


def test_the_scale_note_carries_both_measured_numbers():
    note = _vf().MEASURED_SCALE_NOTE
    assert "800,349" in note and "597" in note and "2026-09-15" in note


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
