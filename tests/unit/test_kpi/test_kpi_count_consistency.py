"""Regression guard for KPI registry count consistency (issue #1072).

Background
----------
PR #1068 removed WS1-MP-008 ("Fairness Gap (ΔRecall)") from the registry and the
gold-standard scorer — it needs protected-group ``fairness_metrics`` the synthetic
substrate does not populate. The framework therefore defines **45** calculable
KPIs, not 46. #1068 left a tail of stale "46 KPIs" prose across unrelated
subsystems (and the source-of-truth config summary block) which this test locks
down so the documented count can never silently drift from the live registry
again.

These assertions are deliberately *count-agnostic*: they bind the documented
summary to whatever the registry actually loads, so adding/removing a future KPI
keeps the doc honest without editing this test.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.kpi.registry import get_registry

REPO_ROOT = Path(__file__).resolve().parents[3]
KPI_YAML = REPO_ROOT / "config" / "kpi_definitions.yaml"

# Workstream sections that hold KPI entries in the YAML.
WORKSTREAM_SECTIONS = [
    "ws1_data_quality",
    "ws1_model_performance",
    "ws2_triggers",
    "ws3_business",
    "brand_specific",
    "causal_metrics",
]


def _load_yaml() -> dict:
    return yaml.safe_load(KPI_YAML.read_text())


def _section_entry_count(data: dict, section: str) -> int:
    return sum(1 for v in (data.get(section) or {}).values() if isinstance(v, dict) and "id" in v)


def _total_entry_count(data: dict) -> int:
    return sum(_section_entry_count(data, s) for s in WORKSTREAM_SECTIONS)


def test_registry_loads_actual_yaml_entry_count() -> None:
    """The live registry size must equal the number of YAML entries."""
    data = _load_yaml()
    registry = get_registry()
    assert len(registry.get_all()) == _total_entry_count(data)


def test_yaml_summary_total_matches_registry() -> None:
    """summary.total_kpis must equal the live registry size (currently 45)."""
    data = _load_yaml()
    registry = get_registry()
    n = len(registry.get_all())
    assert data["summary"]["total_kpis"] == n, (
        f"summary.total_kpis={data['summary']['total_kpis']} but registry loads {n}"
    )


def test_yaml_summary_subcounts_are_internally_consistent() -> None:
    """by_workstream and direct/derived splits must sum to total_kpis."""
    summary = _load_yaml()["summary"]
    total = summary["total_kpis"]
    assert sum(summary["by_workstream"].values()) == total
    assert summary["direct_calculable"] + summary["derived_calculable"] == total


def test_yaml_direct_derived_match_actual_calculation_types() -> None:
    """The direct/derived split must match the actual per-entry calculation_type
    tallies — not merely sum to the total. Guards against a stale split (e.g.
    28/17) that happens to add up but misrepresents the entries."""
    data = _load_yaml()
    summary = data["summary"]
    tallies: dict[str, int] = {}
    for section in WORKSTREAM_SECTIONS:
        for v in (data.get(section) or {}).values():
            if isinstance(v, dict) and "id" in v:
                ct = v.get("calculation_type", "")
                tallies[ct] = tallies.get(ct, 0) + 1
    assert summary["direct_calculable"] == tallies.get("direct", 0), (
        f"summary.direct_calculable={summary['direct_calculable']} but actual "
        f"direct entries={tallies.get('direct', 0)}"
    )
    assert summary["derived_calculable"] == tallies.get("derived", 0), (
        f"summary.derived_calculable={summary['derived_calculable']} but actual "
        f"derived entries={tallies.get('derived', 0)}"
    )


def test_yaml_by_workstream_matches_actual_sections() -> None:
    """Each by_workstream count must match its actual entry count."""
    data = _load_yaml()
    summary = data["summary"]
    for section in WORKSTREAM_SECTIONS:
        actual = _section_entry_count(data, section)
        assert summary["by_workstream"][section] == actual, (
            f"{section}: summary says {summary['by_workstream'][section]}, actual entries {actual}"
        )


# Files that historically hardcoded the registry size as "46 KPIs". The fix makes
# them count-agnostic; this guard ensures the stale number cannot reappear.
_FORBIDDEN = re.compile(r"\b46\b\s*\+?\s*(?:KPIs?|calculable|defined)", re.IGNORECASE)
_SCANNED_FILES = [
    "config/kpi_definitions.yaml",
    "src/api/routes/chatbot_tools.py",
    "src/services/kpi_resolution.py",
    "src/kpi/__init__.py",
    "src/repositories/sample_data.py",
    "tests/unit/test_services/test_kpi_resolution.py",
]


def test_no_stale_hardcoded_kpi_count_references() -> None:
    offenders: list[str] = []
    for rel in _SCANNED_FILES:
        text = (REPO_ROOT / rel).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if _FORBIDDEN.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "Stale hardcoded KPI count references:\n" + "\n".join(offenders)
