"""Regression guard for KPI registry count consistency (issues #1072, T8).

Background
----------
PR #1068 removed WS1-MP-008 ("Fairness Gap (ΔRecall)") from the registry and the
gold-standard scorer — it needs protected-group ``fairness_metrics`` the synthetic
substrate does not populate. T8 then removed WS1-DQ-008 ("Label Quality (IAA)") by
product decision — a *working* metric (corpus Fleiss κ ≈ 0.76) that the user chose
to drop from the live KPI set. The framework therefore defines **44** calculable
KPIs, not 45 (and not the original 46). Each removal left a tail of stale count
prose ("46 KPIs", "45 KPIs", "45/45 MAPPED") across unrelated subsystems (and the
source-of-truth config summary block) which this test locks down so the documented
count can never silently drift from the live registry again.

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


# --- T8: WS1-DQ-008 (Label Quality / IAA) removed from the live KPI set ---
# Removed by product decision (a WORKING metric, corpus κ≈0.76 — not a data limit).
# DB objects (v_kpi_label_quality, ml_annotations) are intentionally retained; only the
# live registry / calculator / coverage-tooling / FE surfaces are dropped, mirroring the
# WS1-MP-008 decommission (#1068).
_REMOVED_KPI_ID = "WS1-DQ-008"
_REMOVED_KPI_KEY = "label_quality_iaa"


def test_label_quality_iaa_absent_from_yaml() -> None:
    """The ``label_quality_iaa`` entry (WS1-DQ-008) must be gone from the YAML registry."""
    data = _load_yaml()
    dq = data.get("ws1_data_quality") or {}
    assert _REMOVED_KPI_KEY not in dq, f"{_REMOVED_KPI_KEY} still present in ws1_data_quality"
    all_ids = {
        v.get("id")
        for s in WORKSTREAM_SECTIONS
        for v in (data.get(s) or {}).values()
        if isinstance(v, dict)
    }
    assert _REMOVED_KPI_ID not in all_ids, f"{_REMOVED_KPI_ID} is still defined in the YAML"


def test_label_quality_iaa_absent_from_registry() -> None:
    """CI-faithful: the live registry must not load WS1-DQ-008.

    Import-based — the editable ``.venv`` pins ``src`` to the MAIN checkout, so this is
    authoritative in CI / from the main checkout, not from a git worktree.
    """
    registry = get_registry()
    ids = {k.id for k in registry.get_all()}
    assert _REMOVED_KPI_ID not in ids, f"{_REMOVED_KPI_ID} is still loaded by the registry"


def test_label_quality_calculator_method_removed() -> None:
    """CI-faithful: the WS1-DQ-008 code path is gone, not merely unwired."""
    from src.kpi.calculators.data_quality import DataQualityCalculator

    assert not hasattr(DataQualityCalculator, "_calc_label_quality")
    assert not hasattr(DataQualityCalculator, "_generalized_fleiss_kappa")


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


# Surfaces that asserted a stale calculable count. #1068 removed WS1-MP-008 (46→45);
# T8 removed WS1-DQ-008 (45→44). These guards ensure NEITHER stale number ("46" or
# "45") can reappear in any CURRENT-STATE reference (code, config, coverage tooling,
# and the live framework/reference docs). Dated historical records (completed-issue
# plans/reports, design specs) are intentionally NOT scanned — the older counts were
# true when they were written. The framework reference (06-KPI-REFERENCE.md) keeps the
# two decommissioned KPIs documented as DECOMMISSIONED, so it may still say
# "WS1-MP-008"/"WS1-DQ-008"/"9 KPIs" (designed-count section headers), just not a stale
# *calculable*-count of 45 or 46. The patterns require KPI/calculable/defined adjacency
# (or the "N/N MAPPED" / "TOTAL N" coverage-probe forms) so bare numbers — row indices
# like "| 45 | CM-004", thresholds, dates — never false-match.
_FORBIDDEN_PATTERNS = [
    re.compile(r"\b46\b\s*\+?\s*(?:KPIs?|calculable|defined)", re.IGNORECASE),
    re.compile(r"Total\s+KPIs\D{0,6}46\b", re.IGNORECASE),  # "Total KPIs: 46"
    re.compile(r"\b46/46\b"),  # coverage map "46/46 MAPPED"
    re.compile(r"\bTOTAL\s+46\b"),  # coverage probe "TOTAL 46 MAPPED 46"
    # T8: 44 is now the live calculable count; "45" is the new stale tail.
    re.compile(r"\b45\b\s*\+?\s*(?:KPIs?|calculable|defined)", re.IGNORECASE),
    re.compile(r"Total\s+KPIs\D{0,6}45\b", re.IGNORECASE),  # "Total KPIs: 45"
    re.compile(r"\b45/45\b"),  # coverage map "45/45 MAPPED"
    re.compile(r"\bTOTAL\s+45\b"),  # coverage probe "TOTAL 45 MAPPED 45"
    # Header form where the count trails the label: "Calculable KPIs: 45/46".
    re.compile(r"Calculable\s+KPIs\D{0,6}4[56]\b", re.IGNORECASE),
]
_SCANNED_FILES = [
    "config/kpi_definitions.yaml",
    "src/api/routes/chatbot_tools.py",
    "src/services/kpi_resolution.py",
    "src/kpi/__init__.py",
    "src/repositories/sample_data.py",
    "tests/unit/test_services/test_kpi_resolution.py",
    # Coverage tooling (probes the live calculable set; counts must track the registry).
    "scripts/check_kpi_coverage.py",
    "scripts/validate_kpi_coverage.py",
    # Current-state reference docs (issue #1075).
    "README.md",
    "docs/data/00-INDEX.md",
    "docs/data/04-KNOWLEDGE-GRAPH-ONTOLOGY.md",
    "docs/data/06-KPI-REFERENCE.md",
    "docs/data/kpi_coverage_map_synthetic.md",
    "docs/foundry/e2i_lineage.yaml",
    "docs/foundry/e2i_ontology.yaml",
]


def test_no_stale_hardcoded_kpi_count_references() -> None:
    offenders: list[str] = []
    for rel in _SCANNED_FILES:
        text = (REPO_ROOT / rel).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in _FORBIDDEN_PATTERNS):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "Stale hardcoded KPI count references:\n" + "\n".join(offenders)
