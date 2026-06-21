"""Shard 09 Task 7: the coverage map must list ALL 45 KPIs in
config/kpi_definitions.yaml, each with a MAPPED / N/A verdict. Hermetic (files only).
Also asserts the probe script defines a path (registry or direct) for all 45.

(45, not 46: WS1-MP-008 "Fairness Gap (ΔRecall)" was removed in #1068 — it needs
protected-group fairness_metrics that the synthetic substrate does not populate.)"""

import pathlib

import yaml


def _kpi_ids() -> list[str]:
    cfg = yaml.safe_load(open("config/kpi_definitions.yaml"))
    ids: list[str] = []
    for section in (
        "ws1_data_quality",
        "ws1_model_performance",
        "ws2_triggers",
        "ws3_business",
        "brand_specific",
        "causal_metrics",
    ):
        ids += [v["id"] for v in cfg[section].values()]
    return ids


def test_coverage_map_covers_all_45_kpis():
    ids = _kpi_ids()
    assert len(ids) == 45
    txt = pathlib.Path("docs/data/kpi_coverage_map_synthetic.md").read_text()
    for kid in ids:
        assert kid in txt, f"{kid} missing from coverage map"
        row = next(line for line in txt.splitlines() if line.startswith(f"| {kid} "))
        assert ("MAPPED" in row) or ("N/A:" in row), f"{kid} has no MAPPED/N/A verdict"


def test_probe_script_defines_path_for_all_45():
    from scripts.check_kpi_coverage import DIRECT_PROBES, PROBES

    ids = _kpi_ids()
    covered = set(PROBES) | set(DIRECT_PROBES)
    missing = [k for k in ids if k not in covered]
    assert not missing, f"probe map missing {missing}"
    assert len(covered) == 45
