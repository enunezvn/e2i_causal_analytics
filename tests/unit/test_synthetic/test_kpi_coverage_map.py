"""Shard 09 Task 7: the coverage map must list EVERY calculable KPI in
config/kpi_definitions.yaml, each with a MAPPED / N/A verdict. Hermetic (files only).
Also asserts the probe script defines a path (registry or direct) for every KPI.

Count-agnostic by design: the assertions bind to whatever the YAML actually defines,
so a future add/remove keeps these honest without editing the test. History: the
calculable set was 46 → 45 (#1068 removed WS1-MP-008 "Fairness Gap") → 44 (T8 removed
WS1-DQ-008 "Label Quality (IAA)"). Both decommissioned KPIs are omitted from the
calculable-coverage map (their DB objects are retained)."""

import pathlib

import yaml


def _load_cfg() -> dict:
    return yaml.safe_load(open("config/kpi_definitions.yaml"))


def _kpi_ids(cfg: dict | None = None) -> list[str]:
    cfg = cfg or _load_cfg()
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


def test_coverage_map_covers_all_kpis():
    cfg = _load_cfg()
    ids = _kpi_ids(cfg)
    # Drift-proof: the enumerated entries equal the documented summary total.
    assert len(ids) == cfg["summary"]["total_kpis"]
    txt = pathlib.Path("docs/data/kpi_coverage_map_synthetic.md").read_text()
    for kid in ids:
        assert kid in txt, f"{kid} missing from coverage map"
        row = next(line for line in txt.splitlines() if line.startswith(f"| {kid} "))
        assert ("MAPPED" in row) or ("N/A:" in row), f"{kid} has no MAPPED/N/A verdict"


def test_probe_script_defines_path_for_all_kpis():
    from scripts.check_kpi_coverage import DIRECT_PROBES, PROBES

    ids = _kpi_ids()
    covered = set(PROBES) | set(DIRECT_PROBES)
    missing = [k for k in ids if k not in covered]
    assert not missing, f"probe map missing {missing}"
    # The probe map covers EXACTLY the live KPI set — no orphan probe for a removed KPI.
    assert covered == set(ids), f"orphan probes not in the registry: {covered - set(ids)}"
