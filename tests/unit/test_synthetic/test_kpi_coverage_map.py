"""Shard 09 Task 7: the coverage map must list EVERY calculable KPI in
config/kpi_definitions.yaml, each with a MAPPED / N/A verdict. Hermetic (files only).
Also asserts the probe script defines a path (registry or direct) for every KPI.

Count-agnostic by design: the assertions bind to whatever the YAML actually defines,
so a future add/remove keeps these honest without editing the test. History: the
calculable set was 46 → 45 (#1068 removed WS1-MP-008 "Fairness Gap") → 44 (T8 removed
WS1-DQ-008 "Label Quality (IAA)"). Both decommissioned KPIs are omitted from the
calculable-coverage map (their DB objects are retained)."""

import pathlib
import re

import yaml

#: KPIs whose substrate statement is SHIPPED but not yet APPLIED, so no verdict can
#: honestly be measured yet. PENDING exists because the two verdicts this map had are
#: both false for them: MAPPED would carry over a figure from a substrate they no
#: longer read (WS3-BI-005..008 moved to the canonical `business_metrics` series in
#: migration 143), and "N/A" means not calculable, which they are not — they are
#: calculable the moment 143 is applied.
#:
#: Pinned rather than open-ended so PENDING cannot become a parking lot: a fifth
#: pending row fails this test until someone edits this set deliberately.
#:
#: ⚠ NOTHING HERMETIC CAN NOTICE THAT 143 HAS BEEN APPLIED. This set going stale is
#: caught by `E2I_DB_INTEGRATION=1 scripts/check_kpi_coverage.py`, which exits 1 while
#: any row is EMPTY — and which no CI job runs, so the refresh is a manual step.
PENDING_KPI_IDS = {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007", "WS3-BI-008"}


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
    pending: set[str] = set()
    for kid in ids:
        assert kid in txt, f"{kid} missing from coverage map"
        row = next(line for line in txt.splitlines() if line.startswith(f"| {kid} "))
        # The VERDICT COLUMN, not a substring of the whole row: every row is
        # `| id | name | substrate | verdict | measured |`, and a substring test
        # passes on a row that says MAPPED anywhere — including one whose verdict
        # is something else and whose notes mention a mapped sibling.
        cells = row.split("|")
        assert len(cells) == 7, f"{kid} row is not the 5-column table shape: {row}"
        verdict = cells[4].strip()
        assert (
            verdict == "MAPPED" or verdict.startswith("N/A:") or verdict.startswith("PENDING:")
        ), f"{kid} has no MAPPED/N/A/PENDING verdict, got {verdict!r}"
        if verdict.startswith("PENDING:"):
            # A pending verdict must say what it is waiting for, or it is just an
            # unexplained blank that reads like a measurement.
            assert re.search(r"migration \d+", verdict), (
                f"{kid} is PENDING without naming the migration it waits on: {verdict!r}"
            )
            pending.add(kid)
    assert pending == PENDING_KPI_IDS, (
        f"pending set drifted: {sorted(pending)} vs pinned {sorted(PENDING_KPI_IDS)}. "
        "Re-probe with E2I_DB_INTEGRATION=1 scripts/check_kpi_coverage.py and record "
        "the measured verdict, or update the pin deliberately."
    )


def test_probe_script_defines_path_for_all_kpis():
    from scripts.check_kpi_coverage import DIRECT_PROBES, PROBES

    ids = _kpi_ids()
    covered = set(PROBES) | set(DIRECT_PROBES)
    missing = [k for k in ids if k not in covered]
    assert not missing, f"probe map missing {missing}"
    # The probe map covers EXACTLY the live KPI set — no orphan probe for a removed KPI.
    assert covered == set(ids), f"orphan probes not in the registry: {covered - set(ids)}"
