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
#: Pinned rather than open-ended so PENDING cannot become a parking lot: a new
#: pending row fails this test until someone edits this set deliberately.
#:
#: ⚠ NOTHING HERMETIC CAN NOTICE THAT A MIGRATION HAS BEEN APPLIED — and that is not
#: hypothetical. This set held WS3-BI-005..008 for exactly one hour: migration 143
#: was applied on 2026-09-17 and the four rows were stale the moment it landed, with
#: every test still green. The only thing that notices is
#: `E2I_DB_INTEGRATION=1 scripts/check_kpi_coverage.py`, which exits 1 while any row
#: is EMPTY — and NO CI JOB RUNS IT, so the refresh is a manual step.
PENDING_KPI_IDS: set[str] = set()

#: KPIs whose statement is deployed and runs, but whose substrate has no rows to
#: return. Pinned for the same reason as PENDING and with the same limitation: the
#: pin is the guard, because a spreading EMPTY is a substrate regression and must not
#: land silently. WS3-BI-007 canonical NBRx is the only one — `business_metrics`
#: holds no `nbrx` rows at any date (measured 2026-09-17), so it serves NULL until
#: the reseed populates it.
EMPTY_KPI_IDS = {"WS3-BI-007"}


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
    empty: set[str] = set()
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
            verdict == "MAPPED"
            or verdict.startswith("N/A:")
            or verdict.startswith("PENDING:")
            or verdict.startswith("EMPTY:")
        ), f"{kid} has no MAPPED/N/A/PENDING/EMPTY verdict, got {verdict!r}"
        if verdict.startswith("PENDING:"):
            # A pending verdict must say what it is waiting for, or it is just an
            # unexplained blank that reads like a measurement.
            assert re.search(r"migration \d+", verdict), (
                f"{kid} is PENDING without naming the migration it waits on: {verdict!r}"
            )
            pending.add(kid)
        if verdict.startswith("EMPTY:"):
            # Same rule, different question: PENDING must name what it waits on,
            # EMPTY must say what is missing. A bare "EMPTY:" reads as a measured
            # zero, and a measured zero is a very different claim from "the
            # substrate has no rows to measure".
            assert len(verdict[len("EMPTY:") :].strip()) >= 20, (
                f"{kid} is EMPTY without saying what is missing: {verdict!r}"
            )
            empty.add(kid)
    for found, pinned, label in (
        (pending, PENDING_KPI_IDS, "pending"),
        (empty, EMPTY_KPI_IDS, "empty"),
    ):
        assert found == pinned, (
            f"{label} set drifted: {sorted(found)} vs pinned {sorted(pinned)}. "
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
