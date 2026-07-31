"""#1387 --only-tables load allowlist (the triggers-only backfill entrypoint).

The view-stage backfill regenerates the FULL dataset graph (cross-table draws
must stay coherent) but upserts only ``triggers``. These tests pin the filter
helper's contract (identity when unset, fail-loud on typos, exact allowlist
otherwise) and the load-path frame invariants the backfill depends on:
``view_timestamp`` present on the generated triggers frame and the central
``is_synthetic=True`` stamp intact (the default-false provenance trap, #1389).
"""

import importlib

import pandas as pd
import pytest

from src.ml.synthetic.config import DGPType

load_mod = importlib.import_module("scripts.load_synthetic_data")

_SMALL_SIZES = {
    "hcp": 50,
    "patient": 200,
    "treatment": 200,
    "prediction": 50,
    "trigger": 400,
    "business_metrics": 30,
    "feature_values": 50,
}


def _frames():
    return {
        "triggers": pd.DataFrame({"trigger_id": ["trg_00000"]}),
        "patient_journeys": pd.DataFrame({"patient_id": ["pt_000000"]}),
    }


def test_no_allowlist_is_identity():
    datasets = _frames()
    assert load_mod.filter_datasets_to_tables(datasets, None) is datasets
    assert load_mod.filter_datasets_to_tables(datasets, "") is datasets


def test_allowlist_keeps_only_named_tables():
    kept = load_mod.filter_datasets_to_tables(_frames(), "triggers")
    assert list(kept) == ["triggers"]


def test_unknown_table_fails_loud():
    # a typo must not silently no-op the backfill
    with pytest.raises(ValueError, match="triggerz"):
        load_mod.filter_datasets_to_tables(_frames(), "triggerz")


def test_cli_exposes_only_tables_flag():
    src = open(load_mod.__file__).read()
    assert '"--only-tables"' in src


@pytest.mark.parametrize("mode_flag", ["--refresh-ab", "--append-frontier"])
def test_only_tables_rejected_in_whole_load_modes(mode_flag, monkeypatch):
    """codex iter-1 MED: --refresh-ab purges ALL synthetic AB rows before
    reloading and --append-frontier appends one coherent weekly cohort across
    tables — a filtered subset of either is incoherent (stale AB children
    survive a skipped purge / cross-table cohort references strand). The CLI
    must fail loud BEFORE any generation or purge."""
    monkeypatch.setattr(
        "sys.argv",
        ["load_synthetic_data.py", mode_flag, "--only-tables", "triggers"],
    )
    with pytest.raises(SystemExit) as excinfo:
        load_mod.main()
    assert excinfo.value.code == 2  # argparse parser.error


def test_generated_triggers_frame_carries_view_stage_and_provenance():
    datasets = load_mod.generate_datasets(sizes=_SMALL_SIZES, dgp_type=DGPType.CONFOUNDED, seed=11)
    triggers = datasets["triggers"]
    assert "view_timestamp" in triggers.columns
    viewed = triggers["delivery_status"] == "viewed"
    assert triggers.loc[viewed, "view_timestamp"].notna().all()
    accepted = triggers["acceptance_status"] == "accepted"
    assert (triggers.loc[accepted, "delivery_status"] == "viewed").all()
    # central provenance stamp (generate_datasets) survives the view stage —
    # a reseed must never flip is_synthetic back to default false (#1389 trap)
    assert "is_synthetic" in triggers.columns
    assert triggers["is_synthetic"].all()
