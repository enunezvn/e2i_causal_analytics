"""Lane B (real-data causal estimation, 2026-09-22) — ``scripts/author_cohort_dag.py``
with a fake LM and the dead-Supabase pin (spec §6: "CLI with a fake LM and the
dead-Supabase pin"). A dry run writes the three artefacts and the manifest diff;
``--review`` refuses a fake DAG unless explicitly allowed, and with the unit
tree's dead endpoint (#1420) it must fail loudly (exit 3, files kept, no
``review.json``) instead of pretending a review exists.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "author_cohort_dag.py"

OPTUM_T, OPTUM_Y = "biologic_initiation", "initiated_biologic_180d"


def _load():
    spec = importlib.util.spec_from_file_location("author_cohort_dag", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(240)
def test_fake_dry_run_on_optum_writes_artefacts_and_diffs_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-be-blanked")
    mod = _load()
    rc = mod.main(
        [
            "--manifest",
            "optum",
            "--treatment",
            OPTUM_T,
            "--outcome",
            OPTUM_Y,
            "--features",
            "age_at_index,zip3",
            "--lm",
            "fake",
            "--no-assumption",
            "--diff-manifest-attestations",
            "--out-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert "OPENAI_API_KEY" not in os.environ
    out = tmp_path / f"optum_{OPTUM_T}_{OPTUM_Y}"
    assert {p.name for p in out.iterdir()} == {
        "attestations.json",
        "dag.json",
        "review.md",
        "manifest_diff.json",
    }

    att = json.loads((out / "attestations.json").read_text())
    assert att["meta"]["lm"] == "fake" and att["meta"]["provenance"] == "machine"
    assert [r["feature_name"] for r in att["records"]] == ["age_at_index", "zip3"]
    assert all(r["model_id"] == "dummy" and r["provenance"] == "machine" for r in att["records"])
    # The brief is the Layer-4 input pair with the causal treatment appended.
    dag = json.loads((out / "dag.json").read_text())
    snap = dag["dag_structure_json"]
    assert snap["treatment_nodes"] == [OPTUM_T] and snap["outcome_nodes"] == [OPTUM_Y]
    assert ["age_at_index", OPTUM_T] in snap["edges"] and [OPTUM_T, OPTUM_Y] in snap["edges"]
    assert dag["adjustment_valid"] is True
    assert dag["minimal_adjustment_set"] == ["age_at_index", "zip3"]

    diff = json.loads((out / "manifest_diff.json").read_text())
    rows = {r["feature"]: r for r in diff["rows"]}
    # age_at_index is a confounder in the manifest too → exact agreement.
    assert rows["age_at_index"]["edge_exact"] is True and rows["age_at_index"]["role_agree"] is True
    # zip3 is a manifest INSTRUMENT; the fake author drew a confounder → listed
    # with both rationales, no threshold applied.
    assert rows["zip3"]["role_agree"] is False
    assert (
        rows["zip3"]["manifest_role"] == "instrument"
        and rows["zip3"]["authored_role"] == "confounder"
    )
    assert rows["zip3"]["manifest_only"] == [] and rows["zip3"]["authored_only"] == [
        f"zip3->{OPTUM_Y}"
    ]
    assert rows["zip3"]["manifest_provenance"] == "machine"
    assert "authored_rationale" in rows["zip3"] and "manifest_rationale" in rows["zip3"]
    assert diff["disagreements"] == ["zip3"] and diff["n_compared"] == 2

    review = (out / "review.md").read_text()
    assert "# Cohort DAG review: optum" in review
    assert "## Diff against the manifest's machine attestations" in review
    assert "escalation_decision_point" not in review  # --no-assumption


@pytest.mark.timeout(240)
def test_brief_is_the_layer_4_input_pair_plus_the_treatment_suffix():
    mod = _load()
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_layer_4_inputs,
    )
    from src.data.manifests import lookup_feature_contract

    [brief] = mod.build_briefs(
        manifest="optum_mart",
        treatment="treatment_dupixent",
        outcome="persistent_at_180d_g28",
        treatment_label="dupilumab",
        outcome_label="persistence",
        features=["age_at_index"],
    )
    contract = lookup_feature_contract("age_at_index", data_source="optum_mart")
    derivation, context = _build_layer_4_inputs(
        "age_at_index", contract, "persistent_at_180d_g28", "optum_mart"
    )
    assert brief.derivation_pseudocode == derivation
    assert brief.dataset_context.startswith(context)
    assert brief.dataset_context.endswith(
        "; treatment=treatment_dupixent; causal_question=effect of treatment_dupixent on persistent_at_180d_g28"
    )
    assert brief.panel is None


@pytest.mark.timeout(60)
def test_review_of_a_fake_dag_is_refused_unless_allowed(tmp_path):
    mod = _load()
    rc = mod.main(
        [
            "--manifest",
            "optum",
            "--treatment",
            OPTUM_T,
            "--outcome",
            OPTUM_Y,
            "--features",
            "age_at_index",
            "--lm",
            "fake",
            "--review",
            "--out-root",
            str(tmp_path),
        ]
    )
    assert rc == 4
    assert not (tmp_path / f"optum_{OPTUM_T}_{OPTUM_Y}").exists()


@pytest.mark.timeout(240)
def test_review_under_the_dead_supabase_pin_fails_loudly_and_keeps_files(tmp_path):
    assert os.environ.get("SUPABASE_URL", "").startswith("http://127.0.0.1:1")
    mod = _load()
    rc = mod.main(
        [
            "--manifest",
            "optum",
            "--treatment",
            OPTUM_T,
            "--outcome",
            OPTUM_Y,
            "--features",
            "age_at_index",
            "--lm",
            "fake",
            "--review",
            "--allow-fake-review",
            "--no-assumption",
            "--out-root",
            str(tmp_path),
        ]
    )
    assert rc == 3
    out = tmp_path / f"optum_{OPTUM_T}_{OPTUM_Y}"
    assert (out / "attestations.json").exists() and (out / "dag.json").exists()
    assert not (out / "review.json").exists()


@pytest.mark.timeout(60)
def test_real_lm_refuses_without_cost_acceptance(tmp_path):
    mod = _load()
    rc = mod.main(
        [
            "--manifest",
            "optum_mart",
            "--treatment",
            "treatment_dupixent",
            "--outcome",
            "persistent_at_180d_g28",
            "--lm",
            "real",
            "--out-root",
            str(tmp_path),
        ]
    )
    assert rc == 3
    assert not any(tmp_path.iterdir())


@pytest.mark.timeout(60)
def test_panel_for_another_estimand_is_refused(tmp_path):
    mod = _load()
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {"manifest_source": "optum_mart", "treatment": "other", "outcome": "y", "records": {}}
        )
    )
    with pytest.raises(ValueError, match="treatment='other' does not match"):
        mod._load_panel(panel, manifest="optum_mart", treatment="treatment_dupixent", outcome="y")
