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
    # codex r2 HIGH 4: every capture names the tree it ran on.
    assert len(att["meta"]["tree"]["commit"]) == 40
    assert isinstance(att["meta"]["tree"]["dirty_src_scripts_tests"], bool)
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
    # Both rationales, feature-specific (codex r1 MED 6): the author's reasoning
    # and the manifest side's family bullet naming zip3, with its PMIDs and line.
    assert rows["zip3"]["authored_reasoning"].startswith("fake LM (dry run)")
    grounding = rows["zip3"]["manifest_rationale"]
    assert grounding["found"] is True
    assert grounding["source"] == "docs/layer4/optum_initiation_attestation_research.md"
    assert any("`zip3`" in b["text"] and "36481046" in b["pmids"] for b in grounding["bullets"])
    assert all(isinstance(b["line"], int) and b["line"] > 0 for b in grounding["bullets"])
    # A feature from another family gets ITS bullet, not the same text.
    age_grounding = mod.manifest_grounding("age_at_index")
    assert age_grounding["found"] is True
    assert age_grounding["bullets"][0]["text"] != grounding["bullets"][0]["text"]
    assert "`age_at_index`" in age_grounding["bullets"][0]["text"]
    assert mod.manifest_grounding("not_a_feature")["found"] is False
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
def test_real_lm_without_a_panel_is_refused_unless_overridden(tmp_path):
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
            "--i-accept-cost",
            "--out-root",
            str(tmp_path),
        ]
    )
    assert rc == 4
    assert not any(tmp_path.iterdir())


def _panel_payload(records):
    return {
        "manifest_source": "optum",
        "treatment": OPTUM_T,
        "outcome": OPTUM_Y,
        "records": records,
    }


def _record(feature, **over):
    rec = {
        "feature": feature,
        "layer_1": {"verdict": "pre_index"},
        "layer_2": {},
        "layer_3": {"ran": False},
        "layer_4": {"fired": False},
        "ensemble": {"final_role": None, "decided_by": "abstain"},
        "leak_verdict": False,
        "leak_source": None,
        "review_required": False,
    }
    rec.update(over)
    return rec


@pytest.mark.timeout(60)
def test_panel_records_are_validated_strictly(tmp_path):
    """codex r1 MED 4: a malformed or incomplete panel must not silently
    become "no panel" for a feature (that would drop the Lane E constraints)."""
    mod = _load()
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(_panel_payload({"age_at_index": _record("age_at_index", leak_verdict="true")}))
    )
    with pytest.raises(ValueError, match="leak_verdict is not a bool"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    panel.write_text(json.dumps(_panel_payload({"age_at_index": _record("zip3")})))
    with pytest.raises(ValueError, match="carries feature='zip3'"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    panel.write_text(json.dumps(_panel_payload({"age_at_index": "nope"})))
    with pytest.raises(ValueError, match="is not an object"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    # codex r2 HIGH 2: a record that OMITS a safety-critical field is refused,
    # never defaulted (that would drop the veto / leak exclusion / cross-check).
    for missing in (
        "layer_1",
        "layer_3",
        "ensemble",
        "leak_verdict",
        "leak_source",
        "review_required",
    ):
        rec = _record("age_at_index")
        del rec[missing]
        panel.write_text(json.dumps(_panel_payload({"age_at_index": rec})))
        with pytest.raises(ValueError, match=f"lacks '{missing}'"):
            mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    panel.write_text(
        json.dumps(_panel_payload({"age_at_index": _record("age_at_index", layer_1={})}))
    )
    with pytest.raises(ValueError, match="layer_1.verdict=None is not one of"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    panel.write_text(
        json.dumps(_panel_payload({"age_at_index": _record("age_at_index", ensemble={})}))
    )
    with pytest.raises(ValueError, match="ensemble lacks final_role/decided_by"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    panel.write_text(
        json.dumps(_panel_payload({"age_at_index": _record("age_at_index", leak_source="layer_9")}))
    )
    with pytest.raises(ValueError, match="leak_source='layer_9' is not one of"):
        mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    # Valid panel, but it does not cover every requested feature → refused.
    panel.write_text(json.dumps(_panel_payload({"age_at_index": _record("age_at_index")})))
    loaded = mod._load_panel(panel, manifest="optum", treatment=OPTUM_T, outcome=OPTUM_Y)
    with pytest.raises(ValueError, match=r"no record for 1 feature\(s\): \['zip3'\]"):
        mod.build_briefs(
            manifest="optum",
            treatment=OPTUM_T,
            outcome=OPTUM_Y,
            treatment_label="t",
            outcome_label="y",
            features=["age_at_index", "zip3"],
            panel=loaded,
        )


class _CasRepo:
    """An in-memory expert_reviews store that enforces the SAME version-pair
    guard as ``ExpertReviewRepository.update_agent_assessment`` (both halves
    filter the UPDATE; a None adjustment half means IS NULL)."""

    def __init__(self):
        self.rows = {}

    async def create_review(self, **row):
        review_id = "44444444-4444-4444-4444-444444444444"
        self.rows[review_id] = {
            "review_id": review_id,
            "approval_status": "pending",
            "valid_until": None,
            "dag_structure_json": row.get("dag_structure"),
            "agent_assessment_json": None,
            **{k: v for k, v in row.items() if k != "dag_structure"},
        }
        return review_id

    async def update_agent_assessment(
        self, review_id, assessment, *, for_dag_version_hash=None, for_adjustment_set_hash=None
    ):
        row = self.rows.get(review_id)
        if row is None:
            return False
        if for_dag_version_hash is not None:
            if row.get("dag_version_hash") != for_dag_version_hash:
                return False
            if row.get("adjustment_set_hash") != for_adjustment_set_hash:
                return False  # IS NULL semantics for None: a non-null row never matches
        row["agent_assessment_json"] = assessment
        return True


@pytest.mark.timeout(240)
def test_opened_review_seeds_a_prior_once_approved(tmp_path):
    """Producer → loader round trip (codex r1 HIGH 1): the review the CLI mints
    carries a non-null adjustment hash, so the evidence write must pass BOTH
    halves of the version pair or it matches no row; once a human approves the
    row, the loader derives the prior from the stored snapshot."""
    import asyncio

    from src.data.kg.structural_author import author_feature
    from src.data.kg.structural_prior_loader import structural_prior_from_review_row
    from src.ml.causal_role_dgp.assembler import assemble_cohort_dag

    mod = _load()
    from dspy.utils.dummies import DummyLM

    records = []
    for feat, edges in (
        ("age_at_index", [["age_at_index", "T"], ["age_at_index", "Y"], ["T", "Y"]]),
        ("zip3", [["zip3", "T"], ["T", "Y"]]),
    ):
        [brief] = mod.build_briefs(
            manifest="optum",
            treatment=OPTUM_T,
            outcome=OPTUM_Y,
            treatment_label="t",
            outcome_label="y",
            features=[feat],
        )
        lm = DummyLM(
            [
                {
                    "reasoning": "r",
                    "edges": json.dumps(edges),
                    "edge_rationales": "[]",
                    "entity_names": "{}",
                    "expected_role": "confounder",
                    "ambiguous": "false",
                }
            ]
        )
        records.append(author_feature(brief, resolver=mod.OfflineResolver(), lm=lm))
    dag = assemble_cohort_dag(records, treatment=OPTUM_T, outcome=OPTUM_Y)
    repo = _CasRepo()
    result = asyncio.run(
        mod.open_review(
            dag=dag,
            attestations_path=tmp_path / "attestations.json",
            manifest="optum",
            treatment=OPTUM_T,
            outcome=OPTUM_Y,
            treatment_label="t",
            outcome_label="y",
            brand=None,
            model_id="dummy",
            prompt_hash="p" * 64,
            guide_hash="g" * 64,
            assumption=None,
            repo=repo,
        )
    )
    assert result["assessment_persisted"] is True
    row = repo.rows[result["review_id"]]
    assert row["review_type"] == "initial_dag"
    assert row["dag_version_hash"] == result["dag_version_hash"]
    assert row["adjustment_set_hash"] == result["adjustment_set_hash"]
    assert "structural_author" in row["agent_assessment_json"]
    # Pending: never a prior. Approved: the prior comes from the stored snapshot.
    assert structural_prior_from_review_row(row) is None
    row["approval_status"] = "approved"
    prior = structural_prior_from_review_row(row)
    assert prior is not None
    assert prior.anchored_confounders == ["age_at_index"]
    assert prior.instruments == ["zip3"]
    assert prior.roles == {"age_at_index": "confounder", "zip3": "instrument"}


@pytest.mark.timeout(240)
def test_open_review_fails_loudly_when_the_evidence_write_matches_no_row(tmp_path):
    import asyncio

    from src.data.kg.structural_author import author_feature
    from src.ml.causal_role_dgp.assembler import assemble_cohort_dag

    mod = _load()
    from dspy.utils.dummies import DummyLM

    [brief] = mod.build_briefs(
        manifest="optum",
        treatment=OPTUM_T,
        outcome=OPTUM_Y,
        treatment_label="t",
        outcome_label="y",
        features=["age_at_index"],
    )
    lm = DummyLM(
        [
            {
                "reasoning": "r",
                "edges": json.dumps([["age_at_index", "T"], ["age_at_index", "Y"], ["T", "Y"]]),
                "edge_rationales": "[]",
                "entity_names": "{}",
                "expected_role": "confounder",
                "ambiguous": "false",
            }
        ]
    )
    dag = assemble_cohort_dag(
        [author_feature(brief, resolver=mod.OfflineResolver(), lm=lm)],
        treatment=OPTUM_T,
        outcome=OPTUM_Y,
    )

    class _MovedRepo(_CasRepo):
        async def create_review(self, **row):
            rid = await super().create_review(**row)
            self.rows[rid]["adjustment_set_hash"] = "moved"  # a concurrent advance
            return rid

    with pytest.raises(RuntimeError, match="NOT persisted"):
        asyncio.run(
            mod.open_review(
                dag=dag,
                attestations_path=tmp_path / "a.json",
                manifest="optum",
                treatment=OPTUM_T,
                outcome=OPTUM_Y,
                treatment_label="t",
                outcome_label="y",
                brand=None,
                model_id="dummy",
                prompt_hash="p" * 64,
                guide_hash="g" * 64,
                assumption=None,
                repo=_MovedRepo(),
            )
        )


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
