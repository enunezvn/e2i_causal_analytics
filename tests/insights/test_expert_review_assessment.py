"""Tests for the advisory expert-review DAG assessment (mig 097).

The module grades the six reviewer-checklist questions from machine evidence
only (DAG structure + refutation rows). The conftest forces the deterministic
fallback path (no live LLM), mirroring CI; the LM parse/vouching layer is
tested by faking ``run_signature`` predictions.
"""

from src.insights.expert_review_assessment import (
    CHECKLIST_QUESTIONS,
    build_grounding,
    compute_structural_facts,
    generate_assessment,
)

# t -> m -> y with confounder c (c->t, c->y), direct t->y, and a
# discovery-augmented x -> y edge.
STRUCTURE = {
    "nodes": ["t", "y", "m", "c", "x"],
    "edges": [["t", "m"], ["m", "y"], ["c", "t"], ["c", "y"], ["t", "y"], ["x", "y"]],
    "treatment_nodes": ["t"],
    "outcome_nodes": ["y"],
    "augmented_edges": [["x", "y"]],
    "discovery_gate_decision": "augment",
    "adjustment_sets": [["c"]],
}

VALIDATIONS = [
    {
        "test_type": "random_common_cause",
        "status": "passed",
        "original_effect": 0.12,
        "refuted_effect": 0.118,
        "delta_percent": 1.7,
    },
    {"test_type": "sensitivity_e_value", "status": "passed", "p_value": 0.03},
    {
        "test_type": "data_subset",
        "status": "failed",
        "original_effect": 0.12,
        "refuted_effect": 0.05,
        "delta_percent": 58.3,
    },
    {"test_type": "placebo_treatment", "status": "passed"},
]

REVIEW_ROW = {
    "review_id": "rev-1",
    "brand": "Kisqali",
    "treatment_variable": "t",
    "outcome_variable": "y",
    "analysis_context": "confidence=0.60, gate=review",
    "dag_structure_json": STRUCTURE,
}


def _items_by_id(out):
    return {item["id"]: item for item in out["items"]}


class TestStructuralFacts:
    def test_mediators_path_and_acyclicity(self):
        facts = compute_structural_facts(STRUCTURE)
        assert facts["has_structure"] is True
        assert facts["is_acyclic"] is True
        assert facts["has_treatment_outcome_path"] is True
        assert facts["mediators"] == ["m"]
        assert facts["outcome_to_treatment_edge"] is False

    def test_detects_outcome_to_treatment_edge(self):
        s = {**STRUCTURE, "edges": STRUCTURE["edges"] + [["y", "t"]]}
        facts = compute_structural_facts(s)
        assert facts["outcome_to_treatment_edge"] is True
        # y -> t plus t -> y is also a cycle.
        assert facts["is_acyclic"] is False

    def test_detects_missing_treatment_outcome_path(self):
        s = {
            "nodes": ["t", "y", "c"],
            "edges": [["c", "t"], ["c", "y"]],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
        }
        facts = compute_structural_facts(s)
        assert facts["has_treatment_outcome_path"] is False
        assert facts["mediators"] == []

    def test_none_structure(self):
        facts = compute_structural_facts(None)
        assert facts["has_structure"] is False


class TestBuildGrounding:
    def test_accepts_dict_or_json_string_structure(self):
        import json

        g_dict = build_grounding(REVIEW_ROW, VALIDATIONS)
        g_str = build_grounding(
            {**REVIEW_ROW, "dag_structure_json": json.dumps(STRUCTURE)}, VALIDATIONS
        )
        assert g_dict["has_dag_structure"] is True
        assert g_str["has_dag_structure"] is True
        assert "t -> m" in g_dict["dag_summary"]

    def test_refutation_evidence_lists_tests(self):
        g = build_grounding(REVIEW_ROW, VALIDATIONS)
        assert "random_common_cause" in g["refutation_evidence"]
        assert "data_subset" in g["refutation_evidence"]
        assert "failed" in g["refutation_evidence"]

    def test_no_structure_no_validations(self):
        g = build_grounding({"review_id": "rev-2"}, [])
        assert g["has_dag_structure"] is False
        assert g["validations_used"] == 0


class TestFallbackAssessment:
    """conftest forces the no-LM path: verdicts must be the deterministic,
    evidence-derived ones — never fabricated."""

    def test_covers_all_six_questions(self):
        out = generate_assessment(build_grounding(REVIEW_ROW, VALIDATIONS))
        assert out["is_fallback"] is True
        assert [i["id"] for i in out["items"]] == [q["id"] for q in CHECKLIST_QUESTIONS]

    def test_verdicts_follow_evidence(self):
        out = generate_assessment(build_grounding(REVIEW_ROW, VALIDATIONS))
        items = _items_by_id(out)
        # Both confounder-sensitivity tests passed.
        assert items["conf_complete"]["verdict"] == "supports"
        # data_subset FAILED -> overlap/stability concern.
        assert items["positivity"]["verdict"] == "concern"
        # Acyclic, no outcome->treatment edge -> structural pass.
        assert items["no_forbidden"]["verdict"] == "supports"
        # Mediator listed for human verification, never auto-approved.
        assert items["mediators_correct"]["verdict"] == "unclear"
        assert "m" in items["mediators_correct"]["rationale"]
        # Discovery-augmented edges -> flagged for domain verification.
        assert items["edge_plausible"]["verdict"] == "unclear"
        # SUTVA is not machine-assessable.
        assert items["sutva_plausible"]["verdict"] == "no_evidence"

    def test_no_evidence_everywhere_when_row_is_bare(self):
        out = generate_assessment(build_grounding({"review_id": "rev-2"}, []))
        assert {i["verdict"] for i in out["items"]} == {"no_evidence"}

    def test_missing_treatment_outcome_path_is_a_concern(self):
        s = {
            "nodes": ["t", "y"],
            "edges": [["y", "t"]],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
        }
        row = {**REVIEW_ROW, "dag_structure_json": s}
        items = _items_by_id(generate_assessment(build_grounding(row, [])))
        assert items["mediators_correct"]["verdict"] == "concern"
        assert items["no_forbidden"]["verdict"] == "concern"

    def test_warning_status_yields_unclear(self):
        vals = [{"test_type": "data_subset", "status": "warning"}]
        items = _items_by_id(generate_assessment(build_grounding(REVIEW_ROW, vals)))
        assert items["positivity"]["verdict"] == "unclear"


class _FakePred:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class TestLmParseAndVouching:
    """When the LM path runs, each answer must parse to a valid verdict and its
    rationale digits must be vouched by the grounding text — otherwise that
    item falls back to the deterministic one."""

    def test_valid_lm_answers_are_kept(self, monkeypatch):
        pred = _FakePred(
            conf_complete="supports — both confounder-sensitivity refuters passed",
            edge_plausible="unclear — the augmented x edge needs domain sign-off",
            no_forbidden="supports — the graph is acyclic",
            mediators_correct="unclear — verify m sits between treatment and outcome",
            sutva_plausible="no_evidence — interference is not machine-assessable",
            positivity="concern — the data_subset refuter failed",
        )
        monkeypatch.setattr(
            "src.insights.expert_review_assessment.run_signature", lambda *a, **k: pred
        )
        out = generate_assessment(build_grounding(REVIEW_ROW, VALIDATIONS))
        assert out["is_fallback"] is False
        items = _items_by_id(out)
        assert items["positivity"]["verdict"] == "concern"
        assert "data_subset" in items["positivity"]["rationale"]

    def test_unvouched_digit_falls_back_per_item(self, monkeypatch):
        pred = _FakePred(
            # 3.2 appears NOWHERE in the grounding -> must be rejected.
            conf_complete="supports — effect moved only 3.2% under the fake confounder",
            edge_plausible="unclear — verify augmented edges",
            no_forbidden="supports — acyclic",
            mediators_correct="unclear — m is a mediator",
            sutva_plausible="no_evidence — requires domain judgment",
            positivity="concern — data_subset failed",
        )
        monkeypatch.setattr(
            "src.insights.expert_review_assessment.run_signature", lambda *a, **k: pred
        )
        g = build_grounding(REVIEW_ROW, VALIDATIONS)
        items = _items_by_id(generate_assessment(g))
        # The tainted item is replaced by its deterministic fallback...
        assert "3.2" not in items["conf_complete"]["rationale"]
        # ...while untainted items keep their LM text verbatim.
        assert items["positivity"]["rationale"] == "data_subset failed"

    def test_invalid_verdict_token_falls_back_per_item(self, monkeypatch):
        pred = _FakePred(
            conf_complete="definitely fine!",  # not a valid verdict token
            edge_plausible="unclear — verify augmented edges",
            no_forbidden="supports — acyclic",
            mediators_correct="unclear — m is a mediator",
            sutva_plausible="no_evidence — requires domain judgment",
            positivity="concern — data_subset failed",
        )
        monkeypatch.setattr(
            "src.insights.expert_review_assessment.run_signature", lambda *a, **k: pred
        )
        items = _items_by_id(generate_assessment(build_grounding(REVIEW_ROW, VALIDATIONS)))
        assert items["conf_complete"]["verdict"] == "supports"  # deterministic result
        assert "refuter" in items["conf_complete"]["rationale"] or "test" in (
            items["conf_complete"]["rationale"].lower()
        )
