"""Lane B (real-data causal estimation, 2026-09-22) — structural author.

Spec §6 (Lane B line): "parser and grader on fixed model outputs". Every LM
here is a ``dspy.utils.dummies.DummyLM`` with a scripted answer; the resolver
is a stub with canned abstracts. Nothing paid, nothing networked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.kg.structural_author import (
    GUIDE_FIRST_HEADING,
    GUIDE_HASH,
    GUIDE_PATH,
    GUIDE_SECTIONS_0_TO_6,
    AuthoredAttestation,
    PanelRecordView,
    StructuralAuthorError,
    author_feature,
    build_brief,
    grade_edges,
    parse_author_output,
    parse_citation,
    prompt_hash,
)
from src.data.kg.types import CitationVerdict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Fixtures: a stub resolver with canned abstracts, briefs, scripted LM answers
# ---------------------------------------------------------------------------


class _StubResolver:
    """``verify_citation`` with the production scoring shape on canned text."""

    def __init__(self, abstracts: dict[str, str], *, raise_on: set[str] | None = None):
        self.abstracts = abstracts
        self.raise_on = raise_on or set()
        self.calls: list[tuple[str, str, str, str]] = []

    def verify_citation(self, identifier, *, identifier_kind, subject_name, object_name):
        self.calls.append((identifier, identifier_kind, subject_name, object_name))
        if identifier in self.raise_on:
            raise RuntimeError("europe pmc down")
        text = self.abstracts.get(identifier)
        if text is None:
            return CitationVerdict(
                identifier=identifier, identifier_kind=identifier_kind, abstract_resolved=False
            )
        low = text.lower()
        found = tuple(n for n in (subject_name, object_name) if n.lower() in low)
        cue = "causes" if "causes" in low else None
        conf = 0.0
        if len(found) == 2:
            conf = 0.5 + (0.5 if cue else 0.0)
        return CitationVerdict(
            identifier=identifier,
            identifier_kind=identifier_kind,
            abstract_resolved=True,
            entities_found=found,
            causal_cue_found=cue,
            overall_confidence=conf,
        )


def _resolver() -> _StubResolver:
    return _StubResolver(
        {
            "1001": "Baseline severity causes escalation to dupilumab in refractory CSU.",
            "1002": "Baseline severity was associated with persistence on therapy.",
            "1003": "An unrelated abstract about something else entirely.",
        }
    )


def _panel(feature="baseline_uas7", **over):
    rec = {
        "feature": feature,
        "layer_1": {"verdict": "pre_index", "temporal_status": "pre_index"},
        "layer_2": {"signal": "no_signal"},
        "layer_3": {"ran": True, "severity_pre_joint_check": "info", "z_score": 0.4},
        "layer_4": {"fired": False, "role": None, "mechanism": None},
        "ensemble": {"decided_by": "abstain", "final_role": None, "confidence": None},
        "leak_verdict": False,
        "leak_source": None,
        "review_required": False,
    }
    rec.update(over)
    return rec


def _brief(feature="baseline_uas7", panel=None):
    return build_brief(
        feature,
        derivation_pseudocode="source=claims; derivation_inputs=['uas7']; aggregation=None; "
        "window_days=None; knowable_at=index_date",
        dataset_context="cohort=optum_mart; target=persistent_at_180d_g28; "
        "prediction_anchor=index_date; treatment=treatment_dupixent; "
        "causal_question=effect of treatment_dupixent on persistent_at_180d_g28",
        treatment_label="dupilumab",
        outcome_label="persistence",
        panel_record=panel,
    )


def _confounder_answer(feature="baseline_uas7", **over):
    ans = {
        "reasoning": "Pre-index severity drives escalation and prognosis.",
        "edges": json.dumps([[feature, "T"], [feature, "Y"], ["T", "Y"]]),
        "edge_rationales": json.dumps(
            [
                {
                    "from_node": feature,
                    "to_node": "T",
                    "rationale": "severity drives escalation",
                    "citations": ["PMID:1001"],
                },
                {
                    "from_node": feature,
                    "to_node": "Y",
                    "rationale": "severity is prognostic",
                    "citations": ["1002"],
                },
            ]
        ),
        "entity_names": json.dumps(
            {feature: "baseline severity", "T": "dupilumab", "Y": "persistence"}
        ),
        "expected_role": "confounder",
        "ambiguous": "false",
    }
    ans.update(over)
    return ans


def _dummy(*answers):
    from dspy.utils.dummies import DummyLM

    return DummyLM(list(answers))


# ---------------------------------------------------------------------------
# The instructions ARE the guide (sections 0–6, verbatim)
# ---------------------------------------------------------------------------


def test_guide_sections_are_verbatim():
    guide = (PROJECT_ROOT / GUIDE_PATH).read_text(encoding="utf-8")
    expected = guide[guide.index(GUIDE_FIRST_HEADING) :]
    assert GUIDE_SECTIONS_0_TO_6 == expected
    assert GUIDE_HASH == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    for heading in ("## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in GUIDE_SECTIONS_0_TO_6


def test_signature_instructions_are_the_guide_and_prompt_hash_is_stable():
    from src.data.kg.structural_author import StructuralAttestationSignature

    # dspy runs the class docstring through inspect.cleandoc, which drops the
    # guide's trailing newline and nothing else (measured: 16545 vs 16546 chars).
    assert StructuralAttestationSignature.instructions == GUIDE_SECTIONS_0_TO_6.strip()
    assert GUIDE_SECTIONS_0_TO_6.strip() in GUIDE_SECTIONS_0_TO_6
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 64
    assert prompt_hash() != GUIDE_HASH  # the prompt is more than the guide


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_accepts_confounder_fragment_with_latent_and_citations():
    raw = _confounder_answer()
    raw["edges"] = json.dumps(
        [["baseline_uas7", "T"], ["U_severity", "baseline_uas7"], ["U_severity", "Y"], ["T", "Y"]]
    )
    raw["edge_rationales"] = json.dumps(
        [
            {
                "from_node": "U_severity",
                "to_node": "Y",
                "rationale": "latent severity drives prognosis",
                "citations": ["https://pubmed.ncbi.nlm.nih.gov/1002/", "doi:10.1111/all.15090"],
            }
        ]
    )
    frag = parse_author_output(raw, feature_name="baseline_uas7")
    assert frag.latents == ("U_severity",)
    assert ("T", "Y") in frag.edges
    cits = frag.citations[("U_severity", "Y")]
    assert [(c.kind, c.identifier) for c in cits] == [
        ("pmid", "1002"),
        ("doi", "10.1111/all.15090"),
    ]
    assert frag.expected_role == "confounder"
    assert frag.ambiguous is False


@pytest.mark.parametrize(
    "edges, reason",
    [
        ([["baseline_uas7", "T"], ["baseline_uas7", "Y"]], "estimand edge"),
        ([["T", "Y"]], "appears in no edge"),
        ([["baseline_uas7", "T"], ["T", "baseline_uas7"], ["T", "Y"]], "cycle"),
        ([["baseline_uas7", "T"], ["age", "Y"], ["T", "Y"]], "not the feature"),
        ([["baseline_uas7", "baseline_uas7"], ["T", "Y"]], "self-loop"),
        ([], "empty"),
    ],
)
def test_parser_rejects_malformed_fragments(edges, reason):
    raw = _confounder_answer(edges=json.dumps(edges), edge_rationales="[]")
    with pytest.raises(StructuralAuthorError, match=reason):
        parse_author_output(raw, feature_name="baseline_uas7")


def test_parser_rejects_rationale_for_an_edge_not_drawn():
    raw = _confounder_answer(
        edge_rationales=json.dumps(
            [{"from_node": "baseline_uas7", "to_node": "U_x", "rationale": "", "citations": []}]
        )
    )
    with pytest.raises(StructuralAuthorError, match="not in edges"):
        parse_author_output(raw, feature_name="baseline_uas7")


def test_parse_citation_kinds():
    assert parse_citation("PMID: 34536239").kind == "pmid"
    assert parse_citation("34536239").identifier == "34536239"
    assert parse_citation("10.1111/all.15090").kind == "doi"
    assert parse_citation("https://doi.org/10.1111/all.15090").identifier == "10.1111/all.15090"
    assert parse_citation("NCT04109313").kind == "nct"
    assert parse_citation("https://example.org/paper").kind == "url"
    assert parse_citation("Zuberbier 2021").kind == "unknown"


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def test_grader_direct_family_unsupported_and_estimand():
    resolver = _resolver()
    frag = parse_author_output(_confounder_answer(), feature_name="baseline_uas7")
    graded = {
        (e.from_node, e.to_node): e
        for e in grade_edges(
            frag, resolver=resolver, treatment_label="dupilumab", outcome_label="persistence"
        )
    }
    # 1001: both entities + "causes" → direct.
    assert graded[("baseline_uas7", "T")].grade == "direct"
    # 1002: both entities, no causal cue → family.
    assert graded[("baseline_uas7", "Y")].grade == "family"
    # The estimand edge is never sent to the resolver.
    assert graded[("T", "Y")].grade == "estimand"
    assert all(c[0] != "T" for c in resolver.calls)
    # The resolver was asked with the AUTHOR's entity names, not node labels.
    assert ("1001", "pmid", "baseline severity", "dupilumab") in resolver.calls


def test_grader_unsupported_on_unresolved_unrelated_nct_and_outage():
    resolver = _StubResolver({"1003": "An unrelated abstract."}, raise_on={"1004"})
    raw = _confounder_answer(
        edge_rationales=json.dumps(
            [
                {
                    "from_node": "baseline_uas7",
                    "to_node": "T",
                    "rationale": "",
                    "citations": ["1003", "9999", "NCT04109313", "1004"],
                },
            ]
        )
    )
    frag = parse_author_output(raw, feature_name="baseline_uas7")
    graded = {
        (e.from_node, e.to_node): e
        for e in grade_edges(
            frag, resolver=resolver, treatment_label="dupilumab", outcome_label="persistence"
        )
    }
    t_edge = graded[("baseline_uas7", "T")]
    assert t_edge.grade == "unsupported"
    kinds = {c["identifier"]: c for c in t_edge.citations}
    assert kinds["1003"]["abstract_resolved"] is True and kinds["1003"]["entities_found"] == []
    assert kinds["9999"]["abstract_resolved"] is False and kinds["9999"]["error"] is None
    assert "unverifiable" in kinds["NCT04109313"]["error"]
    assert "resolver raised" in kinds["1004"]["error"]
    # No citation at all → unsupported too.
    assert graded[("baseline_uas7", "Y")].grade == "unsupported"


# ---------------------------------------------------------------------------
# End to end with a scripted LM
# ---------------------------------------------------------------------------


def test_author_feature_end_to_end_with_dummy_lm():
    lm = _dummy(_confounder_answer())
    rec = author_feature(_brief(panel=_panel()), resolver=_resolver(), lm=lm)
    assert isinstance(rec, AuthoredAttestation)
    assert rec.derived_role == "confounder"
    assert rec.expected_role == "confounder"
    assert rec.ambiguous is False
    assert rec.review_required is False
    assert rec.edges == [["baseline_uas7", "T"], ["baseline_uas7", "Y"], ["T", "Y"]]
    assert [e.grade for e in rec.edge_provenance] == ["direct", "family", "estimand"]
    # Stamps.
    assert rec.model_id == "dummy"
    assert rec.prompt_hash == prompt_hash()
    assert rec.guide_hash == GUIDE_HASH
    assert rec.authored_at.endswith("+00:00")
    assert rec.reasoning == "Pre-index severity drives escalation and prognosis."
    # Machine output: audit-only until a human approves it.
    assert rec.provenance == "machine"
    att = rec.to_attestation()
    assert att is not None and att.may_decide() is False
    assert att.edges == (("baseline_uas7", "T"), ("baseline_uas7", "Y"), ("T", "Y"))
    # Round-trips through JSON.
    again = AuthoredAttestation.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert again.to_dict() == rec.to_dict()


def test_author_expected_role_disagreement_routes_to_review_and_extractor_wins():
    """codex r2 MED 4: a fragment whose own story contradicts what its edges
    derive is ambiguous and a review item, not just a logged string."""
    lm = _dummy(_confounder_answer(expected_role="instrument"))
    rec = author_feature(_brief(panel=_panel()), resolver=_resolver(), lm=lm)
    assert rec.derived_role == "confounder"
    assert rec.expected_role == "instrument"
    assert rec.ambiguous is True
    assert rec.review_required is True
    assert any("author expected 'instrument'" in r for r in rec.review_reasons)


def test_author_invalid_expected_role_is_rejected_to_review():
    # The parser refuses a role outside the six (strict, codex r2 MED 4) ...
    with pytest.raises(StructuralAuthorError, match="expected_role 'proxy' is not one of"):
        parse_author_output(_confounder_answer(expected_role="proxy"), feature_name="baseline_uas7")
    # ... and end to end the feature lands in review with no fragment (dspy's
    # own Literal typing already rejects the value before the parser sees it).
    rec = author_feature(
        _brief(), resolver=_resolver(), lm=_dummy(_confounder_answer(expected_role="proxy"))
    )
    assert rec.edges == [] and rec.review_required is True
    assert rec.review_reasons and rec.review_reasons[0].startswith("LM call failed")


def test_panel_cross_check_disagreement_sets_ambiguous_and_review():
    """Lane E 3(c): derived role vs the ensemble's final_role."""
    lm = _dummy(_confounder_answer())
    panel = _panel(ensemble={"decided_by": "llm", "final_role": "instrument", "confidence": 0.9})
    rec = author_feature(_brief(panel=panel), resolver=_resolver(), lm=lm)
    assert rec.cross_check == {
        "derived_role": "confounder",
        "panel_final_role": "instrument",
        "agrees": False,
    }
    assert rec.ambiguous is True
    assert rec.review_required is True


def test_layer_1_post_index_forbids_feature_to_treatment():
    """Lane E 3(b): a post-index verdict forbids feature -> T; the authored edge
    is kept verbatim (author-once) but flagged, and the feature goes to review."""
    lm = _dummy(_confounder_answer())
    panel = _panel(
        layer_1={"verdict": "post_index", "temporal_status": "post_index"},
        leak_verdict=True,
        leak_source="layer_1_post_index",
    )
    rec = author_feature(_brief(panel=panel), resolver=_resolver(), lm=lm)
    assert rec.constraint_violations == ["layer_1_post_index_forbids_feature_to_T"]
    assert rec.review_required is True
    flagged = [e for e in rec.edge_provenance if e.constraint_violation]
    assert [(e.from_node, e.to_node) for e in flagged] == [("baseline_uas7", "T")]
    assert rec.panel_summary["leak_verdict"] is True
    assert any("leak verdict" in r for r in rec.review_reasons)


def test_unclassifiable_fragment_routes_to_review_not_exception():
    # feature -> U_x only, and U_x reaches nothing: the extractor cannot classify.
    ans = _confounder_answer(
        edges=json.dumps([["baseline_uas7", "U_x"], ["T", "Y"]]), edge_rationales="[]"
    )
    rec = author_feature(_brief(), resolver=_resolver(), lm=_dummy(ans))
    assert rec.derived_role is None
    assert rec.review_required is True
    assert any("unclassifiable" in r for r in rec.review_reasons)
    assert rec.edges == [["baseline_uas7", "U_x"], ["T", "Y"]]  # kept verbatim for the reviewer


def test_rejected_output_routes_to_review_with_reason():
    ans = _confounder_answer(
        edges=json.dumps([["baseline_uas7", "T"], ["baseline_uas7", "Y"]]), edge_rationales="[]"
    )
    rec = author_feature(_brief(), resolver=_resolver(), lm=_dummy(ans))
    assert rec.edges == []
    assert rec.review_required is True
    assert rec.to_attestation() is None
    assert any("estimand edge" in r for r in rec.review_reasons)
    assert rec.model_id == "dummy" and rec.guide_hash == GUIDE_HASH


def test_lm_failure_routes_to_review_not_exception():
    class _Boom:
        model = "boom/model"

        def __call__(self, *a, **k):
            raise RuntimeError("provider down")

    import dspy

    rec = author_feature(
        _brief(), resolver=_resolver(), lm=dspy.LM("openai/never-called"), program=_Boom()
    )
    assert rec.review_required is True
    assert rec.edges == []
    assert any("LM call failed" in r and "provider down" in r for r in rec.review_reasons)


def test_no_lm_configured_routes_to_review(monkeypatch):
    import dspy

    monkeypatch.setattr(dspy.settings, "lm", None, raising=False)
    rec = author_feature(_brief(), resolver=_resolver(), lm=None)
    assert rec.review_required is True
    assert rec.review_reasons == ["no DSPy LM configured"]


def test_panel_record_view_reads_lane_e_shape():
    view = PanelRecordView.from_record(
        _panel(layer_4={"fired": True, "role": "confounder", "mechanism": "m"})
    )
    assert view.feature == "baseline_uas7"
    assert view.layer_1_verdict == "pre_index"
    assert view.layer_4_role == "confounder"
    assert view.layer_3_severity == "info"
    text = view.brief_text()
    assert "layer_1: verdict=pre_index" in text and "leak_verdict=False" in text


def test_build_brief_refuses_a_panel_record_for_another_feature():
    with pytest.raises(StructuralAuthorError, match="panel record is for"):
        _brief(feature="age", panel=_panel(feature="baseline_uas7"))


def test_tree_identity_names_the_commit_and_the_dirty_state(tmp_path):
    """codex r2 HIGH 4: evidence captured from a dirty tree must say so. The
    stamp is git-derived; where git or a repo is absent it is None, never a
    fabricated identity and never a crash (the API image has no .git)."""
    from src.data.kg.structural_author import tree_identity

    ident = tree_identity(Path(__file__).resolve().parents[4])
    assert isinstance(ident["commit"], str) and len(ident["commit"]) == 40
    assert isinstance(ident["dirty_src_scripts_tests"], bool)
    assert tree_identity(tmp_path) == {"commit": None, "dirty_src_scripts_tests": None}


@pytest.mark.parametrize(
    "over,match",
    [
        ({"edge_rationales": 42}, "edge_rationales"),
        (
            {
                "edge_rationales": [
                    {"from": "baseline_uas7", "to": "T", "rationale": "x", "citations": 7}
                ]
            },
            "citations",
        ),
        ({"entity_names": [1, 2]}, "entity_names"),
        ({"ambiguous": None}, "ambiguous"),
        ({"ambiguous": "maybe"}, "ambiguous"),
    ],
    ids=[
        "rationales_int",
        "citations_int",
        "entity_names_list",
        "ambiguous_missing",
        "ambiguous_word",
    ],
)
def test_parser_refuses_malformed_containers_and_an_unreadable_ambiguous(over, match):
    """codex r3 MED 3: every malformed model output is a StructuralAuthorError
    (-> review), never a TypeError/ValueError that aborts the cohort CLI; and
    ``ambiguous`` must be a real boolean, not defaulted to False."""
    ans = _confounder_answer()
    ans.update(over)
    if over.get("ambiguous", "x") is None:
        del ans["ambiguous"]
    with pytest.raises(StructuralAuthorError, match=match):
        parse_author_output(ans, feature_name="baseline_uas7")


def test_post_processing_failure_routes_to_review_not_an_exception(monkeypatch):
    """codex r3 MED 3: a failure AFTER parsing (grading / extraction) must yield a
    review-only record like an LM failure does (spec §5), not abort the run."""
    import src.data.kg.structural_author as mod

    def _boom(*a, **k):
        raise RuntimeError("grader exploded")

    monkeypatch.setattr(mod, "postprocess_fragment", _boom)
    rec = author_feature(_brief(), resolver=_resolver(), lm=_dummy(_confounder_answer()))
    assert rec.edges == [] and rec.review_required is True
    assert rec.review_reasons and rec.review_reasons[0].startswith(
        "post-processing failed: RuntimeError"
    )
