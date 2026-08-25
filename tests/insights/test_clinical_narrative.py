"""Unit tests for the clinical-narrative insight module (spec 2026-08-24).

The fallback path is forced by tests/insights/conftest.py (no live LLM). These
tests pin the DERIVED grounding strings — never bare booleans — so a silent
composition change fails loudly (wave-27: assert the derivation, not the
decision)."""

from types import SimpleNamespace

from src.insights import clinical_narrative


def _payload(**overrides):
    """A real-shaped ClinicalContextService.get_context payload (remibrutinib)."""
    base = {
        "brand": "Remibrutinib",
        "drug_name": "remibrutinib",
        "disease": "Chronic spontaneous urticaria",
        "our_outcome": "adopted",
        "our_treatment": "treatment_arm",
        "mapped_endpoint": None,
        "treatment_context": {
            "column": "treatment_arm",
            "label": "on remibrutinib therapy",
            "framing": "being on remibrutinib",
            "kind": "drug_therapy",
            "source": "curated",
        },
        "analysis_framing": "This analysis asks what being on remibrutinib does to prescriber adoption.",
        "analysis_grounding": {
            "label_considerations": [],
            "competitive_context": "At initiation the choice is between remibrutinib and two injectable biologics.",
            "note": None,
            "outcome_theme": None,
        },
        "mechanism": {
            "mechanism_of_action": "Bruton tyrosine kinase (BTK) inhibitor",
            "source": "chembl",
        },
        "pivotal_endpoints": {
            "endpoints": [
                {"measure": "Change from baseline in UAS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
                {"measure": "Change from baseline in ISS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
                {"measure": "Change from baseline in HSS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
            ],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": None,
        "approved_indications": {
            "indications": [
                "RHAPSIDO is indicated for the treatment of chronic spontaneous urticaria "
                "in adults who remain symptomatic despite H1 antihistamine treatment."
            ],
            "limitations_of_use": None,
            "boxed_warning": None,
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Xolair (omalizumab)", "Dupixent (dupilumab)"],
            "count": 2,
            "source": "curated",
        },
        "causal_evidence": {
            "status": "found",
            "indication_edge": {
                "predicate": "associated_with",
                "drug_id": "CHEMBL4650485",
                "drug_name": "remibrutinib",
                "disease_id": "EFO_0005854",
                "disease_name": "chronic spontaneous urticaria",
                "max_clinical_stage": "PHASE_3",
                "source": "open_targets",
            },
            "sources_unavailable": [],
            "citations": [],
            "note": None,
        },
        "honesty_label": "Effect estimate = a SYNTHETIC patient cohort ...",
    }
    base.update(overrides)
    return base


def _grounding(**overrides):
    kwargs = {"grain": "hcp", "ate": 0.14, "ate_ci_lower": 0.05, "ate_ci_upper": 0.23, "gate_decision": "proceed"}
    kwargs.update({k: overrides.pop(k) for k in list(overrides) if k in kwargs})
    return clinical_narrative.build_grounding(_payload(**overrides), **kwargs)


class TestBuildGrounding:
    def test_result_string_pins_signed_ate_ci_and_gate_phrase(self):
        g = _grounding()
        assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in g["result"]
        assert "Robustness gate: proceed — the estimate survived all robustness checks." in g["result"]
        assert "synthetic patient cohort" in g["result"]

    def test_gate_phrases_review_block_and_missing(self):
        assert "needs review (mixed robustness)" in _grounding(gate_decision="review")["result"]
        assert "failed robustness checks" in _grounding(gate_decision="block")["result"]
        assert "Robustness gate: not reported." in _grounding(gate_decision=None)["result"]

    def test_missing_ate_is_reported_not_invented(self):
        g = _grounding(ate=None, ate_ci_lower=None, ate_ci_upper=None)
        assert "No effect estimate was provided for treatment_arm -> adopted." in g["result"]

    def test_analysis_carries_framing_kind_and_grain(self):
        g = _grounding()
        assert "prescriber adoption" in g["analysis"]
        assert "therapy contrast" in g["analysis"]
        assert "Analysis grain: hcp." in g["analysis"]

    def test_clinical_covariate_gets_the_observational_sentence(self):
        payload = _payload(
            treatment_context={
                "column": "disease_severity",
                "label": "high disease severity",
                "framing": "severe disease",
                "kind": "clinical_covariate",
                "source": "curated",
            }
        )
        g = clinical_narrative.build_grounding(
            payload, grain="patient", ate=0.05, ate_ci_lower=None, ate_ci_upper=None, gate_decision=None
        )
        assert (
            "The treatment 'high disease severity' is a patient-state variable "
            "used as an observational treatment." in g["analysis"]
        )

    def test_commercial_lever_gets_the_boundary_sentence(self):
        payload = _payload(
            treatment_context={
                "column": "copay_support",
                "label": "copay support active",
                "framing": "copay support",
                "kind": "commercial",
                "source": "curated",
            }
        )
        g = clinical_narrative.build_grounding(
            payload, grain="patient", ate=0.02, ate_ci_lower=None, ate_ci_upper=None, gate_decision="review"
        )
        assert "commercial (access/promotion) lever" in g["analysis"]
        assert "never this lever" in g["analysis"]

    def test_unmapped_outcome_is_stated(self):
        g = _grounding()
        assert "Our outcome 'adopted' is not mapped to any registered endpoint." in g["trial_endpoints"]
        assert "Change from baseline in UAS7 at Week 12" in g["trial_endpoints"]

    def test_mapped_outcome_names_the_endpoint(self):
        g = _grounding(mapped_endpoint="Treatment persistence / duration of therapy")
        assert (
            "Our outcome 'adopted' maps to the real endpoint: "
            "Treatment persistence / duration of therapy." in g["trial_endpoints"]
        )

    def test_endpoint_list_is_capped_with_honest_overflow(self):
        eps = [
            {"measure": f"Endpoint {i}", "time_frame": None, "nct_id": None} for i in range(7)
        ]
        g = _grounding(pivotal_endpoints={"endpoints": eps, "source": "clinicaltrials.gov"})
        assert "Endpoint 4" in g["trial_endpoints"]
        assert "Endpoint 5" not in g["trial_endpoints"]
        assert "(+2 more)" in g["trial_endpoints"]

    def test_rwe_absence_is_woven_not_blank(self):
        assert "No real-world evidence names this brand yet" in _grounding()["evidence"]

    def test_rwe_presence_carries_title_and_pmid(self):
        g = _grounding(
            real_world_evidence={
                "pmid": "35642282",
                "title": "CDK4/6 inhibitor treatment use in women with advanced breast cancer.",
                "journal": "J Oncol Pharm Pract",
                "pubdate": "2023 Jul",
                "doi": None,
                "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
                "source": "pubmed",
                "search_term": None,
            }
        )
        assert "CDK4/6 inhibitor treatment use" in g["evidence"]
        assert "(PMID 35642282)" in g["evidence"]
        assert "No real-world evidence" not in g["evidence"]

    def test_label_read_vs_unreadable_are_different_claims(self):
        read = _grounding()  # openfda source, no considerations
        assert "The FDA label was read and carries nothing bearing on this outcome." in read["evidence"]
        unreadable = _grounding(
            approved_indications={
                "indications": ["curated indication text"],
                "limitations_of_use": None,
                "boxed_warning": None,
                "source": "static_fallback",
            }
        )
        assert "The FDA label could not be read for this analysis" in unreadable["evidence"]

    def test_label_considerations_render_verbatim(self):
        g = _grounding(
            analysis_grounding={
                "label_considerations": [
                    {
                        "title": "Antihistamine-refractory population",
                        "detail": "Indicated only after H1 antihistamines.",
                        "section": "indications",
                        "references": "1",
                        "source": "openfda",
                    }
                ],
                "competitive_context": None,
                "note": None,
                "outcome_theme": None,
            }
        )
        assert (
            "Label consideration (indications): Antihistamine-refractory population — "
            "Indicated only after H1 antihistamines." in g["evidence"]
        )

    def test_open_targets_edge_is_composed(self):
        g = _grounding()
        assert (
            "Open Targets records remibrutinib as in development for "
            "chronic spontaneous urticaria (max clinical stage: PHASE_3)." in g["evidence"]
        )

    def test_open_targets_approved_edge_says_approved_therapy(self):
        g = _grounding(
            causal_evidence={
                "status": "found",
                "indication_edge": {
                    "predicate": "treats",
                    "drug_id": "CHEMBL4650485",
                    "drug_name": "remibrutinib",
                    "disease_id": "EFO_0005854",
                    "disease_name": "chronic spontaneous urticaria",
                    "max_clinical_stage": "APPROVED",
                    "source": "open_targets",
                },
                "sources_unavailable": [],
                "citations": [],
                "note": None,
            }
        )
        assert (
            "Open Targets records remibrutinib as an approved therapy for "
            "chronic spontaneous urticaria (max clinical stage: APPROVED)." in g["evidence"]
        )

    def test_clinical_position_carries_moa_indication_and_positioning(self):
        g = _grounding()
        assert "Bruton tyrosine kinase (BTK) inhibitor" in g["clinical_position"]
        assert "RHAPSIDO is indicated" in g["clinical_position"]
        # Curated positioning from src.insights.clinical_context._CLINICAL_POSITIONING:
        assert "antihistamine-refractory, later-line population" in g["clinical_position"]

    def test_competitive_position_carries_framing_and_rivals(self):
        g = _grounding()
        assert "two injectable biologics" in g["competitive_position"]
        assert "Curated rivals: Xolair (omalizumab); Dupixent (dupilumab)." in g["competitive_position"]

    def test_grounding_chips(self):
        g = _grounding()
        chips = {c["label"]: c["value"] for c in g["grounding"]}
        assert chips["Brand"] == "Remibrutinib"
        assert chips["Analysis"] == "treatment_arm -> adopted"
        assert chips["Gate"] == "proceed"
        # chembl + clinicaltrials.gov + openfda live; RWE is None -> 3/4
        assert chips["Live sources"] == "3/4"
        assert g["context_unavailable"] is False


class TestResultOnlyGrounding:
    def test_marks_context_unavailable_and_still_pins_the_result(self):
        g = clinical_narrative.build_result_only_grounding(
            brand="Remibrutinib",
            grain="hcp",
            treatment="treatment_arm",
            outcome="adopted",
            ate=0.14,
            ate_ci_lower=0.05,
            ate_ci_upper=0.23,
            gate_decision="proceed",
        )
        assert g["context_unavailable"] is True
        assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in g["result"]
        assert "Causal analysis of treatment_arm -> adopted for Remibrutinib" in g["analysis"]
        unavailable = "The clinical-context sources could not be fetched for this analysis."
        assert g["clinical_position"] == unavailable
        assert g["competitive_position"] == unavailable
        assert g["trial_endpoints"] == unavailable
        assert g["evidence"] == unavailable
        assert {"label": "Clinical context", "value": "unavailable"} in g["grounding"]


class TestFallback:
    def test_fallback_composes_the_grounding_strings(self):
        g = _grounding()
        out = clinical_narrative.fallback(g)
        assert out["is_fallback"] is True
        assert out["key_takeaways"] == []
        assert g["result"] in out["insight"]
        assert g["clinical_position"] in out["insight"]
        assert "(Factual summary — LLM narrative unavailable.)" in out["insight"]

    def test_result_only_fallback_says_sources_unfetchable(self):
        g = clinical_narrative.build_result_only_grounding(
            brand="Remibrutinib",
            grain="hcp",
            treatment="treatment_arm",
            outcome="adopted",
            ate=None,
            ate_ci_lower=None,
            ate_ci_upper=None,
            gate_decision=None,
        )
        out = clinical_narrative.fallback(g)
        assert out["is_fallback"] is True
        assert "could not be fetched" in out["insight"]
        # The unavailable filler strings must not be repeated as if they were facts.
        assert out["insight"].count("could not be fetched") == 1


class TestGenerateInsight:
    def test_no_lm_returns_fallback(self):
        # conftest forces ensure_dspy_configured -> False, so run_signature -> None.
        out = clinical_narrative.generate_insight(_grounding())
        assert out["is_fallback"] is True

    def test_good_narrative_passes(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="Remibrutinib, a BTK inhibitor, showed ATE +0.1400 on adoption."
            ),
        )
        out = clinical_narrative.generate_insight(_grounding())
        assert out["is_fallback"] is False
        assert out["insight"].startswith("Remibrutinib, a BTK inhibitor")
        assert out["key_takeaways"] == []

    def test_empty_narrative_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative, "run_signature", lambda *a, **k: SimpleNamespace(narrative="   ")
        )
        assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_fabricated_pmid_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="A registry study (PMID 99999999) proved adoption doubles."
            ),
        )
        assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_fabricated_nct_and_url_are_rejected(self, monkeypatch):
        for bad in (
            "See trial NCT99999999 for confirmation.",
            "Details at https://example.com/made-up.",
        ):
            monkeypatch.setattr(
                clinical_narrative, "run_signature", lambda *a, _b=bad, **k: SimpleNamespace(narrative=_b)
            )
            assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_grounded_pmid_passes_the_guard(self, monkeypatch):
        g = _grounding(
            real_world_evidence={
                "pmid": "35642282",
                "title": "CDK4/6 inhibitor treatment use in women with advanced breast cancer.",
                "journal": None,
                "pubdate": None,
                "doi": None,
                "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
                "source": "pubmed",
                "search_term": None,
            }
        )
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="Real-world use is documented (PMID 35642282) alongside the estimate."
            ),
        )
        out = clinical_narrative.generate_insight(g)
        assert out["is_fallback"] is False
