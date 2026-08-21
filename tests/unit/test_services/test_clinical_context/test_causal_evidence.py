"""#1763 Phase 2 — the causal-evidence provider.

Given the analysis (brand + treatment -> outcome) it returns, from the public
knowledge-graph stack:

- the Open Targets drug -> indication edge with its clinical stage, EXACT
  disease-node match only (a loose name match would let an unrelated indication
  speak for this one);
- literature the CitationResolver actually verified against the abstract (both
  entities named), capped;

and, when the treatment is a commercial lever, an honest "these sources do not
speak to this treatment side" state instead of the drug's evidence dressed up as
evidence for the lever.

No live HTTP: the Open Targets / PubMed / citation-resolver collaborators are
injected as fakes.
"""

from __future__ import annotations

import pytest

from src.data.kg.types import CitationVerdict
from src.services.clinical_context.brand_map import (
    resolve_brand_profile,
    treatment_context_for,
)
from src.services.clinical_context.causal_evidence import CausalEvidenceProvider

# The shape Open Targets returns (verified live 2026-08-21): drug.indications.rows
# carry the disease node + that indication's own maxClinicalStage.
_KISQALI_OT = {
    "drug": {
        "id": "CHEMBL3545110",
        "name": "RIBOCICLIB",
        "maximumClinicalStage": "APPROVAL",
        "indications": {
            "count": 3,
            "rows": [
                {"disease": {"id": "MONDO_0007254", "name": "breast cancer"}, "maxClinicalStage": "PHASE_3"},
                {"disease": {"id": "MONDO_0008315", "name": "prostate cancer"}, "maxClinicalStage": "PHASE_1_2"},
                {"disease": {"id": "MONDO_0011962", "name": "endometrial cancer"}, "maxClinicalStage": "PHASE_2"},
            ],
        },
    }
}

_FABHALTA_OT = {
    "drug": {
        "id": "CHEMBL4594448",
        "name": "IPTACOPAN",
        "maximumClinicalStage": "APPROVAL",
        "indications": {
            "count": 1,
            "rows": [
                {
                    "disease": {
                        "id": "MONDO_0100244",
                        "name": "paroxysmal nocturnal hemoglobinuria",
                    },
                    "maxClinicalStage": "APPROVAL",
                }
            ],
        },
    }
}


class _FakeOpenTargets:
    def __init__(self, drug_id="CHEMBL3545110", disease_id="MONDO_0007254", payload=None, boom=None):
        self._drug_id = drug_id
        self._disease_id = disease_id
        self._payload = payload if payload is not None else _KISQALI_OT
        self._boom = boom
        self.calls = 0

    def search_drug(self, name):
        return self._drug_id

    def search_disease(self, name):
        return self._disease_id

    def drug_disease_evidence(self, drug_chembl_id, disease_efo_id):
        self.calls += 1
        if self._boom is not None:
            raise self._boom
        return self._payload


class _FakePubMedSearch:
    def __init__(self, pmids=(), boom=None):
        self._pmids = list(pmids)
        self._boom = boom
        self.terms = []

    def search_pmids(self, term, *, retmax=5):
        self.terms.append((term, retmax))
        if self._boom is not None:
            raise self._boom
        return list(self._pmids[:retmax])

    def fetch_by_pmid(self, pmid):
        from src.services.clinical_context.clients import PubMedArticle

        return PubMedArticle(pmid=pmid, title=f"Study {pmid}", journal="J", pubdate="2024")


class _FakeResolver:
    """Verdicts keyed by PMID; anything unlisted comes back unresolved."""

    def __init__(self, verdicts=None):
        self._verdicts = verdicts or {}
        self.calls = []

    def verify_citation(self, identifier, *, identifier_kind="pmid", subject_name, object_name, **kw):
        self.calls.append((identifier, subject_name, object_name))
        verdict = self._verdicts.get(identifier)
        if verdict is not None:
            return verdict
        return CitationVerdict(
            identifier=identifier,
            identifier_kind=identifier_kind,
            abstract_resolved=False,
            entities_found=(),
            causal_cue_found=None,
            overall_confidence=0.0,
            error="abstract unavailable",
        )


def _verdict(pmid, confidence, entities=("ribociclib", "breast cancer")):
    return CitationVerdict(
        identifier=pmid,
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=tuple(entities),
        causal_cue_found=None,
        overall_confidence=confidence,
        error=None,
    )


def _provider(open_targets=None, pubmed=None, resolver=None):
    return CausalEvidenceProvider(
        open_targets=open_targets if open_targets is not None else _FakeOpenTargets(),
        pubmed=pubmed if pubmed is not None else _FakePubMedSearch(),
        resolver=resolver if resolver is not None else _FakeResolver(),
    )


@pytest.mark.unit
def test_indication_edge_uses_the_exact_disease_node_stage_not_the_drug_wide_stage():
    """Open Targets reports APPROVAL at the DRUG level (ribociclib is approved for
    something) while the breast-cancer node itself is PHASE_3. Reading the drug-wide
    stage would assert an approval Open Targets never made for this indication."""
    profile = resolve_brand_profile("Kisqali")
    frag = _provider().evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.disease_id == "MONDO_0007254"
    assert frag.indication_edge.max_clinical_stage == "PHASE_3"
    assert frag.indication_edge.predicate == "associated_with"
    assert frag.indication_edge.source == "open_targets"


@pytest.mark.unit
def test_treats_predicate_only_on_an_approval_stage_indication():
    profile = resolve_brand_profile("Fabhalta")
    provider = _provider(
        open_targets=_FakeOpenTargets(
            drug_id="CHEMBL4594448", disease_id="MONDO_0100244", payload=_FABHALTA_OT
        )
    )
    frag = provider.evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Fabhalta", "treatment_arm"),
        search_term="iptacopan pnh persistence",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.predicate == "treats"
    assert frag.indication_edge.max_clinical_stage == "APPROVAL"


@pytest.mark.unit
def test_a_non_approval_edge_defers_to_the_fda_label_in_its_note():
    """Open Targets staging lags the FDA label. When the edge is below APPROVAL the
    payload must say the label is the approval authority, or the panel reads as if
    the drug were not approved."""
    profile = resolve_brand_profile("Kisqali")
    frag = _provider().evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert "label" in frag.note.lower()


@pytest.mark.unit
def test_a_loose_disease_name_match_never_stands_in_for_the_indication():
    """search_disease returned an id that is not among the indication rows. The other
    'cancer' rows (prostate, endometrial) must NOT be used — an unrelated indication
    cannot speak for this one."""
    profile = resolve_brand_profile("Kisqali")
    provider = _provider(
        open_targets=_FakeOpenTargets(disease_id="MONDO_9999999", payload=_KISQALI_OT)
    )
    frag = provider.evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is None


@pytest.mark.unit
def test_falls_back_to_a_full_disease_name_match_when_the_id_lookup_misses():
    """No id from search_disease at all: a row whose name IS the disease term (in
    full, not a shared word) is still the right node."""
    profile = resolve_brand_profile("Kisqali")  # disease_search_term = "breast cancer"
    provider = _provider(open_targets=_FakeOpenTargets(disease_id=None, payload=_KISQALI_OT))
    frag = provider.evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.disease_name == "breast cancer"


@pytest.mark.unit
def test_only_verified_citations_are_surfaced_and_they_are_capped():
    profile = resolve_brand_profile("Kisqali")
    pubmed = _FakePubMedSearch(pmids=["1", "2", "3", "4", "5"])
    resolver = _FakeResolver(
        {
            "1": _verdict("1", 0.5),
            "2": _verdict("2", 0.3),  # below the bar -> dropped
            "3": _verdict("3", 1.0),
            "4": _verdict("4", 0.8),
            "5": _verdict("5", 0.9),
        }
    )
    frag = _provider(pubmed=pubmed, resolver=resolver).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert [c.pmid for c in frag.citations] == ["1", "3", "4"]
    assert all(c.confidence >= 0.5 for c in frag.citations)
    # Verification stopped once the cap was met — PMID 5 was never fetched.
    assert [pmid for pmid, _, _ in resolver.calls] == ["1", "2", "3", "4"]
    # The entities checked against the abstract are the drug and the plain-language
    # disease (the SSOT coding string never appears in an abstract).
    assert resolver.calls[0][1] == "ribociclib"
    assert resolver.calls[0][2] == "breast cancer"


@pytest.mark.unit
def test_the_literature_search_uses_the_analysis_query():
    """The evidence block must not run its own brand-level query — it searches the
    same analysis-composed term the citation provider uses."""
    profile = resolve_brand_profile("Kisqali")
    pubmed = _FakePubMedSearch(pmids=[])
    _provider(pubmed=pubmed).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert [term for term, _ in pubmed.terms] == [
        "ribociclib breast cancer persistence real-world"
    ]


@pytest.mark.unit
def test_commercial_levers_get_an_honest_not_covered_state():
    profile = resolve_brand_profile("Kisqali")
    ot = _FakeOpenTargets()
    resolver = _FakeResolver()
    frag = _provider(open_targets=ot, resolver=resolver).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "copay_support"),
        search_term="ribociclib breast cancer persistence copay assistance real-world",
    )
    assert frag.status == "commercial_lever"
    assert frag.indication_edge is None
    assert frag.citations == []
    assert "copay" in frag.note.lower() or "commercial" in frag.note.lower()
    # Nothing was asked of the clinical sources — no evidence to dress up.
    assert ot.calls == 0
    assert resolver.calls == []


@pytest.mark.unit
def test_clinical_covariate_treatments_do_get_evidence():
    """#1321 axes (advanced-line disease, UAS7 severity) are patient-state variables,
    not commercial levers: the clinical literature does speak to them."""
    profile = resolve_brand_profile("Kisqali")
    pubmed = _FakePubMedSearch(pmids=["7"])
    resolver = _FakeResolver({"7": _verdict("7", 0.9)})
    frag = _provider(pubmed=pubmed, resolver=resolver).evidence(
        profile,
        outcome="discontinued_180d",
        treatment_context=treatment_context_for("Kisqali", "disease_stage"),
        search_term="ribociclib breast cancer discontinuation metastatic advanced disease",
    )
    assert frag.status == "evidence"
    assert [c.pmid for c in frag.citations] == ["7"]


@pytest.mark.unit
def test_degrades_honestly_when_every_source_fails():
    from src.data.kg.open_targets import OpenTargetsError

    profile = resolve_brand_profile("Kisqali")
    frag = _provider(
        open_targets=_FakeOpenTargets(boom=OpenTargetsError("upstream 500")),
        pubmed=_FakePubMedSearch(boom=RuntimeError("pubmed down")),
    ).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.status == "unavailable"
    assert frag.indication_edge is None
    assert frag.citations == []
    assert frag.note


@pytest.mark.unit
def test_unmapped_treatment_yields_no_evidence_rather_than_brand_evidence():
    profile = resolve_brand_profile("Kisqali")
    ot = _FakeOpenTargets()
    frag = _provider(open_targets=ot).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=None,
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.status == "unavailable"
    assert ot.calls == 0


@pytest.mark.unit
def test_citations_that_resolve_no_abstract_are_dropped_not_shown_as_weak():
    """A Europe PMC timeout leaves abstract_resolved False. That is 'we could not
    check', not 'we checked and it is weak' — it must not reach the panel."""
    profile = resolve_brand_profile("Kisqali")
    pubmed = _FakePubMedSearch(pmids=["9"])
    frag = _provider(pubmed=pubmed, resolver=_FakeResolver()).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.citations == []


@pytest.mark.unit
def test_verification_stops_at_the_wall_clock_budget(monkeypatch):
    """Europe PMC can hang. The panel must come back with what verified rather than
    holding the user for one round trip per candidate."""
    import src.services.clinical_context.causal_evidence as ev_mod

    monkeypatch.setattr(ev_mod, "_VERIFICATION_BUDGET_S", 0.0)
    profile = resolve_brand_profile("Kisqali")
    resolver = _FakeResolver({"1": _verdict("1", 0.9), "2": _verdict("2", 0.9)})
    frag = _provider(pubmed=_FakePubMedSearch(pmids=["1", "2"]), resolver=resolver).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    # The budget was already spent on entry, so no candidate was fetched at all.
    assert resolver.calls == []
    assert frag.citations == []


# --- codex iter-1 findings ------------------------------------------------------


@pytest.mark.unit
def test_a_patient_state_treatment_gets_no_drug_indication_edge():
    """codex HIGH. TreatmentContext documents clinical_covariate as 'not a therapy:
    no drug-indication claim belongs to it', but the provider attached the drug's
    indication edge to it anyway — the therapy's evidence rendered under an
    'evidence for this analysis' heading for an analysis about disease stage. That
    is the borrowed-relevance failure #1763 is about, one notch milder."""
    profile = resolve_brand_profile("Kisqali")
    ot = _FakeOpenTargets()
    pubmed = _FakePubMedSearch(pmids=["7"])
    resolver = _FakeResolver({"7": _verdict("7", 0.9)})
    frag = _provider(open_targets=ot, pubmed=pubmed, resolver=resolver).evidence(
        profile,
        outcome="discontinued_180d",
        treatment_context=treatment_context_for("Kisqali", "disease_stage"),
        search_term="ribociclib breast cancer discontinuation metastatic advanced disease",
    )
    assert frag.status == "evidence"
    assert frag.indication_edge is None
    assert ot.calls == 0  # the edge was not even fetched
    # The literature IS on-topic (the covariate theme is in the query) and is kept,
    # but the note must say what was actually verified.
    assert [c.pmid for c in frag.citations] == ["7"]
    assert "patient-state" in frag.note.lower() or "not a therapy" in frag.note.lower()


@pytest.mark.unit
def test_a_drug_therapy_treatment_still_gets_the_edge():
    profile = resolve_brand_profile("Fabhalta")
    frag = _provider(
        open_targets=_FakeOpenTargets(
            drug_id="CHEMBL4594448", disease_id="MONDO_0100244", payload=_FABHALTA_OT
        )
    ).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Fabhalta", "complement_inhibitor_status"),
        search_term="iptacopan pnh switch",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.predicate == "treats"


@pytest.mark.unit
def test_verification_reserves_room_for_a_full_timeout_before_starting_one(monkeypatch):
    """codex MEDIUM. Checking 'elapsed < budget' BEFORE a call that can itself burn a
    full client timeout does not bound the wait: with a 20s budget and an 8s timeout
    a call could start at 19.9s and end at 27.9s. Only start a candidate when a whole
    timeout still fits."""
    import src.services.clinical_context.causal_evidence as ev_mod

    # Budget smaller than one client timeout: not even the first candidate may start.
    monkeypatch.setattr(ev_mod, "_VERIFICATION_BUDGET_S", 5.0)
    monkeypatch.setattr(ev_mod, "_EUROPE_PMC_TIMEOUT_S", 8.0)
    profile = resolve_brand_profile("Kisqali")
    resolver = _FakeResolver({"1": _verdict("1", 0.9)})
    frag = _provider(pubmed=_FakePubMedSearch(pmids=["1"]), resolver=resolver).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert resolver.calls == []
    assert frag.citations == []


# --- adversarial review findings ------------------------------------------------


@pytest.mark.unit
def test_the_edge_is_dropped_when_open_targets_answered_about_another_molecule():
    """HIGH. `search_drug` is a relevance-ranked search with no name comparison, and
    the panel attributes the edge to the CURATED drug name — so a mis-resolved
    molecule would render a regulatory-sounding claim about the wrong drug, and
    nothing would show it. The disproving field (drug.name) is already fetched."""
    profile = resolve_brand_profile("Kisqali")
    wrong = {
        "drug": {
            "id": "CHEMBL189963",
            "name": "PALBOCICLIB",
            "maximumClinicalStage": "APPROVAL",
            "indications": {
                "count": 1,
                "rows": [
                    {
                        "disease": {"id": "MONDO_0007254", "name": "breast cancer"},
                        "maxClinicalStage": "APPROVAL",
                    }
                ],
            },
        }
    }
    frag = _provider(open_targets=_FakeOpenTargets(payload=wrong)).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is None


@pytest.mark.unit
def test_the_edge_names_the_molecule_that_actually_answered():
    profile = resolve_brand_profile("Kisqali")
    frag = _provider().evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.drug_id == "CHEMBL3545110"
    assert frag.indication_edge.drug_name.lower() == "ribociclib"


@pytest.mark.unit
def test_a_salt_form_of_the_same_molecule_still_matches():
    profile = resolve_brand_profile("Kisqali")
    salt = {
        "drug": {
            "id": "CHEMBL3545110",
            "name": "RIBOCICLIB SUCCINATE",
            "maximumClinicalStage": "APPROVAL",
            "indications": {
                "count": 1,
                "rows": [
                    {
                        "disease": {"id": "MONDO_0007254", "name": "breast cancer"},
                        "maxClinicalStage": "PHASE_3",
                    }
                ],
            },
        }
    }
    frag = _provider(open_targets=_FakeOpenTargets(payload=salt)).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.max_clinical_stage == "PHASE_3"


@pytest.mark.unit
def test_a_row_with_no_disease_cannot_match_a_missing_disease_id():
    """LOW. `row_id == disease_id` with both None matched a malformed row, which also
    suppressed the full-name fallback — the result was a blank-disease APPROVAL
    claim."""
    profile = resolve_brand_profile("Kisqali")
    malformed = {
        "drug": {
            "id": "CHEMBL3545110",
            "name": "RIBOCICLIB",
            "maximumClinicalStage": "APPROVAL",
            "indications": {
                "count": 2,
                "rows": [
                    {"disease": None, "maxClinicalStage": "APPROVAL"},
                    {
                        "disease": {"id": "MONDO_0007254", "name": "breast cancer"},
                        "maxClinicalStage": "PHASE_3",
                    },
                ],
            },
        }
    }
    frag = _provider(
        open_targets=_FakeOpenTargets(disease_id=None, payload=malformed)
    ).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.indication_edge is not None
    assert frag.indication_edge.disease_name == "breast cancer"
    assert frag.indication_edge.max_clinical_stage == "PHASE_3"


@pytest.mark.unit
def test_a_source_that_errored_is_disclosed_not_silently_read_as_absence():
    """HIGH. With Open Targets down and PubMed up, the fragment used to come back
    status='evidence' with no edge — indistinguishable from 'no indication edge
    exists'. An outage must be visible, and must not be cached as a settled result."""
    from src.data.kg.open_targets import OpenTargetsError

    profile = resolve_brand_profile("Kisqali")
    pubmed = _FakePubMedSearch(pmids=["1"])
    resolver = _FakeResolver({"1": _verdict("1", 0.9)})
    frag = _provider(
        open_targets=_FakeOpenTargets(boom=OpenTargetsError("502")),
        pubmed=pubmed,
        resolver=resolver,
    ).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.status == "evidence"
    assert frag.indication_edge is None
    assert frag.sources_unavailable == ("open_targets",)
    assert "unreachable" in frag.note.lower()
    assert [c.pmid for c in frag.citations] == ["1"]


@pytest.mark.unit
def test_a_literature_failure_is_disclosed_too():
    profile = resolve_brand_profile("Kisqali")
    frag = _provider(pubmed=_FakePubMedSearch(boom=RuntimeError("pubmed down"))).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert frag.sources_unavailable == ("pubmed",)


@pytest.mark.unit
def test_the_fda_label_note_does_not_point_in_a_direction_the_panel_does_not_render():
    """LOW. The note said the approval status comes from "the label section above",
    but the panel renders the approved-use section BELOW the evidence block."""
    import src.services.clinical_context.causal_evidence as ev_mod

    assert "above" not in ev_mod._FDA_LABEL_NOTE.lower()
    assert "approved-use" in ev_mod._FDA_LABEL_NOTE.lower()


@pytest.mark.unit
def test_the_real_provider_wiring_constructs():
    """MEDIUM. Nothing exercised the production wiring, so a constructor-signature
    drift in EuropePMCClient / CitationResolver / OpenTargetsClient would first
    surface as a swallowed warning in prod. No network: constructing the clients
    opens no connection."""
    from src.services.clinical_context.causal_evidence import (
        CausalEvidenceProvider,
        default_causal_evidence_provider,
    )

    provider = default_causal_evidence_provider()
    assert isinstance(provider, CausalEvidenceProvider)
    assert callable(provider._open_targets.search_drug)
    assert callable(provider._pubmed.search_pmids)
    assert callable(provider._resolver.verify_citation)
