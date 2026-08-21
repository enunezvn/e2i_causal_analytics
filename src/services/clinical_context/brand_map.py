"""Brand -> drug / disease / static-fallback profile for clinical-context
enrichment.

Joins the canonical SSOT in ``src/ml/synthetic/clinical_codes.py`` (drug_name,
drug_class, disease desc) with the enrichment-only extras that SSOT does not
carry: the precise static MoA fallback strings the redesign spec pins (used when
ChEMBL is unreachable), the static pivotal-endpoint fallback (used when
ClinicalTrials.gov is down OR returns only safety endpoints), the PubMed RWE
search term, and the map from OUR synthetic outcome -> the real pivotal endpoint
framing.

All fallback strings are REAL clinical facts (MoA per ChEMBL mechanism rows;
pivotal endpoints verified live 2026-06-19 against ClinicalTrials.gov API v2:
breast cancer = OS/PFS/iDFS; PNH = transfusion-avoidance/LDH/Hb-stabilization;
CSU = UAS7/UCT7/ISS7) — never invented placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ml.synthetic.clinical_codes import BRAND_DIAGNOSIS, BRAND_NDC


@dataclass(frozen=True)
class BrandClinicalProfile:
    """The clinical facts + static fallbacks for one brand.

    ``moa_fallback`` / ``pivotal_endpoints_fallback`` are used ONLY when the live
    API is unreachable or unhelpful; the providers prefer the live API value and
    fall back to these. ``outcome_endpoint_map`` maps our synthetic outcome
    column -> the real pivotal-endpoint framing it stands in for.

    Therapy-label + competitor fields (added 2026-06-20):
    ``indications_fallback`` — curated approved indication strings (used when
    OpenFDA label is unreachable); ``limitations_fallback`` / ``boxed_warning_fallback``
    — curated LoU / boxed-warning text or None when absent for this drug;
    ``competitor_map`` — dict keyed by disease string (lowercased) -> list of
    competitor ``"Brand (generic)"`` strings within the same therapeutic class.
    """

    brand: str
    drug_name: str
    disease: str
    drug_class: str
    moa_fallback: str
    pivotal_endpoints_fallback: List[str]
    rwe_search_term: str
    rwe_seed_pmid: Optional[str]
    outcome_endpoint_map: Dict[str, str] = field(default_factory=dict)
    # Therapy-label fallback fields — default-empty so existing construction is unaffected.
    indications_fallback: List[str] = field(default_factory=list)
    limitations_fallback: Optional[str] = None
    boxed_warning_fallback: Optional[str] = None
    # key = disease string lowercased → list of competitor "Brand (generic)" strings
    competitor_map: Dict[str, List[str]] = field(default_factory=dict)
    # Curated brand-SPECIFIC seminal real-world-evidence citation (verified against
    # PubMed). Surfaced as its OWN "seminal RWE" link so the brand of interest always
    # has a brand-faithful reference, INDEPENDENT of what the live PubMed relevance
    # search returns — that search can rank a competitor / CDK4-6-class-comparison
    # paper first (the reported abemaciclib-for-Kisqali confusion). Keys:
    # pmid, title, journal, year, doi. None when no seminal RWE is curated yet.
    seminal_rwe: Optional[Dict[str, str]] = None
    # Plain-language disease term for LITERATURE search. The SSOT disease string is a
    # clinical-coding description ("Malignant neoplasm of breast"); clinicians publish
    # under "breast cancer". Kept separate so neither use distorts the other.
    disease_search_term: str = ""
    # #1763: curated clinical framing per synthetic TREATMENT column (the mirror of
    # outcome_endpoint_map for the treatment side of the analysis). Keyed by column.
    treatment_context: Dict[str, "TreatmentContext"] = field(default_factory=dict)
    # #1763: the analysis-specific PubMed term, composed per (outcome, treatment) by
    # the service and attached to a per-request copy of the profile. None on the
    # brand-level view, where the curated ``rwe_search_term`` is the right query.
    analysis_rwe_search_term: Optional[str] = None


@dataclass(frozen=True)
class TreatmentContext:
    """Curated clinical framing for ONE synthetic treatment column (#1763).

    ``kind`` is what lets the evidence layer stay honest about what the public
    clinical APIs can and cannot speak to:

    - ``drug_therapy`` — the treatment IS a therapy (the brand's regimen, its
      initiation, a prior-therapy switch). Drug-disease and drug-outcome clinical
      evidence is on-topic.
    - ``clinical_covariate`` — a patient-state variable used as an observational
      treatment (disease stage, UAS7 severity). Clinical literature speaks to it,
      but it is not a therapy: no drug-indication claim belongs to it.
    - ``commercial`` — an access / promotion lever (copay, PSP, detailing,
      sampling, NBA triggers). Biomedical APIs do NOT speak to this treatment
      side; the evidence layer must say so rather than attach clinical evidence
      that is really about the drug.

    ``literature_theme`` is the phrase (if any) this treatment contributes to the
    real-world-evidence search. ``None`` = the lever has no clinical-literature
    analogue, so it adds nothing to the query rather than skewing it.
    """

    column: str
    label: str
    framing: str
    kind: str
    literature_theme: Optional[str] = None


# Our synthetic outcome column -> the phrase used in the analysis framing sentence.
# Unmapped outcomes fall back to the raw column name (honest, never invented).
_OUTCOME_FRAMING: Dict[str, str] = {
    "persistent_180d": "180-day treatment persistence",
    "discontinued_180d": "180-day treatment discontinuation",
    "treatment_initiated": "treatment initiation",
    "adherent_180d": "180-day adherence",
    "low_gap_180d": "gap-free refill adherence",
    "adopted": "prescriber adoption",
}

# Our synthetic outcome -> the theme it contributes to the literature search.
_OUTCOME_LITERATURE_THEME: Dict[str, str] = {
    "persistent_180d": "persistence",
    "discontinued_180d": "discontinuation",
    "treatment_initiated": "treatment initiation",
    "adherent_180d": "adherence",
    "low_gap_180d": "adherence",
}

# Access / promotion levers. Identical across brands (they are commercial, not
# clinical), so they are curated once and merged into every brand's map.
# copay assistance and patient-support programmes DO have a real health-services
# literature, so they carry a search theme; rep detailing, sampling, NBA triggers
# and prescriber peer influence do not — they contribute no query term.
_COMMERCIAL_TREATMENT_CONTEXT: Dict[str, Dict[str, Optional[str]]] = {
    "copay_support": {
        "label": "Copay support",
        "framing": "receiving copay assistance",
        "literature_theme": "copay assistance",
    },
    "psp_enrolled": {
        "label": "Patient support program",
        "framing": "being enrolled in a patient support program",
        "literature_theme": "patient support program",
    },
    "rep_detailing_high": {
        "label": "High rep detailing",
        "framing": "high sales-representative detailing",
        "literature_theme": None,
    },
    "sample_dropped": {
        "label": "Sample dropped",
        "framing": "receiving a product sample",
        "literature_theme": None,
    },
    "trigger_accepted": {
        "label": "NBA trigger accepted",
        "framing": "the prescriber acting on a next-best-action trigger",
        "literature_theme": None,
    },
    "peer_influence_score": {
        "label": "Prescriber peer influence",
        "framing": "prescriber peer influence in the referral network",
        "literature_theme": None,
    },
}


# Enrichment-only static facts keyed by brand. The drug_name / disease / drug_class
# are pulled from the clinical_codes SSOT at construction time (below) so they can
# never drift from it.
_STATIC_ENRICHMENT: Dict[str, Dict[str, object]] = {
    "Kisqali": {
        "moa_fallback": "CDK4/6 inhibitor",
        "pivotal_endpoints_fallback": [
            "Overall Survival (OS)",
            "Progression-Free Survival (PFS)",
            "Invasive Disease-Free Survival (iDFS)",
        ],
        "disease_search_term": "breast cancer",
        # #1763: the treatment side of each analysis. treatment_arm is the CDK4/6
        # regimen itself (combination with an AI / fulvestrant per label), so the
        # framing says "regimen", not the bare molecule.
        "treatment_context": {
            "treatment_arm": {
                "label": "Treatment arm",
                "framing": "being on a ribociclib-containing regimen",
                "kind": "drug_therapy",
                "literature_theme": None,
            },
            "treatment_initiated": {
                "label": "Treatment initiated",
                "framing": "initiating ribociclib",
                "kind": "drug_therapy",
                "literature_theme": "treatment initiation",
            },
            # #1321 Kisqali axis: advanced-line disease as an observational treatment.
            "disease_stage": {
                "label": "Advanced line (metastatic / stage IV)",
                "framing": "advanced-line disease (metastatic / stage IV)",
                "kind": "clinical_covariate",
                "literature_theme": "metastatic advanced disease",
            },
        },
        "rwe_search_term": "ribociclib persistence adherence breast cancer real-world",
        "rwe_seed_pmid": "35642282",
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation / time-to-treatment-start",
            "persistent_180d": "Treatment persistence / duration of therapy (proxy for PFS-supporting adherence)",
            "discontinued_180d": "Treatment discontinuation / early termination",
        },
        # Therapy-label fallback (OpenFDA SPL / prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "HR+/HER2- advanced or metastatic breast cancer (with an aromatase inhibitor or fulvestrant)",
            "HR+/HER2- node-positive early breast cancer, adjuvant (with an aromatase inhibitor)",
        ],
        "limitations_fallback": None,
        "boxed_warning_fallback": None,
        # disease key = "malignant neoplasm of breast" (BRAND_DIAGNOSIS["Kisqali"]["desc"].lower())
        "competitor_map": {
            "malignant neoplasm of breast": [
                "Ibrance (palbociclib)",
                "Verzenio (abemaciclib)",
            ],  # ATC L01EF CDK4/6 inhibitors (OpenFDA/RxClass probe-confirmed)
        },
        # Curated brand-SPECIFIC seminal RWE (ribociclib-only, no competitor in the
        # title — verified against PubMed 2026-07-10). A real-world managed-access
        # study mirroring the MONALEESA-7 pivotal population. Chosen over a live
        # relevance hit precisely because relevance ranks CDK4/6-class comparisons
        # (palbociclib/abemaciclib) first, which read as competitor papers.
        "seminal_rwe": {
            "pmid": "36135090",
            "title": (
                "Real-World Clinical Outcomes of Ribociclib in Combination with a "
                "Non-Steroidal Aromatase Inhibitor and a Luteinizing Hormone-Releasing "
                "Hormone Agonist in Premenopausal HR+/HER2- Advanced Breast Cancer "
                "Patients: An Italian Managed Access Program"
            ),
            "journal": "Current Oncology",
            "year": "2022",
            "doi": "10.3390/curroncol29090521",
        },
    },
    "Remibrutinib": {
        "moa_fallback": "BTK inhibitor",
        "pivotal_endpoints_fallback": [
            "Change from baseline in UAS7 (Urticaria Activity Score over 7 days)",
            "UCT7 (Urticaria Control Test)",
            "ISS7 (Itch Severity Score) / WI-NRS",
        ],
        "disease_search_term": "chronic spontaneous urticaria",
        "treatment_context": {
            "treatment_arm": {
                "label": "Treatment arm",
                "framing": "being on remibrutinib",
                "kind": "drug_therapy",
                "literature_theme": None,
            },
            "treatment_initiated": {
                "label": "Treatment initiated",
                "framing": "initiating remibrutinib after antihistamine failure",
                "kind": "drug_therapy",
                "literature_theme": "treatment initiation",
            },
            # #1321 Remibrutinib axis: uncontrolled CSU as an observational treatment.
            "urticaria_severity_uas7": {
                "label": "Uncontrolled CSU (UAS7 >= 28)",
                "framing": "uncontrolled disease activity (UAS7 >= 28)",
                "kind": "clinical_covariate",
                "literature_theme": "disease activity UAS7",
            },
        },
        "rwe_search_term": "remibrutinib chronic spontaneous urticaria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (BTKi start after antihistamine failure)",
            "persistent_180d": "Treatment persistence / sustained UAS7 control",
            "discontinued_180d": "Treatment discontinuation",
        },
        # Therapy-label fallback (FDA prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "Chronic spontaneous urticaria (CSU) in adults who remain symptomatic despite H1-antihistamine treatment",
        ],
        "limitations_fallback": "Not indicated for other forms of urticaria.",
        "boxed_warning_fallback": None,
        # disease key = "chronic spontaneous urticaria" (BRAND_DIAGNOSIS["Remibrutinib"]["desc"].lower())
        "competitor_map": {
            "chronic spontaneous urticaria": [
                "Xolair (omalizumab)",
                "Dupixent (dupilumab)",
            ],  # CSU biologics approved for CSU (omalizumab FDA 2014; dupilumab FDA 2025)
        },
    },
    "Fabhalta": {
        "moa_fallback": "complement Factor B inhibitor",
        "pivotal_endpoints_fallback": [
            "Transfusion avoidance (proportion not requiring RBC transfusion)",
            "Sustained hemoglobin stabilization (increase >= 2 g/dL without transfusion)",
            "Change from baseline in lactate dehydrogenase (LDH)",
        ],
        "disease_search_term": "paroxysmal nocturnal hemoglobinuria",
        "treatment_context": {
            "treatment_arm": {
                "label": "Treatment arm",
                "framing": "being on iptacopan",
                "kind": "drug_therapy",
                "literature_theme": None,
            },
            "treatment_initiated": {
                "label": "Treatment initiated",
                "framing": "initiating iptacopan",
                "kind": "drug_therapy",
                "literature_theme": "treatment initiation",
            },
            # #1321 Fabhalta pilot: switching off a prior C5 inhibitor is itself a
            # therapy contrast, so it is drug_therapy, not a covariate.
            "complement_inhibitor_status": {
                "label": "Prior C5-inhibitor (switch)",
                "framing": "switching from a prior C5 inhibitor (eculizumab / ravulizumab)",
                "kind": "drug_therapy",
                "literature_theme": "complement inhibitor switch",
            },
        },
        "rwe_search_term": "iptacopan paroxysmal nocturnal hemoglobinuria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (complement-inhibitor start/switch)",
            "persistent_180d": "Treatment persistence / sustained Hb stabilization",
            "discontinued_180d": "Treatment discontinuation",
        },
        # Therapy-label fallback (FDA prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "Paroxysmal nocturnal hemoglobinuria (PNH)",
            "Primary IgA nephropathy (IgAN), to reduce proteinuria",
        ],
        "limitations_fallback": None,
        "boxed_warning_fallback": (
            "Serious infections caused by encapsulated bacteria (e.g., S. pneumoniae, "
            "N. meningitidis, H. influenzae) can occur; complete or update vaccinations "
            "before initiation."
        ),
        # disease key = "paroxysmal nocturnal hemoglobinuria" (BRAND_DIAGNOSIS["Fabhalta"]["desc"].lower())
        # A second key is included for IgAN so providers can resolve either indication.
        "competitor_map": {
            "paroxysmal nocturnal hemoglobinuria": [
                "Soliris (eculizumab)",
                "Ultomiris (ravulizumab)",
                "Empaveli (pegcetacoplan)",
                "Voydeya (danicopan)",
            ],  # Complement inhibitors approved for PNH (FDA/EMA probe-confirmed)
            "primary iga nephropathy": [
                "Tarpeyo (budesonide)",
                "Filspari (sparsentan)",
            ],  # IgAN-approved therapies (FDA 2023-2024)
        },
    },
}


def _build_treatment_context(
    brand_specific: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, TreatmentContext]:
    """Merge the shared commercial levers with the brand's own therapy / covariate
    treatments into one column -> TreatmentContext map."""
    out: Dict[str, TreatmentContext] = {}
    for column, spec in _COMMERCIAL_TREATMENT_CONTEXT.items():
        out[column] = TreatmentContext(
            column=column,
            label=str(spec["label"]),
            framing=str(spec["framing"]),
            kind="commercial",
            literature_theme=spec.get("literature_theme"),
        )
    for column, spec in brand_specific.items():
        out[column] = TreatmentContext(
            column=column,
            label=str(spec["label"]),
            framing=str(spec["framing"]),
            kind=str(spec["kind"]),
            literature_theme=spec.get("literature_theme"),
        )
    return out


def _build_map() -> Dict[str, BrandClinicalProfile]:
    out: Dict[str, BrandClinicalProfile] = {}
    for brand, extra in _STATIC_ENRICHMENT.items():
        ndc = BRAND_NDC[brand]
        dx = BRAND_DIAGNOSIS[brand]
        out[brand] = BrandClinicalProfile(
            brand=brand,
            drug_name=str(ndc["drug_name"]),
            disease=str(dx["desc"]),
            # drug_class lives in clinical_codes.BRAND_DRUG_CLASS, but the
            # enrichment MoA-fallback string is the precise spec-pinned phrasing;
            # we surface the fallback as the authoritative static MoA.
            drug_class=str(extra["moa_fallback"]),
            moa_fallback=str(extra["moa_fallback"]),
            pivotal_endpoints_fallback=list(extra["pivotal_endpoints_fallback"]),  # type: ignore[call-overload]
            rwe_search_term=str(extra["rwe_search_term"]),
            rwe_seed_pmid=(str(extra["rwe_seed_pmid"]) if extra["rwe_seed_pmid"] else None),
            outcome_endpoint_map=dict(extra["outcome_endpoint_map"]),  # type: ignore[call-overload]
            # Therapy-label + competitor fields (Task 2, 2026-06-20)
            indications_fallback=list(extra.get("indications_fallback", [])),  # type: ignore[call-overload]
            limitations_fallback=(
                str(extra["limitations_fallback"]) if extra.get("limitations_fallback") else None
            ),
            boxed_warning_fallback=(
                str(extra["boxed_warning_fallback"])
                if extra.get("boxed_warning_fallback")
                else None
            ),
            competitor_map={k: list(v) for k, v in extra.get("competitor_map", {}).items()},  # type: ignore[attr-defined]
            disease_search_term=str(extra["disease_search_term"]),
            treatment_context=_build_treatment_context(extra.get("treatment_context", {})),  # type: ignore[arg-type]
            seminal_rwe=(
                dict(extra["seminal_rwe"]) if extra.get("seminal_rwe") else None  # type: ignore[call-overload]
            ),
        )
    return out


BRAND_CLINICAL_MAP: Dict[str, BrandClinicalProfile] = _build_map()


def resolve_brand_profile(brand: str) -> BrandClinicalProfile:
    """Return the clinical profile for ``brand``.

    Raises ``KeyError`` on an unsupported brand so callers fail closed rather
    than emit a wrong indication.
    """
    return BRAND_CLINICAL_MAP[brand]


def endpoint_mapping_for_outcome(brand: str, outcome: str) -> Optional[str]:
    """Map our synthetic ``outcome`` column -> the real pivotal-endpoint framing
    for ``brand``. Returns None when there is no curated mapping (honest: we do
    not fabricate a clinical claim for an outcome we have not mapped)."""
    profile = BRAND_CLINICAL_MAP.get(brand)
    if profile is None:
        return None
    return profile.outcome_endpoint_map.get(outcome)


def treatment_context_for(brand: str, treatment: Optional[str]) -> Optional[TreatmentContext]:
    """Return the curated clinical framing for ``treatment`` under ``brand``.

    ``None`` when the brand is unknown, no treatment was supplied, or the column
    has no curated framing — the caller then omits the analysis frame entirely
    rather than emitting a sentence with an invented treatment in it. Brand-distinct
    treatments (#1321 axes) resolve ONLY under their own brand, so a Fabhalta-only
    column can never be framed as a Kisqali analysis.
    """
    if not treatment:
        return None
    profile = BRAND_CLINICAL_MAP.get(brand)
    if profile is None:
        return None
    return profile.treatment_context.get(treatment)


def outcome_framing_for(outcome: str) -> str:
    """Plain-language phrase for our synthetic ``outcome`` column; the raw column
    name when we have no curated phrase (honest, never invented)."""
    return _OUTCOME_FRAMING.get(outcome, outcome)


def compose_rwe_search_term(
    profile: BrandClinicalProfile, outcome: str, treatment: Optional[str]
) -> str:
    """Compose the analysis-specific PubMed query: drug + plain-language disease +
    the outcome's theme + the treatment's theme.

    Falls back to the curated brand term when NEITHER side contributes a theme —
    a half-built query would search worse than the curated one, and a lever with
    no clinical-literature analogue (rep detailing, sampling) must not have a
    theme invented for it.
    """
    outcome_theme = _OUTCOME_LITERATURE_THEME.get(outcome)
    context = treatment_context_for(profile.brand, treatment)
    treatment_theme = context.literature_theme if context is not None else None
    if not outcome_theme and not treatment_theme:
        return profile.rwe_search_term
    parts = [profile.drug_name, profile.disease_search_term]
    if outcome_theme:
        parts.append(outcome_theme)
    if treatment_theme:
        parts.append(treatment_theme)
    parts.append("real-world")
    return " ".join(p for p in parts if p)


def analysis_framing_sentence(
    profile: BrandClinicalProfile, outcome: str, treatment: Optional[str]
) -> Optional[str]:
    """One deterministic sentence naming the analysis the panel is grounding:
    treatment -> outcome, for this drug in this disease.

    ``None`` when the treatment has no curated framing — the panel then shows the
    brand-level context it always showed, with no claim about an analysis.
    """
    context = treatment_context_for(profile.brand, treatment)
    if context is None:
        return None
    return (
        f"This analysis estimates the effect of {context.framing} on "
        f"{outcome_framing_for(outcome)} for {profile.drug_name} in {profile.disease}."
    )
