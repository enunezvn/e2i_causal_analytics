"""4 hand-authored DAG scenarios for the S12a golden set (plan §3.2).

Each scenario carries:

- An explicit DAG (``nx.DiGraph``) with named treatment ``T`` and outcome ``Y``.
- Per-node feature entries with snake_case feature names, production-shape
  ``derivation_pseudocode`` strings (matching the f-string at
  ``adaptive_validity_check.py:879-885``), cohort-only ``dataset_context``,
  ground-truth role (mechanically extracted via :mod:`extractor`), and
  Pearl-Lauritzen rationale.

Family A is built directly here as cohort-only entries; Family B
((T, Y)-explicit re-emissions) is generated at the golden-set assembly
stage in :mod:`golden_set`.

Realized role totals across A1-A4 (codex-verified iter-1):
ancestor=7, confounder=6, mediator=4, collider=6, descendant=8,
instrument=6 = 37 Family A entries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx

from src.ml.causal_role_dgp.extractor import extract_role


@dataclass(frozen=True)
class FeatureEntry:
    """One row of the golden set (Family A; Family B is generated downstream)."""

    node_name: str
    feature_name: str
    derivation_pseudocode: str
    dataset_context: str
    ground_truth_role: str
    rationale: str
    treatment_explicit: bool = False


@dataclass(frozen=True)
class SyntheticScenario:
    """A scenario = DAG + (T, Y) + per-node Family A entries."""

    name: str
    treatment_node: str
    outcome_node: str
    dag: nx.DiGraph = field(repr=False)
    entries: tuple[FeatureEntry, ...]


# ---------------------------------------------------------------------------
# Feature naming convention helpers
# ---------------------------------------------------------------------------


def _build_derivation(
    *,
    source: str,
    derivation_inputs: tuple[str, ...],
    aggregation: str | None,
    window_days: int | None,
    knowable_at_str: str,
) -> str:
    """Render the production-shape f-string verified live in plan §3.4."""
    return (
        f"source={source}; "
        f"derivation_inputs={list(derivation_inputs)}; "
        f"aggregation={aggregation}; "
        f"window_days={window_days}; "
        f"knowable_at={knowable_at_str}"
    )


def _cohort_context(*, scenario_name: str, target: str) -> str:
    return f"cohort={scenario_name}; target={target}; prediction_anchor=index_date"


# ---------------------------------------------------------------------------
# A1: confounder-heavy
# Edges: Z1→T, Z1→Y, Z2→T, Z2→Y, Z3→T, Z3→Y, T→Y, A1→Y, A2→Y, D1←T,
#        D2←T, V1←T, V1←Y, IV1→T
# Realized: confounder=3, ancestor=2, descendant=2, collider=1, instrument=1
# ---------------------------------------------------------------------------


def _build_a1() -> SyntheticScenario:
    name = "A1_confounder_heavy"
    T = "treatment_initiated"
    Y = "disease_progression_180d"

    edges: list[tuple[str, str]] = [
        ("Z1_baseline_severity", T),
        ("Z1_baseline_severity", Y),
        ("Z2_comorbidity_burden", T),
        ("Z2_comorbidity_burden", Y),
        ("Z3_age_at_index", T),
        ("Z3_age_at_index", Y),
        (T, Y),
        ("A1_baseline_biomarker", Y),
        ("A2_family_history", Y),
        (T, "D1_post_index_fills"),
        (T, "D2_treatment_followup_visits"),
        (T, "V1_alive_at_180d"),
        (Y, "V1_alive_at_180d"),
        ("IV1_provider_preference_score", T),
    ]
    G = nx.DiGraph(edges)

    node_specs = [
        # (node, feature_name, source, inputs, agg, win_d, knowable_at_str, rationale)
        (
            "Z1_baseline_severity",
            "baseline_severity_score_preindex",
            "lab_events",
            ("severity_score",),
            "max",
            180,
            "index_date-1d",
            "Pre-index disease severity directly drives clinician's "
            "treatment decision (Z1→T) AND independently determines disease "
            "progression risk (Z1→Y). Direct parent of both T and Y → confounder.",
        ),
        (
            "Z2_comorbidity_burden",
            "comorbidity_burden_preindex",
            "diagnosis_events",
            ("icd10cm_code",),
            "count",
            365,
            "index_date-1d",
            "Comorbidity burden modifies treatment selection (sicker patients "
            "get different regimens, Z2→T) AND raises baseline progression hazard "
            "via competing-risk pathways (Z2→Y). Direct parent of both → confounder.",
        ),
        (
            "Z3_age_at_index",
            "age_at_index",
            "demographics",
            ("date_of_birth",),
            None,
            None,
            "index_date",
            "Age affects prescriber's treatment choice (geriatric vs younger "
            "dosing, Z3→T) AND independently predicts progression risk (Z3→Y). "
            "Direct parent of both → confounder.",
        ),
        (
            "A1_baseline_biomarker",
            "baseline_inflammatory_marker_preindex",
            "lab_events",
            ("crp_value",),
            "max",
            180,
            "index_date-1d",
            "Baseline inflammation drives disease-progression risk (A1→Y) but "
            "does not directly determine treatment selection in this cohort. "
            "Parent of Y only, not T → ancestor of Y.",
        ),
        (
            "A2_family_history",
            "family_history_autoimmune",
            "demographics",
            ("family_history_flag",),
            None,
            None,
            "enrollment_date",
            "Family history is a stable pre-index trait that predicts genetic "
            "predisposition to disease progression (A2→Y) but does not directly "
            "affect the index treatment decision. Parent of Y only → ancestor.",
        ),
        (
            "D1_post_index_fills",
            "post_index_medication_fills",
            "medication_events",
            ("fill_date",),
            "count",
            180,
            "index_date+180d",
            "Post-index pharmacy fills are causally downstream of treatment "
            "initiation (T→D1) with no path back to Y → descendant. Classic "
            "leakage pattern: aggregates events AFTER prediction anchor.",
        ),
        (
            "D2_treatment_followup_visits",
            "treatment_followup_visit_count",
            "encounter_events",
            ("visit_date",),
            "count",
            180,
            "index_date+180d",
            "Follow-up visits scheduled in response to treatment initiation "
            "(T→D2). Counted in the post-index window with no causal arrow "
            "back to Y in this DAG → descendant.",
        ),
        (
            "V1_alive_at_180d",
            "alive_at_180d_observation_window",
            "demographics",
            ("death_date",),
            None,
            None,
            "index_date+180d",
            "Conditioning on survival at 180d is a Greenland-Pearl-Robins "
            "M-structure: both T (treatment toxicity affects survival) AND Y "
            "(disease progression affects survival) point INTO V1. Common "
            "descendant of T and Y → collider; M-structure under restriction.",
        ),
        (
            "IV1_provider_preference_score",
            "index_provider_preference_score",
            "claims_provider",
            ("provider_id", "prior_initiation_count"),
            "mean",
            365,
            "index_date-1d",
            "Brookhart-Schneeweiss 2007 preference-based IV. Z→T via "
            "physician prescribing pattern; no Z→Y path bypassing T (verified "
            "by removing T from DAG); no common ancestor with Y → instrument.",
        ),
    ]

    entries = tuple(
        FeatureEntry(
            node_name=node,
            feature_name=feat,
            derivation_pseudocode=_build_derivation(
                source=src,
                derivation_inputs=inp,
                aggregation=agg,
                window_days=win,
                knowable_at_str=ka,
            ),
            dataset_context=_cohort_context(scenario_name=name, target=Y),
            ground_truth_role=extract_role(node, T, Y, G),
            rationale=rat,
        )
        for node, feat, src, inp, agg, win, ka, rat in node_specs
    )
    return SyntheticScenario(name=name, treatment_node=T, outcome_node=Y, dag=G, entries=entries)


# ---------------------------------------------------------------------------
# A2: mediator-heavy
# Edges: Z1→T, Z1→Y, T→M1, M1→Y, T→M2, M2→Y, T→M3, M3→Y, T→D1, T→D2,
#        V1←T, V1←Y, A1→Y, A2→Y, IV1→T
# Realized: confounder=1, mediator=3, descendant=2, collider=1, ancestor=2, instrument=1
# ---------------------------------------------------------------------------


def _build_a2() -> SyntheticScenario:
    name = "A2_mediator_heavy"
    T = "treatment_initiated"
    Y = "clinical_response_180d"

    edges = [
        ("Z1_baseline_severity", T),
        ("Z1_baseline_severity", Y),
        (T, "M1_drug_concentration_30d"),
        ("M1_drug_concentration_30d", Y),
        (T, "M2_target_engagement_marker"),
        ("M2_target_engagement_marker", Y),
        (T, "M3_inflammatory_response_60d"),
        ("M3_inflammatory_response_60d", Y),
        (T, "D1_concomitant_med_fills"),
        (T, "D2_routine_monitoring_labs"),
        (T, "V1_complete_followup_flag"),
        (Y, "V1_complete_followup_flag"),
        ("A1_genetic_marker", Y),
        ("A2_baseline_disease_duration", Y),
        ("IV1_supply_side_index", T),
    ]
    G = nx.DiGraph(edges)

    node_specs = [
        (
            "Z1_baseline_severity",
            "baseline_severity_score_preindex",
            "lab_events",
            ("severity_score",),
            "max",
            180,
            "index_date-1d",
            "Pre-index severity affects treatment decision (Z1→T) AND clinical "
            "response (Z1→Y, sicker patients respond differently). Direct parent of both → confounder.",
        ),
        (
            "M1_drug_concentration_30d",
            "plasma_drug_concentration_30d",
            "lab_events",
            ("drug_level_value", "lab_date"),
            "mean",
            30,
            "index_date+30d",
            "Plasma drug level is the proximal pharmacologic intermediate: "
            "T→M1 (initiation drives exposure) AND M1→Y (exposure determines "
            "response). On the T→Y directed path → mediator.",
        ),
        (
            "M2_target_engagement_marker",
            "target_engagement_biomarker_60d",
            "lab_events",
            ("biomarker_value",),
            "mean",
            60,
            "index_date+60d",
            "Target engagement biomarker reflects whether the drug hit its "
            "molecular target (T→M2) which in turn drives clinical benefit "
            "(M2→Y). On the T→Y path → mediator.",
        ),
        (
            "M3_inflammatory_response_60d",
            "inflammatory_response_delta_60d",
            "lab_events",
            ("crp_value", "esr_value"),
            "mean",
            60,
            "index_date+60d",
            "Post-treatment inflammatory response is downstream of T and upstream "
            "of clinical response (T→M3→Y) → mediator.",
        ),
        (
            "D1_concomitant_med_fills",
            "concomitant_med_fills_followup",
            "medication_events",
            ("fill_date",),
            "count",
            180,
            "index_date+180d",
            "Concomitant medication fills in the post-index window are downstream "
            "of treatment initiation (T→D1) with no causal arrow to Y in this "
            "DAG → descendant.",
        ),
        (
            "D2_routine_monitoring_labs",
            "routine_monitoring_lab_count_followup",
            "lab_events",
            ("lab_date",),
            "count",
            180,
            "index_date+180d",
            "Routine monitoring labs ordered per treatment protocol (T→D2); "
            "no path to Y. Pure descendant.",
        ),
        (
            "V1_complete_followup_flag",
            "complete_followup_at_180d",
            "encounter_events",
            ("last_encounter_date",),
            None,
            None,
            "index_date+180d",
            "Conditioning on complete follow-up is a sample-inclusion collider: "
            "treatment toxicity affects dropout (T→V1) AND disease progression "
            "affects follow-up retention (Y→V1). Common descendant → collider.",
        ),
        (
            "A1_genetic_marker",
            "pharmacogenetic_marker",
            "demographics",
            ("genotype_call",),
            None,
            None,
            "enrollment_date",
            "Pharmacogenetic marker predicts clinical response intensity (A1→Y) "
            "but is not used as a prescribing input in this cohort (no A1→T). "
            "Parent of Y only → ancestor.",
        ),
        (
            "A2_baseline_disease_duration",
            "baseline_disease_duration_months",
            "diagnosis_events",
            ("first_dx_date",),
            None,
            None,
            "index_date-1d",
            "Disease duration is a stable pre-index attribute that predicts "
            "response trajectory (A2→Y) without driving the treatment decision "
            "directly. Parent of Y only → ancestor.",
        ),
        (
            "IV1_supply_side_index",
            "regional_supply_availability_index",
            "claims_provider",
            ("region_code", "formulary_tier"),
            "max",
            365,
            "index_date-1d",
            "Brookhart supply-side IV: regional formulary tier drives treatment "
            "availability (Z→T) but does not directly affect response biology "
            "(no Z→Y bypassing T; no common ancestor with Y). → instrument.",
        ),
    ]

    entries = tuple(
        FeatureEntry(
            node_name=node,
            feature_name=feat,
            derivation_pseudocode=_build_derivation(
                source=src,
                derivation_inputs=inp,
                aggregation=agg,
                window_days=win,
                knowable_at_str=ka,
            ),
            dataset_context=_cohort_context(scenario_name=name, target=Y),
            ground_truth_role=extract_role(node, T, Y, G),
            rationale=rat,
        )
        for node, feat, src, inp, agg, win, ka, rat in node_specs
    )
    return SyntheticScenario(name=name, treatment_node=T, outcome_node=Y, dag=G, entries=entries)


# ---------------------------------------------------------------------------
# A3: descendant- and collider-rich
# Edges: Z1→T, Z1→Y, T→D1, T→D2, T→D3, V1←T, V1←Y, V2←T, V2←Y, V3←T, V3←Y,
#        A1→Y, A2→Y, IV1→T
# Realized: confounder=1, descendant=3, collider=3, ancestor=2, instrument=1
# ---------------------------------------------------------------------------


def _build_a3() -> SyntheticScenario:
    name = "A3_descendant_collider_rich"
    T = "treatment_initiated"
    Y = "adverse_event_180d"

    edges = [
        ("Z1_baseline_risk_score", T),
        ("Z1_baseline_risk_score", Y),
        (T, "D1_drug_fills_followup"),
        (T, "D2_dose_titration_count"),
        (T, "D3_treatment_persistence_days"),
        (T, "V1_hospitalization_count_followup"),
        (Y, "V1_hospitalization_count_followup"),
        (T, "V2_diagnostic_workup_count"),
        (Y, "V2_diagnostic_workup_count"),
        (T, "V3_ed_visit_count_followup"),
        (Y, "V3_ed_visit_count_followup"),
        ("A1_age_at_index", Y),
        ("A2_baseline_renal_function", Y),
        ("IV1_urban_rural_code", T),
    ]
    G = nx.DiGraph(edges)

    node_specs = [
        (
            "Z1_baseline_risk_score",
            "baseline_adverse_event_risk_score",
            "lab_events",
            ("risk_panel_value",),
            "max",
            180,
            "index_date-1d",
            "Baseline AE-risk score informs both treatment intensity (Z1→T) "
            "AND independently predicts adverse-event likelihood (Z1→Y). "
            "Direct parent of both → confounder.",
        ),
        (
            "D1_drug_fills_followup",
            "drug_fill_count_followup",
            "medication_events",
            ("fill_date",),
            "count",
            180,
            "index_date+180d",
            "Post-index drug fills are downstream of treatment initiation (T→D1) "
            "with no path back to Y → descendant.",
        ),
        (
            "D2_dose_titration_count",
            "dose_titration_event_count_followup",
            "medication_events",
            ("dose_change_date",),
            "count",
            180,
            "index_date+180d",
            "Dose titrations happen only after treatment is initiated (T→D2); "
            "no causal arrow back to Y → descendant.",
        ),
        (
            "D3_treatment_persistence_days",
            "treatment_persistence_days_followup",
            "medication_events",
            ("fill_date", "supply_days"),
            "sum",
            180,
            "index_date+180d",
            "Persistence is by-construction a function of post-index dispensing "
            "(T→D3). No arrow to Y in this DAG → descendant.",
        ),
        (
            "V1_hospitalization_count_followup",
            "hospitalization_count_followup",
            "encounter_events",
            ("admit_date",),
            "count",
            180,
            "index_date+180d",
            "Confounder-collider on hospitalization: T (treatment toxicity) and "
            "Y (adverse-event severity drives admission) both point INTO V1. "
            "Common descendant → collider per Greenland-Pearl-Robins 1999.",
        ),
        (
            "V2_diagnostic_workup_count",
            "diagnostic_workup_count_followup",
            "lab_events",
            ("lab_date",),
            "count",
            180,
            "index_date+180d",
            "Workup count is driven by both protocol-mandated post-T monitoring "
            "(T→V2) AND AE-specific diagnostic chase (Y→V2). Common descendant → collider.",
        ),
        (
            "V3_ed_visit_count_followup",
            "ed_visit_count_followup",
            "encounter_events",
            ("ed_date",),
            "count",
            180,
            "index_date+180d",
            "ED visits respond both to treatment side effects (T→V3) AND to "
            "adverse events themselves (Y→V3). Common descendant → collider.",
        ),
        (
            "A1_age_at_index",
            "age_at_index",
            "demographics",
            ("date_of_birth",),
            None,
            None,
            "index_date",
            "Age affects baseline AE likelihood (older patients accumulate more "
            "events, A1→Y) but does not directly drive the treatment decision in "
            "this cohort. Parent of Y only → ancestor.",
        ),
        (
            "A2_baseline_renal_function",
            "baseline_egfr_preindex",
            "lab_events",
            ("egfr_value",),
            "min",
            180,
            "index_date-1d",
            "Baseline renal function affects drug clearance and hence AE risk "
            "(A2→Y) without being a treatment-selection input in this cohort. "
            "Parent of Y only → ancestor.",
        ),
        (
            "IV1_urban_rural_code",
            "urban_rural_commuting_area_code",
            "demographics",
            ("zip3_code",),
            None,
            None,
            "enrollment_date",
            "Geographic IV (Brookhart): RUCA reflects supply-side specialist "
            "access (IV1→T) with no direct path to AE risk (no IV1→Y bypassing T; "
            "no common ancestor with Y in DAG). → instrument.",
        ),
    ]

    entries = tuple(
        FeatureEntry(
            node_name=node,
            feature_name=feat,
            derivation_pseudocode=_build_derivation(
                source=src,
                derivation_inputs=inp,
                aggregation=agg,
                window_days=win,
                knowable_at_str=ka,
            ),
            dataset_context=_cohort_context(scenario_name=name, target=Y),
            ground_truth_role=extract_role(node, T, Y, G),
            rationale=rat,
        )
        for node, feat, src, inp, agg, win, ka, rat in node_specs
    )
    return SyntheticScenario(name=name, treatment_node=T, outcome_node=Y, dag=G, entries=entries)


# ---------------------------------------------------------------------------
# A4: instrument-rich
# Edges: IV1→T, IV2→T, IV3→T, Z1→T, Z1→Y, T→Y, T→M1, M1→Y, A1→Y, D1←T,
#        V1←T, V1←Y
# Realized: instrument=3, confounder=1, mediator=1, ancestor=1, descendant=1, collider=1
# ---------------------------------------------------------------------------


def _build_a4() -> SyntheticScenario:
    name = "A4_instrument_rich"
    T = "biologic_initiation_180d"
    Y = "hospitalization_180d"

    edges = [
        ("IV1_provider_preference_score", T),
        ("IV2_geographic_region", T),
        ("IV3_index_provider_volume", T),
        ("Z1_baseline_severity", T),
        ("Z1_baseline_severity", Y),
        (T, Y),
        (T, "M1_inflammation_decrease"),
        ("M1_inflammation_decrease", Y),
        ("A1_baseline_oxygen_dependence", Y),
        (T, "D1_post_index_lab_count"),
        (T, "V1_alive_at_180d"),
        (Y, "V1_alive_at_180d"),
    ]
    G = nx.DiGraph(edges)

    node_specs = [
        (
            "IV1_provider_preference_score",
            "index_provider_biologic_preference",
            "claims_provider",
            ("provider_id", "prior_biologic_count"),
            "mean",
            365,
            "index_date-1d",
            "Brookhart-Schneeweiss preference-based IV: provider's historical "
            "biologic prescribing rate predicts T but not hospitalization "
            "biology (no IV1→Y bypassing T; no shared ancestor). → instrument.",
        ),
        (
            "IV2_geographic_region",
            "geographic_region_four_level",
            "demographics",
            ("zip3_code",),
            None,
            None,
            "enrollment_date",
            "Brookhart-style regional IV: regional formulary heterogeneity "
            "drives biologic access (IV2→T) without affecting hospitalization "
            "risk biology (no IV2→Y, no shared ancestor with Y). → instrument.",
        ),
        (
            "IV3_index_provider_volume",
            "index_provider_biologic_volume_prior_year",
            "claims_provider",
            ("provider_id", "prior_init_count"),
            "count",
            365,
            "index_date-1d",
            "Provider-volume IV (distinct from preference-fraction IV1): "
            "absolute volume of biologic inits drives prescribing readiness "
            "(IV3→T) with no direct path to hospitalization. → instrument.",
        ),
        (
            "Z1_baseline_severity",
            "baseline_disease_severity_preindex",
            "lab_events",
            ("severity_score",),
            "max",
            180,
            "index_date-1d",
            "Pre-index severity drives both treatment selection (Z1→T) AND "
            "baseline hospitalization risk (Z1→Y). Direct parent of both → confounder.",
        ),
        (
            "M1_inflammation_decrease",
            "inflammatory_marker_delta_60d",
            "lab_events",
            ("crp_value",),
            "mean",
            60,
            "index_date+60d",
            "Post-biologic inflammation drop is the pharmacologic intermediate: "
            "T→M1 (drug reduces inflammation) and M1→Y (less inflammation → "
            "fewer hospitalizations). On the T→Y path → mediator.",
        ),
        (
            "A1_baseline_oxygen_dependence",
            "baseline_oxygen_dependence",
            "encounter_events",
            ("oxygen_rx_flag",),
            None,
            None,
            "enrollment_date",
            "Baseline oxygen dependence is a stable pre-index marker predicting "
            "hospitalization risk (A1→Y) without driving the biologic decision "
            "in this cohort. Parent of Y only → ancestor.",
        ),
        (
            "D1_post_index_lab_count",
            "post_index_routine_lab_count",
            "lab_events",
            ("lab_date",),
            "count",
            180,
            "index_date+180d",
            "Routine post-biologic monitoring labs (LFT/CBC under biologic "
            "protocol). T→D1; no arrow back to Y → descendant.",
        ),
        (
            "V1_alive_at_180d",
            "alive_at_180d_observation_flag",
            "demographics",
            ("death_date",),
            None,
            None,
            "index_date+180d",
            "Survival-at-180d is a sample-inclusion collider: biologic toxicity "
            "affects survival (T→V1) AND hospitalization frequency affects "
            "mortality (Y→V1). Common descendant → collider.",
        ),
    ]

    entries = tuple(
        FeatureEntry(
            node_name=node,
            feature_name=feat,
            derivation_pseudocode=_build_derivation(
                source=src,
                derivation_inputs=inp,
                aggregation=agg,
                window_days=win,
                knowable_at_str=ka,
            ),
            dataset_context=_cohort_context(scenario_name=name, target=Y),
            ground_truth_role=extract_role(node, T, Y, G),
            rationale=rat,
        )
        for node, feat, src, inp, agg, win, ka, rat in node_specs
    )
    return SyntheticScenario(name=name, treatment_node=T, outcome_node=Y, dag=G, entries=entries)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SCENARIO_BUILDERS: dict[str, Callable[[], SyntheticScenario]] = {
    "A1_confounder_heavy": _build_a1,
    "A2_mediator_heavy": _build_a2,
    "A3_descendant_collider_rich": _build_a3,
    "A4_instrument_rich": _build_a4,
}

SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIO_BUILDERS.keys())


def build_scenario(name: str) -> SyntheticScenario:
    """Build a scenario by name. Raises KeyError if unknown."""
    if name not in SCENARIO_BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; available: {SCENARIO_NAMES}")
    return SCENARIO_BUILDERS[name]()
