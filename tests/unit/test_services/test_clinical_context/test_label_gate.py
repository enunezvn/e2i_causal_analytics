"""Label-gate evaluator: deterministic verdict for a segment vs an indicated
population. Pure logic (no network/DB). Verdict trichotomy + `mixed`:

- off_label    : the segment provably VIOLATES a LABEL-EVIDENCED inclusion criterion
                 (or matches a label-evidenced exclusion). Hard de-prioritization.
- mixed        : a banded segment STRADDLES a label-evidenced threshold.
- indeterminate: the segment bears on no criterion, OR only violates a
                 config-unconfirmed criterion (surfaced for review, never a silent
                 hardcoded off-label flag — codex HIGH#1).
- on_label     : every intersecting criterion is satisfied.
"""

import pytest

from src.agents.cohort_constructor.types import Criterion, CriterionType, Operator
from src.services.clinical_context.label_gate import (
    GateCriterion,
    IndicatedPopulation,
    SegmentDescriptor,
    evaluate_segment,
)


def _pop(*gate_criteria, brand="Remibrutinib", indication="csu", source="openfda_evidenced"):
    return IndicatedPopulation(
        brand=brand, indication=indication, criteria=list(gate_criteria), source=source
    )


def _gc(field, op, value, *, evidenced, ctype=CriterionType.INCLUSION):
    return GateCriterion(
        criterion=Criterion(field=field, operator=op, value=value, criterion_type=ctype,
                            clinical_rationale=f"{field} per label"),
        label_evidenced=evidenced,
        label_evidence=("…label snippet…" if evidenced else None),
    )


@pytest.mark.unit
def test_treatment_naive_csu_is_off_label_when_label_evidenced():
    # Remibrutinib: "despite H1 antihistamine treatment" IS in the live label ->
    # prior_antihistamine_therapy==True is label-evidenced. A treatment-naive
    # (False) segment provably violates it -> HARD off_label.
    pop = _pop(_gc("prior_antihistamine_therapy", Operator.EQUAL, True, evidenced=True))
    v = evaluate_segment([SegmentDescriptor(field="prior_antihistamine_therapy", value=False)], pop)
    assert v.verdict == "off_label"
    assert "prior_antihistamine_therapy" in v.failed_criteria
    assert v.confirmed_by_label is True


@pytest.mark.unit
def test_antihistamine_experienced_csu_is_on_label():
    pop = _pop(_gc("prior_antihistamine_therapy", Operator.EQUAL, True, evidenced=True))
    v = evaluate_segment([SegmentDescriptor(field="prior_antihistamine_therapy", value=True)], pop)
    assert v.verdict == "on_label"


@pytest.mark.unit
def test_low_uas7_band_is_indeterminate_when_criterion_unconfirmed():
    # UAS7>=16 is NOT stated numerically in the indication -> config_unconfirmed.
    # A low-severity band must NOT be hard off_label; it surfaces as indeterminate.
    pop = _pop(_gc("urticaria_severity_uas7", Operator.GREATER_EQUAL, 16, evidenced=False))
    v = evaluate_segment([SegmentDescriptor(field="urticaria_severity_uas7", low=5, high=15)], pop)
    assert v.verdict == "indeterminate"
    assert v.confirmed_by_label is False


@pytest.mark.unit
def test_hr_negative_segment_is_off_label_for_kisqali():
    # Kisqali: "HR-positive" IS in the label -> hr_status==positive is evidenced.
    pop = _pop(_gc("hr_status", Operator.EQUAL, "positive", evidenced=True),
               brand="Kisqali", indication="hr_her2_bc")
    v = evaluate_segment([SegmentDescriptor(field="hr_status", value="negative")], pop)
    assert v.verdict == "off_label"


@pytest.mark.unit
def test_region_segment_not_a_criterion_is_indeterminate():
    pop = _pop(_gc("prior_antihistamine_therapy", Operator.EQUAL, True, evidenced=True))
    v = evaluate_segment([SegmentDescriptor(field="region", value="Northeast")], pop)
    assert v.verdict == "indeterminate"


@pytest.mark.unit
def test_band_straddling_evidenced_threshold_is_mixed():
    # An evidenced continuous threshold (proteinuria>=1.5, IgAN "UPCR >= 1.5 g/g")
    # with a band [1.0, 2.0] straddling 1.5 -> mixed.
    pop = _pop(_gc("proteinuria_g_day", Operator.GREATER_EQUAL, 1.5, evidenced=True),
               brand="Fabhalta", indication="igan")
    v = evaluate_segment([SegmentDescriptor(field="proteinuria_g_day", low=1.0, high=2.0)], pop)
    assert v.verdict == "mixed"


@pytest.mark.unit
def test_band_fully_below_evidenced_threshold_is_off_label():
    pop = _pop(_gc("proteinuria_g_day", Operator.GREATER_EQUAL, 1.5, evidenced=True),
               brand="Fabhalta", indication="igan")
    v = evaluate_segment([SegmentDescriptor(field="proteinuria_g_day", low=0.2, high=1.0)], pop)
    assert v.verdict == "off_label"


@pytest.mark.unit
def test_exclusion_criterion_match_is_off_label():
    pop = _pop(_gc("active_serious_infection", Operator.EQUAL, True, evidenced=True,
                   ctype=CriterionType.EXCLUSION), brand="Fabhalta", indication="pnh")
    v = evaluate_segment([SegmentDescriptor(field="active_serious_infection", value=True)], pop)
    assert v.verdict == "off_label"


@pytest.mark.unit
def test_unavailable_population_is_indeterminate():
    pop = _pop(source="unavailable")
    v = evaluate_segment([SegmentDescriptor(field="prior_antihistamine_therapy", value=False)], pop)
    assert v.verdict == "indeterminate"


@pytest.mark.unit
def test_multi_field_segment_any_off_label_wins():
    pop = _pop(_gc("prior_antihistamine_therapy", Operator.EQUAL, True, evidenced=True),
               _gc("age_at_diagnosis", Operator.GREATER_EQUAL, 18, evidenced=True))
    v = evaluate_segment(
        [
            SegmentDescriptor(field="age_at_diagnosis", value=40),  # on
            SegmentDescriptor(field="prior_antihistamine_therapy", value=False),  # off
        ],
        pop,
    )
    assert v.verdict == "off_label"
