"""LabelCriteriaProvider: derive an indication-scoped IndicatedPopulation by
confirming the reviewed cohort_constructor candidate criteria against the REAL
OpenFDA label text. Uses the captured real-label fixtures + the REAL extraction
methods (faithful — not mocked behavior); a separate faithful test hits live OpenFDA.

Expected evidenced/unconfirmed tags (from the real label text):
- Remibrutinib: age, diagnosis, prior_antihistamine EVIDENCED; UAS7 unconfirmed
  (label says "symptomatic despite H1 antihistamine", never a UAS7 number).
- Kisqali: age, diagnosis, hr_status, her2_status, disease_stage EVIDENCED; ECOG unconfirmed.
- Fabhalta (PNH): age, diagnosis EVIDENCED; ldh_ratio, complement_inhibitor unconfirmed.
"""

import json
from pathlib import Path

import pytest

from src.services.clinical_context.clients import _OpenFDAClient
from src.services.clinical_context.label_criteria_provider import LabelCriteriaProvider

_FIX = Path(__file__).resolve().parents[3] / "fixtures" / "openfda_labels"


class _FixtureClient:
    """Returns a captured real label for fetch_label; delegates the extraction
    helpers to the REAL _OpenFDAClient (faithful extraction on real data)."""

    def __init__(self, drug_name: str):
        self._label = json.loads((_FIX / f"{drug_name}.json").read_text())
        self._real = _OpenFDAClient()

    def fetch_label(self, drug_name):
        return self._label

    def approved_indications(self, label):
        return self._real.approved_indications(label)

    def limitations_of_use(self, label):
        return self._real.limitations_of_use(label)

    def boxed_warning(self, label):
        return self._real.boxed_warning(label)


def _derive(brand, drug_name, indication=None):
    provider = LabelCriteriaProvider(openfda_client=_FixtureClient(drug_name))
    return provider.derive(brand, indication=indication)


def _by_field(pop):
    return {gc.criterion.field: gc for gc in pop.criteria}


@pytest.mark.unit
def test_remibrutinib_prior_antihistamine_is_label_evidenced():
    pop = _derive("Remibrutinib", "remibrutinib")
    assert pop.source == "openfda_evidenced"
    assert pop.indication == "csu"
    f = _by_field(pop)
    assert f["prior_antihistamine_therapy"].label_evidenced is True
    assert f["prior_antihistamine_therapy"].label_evidence  # snippet captured
    assert f["age_at_diagnosis"].label_evidenced is True
    assert f["diagnosis_code"].label_evidenced is True
    # UAS7 threshold is NOT stated in the indication -> unconfirmed (honest).
    assert f["urticaria_severity_uas7"].label_evidenced is False


@pytest.mark.unit
def test_kisqali_hr_her2_stage_are_label_evidenced():
    pop = _derive("Kisqali", "ribociclib")
    assert pop.source == "openfda_evidenced"
    f = _by_field(pop)
    assert f["hr_status"].label_evidenced is True
    assert f["her2_status"].label_evidenced is True
    assert f["disease_stage"].label_evidenced is True
    # ECOG is not in the indication text -> unconfirmed.
    assert f["ecog_performance_status"].label_evidenced is False


@pytest.mark.unit
def test_fabhalta_pnh_indication_scoped_not_defaulted_silently():
    # codex HIGH#2: indication must be explicit/resolved, not a silent default.
    pop = _derive("Fabhalta", "iptacopan", indication="pnh")
    assert pop.indication == "pnh"
    f = _by_field(pop)
    assert f["age_at_diagnosis"].label_evidenced is True
    assert f["diagnosis_code"].label_evidenced is True
    # LDH threshold + complement-inhibitor status are not stated in the indication.
    assert f["ldh_ratio"].label_evidenced is False
    assert f["complement_inhibitor_status"].label_evidenced is False


@pytest.mark.unit
def test_unavailable_label_yields_unavailable_source_failopen():
    class _DeadClient:
        def fetch_label(self, drug_name):
            return None

    provider = LabelCriteriaProvider(openfda_client=_DeadClient())
    pop = provider.derive("Remibrutinib")
    assert pop.source == "unavailable"
    # criteria still present (from the reviewed config) but all unconfirmed ->
    # the gate will return indeterminate (no hard flag without label support).
    assert pop.criteria
    assert all(gc.label_evidenced is False for gc in pop.criteria)
