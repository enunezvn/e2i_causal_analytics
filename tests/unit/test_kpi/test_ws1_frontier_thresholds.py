"""WS1 model-performance targets sit at the DGP-aware achievable frontier.

2026-07-23 analysis (binormal simulation at each gold-standard model's
measured holdout AUC + prevalence back-solved from its confusion-matrix
metrics; per-model bootstrap slope CIs) showed three of the four yellow WS1
targets were set beyond what a PERFECT model on this DGP can achieve:

* F1: max-over-cutoffs ceiling is ~0.70-0.72 brand-mean; the models capture
  96-97% of it, yet the old 0.75 target sat ABOVE the ceiling.
* Brier: the perfectly-calibrated floor is 0.174-0.178 brand-mean (only the
  initiation cohort, AUC 0.84+, can reach the old 0.15); recalibration can
  recover <= 0.001 of the gap.
* Calibration-slope fold: E[1 + mean|s-1|] ~= 1.08 for a PERFECTLY calibrated
  brand at the current holdout sizes (3 cohorts n~850 + hcp_adoption n=250),
  so the old +-0.05 green band was statistically unreachable.
* ROC-AUC 0.80 is the deliberate exception: it is recalibration-invariant,
  tracks real discrimination, and stays UNCHANGED — a brand hair-under it
  (Fabhalta 0.7993) honestly reads WARNING.

These tests pin the retuned yaml values and the intended statuses at the
2026-07-23 measured brand values, so a future edit that re-breaks the
frontier (either direction) fails loudly with this context attached.
"""

import pytest

from src.kpi.models import KPIStatus
from src.kpi.registry import KPIRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    KPIRegistry.reset()
    yield
    KPIRegistry.reset()


def _threshold(kpi_id: str):
    kpi = KPIRegistry().get(kpi_id)
    assert kpi is not None and kpi.threshold is not None
    return kpi.threshold


class TestF1FrontierTarget:
    def test_yaml_values(self):
        t = _threshold("WS1-MP-003")
        assert t.target == 0.65
        assert t.critical == 0.45

    def test_measured_brand_values_read_good(self):
        t = _threshold("WS1-MP-003")
        # 2026-07-23 brand means: Fabhalta 0.6786, Kisqali 0.6782, Remi 0.6915
        assert t.evaluate(0.6786) == KPIStatus.GOOD
        assert t.evaluate(0.6782) == KPIStatus.GOOD
        assert t.evaluate(0.6915) == KPIStatus.GOOD

    def test_degradation_still_detected(self):
        t = _threshold("WS1-MP-003")
        assert t.evaluate(0.649) == KPIStatus.WARNING
        assert t.evaluate(0.44) == KPIStatus.CRITICAL


class TestBrierFrontierTarget:
    def test_yaml_values(self):
        t = _threshold("WS1-MP-005")
        assert t.target == 0.185
        # lower-is-better: `warning` is the yellow/red boundary
        assert t.warning == 0.25

    def test_measured_brand_values_read_good(self):
        t = _threshold("WS1-MP-005")
        # 2026-07-23 brand means: Remi 0.1754, Fabhalta 0.1767, Kisqali 0.1788
        assert t.evaluate(0.1754, lower_is_better=True) == KPIStatus.GOOD
        assert t.evaluate(0.1767, lower_is_better=True) == KPIStatus.GOOD
        assert t.evaluate(0.1788, lower_is_better=True) == KPIStatus.GOOD

    def test_degradation_still_detected(self):
        t = _threshold("WS1-MP-005")
        assert t.evaluate(0.19, lower_is_better=True) == KPIStatus.WARNING
        assert t.evaluate(0.26, lower_is_better=True) == KPIStatus.CRITICAL


class TestSlopeFoldNoiseFloorTolerance:
    def test_yaml_values(self):
        t = _threshold("WS1-MP-006")
        assert t.ideal == 1.0
        assert t.good_tolerance == 0.10
        assert t.warning_tolerance == 0.15

    def test_measured_brand_values(self):
        t = _threshold("WS1-MP-006")
        # Fabhalta 1.0526 and Kisqali 1.0792 sit BELOW the ~1.08 noise floor
        # (consistent with perfect calibration) -> GOOD. Remi 1.1350 is driven
        # by a genuine miscalibration (persistence/discontinuation mirror-pair
        # slope 1.236, bootstrap CI excluding 1) -> honestly WARNING.
        assert t.evaluate(1.0526) == KPIStatus.GOOD
        assert t.evaluate(1.0792) == KPIStatus.GOOD
        assert t.evaluate(1.1350) == KPIStatus.WARNING


class TestAucTargetStaysHonest:
    def test_yaml_values_unchanged(self):
        t = _threshold("WS1-MP-001")
        assert t.target == 0.80
        assert t.critical == 0.60

    def test_hair_miss_stays_warning(self):
        t = _threshold("WS1-MP-001")
        assert t.evaluate(0.7993) == KPIStatus.WARNING  # Fabhalta 2026-07-23
        assert t.evaluate(0.8032) == KPIStatus.GOOD  # Remibrutinib
