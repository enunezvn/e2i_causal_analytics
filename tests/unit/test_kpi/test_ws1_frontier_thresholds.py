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
  brand at the pre-union holdout sizes (3 cohorts n~850 + hcp_adoption n=250),
  so the old +-0.05 green band was statistically unreachable. The OOS-union
  eval window (2026-07-23, test+holdout: patient n~1700, hcp n=1000) roughly
  halves that noise floor (~1.056); the 0.10 band still sits above it.
* ROC-AUC 0.80 is the deliberate exception: it is recalibration-invariant,
  tracks real discrimination, and stays UNCHANGED — a brand hair-under it
  honestly reads WARNING (Fabhalta 0.7993 pre-union; Remibrutinib 0.7965 and
  Kisqali 0.7940 on the union window).

These tests pin the retuned yaml values and the intended statuses at the
2026-07-23 measured brand values (both the original holdout-only window and
the OOS-union window that superseded it), so a future edit that re-breaks
the frontier (either direction) fails loudly with this context attached.
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
        # 2026-07-23 OOS-union brand means: Remi 0.6817, Fabhalta 0.6829,
        # Kisqali 0.6760 (holdout-only window: 0.6915 / 0.6786 / 0.6782)
        assert t.evaluate(0.6817) == KPIStatus.GOOD
        assert t.evaluate(0.6829) == KPIStatus.GOOD
        assert t.evaluate(0.6760) == KPIStatus.GOOD

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
        # 2026-07-23 OOS-union brand means: Remi 0.1784, Fabhalta 0.1756,
        # Kisqali 0.1779 (holdout-only window: 0.1754 / 0.1767 / 0.1788)
        assert t.evaluate(0.1784, lower_is_better=True) == KPIStatus.GOOD
        assert t.evaluate(0.1756, lower_is_better=True) == KPIStatus.GOOD
        assert t.evaluate(0.1779, lower_is_better=True) == KPIStatus.GOOD

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
        # OOS-union window (2026-07-23): Remi 1.0878, Fabhalta 1.0131,
        # Kisqali 1.0692 — all GOOD. The pre-union Remi 1.1350 WARNING was a
        # holdout-window draw, not a stable model property: the mirror-pair's
        # slope over its full OOS pool is ~1.12 (same-size random windows of
        # that pool span ~1.0-1.24, sd 0.067 at n=826, P(>=1.236) ~= 5%), and
        # a Platt calibrator fitted on the test split is ~identity (a=1.015)
        # so no legitimate recalibration could move the old headline.
        assert t.evaluate(1.0878) == KPIStatus.GOOD
        assert t.evaluate(1.0131) == KPIStatus.GOOD
        assert t.evaluate(1.0692) == KPIStatus.GOOD
        # The pre-union headline band behavior stays pinned: a 1.1350 fold
        # (deviation 0.135) reads WARNING, beyond-0.15 reads CRITICAL.
        assert t.evaluate(1.1350) == KPIStatus.WARNING
        assert t.evaluate(1.16) == KPIStatus.CRITICAL


class TestAucTargetStaysHonest:
    def test_yaml_values_unchanged(self):
        t = _threshold("WS1-MP-001")
        assert t.target == 0.80
        assert t.critical == 0.60

    def test_hair_miss_stays_warning(self):
        t = _threshold("WS1-MP-001")
        # OOS-union window (2026-07-23): Fabhalta 0.8041 clears the bar;
        # Remibrutinib 0.7965 and Kisqali 0.7940 sit a hair under it and
        # honestly read WARNING (the bar itself stays untouched at 0.80).
        assert t.evaluate(0.8041) == KPIStatus.GOOD  # Fabhalta
        assert t.evaluate(0.7965) == KPIStatus.WARNING  # Remibrutinib
        assert t.evaluate(0.7940) == KPIStatus.WARNING  # Kisqali
