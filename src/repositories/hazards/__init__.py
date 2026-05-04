"""Adversarial synthetic hazards for the tier-0 ml_patients cohort.

This package provides pure-function DataFrame transformers that plant a
single, *known* failure mode into the cohort produced by
``src.repositories.sample_data.SampleDataGenerator.ml_patients()``. Each
hazard is a thin pd.DataFrame -> pd.DataFrame transformer; none mutate the
input.

Operating contract
==================

All hazards consume the ``ml_patients()`` schema, whose columns are:

    patient_journey_id, patient_id, brand, geographic_region, journey_status,
    journey_start_date, journey_end_date, data_quality_score, days_on_therapy,
    hcp_visits, prior_treatments, age_group, discontinuation_flag, created_at

The target is ``discontinuation_flag`` (binary {0,1}); the remaining columns
are features. A few hazards (``unmeasured_confounder``,
``positivity_violation``) reference a treatment column. Because
``ml_patients()`` does not emit a ``treatment_initiated`` column natively,
those hazards add it as a NEW column derived from the planted hazard pattern.
This is the most faithful adversarial reproduction: tier-0's downstream
agents see ``treatment_initiated`` as a regular feature and the hazard
operates through the documented surface.

Each hazard returns a NEW DataFrame and does not mutate the input. They
accept a ``seed`` for reproducibility plus hazard-specific kwargs documented
in each module.

Adversarial detection contracts
===============================

The adversarial integration suite at
``tests/integration/test_adversarial_synthetic.py`` asserts that the tier-0
detectors fire on each hazard:

* ``inject_unmeasured_confounder``  -> elevated CV ROC-AUC fold variance
  (``cv_roc_auc_std > 0.04``) — clean baseline ~0.041 on n=1500.
* ``inject_measurement_error``       -> monotonic CV ROC-AUC degradation
  across noise levels {0.1, 0.2, 0.3} of the target feature's std.
* ``inject_positivity_violation``    -> any of: SHAP shows segment as a
  near-perfect predictor, OR CV ROC-AUC std > 0.07, OR sampling-frame
  audit flags subgroup mismatch.
* ``inject_label_leakage``           -> ``state["leakage_findings"]``
  contains the post-treatment leak feature at severity in
  {high, critical} (single-feature AUC > 0.90 path).
* ``inject_sampling_frame_drift``    -> ``state["sampling_frame_audit_report"
  ]["max_drift_score"] > 0.3`` AND
  ``"sampling_frame_drift:" in str(state["blocking_issues"])``.
"""

from __future__ import annotations

from src.repositories.hazards.label_leakage import inject_label_leakage
from src.repositories.hazards.measurement_error import inject_measurement_error
from src.repositories.hazards.positivity_violation import inject_positivity_violation
from src.repositories.hazards.sampling_frame_drift import inject_sampling_frame_drift
from src.repositories.hazards.unmeasured_confounder import inject_unmeasured_confounder

__all__ = [
    "inject_label_leakage",
    "inject_measurement_error",
    "inject_positivity_violation",
    "inject_sampling_frame_drift",
    "inject_unmeasured_confounder",
]
