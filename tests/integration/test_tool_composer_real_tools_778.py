"""Integration proof for issue #778: refutation_runner runs the REAL DoWhy suite.

Unlike the fail-closed unit tests, this exercises the full reuse path: a real
DataFrame with a recoverable linear treatment->outcome signal flows through
``refutation_runner`` -> ``DoWhyExecutor(run_refutation=True)`` ->
``RefutationRunner.run_all_tests`` (the exact R6-F1 / #740 machinery), producing
a genuine refutation suite with a ``gate_decision``. No mocking of the estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.tool_composer import tool_registrations as tr


def _linear_cohort(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    confounder = rng.normal(0.0, 1.0, n)
    treatment = (0.5 * confounder + rng.normal(0.0, 1.0, n) > 0).astype(int)
    # Linear outcome -> DoWhy linear_regression yields a standard error, so the
    # refutation gate actually runs (rather than honestly skipping for want of a CI).
    outcome = 2.0 * treatment + 1.5 * confounder + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "patient_id": [f"PJ{i:04d}" for i in range(n)],
            "high_engagement": treatment,
            "converted": outcome,
            "disease_severity": confounder,
        }
    )


def test_refutation_runner_runs_real_dowhy_suite():
    df = _linear_cohort(400)
    result = tr.refutation_runner(
        estimate_id="est-real-778",
        estimation_data=df,
        treatment="high_engagement",
        outcome="converted",
        confounders=["disease_severity"],
    )

    # Provenance echoed.
    assert result["estimate_id"] == "est-real-778"
    assert result["n_samples"] == 400

    # A REAL refutation suite was produced (not fabricated).
    suite = result["refutation_results"]
    assert isinstance(suite, dict)
    assert result["gate_decision"] in {"proceed", "review", "block"}
    assert suite["gate_decision"] == result["gate_decision"]
    assert isinstance(suite["individual_tests"], dict)
    # The canonical DoWhy refutation tests are present.
    assert "placebo_treatment" in suite["individual_tests"]
    assert "random_common_cause" in suite["individual_tests"]
    assert isinstance(result["total_tests"], int) and result["total_tests"] >= 1


def test_cohort_builder_end_to_end_real_filtering():
    # Realistic patient_journeys-shaped frame injected as estimation_data; real
    # pandas filtering produces real eligible IDs (no fabricated placeholders).
    df = pd.DataFrame(
        {
            "patient_id": [f"PJ{i:04d}" for i in range(10)],
            "brand": ["Kisqali"] * 10,
            "geographic_region": ["northeast"] * 10,
            "age_at_diagnosis": list(range(45, 55)),
            "treatment_initiated": [i % 2 for i in range(10)],
        }
    )
    out = tr.cohort_builder(
        brand="Kisqali",
        region="Northeast",
        inclusion_criteria=["age_at_diagnosis >= 50"],
        exclusion_criteria=["treatment_initiated == 1"],
        estimation_data=df,
    )
    # age >= 50 -> ids PJ0005..PJ0009 (5), then drop treatment_initiated==1
    # (odd indices 5,7,9) -> PJ0006, PJ0008 remain.
    assert out.total_evaluated == 10
    assert out.eligible_patient_ids == ["PJ0006", "PJ0008"]
    # Anti-mock regression: the audit's fabricated placeholder IDs never appear.
    assert "P001" not in out.eligible_patient_ids
    assert "P002" not in out.eligible_patient_ids
    assert "P003" not in out.eligible_patient_ids
