"""Task 03.5 — GroundTruthStore.to_json_file persists the ground-truth sidecar.

Shard 11 reads TRUE_ATE/CATE from data/synthetic/ground_truth_<run>.json
(INDEX §CANONICAL v1.1) rather than filtering ml_predictions on a non-existent
cohort/brand column.
"""

import json

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.ground_truth.causal_effects import (
    GroundTruthEffect,
    GroundTruthStore,
)


def test_to_json_file_writes_all_effects(tmp_path):
    store = GroundTruthStore()
    store.store(
        GroundTruthEffect(
            brand=Brand.KISQALI,
            dgp_type=DGPType.HETEROGENEOUS,
            true_ate=0.2368,
            tolerance=0.15,
            confounders=["disease_severity", "academic_hcp"],
            treatment_variable="treatment_arm",
            outcome_variable="treatment_initiated",
            cate_by_segment={
                "high_severity": 0.376,
                "medium_severity": 0.215,
                "low_severity": 0.161,
            },
            n_samples=3000,
        )
    )
    store.store(
        GroundTruthEffect(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.HETEROGENEOUS,
            true_ate=0.1719,
            tolerance=0.15,
            confounders=["disease_severity", "academic_hcp"],
            treatment_variable="treatment_arm",
            outcome_variable="treatment_initiated",
            cate_by_segment={
                "high_severity": 0.277,
                "medium_severity": 0.149,
                "low_severity": 0.117,
            },
            n_samples=3000,
        )
    )

    out = tmp_path / "nested" / "ground_truth_testrun.json"
    store.to_json_file(str(out))

    assert out.exists()  # parent dir auto-created
    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) == 2
    brands = {e["brand"] for e in payload}
    assert brands == {"Kisqali", "Remibrutinib"}
    kisq = next(e for e in payload if e["brand"] == "Kisqali")
    assert kisq["true_ate"] == 0.2368
    assert kisq["cate_by_segment"]["high_severity"] > kisq["cate_by_segment"]["low_severity"]
    assert kisq["dgp_type"] == "heterogeneous"
