"""Shard 08 T4 — per-HCP-CATE -> resource_optimizer allocation_targets builder.

The builder maps each HCP's Shard-03 heterogeneous treatment effect
(``cate_estimate``) to a NON-NEGATIVE response coefficient that VARIES across
HCPs (so the solver can prefer high-CATE HCPs), attaches a positive
``current_allocation``, and sets a BINDING budget = current total so the solver
must REALLOCATE rather than expand spend.

The realized Shard-06 per-HCP CATE artifact carries cols [hcp_id, cate_estimate,
is_synthetic] only — there is NO current_spend column anywhere in the synthetic
pipeline (verified against hcp_adoption_artifact.write_per_hcp_cate_artifact and
load_synthetic_data.write_cohort_frames). So the builder must derive a
deterministic, is_synthetic-stamped equal baseline current_allocation when the
frame has no current_spend, and use current_spend when a test supplies it.
"""

import pandas as pd

from src.ml.synthetic.artifacts.allocation_builder import (
    build_allocation_targets,
    targets_from_cate_frame,
)


def test_targets_carry_variance_positive_spend_nonneg_response():
    cate = pd.DataFrame(
        {
            "hcp_id": ["h1", "h2", "h3", "h4"],
            "cate_estimate": [1.2, -0.4, 0.05, 0.8],  # mixed sign — must stay non-negative
            "current_spend": [10000.0, 8000.0, 5000.0, 12000.0],
            "is_synthetic": [True, True, True, True],
        }
    )
    targets, budget = targets_from_cate_frame(cate)
    responses = [t["expected_response"] for t in targets]
    assert all(r >= 0 for r in responses), "expected_response must be non-negative"
    assert len(set(responses)) > 1, "expected_response must VARY across HCPs"
    assert all(t["current_allocation"] > 0 for t in targets)
    # highest-CATE HCP must have the highest response coefficient
    top = max(targets, key=lambda t: t["expected_response"])
    assert top["entity_id"] == "h1"
    assert budget == sum(cate["current_spend"]), "budget binds to current total"
    assert all(t.get("is_synthetic") is True for t in targets)


def test_targets_synthesize_equal_baseline_when_no_spend_column():
    """The realized Shard-06 artifact has no current_spend; the builder must still
    produce positive, equal current_allocation (the strongest reallocation test)."""
    cate = pd.DataFrame(
        {
            "hcp_id": ["h1", "h2", "h3"],
            "cate_estimate": [0.9, 0.1, 0.5],
            "is_synthetic": [True, True, True],
        }
    )
    targets, budget = targets_from_cate_frame(cate)
    allocations = [t["current_allocation"] for t in targets]
    assert all(a > 0 for a in allocations), "must synthesize a positive baseline"
    assert len(set(allocations)) == 1, "equal baseline across HCPs (binding reallocation)"
    assert budget == sum(allocations) and budget > 0
    responses = [t["expected_response"] for t in targets]
    assert len(set(responses)) > 1, "response still varies with CATE"
    assert all(t.get("is_synthetic") is True for t in targets)


def test_build_allocation_targets_unresolved_brand_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "src.ml.synthetic.artifacts.allocation_builder._load_cate_spend_frame",
        lambda brand: None,
    )
    targets, budget = build_allocation_targets(brand=None)
    assert targets == [] and budget == 0.0


def test_build_allocation_targets_missing_artifact_fails_closed(tmp_path, monkeypatch):
    """A real call with a brand but no artifact on disk -> ([], 0.0), no fabrication."""
    monkeypatch.setattr("src.ml.synthetic.artifacts.allocation_builder._CATE_DIR", tmp_path)
    targets, budget = build_allocation_targets(brand="Kisqali")
    assert targets == [] and budget == 0.0


def test_build_allocation_targets_reads_real_artifact_shape(tmp_path, monkeypatch):
    """End-to-end through _load_cate_spend_frame on a parquet with the EXACT realized
    Shard-06 columns [hcp_id, cate_estimate, is_synthetic] (no current_spend)."""
    monkeypatch.setattr("src.ml.synthetic.artifacts.allocation_builder._CATE_DIR", tmp_path)
    pd.DataFrame(
        {
            "hcp_id": ["ohcp_000001", "ohcp_000002", "ohcp_000003"],
            "cate_estimate": [0.30, 0.05, 0.18],
            "is_synthetic": [True, True, True],
        }
    ).to_parquet(tmp_path / "per_hcp_cate_hcp_adoption_Kisqali.parquet", index=False)
    targets, budget = build_allocation_targets(brand="Kisqali")
    assert len(targets) == 3
    assert budget > 0
    assert all(t["current_allocation"] > 0 for t in targets)
    assert {t["entity_id"] for t in targets} == {"ohcp_000001", "ohcp_000002", "ohcp_000003"}
    # highest-CATE HCP must carry the top response coefficient
    top = max(targets, key=lambda t: t["expected_response"])
    assert top["entity_id"] == "ohcp_000001"
