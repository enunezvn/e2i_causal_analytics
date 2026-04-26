"""Unit tests for the ``audit_sampling_frame`` node (Block 6B, finding #15).

The node performs an *advisory* drift audit comparing ``train_df`` to a
``deployment_reference`` declared on ``scope_spec``. These tests lock in:

* the no-reference branch emits an advisory ``status="no_reference_provided"``
  entry without populating ``blocking_issues``;
* identical distributions report ``drift_detected=False``;
* shifted numeric / categorical distributions surface in ``columns_with_drift``;
* the report never adds to ``blocking_issues`` (advisory-only contract);
* the report is JSON-serialisable (no numpy types, no DataFrames).
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.sampling_frame_audit import (
    audit_sampling_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_train_df() -> pd.DataFrame:
    """A 200-row train_df with one numeric and one categorical column."""
    rng = np.random.default_rng(seed=42)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.normal(loc=50.0, scale=10.0, size=n),
            "region": rng.choice(
                ["northeast", "south", "midwest", "west"],
                size=n,
                p=[0.25, 0.25, 0.25, 0.25],
            ),
        }
    )


def _identical_reference(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Build a deployment_reference whose distributions exactly match train."""
    return {
        "distributions": {
            "age": {
                "mean": float(train_df["age"].mean()),
                "std": float(train_df["age"].std(ddof=1)),
                "quantiles": {
                    "q25": float(train_df["age"].quantile(0.25)),
                    "q50": float(train_df["age"].quantile(0.50)),
                    "q75": float(train_df["age"].quantile(0.75)),
                },
            },
            "region": {
                "categorical_freq": {
                    str(k): float(v)
                    for k, v in (
                        train_df["region"].value_counts(normalize=True).items()
                    )
                }
            },
        },
        "n_reference_samples": int(len(train_df)),
    }


# ---------------------------------------------------------------------------
# 1. No-reference branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_skipped_when_no_deployment_reference(base_train_df):
    """No ``deployment_reference`` → advisory pass-through, no blocking."""
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_no_ref",
        "scope_spec": {},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    assert report["status"] == "no_reference_provided"
    assert report["drift_detected"] is False
    assert report["columns_checked"] == 0
    assert report["columns_with_drift"] == []
    assert report["per_column"] == {}
    # Advisory contract: blocking_issues must be untouched.
    assert "blocking_issues" not in result


# ---------------------------------------------------------------------------
# 2. Identical distributions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_detects_no_drift_on_identical_distributions(
    base_train_df,
):
    """Self-comparison must report ``drift_detected=False`` for every column."""
    reference = _identical_reference(base_train_df)
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_identical",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    assert report["status"] == "no_drift"
    assert report["drift_detected"] is False
    assert report["columns_checked"] == 2
    assert report["columns_with_drift"] == []

    # Every per-column entry should have been "checked" with drift_flagged=False.
    for col, entry in report["per_column"].items():
        assert entry["status"] == "checked", col
        assert entry["drift_flagged"] is False, col

    # Cohen's d on identical data is 0 (or extremely close to it).
    age_entry = report["per_column"]["age"]
    assert age_entry["metric"] == "standardized_mean_diff"
    assert age_entry["metric_value"] == pytest.approx(0.0, abs=1e-9)

    # JS divergence on identical frequencies is 0.
    region_entry = report["per_column"]["region"]
    assert region_entry["metric"] == "jensen_shannon_divergence"
    assert region_entry["metric_value"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Numeric drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_flags_numeric_drift(base_train_df):
    """Shifted numeric mean/std vs reference flags drift on the column."""
    # Reference is centred far from train: train ~ N(50, 10), ref mean=80.
    reference = {
        "distributions": {
            "age": {
                "mean": 80.0,
                "std": 10.0,
                "quantiles": {"q25": 73.0, "q50": 80.0, "q75": 87.0},
            },
        },
    }
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_numeric_drift",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    assert report["status"] == "drift_detected"
    assert report["drift_detected"] is True
    assert "age" in report["columns_with_drift"]

    age_entry = report["per_column"]["age"]
    assert age_entry["status"] == "checked"
    assert age_entry["type"] == "numeric"
    assert age_entry["metric"] == "standardized_mean_diff"
    # |50 - 80| / pooled_std(~10) ≈ 3.0, well above the 0.5 default.
    assert age_entry["metric_value"] > 0.5
    assert age_entry["drift_flagged"] is True
    # Quantile diffs should be populated when reference quantiles are present.
    assert set(age_entry["quantile_diffs"].keys()) == {"q25", "q50", "q75"}


# ---------------------------------------------------------------------------
# 4. Categorical drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_flags_categorical_drift(base_train_df):
    """A categorical reference with very different frequencies flags drift."""
    # train_df has ~25/25/25/25 across regions; reference is 95% northeast.
    reference = {
        "distributions": {
            "region": {
                "categorical_freq": {
                    "northeast": 0.95,
                    "south": 0.02,
                    "midwest": 0.02,
                    "west": 0.01,
                },
            },
        },
    }
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_categorical_drift",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    assert report["drift_detected"] is True
    assert "region" in report["columns_with_drift"]

    region_entry = report["per_column"]["region"]
    assert region_entry["status"] == "checked"
    assert region_entry["type"] == "categorical"
    assert region_entry["metric"] == "jensen_shannon_divergence"
    assert region_entry["metric_value"] > 0.2  # default threshold
    assert region_entry["drift_flagged"] is True
    # Frequencies should be normalised to sum ≈ 1 (small float tolerance).
    assert sum(region_entry["train_freq"].values()) == pytest.approx(1.0)
    assert sum(region_entry["reference_freq"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Advisory contract — never blocks the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_failure_does_not_block_pipeline(base_train_df):
    """Drift detection MUST NOT add anything to ``blocking_issues``."""
    reference = {
        "distributions": {
            "age": {"mean": 200.0, "std": 1.0},  # extreme drift
            "region": {
                "categorical_freq": {"northeast": 1.0},  # extreme drift
            },
        },
    }
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_advisory",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    assert report["drift_detected"] is True
    # The advisory contract: the node returns ONLY the report key, never
    # touches blocking_issues, even when extreme drift is observed.
    assert set(result.keys()) == {"sampling_frame_audit_report"}
    assert "blocking_issues" not in result


# ---------------------------------------------------------------------------
# 6. JSON-serialisability of the report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_report_serializable(base_train_df):
    """Reports must round-trip through ``json.dumps`` without converters."""
    reference = _identical_reference(base_train_df)
    state: Dict[str, Any] = {
        "experiment_id": "exp_audit_json_safe",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)
    report = result["sampling_frame_audit_report"]

    # No numpy types, no DataFrames — must serialise with the default encoder.
    serialised = json.dumps(report)
    # Round-trip back to verify structural integrity.
    decoded = json.loads(serialised)
    assert decoded["columns_checked"] == report["columns_checked"]
    assert decoded["status"] == report["status"]
    # Known-bad types should not appear anywhere in the serialised payload.
    assert "numpy" not in serialised
    assert "<class" not in serialised
