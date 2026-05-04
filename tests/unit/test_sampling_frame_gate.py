"""Unit tests for the sampling-frame audit blocking gate (Phase-1 Task 1.3).

The audit was promoted from advisory to blocking: when the worst per-column
drift score exceeds ``sampling_frame_max_drift`` (default 0.3, overridable
via ``scope_spec["sampling_frame_max_drift"]``), the audit node:

* still writes the full report to ``state["sampling_frame_audit_report"]``;
* appends a single descriptive ``"sampling_frame_drift: ..."`` entry to
  ``state["blocking_issues"]``;
* mirrors the structured detail (``kind``, ``severity``, ``divergence``,
  ``threshold``, ``message``) into ``sampling_frame_audit_report["blocking_detail"]``.

These tests cover three scenarios in isolation (the audit node only — no
LangGraph spin-up):

1. **No divergence**: identical distributions → no ``blocking_issues`` entry.
2. **Above threshold**: large numeric shift → exactly one
   ``"sampling_frame_drift"`` entry in ``state["blocking_issues"]``.
3. **Custom threshold**: ``scope_spec["sampling_frame_max_drift"]=0.5`` with
   a drift of ~0.4 → no entry (custom threshold overrides the 0.3 default).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.sampling_frame_audit import (
    DEFAULT_SAMPLING_FRAME_MAX_DRIFT,
    SAMPLING_FRAME_DRIFT_BLOCKING_KIND,
    audit_sampling_frame,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal in-memory DataFrames; no full pipeline.
# ---------------------------------------------------------------------------


@pytest.fixture
def base_train_df() -> pd.DataFrame:
    """A 200-row train_df with one numeric column ``age`` ~ N(50, 10)."""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame({"age": rng.normal(loc=50.0, scale=10.0, size=200)})


def _identical_reference(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Build a deployment_reference whose age stats exactly match train."""
    return {
        "distributions": {
            "age": {
                "mean": float(train_df["age"].mean()),
                "std": float(train_df["age"].std(ddof=1)),
            },
        },
    }


# ---------------------------------------------------------------------------
# Sanity check: the module-level default is what the brief promised (0.3).
# ---------------------------------------------------------------------------


def test_default_blocking_threshold_is_zero_point_three():
    """The Phase-1 Task 1.3 spec pins the default at 0.3."""
    assert DEFAULT_SAMPLING_FRAME_MAX_DRIFT == 0.3


# ---------------------------------------------------------------------------
# Scenario 1: No divergence (identical distributions) → no blocking entry.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_blocking_entry_when_drift_below_threshold(base_train_df):
    """Identical reference → ``max_drift_score=0`` → no blocking entry.

    Verifies the negative case: the audit must NOT touch ``blocking_issues``
    when the divergence is below the configured threshold (here, identical
    distributions yield SMD ≈ 0, well below the 0.3 default).
    """
    state: Dict[str, Any] = {
        "experiment_id": "exp_no_drift",
        "scope_spec": {"deployment_reference": _identical_reference(base_train_df)},
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    # Audit ran and saw no drift.
    assert report["status"] == "no_drift"
    assert report["drift_detected"] is False
    # max_drift_score should be a finite float at or near 0.
    assert report["max_drift_score"] == pytest.approx(0.0, abs=1e-9)
    # No blocking_issues update — the audit only returns the report key.
    assert "blocking_issues" not in result
    # And no blocking_detail mirrored into the report.
    assert "blocking_detail" not in report


# ---------------------------------------------------------------------------
# Scenario 2: Above threshold → exactly one blocking entry.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_blocking_entry_when_drift_above_threshold(base_train_df):
    """Large numeric shift → single ``sampling_frame_drift`` entry appended.

    Train ``age`` ~ N(50, 10); reference centred at 80 → SMD ≈ 3.0, well
    above the 0.3 default blocking threshold. The audit must:
      * append exactly ONE entry to ``blocking_issues``;
      * the entry must start with the ``sampling_frame_drift:`` prefix
        (the stable identifier callers can grep for);
      * mirror structured detail into ``report["blocking_detail"]`` with
        ``kind="sampling_frame_drift"`` and ``severity="high"``;
      * preserve any pre-existing entries in ``blocking_issues``.
    """
    reference = {
        "distributions": {
            "age": {"mean": 80.0, "std": 10.0},
        },
    }
    pre_existing = ["unrelated_pre_existing_entry"]
    state: Dict[str, Any] = {
        "experiment_id": "exp_above_threshold",
        "scope_spec": {"deployment_reference": reference},
        "train_df": base_train_df,
        "blocking_issues": list(pre_existing),
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]
    # Drift was detected and exceeds the blocking threshold.
    assert report["drift_detected"] is True
    assert report["max_drift_score"] is not None
    assert report["max_drift_score"] > DEFAULT_SAMPLING_FRAME_MAX_DRIFT

    # Exactly one new entry appended (the audit's), pre-existing preserved.
    assert "blocking_issues" in result
    blocking = result["blocking_issues"]
    assert len(blocking) == len(pre_existing) + 1
    assert blocking[: len(pre_existing)] == pre_existing
    new_entry = blocking[-1]
    assert new_entry.startswith(f"{SAMPLING_FRAME_DRIFT_BLOCKING_KIND}:"), (
        f"Expected entry to start with '{SAMPLING_FRAME_DRIFT_BLOCKING_KIND}:', got {new_entry!r}"
    )

    # Structured detail mirrored into the report.
    detail = report["blocking_detail"]
    assert detail["kind"] == SAMPLING_FRAME_DRIFT_BLOCKING_KIND
    assert detail["severity"] == "high"
    assert detail["threshold"] == pytest.approx(DEFAULT_SAMPLING_FRAME_MAX_DRIFT)
    assert detail["divergence"] == report["max_drift_score"]
    assert detail["worst_column"] == "age"
    assert "age" in detail["columns_with_drift"]


# ---------------------------------------------------------------------------
# Scenario 3: Custom threshold (0.5) overrides default → no blocking at SMD ~0.4.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_threshold_overrides_default_and_suppresses_blocking(
    base_train_df,
):
    """``scope_spec["sampling_frame_max_drift"]=0.5`` lets SMD ≈ 0.4 pass.

    Constructs a reference where the SMD is ~0.4 (between the 0.3 default
    and the custom 0.5 override). With the override:
      * the audit MUST NOT append to ``blocking_issues``;
      * the report's threshold metadata must reflect the override (0.5);
      * the report must surface ``max_drift_score`` < the override.

    The override is read from ``scope_spec["sampling_frame_max_drift"]``,
    not from ``scope_spec["sampling_frame_audit"]`` (which controls the
    per-metric ``drift_flagged`` threshold, distinct from the blocking
    gate).
    """
    # Train ``age`` is N(50, 10). A reference centred at 54 with std 10 gives
    # SMD = |50-54| / sqrt((100+100)/2) = 4 / 10 = 0.4 — between 0.3 and 0.5.
    reference = {
        "distributions": {
            "age": {"mean": 54.0, "std": 10.0},
        },
    }
    state: Dict[str, Any] = {
        "experiment_id": "exp_custom_threshold",
        "scope_spec": {
            "deployment_reference": reference,
            "sampling_frame_max_drift": 0.5,  # custom blocking threshold
        },
        "train_df": base_train_df,
        "blocking_issues": [],
    }

    result = await audit_sampling_frame(state)

    report = result["sampling_frame_audit_report"]

    # The override propagates into the report so consumers can see it.
    assert report["thresholds"]["sampling_frame_max_drift"] == 0.5

    # SMD must land strictly between the default (0.3) and the override (0.5);
    # otherwise the test pre-condition is wrong and the assertion below would
    # give a false sense of security.
    smd = report["max_drift_score"]
    assert smd is not None
    assert DEFAULT_SAMPLING_FRAME_MAX_DRIFT < smd < 0.5, (
        f"Test pre-condition violated: expected 0.3 < SMD < 0.5, got {smd!r}"
    )

    # Custom threshold (0.5) > observed SMD (~0.4) → no blocking entry.
    assert "blocking_issues" not in result
    assert "blocking_detail" not in report
