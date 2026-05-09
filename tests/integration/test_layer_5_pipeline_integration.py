"""Integration test: Layer 5 (adaptive_validity_check) inside the data_preparer
pipeline.

Validates the full path on rwd_realistic synthetic data WITH a known-injected
leak: detect_leakage → adaptive_validity_check → severity escalation →
leakage_remediation routing.

Acceptance criterion #2 of adaptive_temporal_validity_redesign.md:
"All injected leakage patterns caught by at least one of Layers 1-3 (the
layer that catches each is logged)."

Specifically proves:
1. detect_leakage runs first and emits its own findings.
2. adaptive_validity_check then runs Layer 3 against every numeric feature
   and catches the injected leak when detect_leakage's hardcoded thresholds
   miss it.
3. leakage_severity is escalated to "high" so the routing function dispatches
   the leak through the existing remediation flow.
4. The merged leaked_features list contains the injected leak.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rwd_realistic_post_index_leak_caught_by_layer_5_pipeline():
    """End-to-end pipeline integration with injected leak.

    Generate rwd_realistic data with leakage_pattern="post_index_aggregation",
    which adds a `post_index_med_count_LEAK` column. Then run detect_leakage
    + adaptive_validity_check sequentially, mirroring the graph wiring, and
    assert the adaptive layer catches what the legacy detector misses.

    Async + ``pytest.mark.asyncio`` (was sync + ``asyncio.run``): under
    xdist, an earlier test in the same worker may have triggered
    ``nest_asyncio.apply()`` (e.g., via ``experiment_designer.graph``),
    which monkey-patches asyncio globally. Subsequent ``asyncio.run``
    calls then route through nest_asyncio's wrapper which references a
    closed loop. ``pytest.mark.asyncio`` manages a fresh loop per
    test, bypassing the patched runner. Same fix pattern as PR #89
    commit ``5afa67a`` (kg ``__init__`` lazy-import for the related
    httpx-side-effect flake).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )
    from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
        detect_leakage,
    )
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    df = generate_rwd_realistic(
        RwdRealisticConfig(
            n_patients=2000,
            leakage_pattern="post_index_aggregation",
            seed=42,
        )
    )

    # Build the minimal state the two nodes need. Pass numeric-only features
    # so detect_leakage's structural checks can run; this matches what the
    # real pipeline does after schema_validator narrows the surface.
    numeric_cols = [
        c for c in df.columns if df[c].dtype.kind in "biufc" and c != "treatment_initiated"
    ]
    train_df = df[numeric_cols + ["treatment_initiated"]].copy()

    state = {
        "experiment_id": "test-layer-5-integration",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": numeric_cols,
            "excluded_features": [],
        },
    }

    # Step 1: legacy detect_leakage runs first
    det_result = await detect_leakage(state)
    state.update(det_result)
    legacy_severity = state.get("leakage_severity", "none")
    legacy_leaked = set(state.get("leaked_features") or [])

    # Step 2: adaptive layer runs after, augments
    adp_result = await adaptive_validity_check(state)
    state.update(adp_result)

    final_severity = state.get("leakage_severity", "none")
    flagged = set(state.get("adaptive_flagged_features") or [])

    # The injected leak column
    leak_col = "post_index_med_count_LEAK"
    assert leak_col in train_df.columns, (
        "Test fixture: synthetic generator should have produced the leak column"
    )

    # Adaptive layer catches the leak (at least one layer catches it)
    catches = (leak_col in legacy_leaked) or (leak_col in flagged)
    assert catches, (
        f"Layer 5 pipeline missed the injected leak. "
        f"legacy_leaked={legacy_leaked}, adaptive_flagged={flagged}"
    )

    # Final severity is at least 'high' since at least one layer flagged
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "info": 1, "none": 0}
    assert severity_rank[final_severity] >= severity_rank["high"], (
        f"Expected severity >= 'high' after Layer 5; got {final_severity}. "
        f"legacy={legacy_severity}, adaptive_flagged={flagged}"
    )

    # Audit trail: every numeric feature has a verdict
    verdicts = state.get("adaptive_verdicts") or []
    verdict_features = {v["feature"] for v in verdicts}
    assert leak_col in verdict_features, (
        f"Audit trail missing the leak column verdict: {verdict_features}"
    )

    # The leak's verdict should be 'high'/'drop'
    leak_verdict = next(v for v in verdicts if v["feature"] == leak_col)
    assert leak_verdict["severity"] == "high"
    assert leak_verdict["remediation"] == "drop"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rwd_realistic_no_leak_injection_does_not_falsely_flag():
    """Without injected leakage, Layer 5 must NOT escalate severity.

    Generate clean rwd_realistic data (leakage_pattern="none"), run the same
    two-node pipeline, and verify that no feature is flagged at z > 5σ.
    This is the false-positive guard for acceptance criterion #4.

    Async per the same nest_asyncio rationale as
    ``test_rwd_realistic_post_index_leak_caught_by_layer_5_pipeline``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    df = generate_rwd_realistic(
        RwdRealisticConfig(
            n_patients=2000,
            leakage_pattern="none",
            seed=42,
        )
    )

    numeric_cols = [
        c for c in df.columns if df[c].dtype.kind in "biufc" and c != "treatment_initiated"
    ]
    train_df = df[numeric_cols + ["treatment_initiated"]].copy()

    state = {
        "experiment_id": "test-layer-5-no-leak",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": numeric_cols,
            "excluded_features": [],
        },
        "leakage_severity": "none",
        "leakage_findings": [],
        "leaked_features": [],
    }

    adp_result = await adaptive_validity_check(state)
    flagged = set(adp_result.get("adaptive_flagged_features") or [])
    assert flagged == set(), f"False-positive: clean data flagged features: {flagged}"
