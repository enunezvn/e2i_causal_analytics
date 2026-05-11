"""v5 Gate C2 — synthetic borderline_genuine HBLP-contrast sanity-check.

ENGINEERING CI SANITY-CHECK ONLY — NOT RWD positive-evidence.

Per v5 plan §2 C2 + codex pass-3 MEDIUM-7: the synthetic generator can
produce any feature AUC by construction; this test does not establish
quality uplift for any RWD cohort. What it pins is that the pipeline
routing in ``adaptive_validity_check`` decides correctly at the HBLP
variance-relaxation band boundary:

  * Legacy arm (no ``feature_manifest_source``): the 5σ legacy threshold
    fires, severity escalates to ``high``, the feature is dropped.
  * HBLP arm (``feature_manifest_source="synthetic"``): the manifest
    declares the feature ``knowable_at=index_date``, so HBLP's
    ``layer_1_declared_safe`` prior applies the 1.5× threshold
    multiplier; at n_pos >> 50 the variance-inflation factor is 1.0 so
    the effective threshold is exactly 5σ × 1.5 = 7.5σ. The injected
    feature's z (~6σ at default parameters) lands below 7.5σ and is
    retained (severity drops to ``moderate`` — queued for Layer 4 causal
    review, not dropped).

The injected feature's z-value is also pinned within a tolerance band so
a drift in either ``compute_adversarial_score`` or the generator's
calibration constants surfaces in this test before it can cause a silent
regression in the contrast.

Reference:
- ``.claude/plans/disease_agnostic_quality_uplift_v5.md`` §2 C2
- ``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py``
  ``hblp_effective_z_threshold`` + ``T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER``
"""

from __future__ import annotations

import pytest

from src.repositories.synthetic_rwd_realistic import (
    BORDERLINE_GENUINE_DEFAULT_N_PATIENTS,
    BORDERLINE_GENUINE_DEFAULT_SEED,
    BORDERLINE_GENUINE_FEATURE_NAME,
    RwdRealisticConfig,
    generate_rwd_realistic,
)

# Z-value calibration record from 2026-05-11 (treated_mean=0.06,
# n_patients=20000, generator seed=42, n_permutations=200, adversarial
# seed=7 — adaptive_validity_check's default). The contrast band is
# (5.0, 7.5); the empirical value lands at ~6.10σ. Use a generous
# tolerance (±1.0σ) so minor numpy / scipy version drift doesn't flake
# the test, but tight enough to catch a real regression.
EXPECTED_Z_LOW = 5.1
EXPECTED_Z_HIGH = 7.4
HBLP_EFFECTIVE_THRESHOLD = 7.5  # 5σ × 1.5 (declared_safe prior) at n_pos>=50
LEGACY_THRESHOLD = 5.0


def _build_train_df():
    df = generate_rwd_realistic(
        RwdRealisticConfig(
            n_patients=BORDERLINE_GENUINE_DEFAULT_N_PATIENTS,
            leakage_pattern="borderline_genuine",
            seed=BORDERLINE_GENUINE_DEFAULT_SEED,
        )
    )
    numeric_cols = [
        c for c in df.columns if df[c].dtype.kind in "biufc" and c != "treatment_initiated"
    ]
    return df[numeric_cols + ["treatment_initiated"]].copy(), numeric_cols


def _scope_spec(numeric_cols, *, manifest_source: str | None):
    spec: dict = {
        "prediction_target": "treatment_initiated",
        "required_features": numeric_cols,
        "excluded_features": [],
    }
    if manifest_source is not None:
        spec["feature_manifest_source"] = manifest_source
    return spec


@pytest.mark.integration
@pytest.mark.asyncio
async def test_v5_c2_legacy_arm_drops_borderline_genuine_feature():
    """Legacy arm (no manifest): borderline_genuine feature is dropped.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    Without ``feature_manifest_source`` in scope_spec, Layer 1 cannot
    declare any feature as ``knowable_at=index_date``, so HBLP's
    ``layer_1_declared_safe`` prior is False. The effective threshold
    collapses to the base 5σ; the injected feature's z (~6σ) crosses it
    and the feature is flagged with ``severity=high``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    train_df, numeric_cols = _build_train_df()
    state = {
        "experiment_id": "v5-c2-legacy",
        "train_df": train_df,
        "scope_spec": _scope_spec(numeric_cols, manifest_source=None),
    }
    result = await adaptive_validity_check(state)

    feature = BORDERLINE_GENUINE_FEATURE_NAME
    assert feature in train_df.columns, "fixture: generator must produce the borderline feature"

    flagged = set(result.get("adaptive_flagged_features") or [])
    assert feature in flagged, (
        f"Legacy arm should flag {feature!r} at z > 5σ; flagged={flagged}"
    )

    verdicts = result.get("adaptive_verdicts") or []
    verdict = next(v for v in verdicts if v["feature"] == feature)
    assert verdict["severity"] == "high", (
        f"Legacy arm should classify {feature!r} as severity=high; got {verdict['severity']}"
    )
    assert verdict["layer"] == "3", (
        f"Legacy arm verdict should come from Layer 3 (adversarial); got layer={verdict['layer']}"
    )
    z = verdict.get("z_score")
    assert z is not None and EXPECTED_Z_LOW <= z <= EXPECTED_Z_HIGH, (
        f"Calibration drift: z={z} fell outside expected band "
        f"[{EXPECTED_Z_LOW}, {EXPECTED_Z_HIGH}] — re-tune generator constants"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_v5_c2_hblp_arm_retains_borderline_genuine_feature():
    """HBLP arm (synthetic manifest): borderline_genuine feature is retained.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    With ``feature_manifest_source="synthetic"``, the synthetic manifest
    declares the feature ``knowable_at=index_date``; HBLP applies the
    1.5× ``layer_1_declared_safe`` multiplier; the effective high
    threshold becomes 5σ × 1.5 = 7.5σ at the cohort's n_pos. The
    injected z (~6σ) is below 7.5σ so the feature is NOT in the
    flagged set; its verdict downgrades to ``moderate`` (queued for
    Layer 4 causal review) rather than ``high``/drop.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    train_df, numeric_cols = _build_train_df()
    state = {
        "experiment_id": "v5-c2-hblp",
        "train_df": train_df,
        "scope_spec": _scope_spec(numeric_cols, manifest_source="synthetic"),
    }
    result = await adaptive_validity_check(state)

    feature = BORDERLINE_GENUINE_FEATURE_NAME
    flagged = set(result.get("adaptive_flagged_features") or [])
    assert feature not in flagged, (
        f"HBLP arm should NOT flag {feature!r} at z < 7.5σ; flagged={flagged}"
    )

    verdicts = result.get("adaptive_verdicts") or []
    verdict = next(v for v in verdicts if v["feature"] == feature)
    assert verdict["severity"] in {"moderate", "info"}, (
        f"HBLP arm should classify {feature!r} as moderate (queued for L4) or info "
        f"(below moderate threshold); got severity={verdict['severity']}"
    )
    assert verdict["layer"] == "3", (
        f"HBLP arm verdict should come from Layer 3 (adversarial); got layer={verdict['layer']}"
    )

    # The HBLP-effective threshold annotation should reflect the 1.5×
    # declared-safe multiplier; pull from the evidence string. This is
    # a sanity check that we're actually exercising the HBLP-relaxed
    # branch and not silently routing through the legacy fallback.
    evidence = verdict.get("evidence", "")
    assert "HBLP-relaxed" in evidence or "HBLP-effective" in evidence, (
        f"HBLP arm verdict evidence should annotate HBLP relaxation; got {evidence!r}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_v5_c2_contrast_pin_legacy_drops_hblp_retains():
    """Pin the full contrast: same data + same z, opposite verdicts.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    Runs both arms against the same train_df and asserts:
      * Legacy arm: in ``adaptive_flagged_features`` (dropped)
      * HBLP arm: NOT in ``adaptive_flagged_features`` (retained)
      * Both arms see the same z_score (the difference is the threshold,
        not the statistic — proves the contrast is about the manifest
        declaration, not data noise).

    The single-test framing is what closes v5 §2 C2 acceptance: "HBLP
    arm RETAINS the feature; legacy arm DROPS it. Integration test pins
    this contrast." (plan acceptance language).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    train_df, numeric_cols = _build_train_df()
    feature = BORDERLINE_GENUINE_FEATURE_NAME

    legacy_state = {
        "experiment_id": "v5-c2-contrast-legacy",
        "train_df": train_df,
        "scope_spec": _scope_spec(numeric_cols, manifest_source=None),
    }
    hblp_state = {
        "experiment_id": "v5-c2-contrast-hblp",
        "train_df": train_df,
        "scope_spec": _scope_spec(numeric_cols, manifest_source="synthetic"),
    }

    legacy_result = await adaptive_validity_check(legacy_state)
    hblp_result = await adaptive_validity_check(hblp_state)

    legacy_flagged = set(legacy_result.get("adaptive_flagged_features") or [])
    hblp_flagged = set(hblp_result.get("adaptive_flagged_features") or [])

    assert feature in legacy_flagged, (
        f"v5 C2 contrast violation: legacy arm should DROP {feature!r}; "
        f"flagged={legacy_flagged}"
    )
    assert feature not in hblp_flagged, (
        f"v5 C2 contrast violation: HBLP arm should RETAIN {feature!r}; "
        f"flagged={hblp_flagged}"
    )

    # Both arms must see the same underlying statistic. The contrast is
    # threshold-driven, not data-driven.
    legacy_verdict = next(
        v for v in legacy_result["adaptive_verdicts"] if v["feature"] == feature
    )
    hblp_verdict = next(v for v in hblp_result["adaptive_verdicts"] if v["feature"] == feature)
    assert legacy_verdict["z_score"] == pytest.approx(hblp_verdict["z_score"], rel=1e-6), (
        f"z_score must be identical across arms (threshold is the only difference). "
        f"legacy={legacy_verdict['z_score']}, hblp={hblp_verdict['z_score']}"
    )
    # Severities differ across arms even though z is identical.
    assert legacy_verdict["severity"] == "high"
    assert hblp_verdict["severity"] in {"moderate", "info"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_v5_c2_synthetic_manifest_registers_borderline_feature():
    """The synthetic manifest must register the borderline feature as pre-anchor.

    Direct unit-style assertion against the manifest registry. Guards
    against accidental rename/removal of the FeatureContract that drives
    the HBLP-arm relaxation.
    """
    from src.data.manifests import MANIFEST_SOURCES, lookup_feature_contract

    assert "synthetic" in MANIFEST_SOURCES, (
        "v5 C2: 'synthetic' must be registered in MANIFEST_SOURCES"
    )

    contract = lookup_feature_contract(
        BORDERLINE_GENUINE_FEATURE_NAME, data_source="synthetic"
    )
    assert contract is not None, (
        f"v5 C2: synthetic manifest must register {BORDERLINE_GENUINE_FEATURE_NAME!r}"
    )
    assert contract.knowable_at.is_pre_or_at_index(), (
        f"v5 C2: borderline feature contract must be pre-anchor; "
        f"got knowable_at={contract.knowable_at}"
    )
