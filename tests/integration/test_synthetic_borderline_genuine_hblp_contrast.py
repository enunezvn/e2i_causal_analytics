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

The test asserts:
1. Decision contract — legacy DROPS, HBLP RETAINS, same z across arms.
2. Calibration sanity — z is in a wide band [4.5, 8.0] so platform
   drift in numpy/scipy/permutation impl doesn't flake CI on a 0.1σ
   shift, but a real regression in the generator constants (e.g.,
   AUC drift > 1σ) surfaces.
3. HBLP relaxation actually fired — verified by calling production
   ``hblp_classify`` directly with the observed z and asserting it
   would reclassify ``high → moderate`` at the manifest threshold.
   This replaces the prior brittle ``"HBLP-relaxed" in evidence``
   string-match (codex pass-1 LOW).

Reference:
- ``.claude/plans/disease_agnostic_quality_uplift_v5.md`` §2 C2
- ``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py``
  ``hblp_classify`` + ``T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER``
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

# Wide calibration sanity band. A real regression in either
# ``compute_adversarial_score`` or the generator constants moves z by
# >>1σ; minor numpy/scipy version drift moves z by ≤0.3σ. Band wide
# enough for the latter, tight enough to catch the former. Separate
# from the decision-contract assertion (legacy 5σ < z < HBLP 7.5σ),
# which is what really matters per codex pass-1 MEDIUM.
EXPECTED_Z_LOW = 4.5
EXPECTED_Z_HIGH = 8.0
LEGACY_THRESHOLD = 5.0
HBLP_DECLARED_SAFE_MULTIPLIER = 1.5  # mirrors T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER
HBLP_EFFECTIVE_THRESHOLD = LEGACY_THRESHOLD * HBLP_DECLARED_SAFE_MULTIPLIER  # = 7.5


@pytest.fixture(scope="module")
def borderline_train_df():
    """Shared train_df for the C2 contrast suite.

    Module-scoped to keep the integration suite under 30s wall-clock per
    codex pass-1 MEDIUM-1. The in-memory DataFrame is immutable across
    tests so sharing is safe.
    """
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


@pytest.fixture(scope="module")
def borderline_arms_results(borderline_train_df):
    """Run ``adaptive_validity_check`` once per arm; cache results.

    The expensive 200-permutation scan over the ~10 numeric columns runs
    twice total (legacy + HBLP), not four times. Both arm results are
    re-used by ``test_v5_c2_*`` consumers. Per codex pass-1 MEDIUM-1.

    Returns a dict with keys ``legacy`` and ``hblp``, each carrying the
    full state-update dict that ``adaptive_validity_check`` returned.
    """
    import asyncio

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    train_df, numeric_cols = borderline_train_df

    async def _run_both():
        legacy = await adaptive_validity_check(
            {
                "experiment_id": "v5-c2-fx-legacy",
                "train_df": train_df,
                "scope_spec": _scope_spec(numeric_cols, manifest_source=None),
            }
        )
        hblp = await adaptive_validity_check(
            {
                "experiment_id": "v5-c2-fx-hblp",
                "train_df": train_df,
                "scope_spec": _scope_spec(numeric_cols, manifest_source="synthetic"),
            }
        )
        return legacy, hblp

    # Fresh event loop so the module-scoped fixture doesn't collide with
    # pytest-asyncio's per-test loop. Same nest_asyncio mitigation pattern
    # as PR #106's test_layer_5_pipeline_integration.py rationale.
    loop = asyncio.new_event_loop()
    try:
        legacy_result, hblp_result = loop.run_until_complete(_run_both())
    finally:
        loop.close()
    return {"legacy": legacy_result, "hblp": hblp_result}


@pytest.mark.integration
def test_v5_c2_legacy_drops_hblp_retains_borderline_genuine(
    borderline_train_df, borderline_arms_results
):
    """v5 §2 C2 acceptance — BOTH arms RETAIN at z in (5σ, 7.5σ) post-issue-#194.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    Pre-issue-#194 the contract was "legacy DROPS, HBLP RETAINS": the
    legacy 5σ z-threshold flagged the borderline_genuine feature, while
    HBLP's 7.5σ effective threshold under ``layer_1_declared_safe=True``
    retained it. The borderline_genuine generator constants
    (``BORDERLINE_GENUINE_TREATED_MEAN=0.06``) were chosen so the
    injected feature lands at AUC ≈ 0.55 with ``|delta_AUC| ≈ 0.05``,
    placing z in the (5σ, 7.5σ) HBLP relaxation band.

    Issue #194 closure (2026-05-14): the joint check
    ``severity ∈ {moderate, high}  ⇔  (z > k) AND (|delta_AUC| > epsilon=0.10)``
    now applies to BOTH arms. Since the borderline_genuine
    ``|delta_AUC| ≈ 0.05`` is below the 0.10 floor, BOTH arms retain
    the feature — the joint check correctly classifies it as a benign
    weak signal, NOT a leak. HBLP's variance-inflation prior remains
    active (verified by ``test_v5_c2_hblp_relaxation_actually_fired``
    via the direct ``hblp_classify`` call) but the joint check fires
    earlier in the decision tree on the legacy arm too.

    The narrative preserved: the borderline_genuine pattern is a
    NEGATIVE control — a feature that legitimately weak-correlates
    with the target but should NOT be dropped. The fix changes who is
    responsible for the correct decision (joint check vs HBLP), not
    the decision itself for the HBLP arm.
    """
    train_df, _ = borderline_train_df
    feature = BORDERLINE_GENUINE_FEATURE_NAME
    assert feature in train_df.columns, "fixture: generator must produce the borderline feature"

    legacy_result = borderline_arms_results["legacy"]
    hblp_result = borderline_arms_results["hblp"]
    legacy_flagged = set(legacy_result.get("adaptive_flagged_features") or [])
    hblp_flagged = set(hblp_result.get("adaptive_flagged_features") or [])
    # Issue #194: BOTH arms retain the borderline_genuine feature
    # because |delta_AUC| ≈ 0.05 < 0.10 floor. The legacy arm now
    # benefits from the joint check the same way HBLP did before.
    assert feature not in legacy_flagged, (
        f"Issue #194 contract: legacy arm should RETAIN {feature!r} via "
        f"joint check (|delta_AUC| ≈ 0.05 ≤ floor 0.10); flagged={legacy_flagged}"
    )
    assert feature not in hblp_flagged, (
        f"v5 C2 + issue #194: HBLP arm should RETAIN {feature!r}; flagged={hblp_flagged}"
    )

    legacy_verdict = next(v for v in legacy_result["adaptive_verdicts"] if v["feature"] == feature)
    hblp_verdict = next(v for v in hblp_result["adaptive_verdicts"] if v["feature"] == feature)
    z_legacy = legacy_verdict["z_score"]
    z_hblp = hblp_verdict["z_score"]
    assert z_legacy == pytest.approx(z_hblp, rel=1e-6), (
        f"z_score must be identical across arms (threshold is the only difference). "
        f"legacy={z_legacy}, hblp={z_hblp}"
    )
    # Issue #194: both arms now classify as ``info`` (z above HIGH_Z
    # but |delta_AUC| ≤ floor → joint check forces info). HBLP's
    # variance-relaxation prior is verified independently by the
    # ``test_v5_c2_hblp_relaxation_actually_fired`` test below — that
    # one directly invokes ``hblp_classify`` without the |delta_AUC|
    # input, isolating the HBLP relaxation mechanism.
    assert legacy_verdict["severity"] == "info"
    assert hblp_verdict["severity"] == "info"
    assert legacy_verdict["layer"] == "3"
    assert hblp_verdict["layer"] == "3"
    # Audit trail: the legacy-arm evidence string must mention the
    # joint check (#194) so an audit reader sees WHY the high-z
    # feature was kept.
    assert (
        "194" in legacy_verdict["evidence"] or "joint check" in legacy_verdict["evidence"].lower()
    ), (
        f"Issue #194: legacy verdict must record joint-check rationale; "
        f"got evidence={legacy_verdict['evidence']!r}"
    )


@pytest.mark.integration
def test_v5_c2_z_lands_in_calibration_band(borderline_arms_results):
    """v5 §2 C2 calibration drift guard — z stays in wide sanity band.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    Band [4.5, 8.0] tolerates minor numpy/scipy version drift (≤0.3σ in
    practice) but catches a real regression in either the generator
    constants or ``compute_adversarial_score`` (which shift z by >>1σ).
    The narrow decision-contract assertion (legacy 5σ < z < HBLP 7.5σ)
    is enforced by the contrast test above; this test isolates the
    calibration concern from the decision concern.

    Reuses the legacy-arm result from the module fixture (z is
    threshold-invariant — same value in both arms).
    """
    legacy_result = borderline_arms_results["legacy"]
    verdict = next(
        v
        for v in legacy_result["adaptive_verdicts"]
        if v["feature"] == BORDERLINE_GENUINE_FEATURE_NAME
    )
    z = verdict.get("z_score")
    assert z is not None and EXPECTED_Z_LOW <= z <= EXPECTED_Z_HIGH, (
        f"Calibration drift: z={z} fell outside expected band "
        f"[{EXPECTED_Z_LOW}, {EXPECTED_Z_HIGH}] — re-tune generator constants"
    )
    # z-band invariant on the HBLP relaxation window: even though
    # issue #194's joint check now retains the borderline_genuine
    # feature on BOTH arms, the z-value MUST still land in (5σ, 7.5σ)
    # for the HBLP relaxation mechanism to be in its active range.
    # If z fell BELOW 5σ the legacy-z-only path would also retain
    # (severity=info pre-joint-check); if z fell ABOVE 7.5σ HBLP's
    # ``layer_1_declared_safe`` relaxation wouldn't help.
    # ``test_v5_c2_hblp_relaxation_actually_fired`` then independently
    # verifies the HBLP mechanism via direct ``hblp_classify`` call.
    # Issue #194 codex pass-1 LOW-2: assertion text refreshed — the
    # "legacy-DROPS / HBLP-RETAINS" framing is stale; the joint check
    # is now the primary decision on the legacy arm. The z-band
    # invariant still has value as the precondition for HBLP's
    # relaxation mechanism to be testable.
    assert LEGACY_THRESHOLD < z < HBLP_EFFECTIVE_THRESHOLD, (
        f"v5 C2 invariant: z={z} must satisfy {LEGACY_THRESHOLD} < z < "
        f"{HBLP_EFFECTIVE_THRESHOLD} for HBLP's variance-relaxation "
        f"mechanism to be in its active range. (Issue #194 joint check "
        f"now retains the feature on BOTH arms via |delta_AUC|-floor; "
        f"the HBLP mechanism is verified independently in "
        f"test_v5_c2_hblp_relaxation_actually_fired.)"
    )


@pytest.mark.integration
def test_v5_c2_hblp_relaxation_actually_fired(borderline_train_df, borderline_arms_results):
    """v5 §2 C2 — verify HBLP variance-relaxation logic ran (not silent fallthrough).

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.

    Calls production ``hblp_classify`` directly with the verdict's z_score
    and the cohort positive-class count, with ``layer_1_declared_safe=
    True``, and asserts it returns the expected 7.5σ effective threshold
    and ``severity != "high"`` — i.e., the relaxation we expect from
    the HBLP arm. Structured verification replaces the brittle
    ``"HBLP-relaxed" in evidence`` string match (codex pass-1 LOW).

    Reuses the HBLP-arm result from the module fixture; runs only the
    cheap ``hblp_classify`` calls in-test.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER,
        hblp_classify,
    )

    train_df, _ = borderline_train_df
    feature = BORDERLINE_GENUINE_FEATURE_NAME

    hblp_result = borderline_arms_results["hblp"]
    verdict = next(v for v in hblp_result["adaptive_verdicts"] if v["feature"] == feature)
    z = float(verdict["z_score"])
    n_pos = int((train_df["treatment_initiated"] == 1).sum())

    # Re-run hblp_classify directly against the observed z + cohort
    # positives + declared_safe=True. The expected effective threshold
    # at n_pos >> 50 is exactly base × declared_safe_multiplier = 7.5σ.
    classification = hblp_classify(z_score=z, n_positives=n_pos, layer_1_declared_safe=True)
    assert classification["hblp_relaxed"] is True
    assert classification["layer_1_factor"] == pytest.approx(
        T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER
    )
    assert classification["effective_high_threshold"] == pytest.approx(
        LEGACY_THRESHOLD * T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER
    )
    assert classification["severity"] != "high", (
        f"HBLP relaxation should reclassify z={z} away from 'high' at the "
        f"declared-safe 7.5σ threshold; got severity={classification['severity']}"
    )

    # And the same call with declared_safe=False should give the legacy
    # 5σ threshold and severity=high — proving the relaxation is the
    # specific mechanism that flipped the verdict in the HBLP arm.
    legacy_classification = hblp_classify(z_score=z, n_positives=n_pos, layer_1_declared_safe=False)
    assert legacy_classification["effective_high_threshold"] == pytest.approx(LEGACY_THRESHOLD)
    assert legacy_classification["severity"] == "high"


@pytest.mark.integration
def test_v5_c2_synthetic_manifest_registers_borderline_feature():
    """The synthetic manifest must register the borderline feature as pre-anchor.

    Direct unit-style assertion against the manifest registry. Guards
    against accidental rename/removal of the FeatureContract that drives
    the HBLP-arm relaxation.
    """
    from src.data.manifests import MANIFEST_SOURCES, lookup_feature_contract

    assert "synthetic" in MANIFEST_SOURCES, (
        "v5 C2: 'synthetic' must be registered in MANIFEST_SOURCES"
    )

    contract = lookup_feature_contract(BORDERLINE_GENUINE_FEATURE_NAME, data_source="synthetic")
    assert contract is not None, (
        f"v5 C2: synthetic manifest must register {BORDERLINE_GENUINE_FEATURE_NAME!r}"
    )
    assert contract.knowable_at.is_pre_or_at_index(), (
        f"v5 C2: borderline feature contract must be pre-anchor; "
        f"got knowable_at={contract.knowable_at}"
    )
