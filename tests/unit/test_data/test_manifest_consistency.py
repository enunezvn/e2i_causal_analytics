"""Cross-manifest consistency tests — Layer 1.3 audit (item I).

When two manifests register a contract for the same feature name (e.g.
``brand``, ``journey_status``, ``treatment_initiated``), they MUST agree on
the temporal-validity claim. A drift here would silently produce divergent
Layer 1 verdicts depending on which cohort opted in via
``scope_spec.feature_manifest_source``.

Pinned 2026-05-07 during the ralph-loop review of PR #84 after the
cross-cohort manifest false-positive bug (commit ``0a35807``) made cohort
gating opt-in. The shared-name set is small (13 names) and the agreement
is currently 13/13. This test guards against a future drift.
"""

from __future__ import annotations

from src.data.manifests.csu_feature_manifest import CSU_FEATURES
from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES


def _by_name(contracts):
    return {c.name: c for c in contracts}


def test_shared_feature_names_have_consistent_knowable_at():
    csu = _by_name(CSU_FEATURES)
    optum = _by_name(OPTUM_FEATURES)
    shared = sorted(set(csu) & set(optum))

    # The shared set must be non-empty (otherwise this test would silently
    # pass even after a refactor that splits all common names apart).
    assert shared, (
        "No shared feature names across CSU + Optum manifests — refactor "
        "that splits common names is suspicious; sanity-check that "
        "treatment_initiated, brand, journey_status, etc. still appear in both."
    )

    drift = []
    for name in shared:
        csu_ref = csu[name].knowable_at.reference
        optum_ref = optum[name].knowable_at.reference
        csu_off = csu[name].knowable_at.offset_days
        optum_off = optum[name].knowable_at.offset_days
        if csu_ref != optum_ref or csu_off != optum_off:
            drift.append(f"{name}: CSU={csu[name].knowable_at}, Optum={optum[name].knowable_at}")
    assert drift == [], (
        "Shared-name temporal-validity drift across CSU + Optum manifests — "
        "either fix the divergence or rename one side to disambiguate:\n  " + "\n  ".join(drift)
    )


def test_shared_post_index_names_include_documented_targets():
    """Anti-regression: the cross-cohort target columns (treatment_initiated,
    discontinuation_flag) MUST be declared post_index in BOTH manifests so
    Layer 5's manifest-driven catch fires for either cohort. The
    cross-manifest false-positive bug (commit ``0a35807``) was about a
    DIFFERENT shared name (``brand``); this test guards the targets.
    """
    csu = _by_name(CSU_FEATURES)
    optum = _by_name(OPTUM_FEATURES)
    expected_post_index = {"treatment_initiated", "discontinuation_flag"}
    for name in expected_post_index:
        assert name in csu, f"CSU manifest missing target column {name!r}"
        assert name in optum, f"Optum manifest missing target column {name!r}"
        assert csu[name].knowable_at.reference == "post_index"
        assert optum[name].knowable_at.reference == "post_index"
