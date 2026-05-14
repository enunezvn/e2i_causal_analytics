"""Synthetic feature manifest — v5 Gate C2 engineering CI sanity-check.

This manifest is INTENTIONALLY narrow. It exists ONLY so the
``synthetic_rwd_realistic`` regime can declare its ``borderline_genuine``
injected feature as ``knowable_at=index_date`` (Layer 1 declared-safe),
which is what unlocks the HBLP variance-relaxation contrast at the
[5σ, 7.5σ] band.

This is a v5 Gate C2 ENGINEERING CI SANITY-CHECK — NOT RWD positive
evidence (v5 plan §2 C2 + codex pass-3 MEDIUM-7). The synthetic generator
can produce any feature AUC by construction; what the integration test
pins is the post-issue-#194 "BOTH arms RETAIN" routing contract — the
legacy arm retains via the Layer 5 joint `|delta_AUC| > 0.10` floor
(borderline_genuine is tuned to `|delta_AUC| ≈ 0.05`); the HBLP arm
retains via the 1.5× declared-safe prior threshold (`5σ × 1.5 = 7.5σ`).
Pre-issue-#194 the contract was "legacy DROPS, HBLP RETAINS" via z
alone. See ``docs/synthetic_v3_design.md`` §3.1.

Disease-agnostic-by-construction note: the synthetic regime is *not* a
disease cohort. The manifest only registers the columns that the
``synthetic_rwd_realistic`` injection helpers emit. A real cohort would
register against the CSU or Optum manifest instead.
"""

from __future__ import annotations

from src.data.feature_contract import FeatureContract, KnowableAt

# ============================================================================
# v5 Gate C2: borderline_genuine_feature is declared knowable_at=index_date
# so HBLP's declared-safe prior (1.5x z-threshold multiplier) applies.
# ============================================================================

# Issue #178: this constant USED to be imported from
# ``src.repositories.synthetic_rwd_realistic``, but importing anything from
# ``src.repositories`` transitively triggers ``src.repositories.__init__``,
# which loads ``ObservabilitySpanRepository`` → ``src.agents.ml_foundation``
# → ``data_preparer/.../adaptive_validity_check`` → BACK into
# ``src.data.manifests``. That cycle made ``from src.data.manifests import
# OPTUM_SAFE_FEATURES`` fail with ImportError on a cold interpreter, which
# is why PR #175 needed ``importlib.util.spec_from_file_location``.
#
# The manifest is the canonical owner of the contracted feature name (it's
# the Layer 1 declaration). The repository imports this constant from the
# manifest, inverting the dependency to its correct direction
# (repositories depend on data contracts, not the reverse).
BORDERLINE_GENUINE_FEATURE_NAME = "borderline_genuine_feature"

_SYNTHETIC_FEATURES = [
    FeatureContract(
        name=BORDERLINE_GENUINE_FEATURE_NAME,
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        # No derivation_inputs declared — the synthetic feature is sampled
        # at-index from a class-conditional Gaussian (no upstream column).
        # Declaring a fictitious input would risk breaking future contract-
        # chain validation that checks input existence (codex pass-1 LOW).
        derivation_inputs=(),
    ),
]


SYNTHETIC_FEATURES: dict[str, FeatureContract] = {fc.name: fc for fc in _SYNTHETIC_FEATURES}

# Proactive defense-in-depth: columns the ``adaptive_validity_check`` node
# excludes from Layer 3 scoring up front (so a Layer 1 bug cannot let them
# reach model training). The synthetic manifest registers no forbidden
# features by design — its only purpose is the pre-anchor declared-safe
# contract on ``borderline_genuine_feature``. The empty list is explicit
# so ``_select_features`` does NOT emit its "unknown manifest_source"
# warning when callers opt into the synthetic manifest.
SYNTHETIC_FORBIDDEN_AS_FEATURES: list[str] = []


def synthetic_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for ``name`` in the synthetic manifest.

    Returns None for any feature not registered — the caller falls through
    to Layer 3 statistical scoring (no HBLP relaxation), which matches the
    pre-v5 behavior for synthetic runs that don't opt into a manifest.
    """
    return SYNTHETIC_FEATURES.get(name)


__all__ = [
    "BORDERLINE_GENUINE_FEATURE_NAME",
    "SYNTHETIC_FEATURES",
    "SYNTHETIC_FORBIDDEN_AS_FEATURES",
    "synthetic_contract_for",
]
