"""Feature manifests — declarative `FeatureContract` registries per data source.

The manifests in this package make explicit, in code, what every feature in
each cohort *claims* about its temporal validity. Layer 1 (declarative
contracts) audits these against the source converters/derivers; Layer 5
(pipeline integration) reads them at runtime and produces structured
LeakageVerdicts whose `layer` field is "1" when a contract violation drives
the decision.

Manifests live here, not next to the converters, so the same registry can be
consumed by:
    - Tests (verifying every emitted feature has a contract)
    - Documentation (auto-generating "what does this column mean?" pages)
    - Adaptive Layer 4 (the DSPy classifier reads contract metadata as input)
    - Future Layer 1.3 audits (catching contract drift when a converter
      changes a derivation)

Disease-agnostic by construction: the *vocabulary* (`KnowableAt`, `source`,
`aggregation`, `window_days`) is universal. The *registry* per source is the
declaration of what features that source emits. Manifest application at
runtime is gated by an explicit ``data_source`` argument — callers that pass
``None`` (or a value not in MANIFEST_SOURCES) get no Layer 1 contracts back.
This prevents cross-cohort false positives where a synthetic run that happens
to use a CSU-canonical column name (e.g., ``brand``) is incorrectly flagged
under the CSU manifest.
"""

from typing import Callable, Mapping

from src.data.feature_contract import FeatureContract

from .csu_feature_manifest import (
    CSU_FEATURES,
    CSU_FORBIDDEN_AS_FEATURES,
    CSU_FORBIDDEN_NON_TARGET,
    CSU_SAFE_FEATURES,
    CSU_TARGETS,
    csu_contract_for,
)
from .optum_feature_manifest import (
    OPTUM_FEATURES,
    OPTUM_FORBIDDEN_AS_FEATURES,
    OPTUM_FORBIDDEN_NON_TARGET,
    OPTUM_SAFE_FEATURES,
    OPTUM_TARGETS,
    optum_contract_for,
)
from .optum_hcp_feature_manifest import (
    OPTUM_HCP_FEATURES,
    OPTUM_HCP_FORBIDDEN_AS_FEATURES,
    OPTUM_HCP_FORBIDDEN_NON_TARGET,
    OPTUM_HCP_SAFE_FEATURES,
    OPTUM_HCP_TARGETS,
    optum_hcp_contract_for,
)
from .optum_mart_feature_manifest import (
    MART_FORBIDDEN_AS_FEATURES,
    MART_FORBIDDEN_NON_TARGET,
    MART_SAFE_FEATURES,
    MART_TARGETS,
    OPTUM_MART_FEATURES,
    optum_mart_contract_for,
)
from .synthetic_feature_manifest import (
    SYNTHETIC_FEATURES,
    SYNTHETIC_FORBIDDEN_AS_FEATURES,
    synthetic_contract_for,
)

# Tag each registered manifest with the canonical data_source string the
# pipeline runner uses to opt in. Adding a new data source is two lines:
# import its ``<source>_contract_for`` above and add the entry below.
#
# v5 Gate C2 note: the ``synthetic`` source is the v5 engineering CI
# manifest that registers only the ``borderline_genuine_feature`` injected
# by ``synthetic_rwd_realistic`` for HBLP-contrast testing. It is NOT a
# disease cohort and emits NO RWD positive-evidence claim (plan §2 C2).
# ``optum_mart`` is the entity-stacked, pre-engineered Optum drop — the SAME
# cohort as ``optum`` but a structurally different schema (4/110 column overlap),
# so it registers its OWN FeatureContract list. Resolved via an explicit
# feature_manifest_source override on a non-``optum``-prefixed cohort path
# (autodetect would otherwise flag ``optum_mart`` ambiguous against ``optum``).
MANIFEST_SOURCES: Mapping[str, Callable[[str], FeatureContract | None]] = {
    "csu": csu_contract_for,
    "optum": optum_contract_for,
    "optum_mart": optum_mart_contract_for,
    # ``optum_hcp`` is the entity-stacked drop's HCP grain (commercial targeting):
    # an adoption target + claims practice-profile predictors, a different schema
    # AND a different unit of analysis from the patient ``optum_mart`` cohort, so
    # it registers its OWN FeatureContract list. Resolved via an explicit
    # feature_manifest_source override on the ``hcp_adoption`` cohort path.
    "optum_hcp": optum_hcp_contract_for,
    "synthetic": synthetic_contract_for,
}


def lookup_feature_contract(
    name: str,
    data_source: str | None = None,
) -> FeatureContract | None:
    """Search registered manifests for the named feature's FeatureContract.

    Args:
        name: Feature/column name to look up.
        data_source: Which manifest to consult. Required to apply Layer 1.
            Callers that don't know their cohort (e.g., synthetic runs that
            never registered a manifest) MUST leave this as ``None``; the
            lookup will return None so Layer 5 falls through to Layer 3
            statistical scoring rather than emit a false-positive Layer 1
            verdict against an unrelated cohort's contract.

    Returns:
        The matching FeatureContract, or None if (a) ``data_source`` is None
        or unknown, or (b) the matching manifest doesn't declare ``name``.
    """
    if data_source is None:
        return None
    fn = MANIFEST_SOURCES.get(data_source)
    if fn is None:
        return None
    return fn(name)


__all__ = [
    "CSU_FEATURES",
    "CSU_FORBIDDEN_AS_FEATURES",
    "CSU_FORBIDDEN_NON_TARGET",
    "CSU_SAFE_FEATURES",
    "CSU_TARGETS",
    "csu_contract_for",
    "MANIFEST_SOURCES",
    "OPTUM_FEATURES",
    "OPTUM_FORBIDDEN_AS_FEATURES",
    "OPTUM_FORBIDDEN_NON_TARGET",
    "OPTUM_SAFE_FEATURES",
    "OPTUM_TARGETS",
    "optum_contract_for",
    "OPTUM_MART_FEATURES",
    "MART_FORBIDDEN_AS_FEATURES",
    "MART_FORBIDDEN_NON_TARGET",
    "MART_SAFE_FEATURES",
    "MART_TARGETS",
    "optum_mart_contract_for",
    "OPTUM_HCP_FEATURES",
    "OPTUM_HCP_FORBIDDEN_AS_FEATURES",
    "OPTUM_HCP_FORBIDDEN_NON_TARGET",
    "OPTUM_HCP_SAFE_FEATURES",
    "OPTUM_HCP_TARGETS",
    "optum_hcp_contract_for",
    "SYNTHETIC_FEATURES",
    "SYNTHETIC_FORBIDDEN_AS_FEATURES",
    "synthetic_contract_for",
    "lookup_feature_contract",
]
