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
declaration of what features that source emits.
"""

from src.data.feature_contract import FeatureContract

from .csu_feature_manifest import (
    CSU_FEATURES,
    CSU_FORBIDDEN_AS_FEATURES,
    CSU_SAFE_FEATURES,
    csu_contract_for,
)
from .optum_feature_manifest import (
    OPTUM_FEATURES,
    OPTUM_FORBIDDEN_AS_FEATURES,
    OPTUM_SAFE_FEATURES,
    optum_contract_for,
)


def lookup_feature_contract(name: str) -> FeatureContract | None:
    """Search all registered manifests for the named feature's FeatureContract.

    Returns the first match. Adding a new data source is one line — append the
    new ``<source>_contract_for`` to the chain below. Disease-agnostic by
    construction: Layer 5 doesn't need to know whether the data is CSU, Optum,
    synthetic, or a future indication.

    Returns:
        The matching FeatureContract, or None if no manifest declares the name.
    """
    for fn in (csu_contract_for, optum_contract_for):
        c = fn(name)
        if c is not None:
            return c
    return None


__all__ = [
    "CSU_FEATURES",
    "CSU_FORBIDDEN_AS_FEATURES",
    "CSU_SAFE_FEATURES",
    "csu_contract_for",
    "OPTUM_FEATURES",
    "OPTUM_FORBIDDEN_AS_FEATURES",
    "OPTUM_SAFE_FEATURES",
    "optum_contract_for",
    "lookup_feature_contract",
]
