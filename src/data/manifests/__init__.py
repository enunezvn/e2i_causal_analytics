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

__all__ = [
    "CSU_FEATURES",
    "CSU_FORBIDDEN_AS_FEATURES",
    "CSU_SAFE_FEATURES",
    "csu_contract_for",
    "OPTUM_FEATURES",
    "OPTUM_FORBIDDEN_AS_FEATURES",
    "OPTUM_SAFE_FEATURES",
    "optum_contract_for",
]
