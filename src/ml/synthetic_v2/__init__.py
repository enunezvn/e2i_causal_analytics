"""Synthetic data generator v2 — package root.

Public API surface (shard 01 §B):

- ``generate_scenario(scenario, *, seed, n_total, train_ratio, val_ratio,
  test_ratio)``: end-to-end entry point.
- ``ScenarioName``: canonical A/B/C identifier enum (with ``from_short()``).
- ``SyntheticDataset``: frozen dataset container.
- ``ScenarioMetadata``: frozen audit-trail dataclass.

The per-scenario builders (Scenario A / B / C) are registered into
``src.ml.synthetic_v2.scenarios.SCENARIO_REGISTRY`` by commits 07 / 08 / 09.
"""

from src.ml.synthetic_v2.api import (
    ScenarioMetadata,
    SyntheticDataset,
    generate_scenario,
)
from src.ml.synthetic_v2.scenarios import ScenarioName

__all__ = [
    "ScenarioMetadata",
    "ScenarioName",
    "SyntheticDataset",
    "generate_scenario",
]
