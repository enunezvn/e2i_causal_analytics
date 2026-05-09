"""Scenarios sub-package (shard 01 §B.1).

Hosts:

- ``ScenarioName``: ``str`` + ``Enum`` of canonical scenario IDs (A/B/C),
  with ``from_short("A"/"B"/"C")`` resolver for CLI flags.
- ``SCENARIO_REGISTRY``: mapping ``ScenarioName -> ScenarioBuilder`` factory.
  Currently an empty dict — populated by commits 07 / 08 / 09 as
  Scenario A / B / C builders land.

The ``ScenarioBuilder`` ABC consumed by registry factories lives in
``scenarios/_base.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.synthetic_v2.scenarios._base import ScenarioBuilder


class ScenarioName(str, Enum):
    """Canonical scenario identifiers consumed by ``generate_scenario`` (shard 01 §B.1).

    Inheriting from ``str`` enables JSON serialization
    (``json.dumps(ScenarioName.A_DIAGNOSTIC_BC_IDFS)`` emits the value
    string) and direct comparison against YAML-loaded strings.
    """

    A_DIAGNOSTIC_BC_IDFS = "scenario_a_diagnostic_ebc_idfs_5y"
    A_DIAGNOSTIC_BC_IDFS_BALANCED = "scenario_a_diagnostic_ebc_idfs_5y_balanced"
    B_SCREENING_IGAN_ESKD = "scenario_b_screening_igan_eskd_5y"
    C_TREATMENT_CSU_RESPONSE = "scenario_c_treatment_decision_csu_remib_response"

    @classmethod
    def from_short(cls, short: str) -> ScenarioName:
        """Resolve ``"A"`` / ``"B"`` / ``"C"`` (case-insensitive) → enum value.

        Used by CLI flags such as
        ``scripts/run_phase1_diagnostic.py --scenario A``.
        """
        mapping = {
            "A": cls.A_DIAGNOSTIC_BC_IDFS,
            "B": cls.B_SCREENING_IGAN_ESKD,
            "C": cls.C_TREATMENT_CSU_RESPONSE,
        }
        key = short.upper()
        if key not in mapping:
            raise ValueError(
                f"Unknown scenario short code {short!r}; choose from {sorted(mapping.keys())}"
            )
        return mapping[key]


# Mapping ``ScenarioName -> factory(no-arg) -> ScenarioBuilder``. Populated by
# commits 07 (Scenario A), 08 (B), 09 (C). Kept as a module-level dict so
# ``api.generate_scenario`` (commit 06) can dispatch without needing to
# import each scenario module directly.
SCENARIO_REGISTRY: dict[ScenarioName, Callable[[], ScenarioBuilder]] = {}


__all__ = ["SCENARIO_REGISTRY", "ScenarioName"]


# Side-effect imports: each scenario module appends itself to SCENARIO_REGISTRY
# at import time. Imports MUST be at the bottom of this file so ScenarioName +
# SCENARIO_REGISTRY are already bound by the time scenario_* execute.
# B and C are added by commits 08 and 09; missing modules cause ImportError on
# package load until those commits land.
from src.ml.synthetic_v2.scenarios import scenario_a as _scenario_a  # noqa: E402, F401
from src.ml.synthetic_v2.scenarios import (  # noqa: E402, F401
    scenario_a_balanced as _scenario_a_balanced,
)
from src.ml.synthetic_v2.scenarios import scenario_b as _scenario_b  # noqa: E402, F401
from src.ml.synthetic_v2.scenarios import scenario_c as _scenario_c  # noqa: E402, F401
