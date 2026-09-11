"""#2014: ``refutation_runner.estimate_id`` is optional.

No composable tool produces an estimate id, so the planner had to fill the required
``estimate_id`` with whatever it could reference (``$step_1.method``, ``$step_1.ate``),
and the output echoed that back as provenance. Nothing reads the id: the live DoWhy
estimate cannot cross the step boundary, so the suite re-estimates from the source data
and ``treatment`` / ``outcome`` / ``confounders``. An id minted by the estimator would
therefore claim a link the refutation does not have; the input is optional instead.

Real DoWhy refutation suite on a real deterministic frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.tool_composer import tool_registrations as tr


def _frame(n: int = 400, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})


def test_estimate_id_is_declared_optional() -> None:
    from src.tool_registry import get_registry

    params = {p.name: p for p in get_registry().get_schema("refutation_runner").input_parameters}

    assert params["estimate_id"].required is False


def test_refutation_runs_without_an_estimate_id() -> None:
    result = tr.refutation_runner(
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=_frame(),
    )

    assert result["estimate_id"] is None
    assert result["refutation_results"]["gate_decision"] in {"proceed", "review", "block"}
