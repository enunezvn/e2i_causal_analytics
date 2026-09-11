"""``risk_scorer`` refuses an ``outcome`` that is not a 0/1 event column (#2003).

#2003 declared ``outcome`` to the planner as the binary risk event. The body cast any
column with ``astype(int)``: a count such as ``refill_count`` became a multi-class target, and
the tool returned the probability of "exactly 1" as a patient's risk score with nothing
saying so; a 0-1 rate such as ``adherence_rate`` truncated to one class and was refused
with a misleading reason. The planner can now name either, and on the patient cohort the
default ``discontinuation_flag`` does not exist, so the planned ``outcome`` is what runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError


def _cohort(n: int = 300, seed: int = 2003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    severity = rng.normal(size=n)
    event = (rng.random(n) < 1.0 / (1.0 + np.exp(-severity))).astype(int)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "disease_severity": severity,
            "engagement_score": rng.normal(size=n),
            "discontinued_180d": event,
            "is_churned": event.astype(bool),
            "refill_count": rng.integers(0, 8, size=n),
            "adherence_rate": rng.uniform(0.2, 1.0, size=n),
        }
    )


@pytest.mark.parametrize("outcome", ["refill_count", "adherence_rate"])
def test_a_non_binary_outcome_is_refused(outcome):
    with pytest.raises(ToolRefusalError, match="binary"):
        tr.risk_scorer(
            entity_type="patient",
            risk_type="discontinuation",
            estimation_data=_cohort(),
            outcome=outcome,
        )


@pytest.mark.parametrize("outcome", ["discontinued_180d", "is_churned"])
def test_a_binary_outcome_is_scored(outcome):
    out = tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=_cohort(),
        outcome=outcome,
    )
    assert len(out.scores) == 300
    assert all(0.0 <= s["risk_score"] <= 1.0 for s in out.scores)
