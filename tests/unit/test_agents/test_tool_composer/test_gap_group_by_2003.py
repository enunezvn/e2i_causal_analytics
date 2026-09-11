"""``gap_calculator`` refuses a ``group_by`` that is not a column instead of regrouping (#2003).

#2003 declared ``group_by`` to the planner. When the supplied name was not a column the
tool silently fell back to ``entity_type`` / heuristic columns, so a planned grouping could
be replaced by another one with nothing in the output saying so. An explicit ``group_by``
must be a real column; omitting it keeps the documented resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame(
        {
            "geographic_region": rng.choice(["northeast", "south", "west"], size=n),
            "territory": rng.choice(["t1", "t2"], size=n),
            "trx": rng.normal(100, 10, size=n),
        }
    )


def test_a_group_by_that_is_not_a_column_is_refused():
    with pytest.raises(ToolRefusalError, match="group_by"):
        tr.gap_calculator(
            metric="trx",
            entity_type="region",
            entities=[],
            estimation_data=_frame(),
            group_by="hcp_specialty",
        )


def test_an_explicit_group_by_is_the_grouping_used():
    out = tr.gap_calculator(
        metric="trx",
        entity_type="region",
        entities=[],
        estimation_data=_frame(),
        group_by="territory",
    )
    assert set(out.entity_values) == {"t1", "t2"}


def test_an_omitted_group_by_keeps_the_entity_type_resolution():
    out = tr.gap_calculator(
        metric="trx", entity_type="region", entities=[], estimation_data=_frame()
    )
    assert set(out.entity_values) == {"northeast", "south", "west"}
