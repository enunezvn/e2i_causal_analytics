"""F16: data_preparer entity-split must combine frames via pd.concat (pandas 2.x).

pandas 2.x removed ``DataFrame.append``; ``_load_from_supabase`` chained
``dataset.train.append(dataset.val).append(dataset.test)`` on the entity-split
branch, which raises ``AttributeError`` at runtime. Red-first: drive that branch
and assert the combined frame reaches ``combined_split`` intact.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes import data_loader


def _frame(n: int, start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [f"e{i % 3}" for i in range(n)],
            "date": pd.date_range(start, periods=n, freq="D").astype(str),
            "x": list(range(n)),
        }
    )


@pytest.mark.asyncio
@patch.object(data_loader, "get_data_splitter")
@patch.object(data_loader, "get_ml_data_loader")
async def test_entity_split_combines_frames_without_dataframe_append(
    mock_get_loader, mock_get_splitter
):
    train, val, test = _frame(5, "2026-01-01"), _frame(3, "2026-02-01"), _frame(2, "2026-03-01")

    loader = MagicMock()
    loader.load_for_training = AsyncMock(
        return_value=SimpleNamespace(train=train, val=val, test=test)
    )
    mock_get_loader.return_value = loader

    captured: dict = {}

    def _combined_split(frame, **kwargs):
        captured["frame"] = frame
        return SimpleNamespace(
            train=frame, val=frame.iloc[0:0], test=frame.iloc[0:0], holdout=None
        )

    splitter = MagicMock()
    splitter.combined_split = MagicMock(side_effect=_combined_split)
    mock_get_splitter.return_value = splitter

    result = await data_loader._load_from_supabase(
        data_source="ml_training_runs",
        filters={},
        date_column="date",
        entity_column="entity_id",
        split_date=None,
        val_days=1,
        test_days=1,
    )

    # The concatenated frame (5 + 3 + 2 = 10 rows) must reach combined_split —
    # proving pd.concat replaced the removed DataFrame.append (no AttributeError).
    assert "frame" in captured, "entity-split branch did not reach combined_split"
    assert len(captured["frame"]) == 10
    assert list(captured["frame"].index) == list(range(10)), "expected a fresh RangeIndex"
    assert "train" in result
