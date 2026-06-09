"""SSOT provenance helper: default-exclude predicate + covariate drop-list."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.repositories.provenance import (
    PROVENANCE_DROP_COLS,
    apply_provenance_filter,
    drop_provenance_cols,
)


@pytest.mark.unit
def test_is_synthetic_in_drop_cols():
    assert "is_synthetic" in PROVENANCE_DROP_COLS


@pytest.mark.unit
def test_drop_provenance_cols_removes_tag_only():
    df = pd.DataFrame(
        {"treatment": [0, 1], "outcome": [1.0, 2.0],
         "is_synthetic": [True, True], "x1": [3, 4]}
    )
    out = drop_provenance_cols(df)
    assert "is_synthetic" not in out.columns
    assert list(out.columns) == ["treatment", "outcome", "x1"]


@pytest.mark.unit
def test_apply_filter_default_excludes():
    q = MagicMock()
    apply_provenance_filter(q, include_synthetic=False)
    q.eq.assert_called_once_with("is_synthetic", False)


@pytest.mark.unit
def test_apply_filter_opt_in_is_noop():
    q = MagicMock()
    out = apply_provenance_filter(q, include_synthetic=True)
    q.eq.assert_not_called()
    assert out is q
