"""#617: ``_impute_column`` must not poison a numeric column with a string.

For an all-null column ``df[column].mode()`` is empty. The ``mode`` branch
fell back to the literal string ``"UNKNOWN"`` and ``fillna``'d it into the
column — flipping a NUMERIC column's dtype to ``object``. ``transform_data``
then raised ``could not convert string to float: 'UNKNOWN'``
(``transformation_error``), which the data_preparer surfaced as a
``RuntimeError`` instead of a graceful QC block (``test_qc_gate_blocks_on_failure``
crashed in slow-tests Job B). The fix keeps the fallback dtype-appropriate.
"""

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.qc_remediation import _impute_column


@pytest.mark.unit
def test_impute_mode_all_null_numeric_stays_numeric() -> None:
    df = pd.DataFrame({"x": [None, None, None]}, dtype="float64")
    out, _ = _impute_column(df, "x", "mode")
    assert pd.api.types.is_numeric_dtype(out["x"]), f"numeric dtype was poisoned: {out['x'].dtype}"
    assert "UNKNOWN" not in out["x"].astype(str).tolist()


@pytest.mark.unit
def test_impute_mode_all_null_object_still_uses_placeholder() -> None:
    # Non-numeric all-null column may still take a string placeholder.
    df = pd.DataFrame({"c": pd.Series([None, None, None], dtype="object")})
    out, _ = _impute_column(df, "c", "mode")
    assert out["c"].isnull().sum() == 0
