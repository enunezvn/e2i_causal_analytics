"""Real-DB integration tests for FeatureBuilder.load_frame + build_for_split (Task 4).

Gate: ``E2I_DB_INTEGRATION=1`` + a reachable async Supabase client (``SUPABASE_URL``
+ key). Mirrors the opt-in used by other integration suites (e.g.
``test_model_deployer_persistence_integration.py``) so unit-only CI lanes never
touch the DB.

NO mocks — all assertions run against the LIVE synthetic patient_journeys rows
(is_synthetic=True, brand=Remibrutinib) seeded in the Supabase docker DB.

Expected data: ~5075 holdout rows, label mean ~0.35, dates 2026-04 to 2026-06,
3 KEEP_COLUMNS present and non-null.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason=(
        "E2I_DB_INTEGRATION!=1; set to 1 to run against the real Supabase "
        "DB. Requires SUPABASE_URL + key in environment."
    ),
)

# ---------------------------------------------------------------------------
# Fixture: fresh async client per test (mirrors test_model_deployer_persistence)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the global cached async client so each test gets a fresh one on its
    own event loop — the cached httpx.AsyncClient is bound to the creating loop
    and would raise 'Event loop is closed' on reuse across per-test loops.
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_frame_holdout_returns_expected_shape_and_columns() -> None:
    """load_frame(splits=['holdout']) returns >0 rows with all required columns."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.cohort_spec import INITIATION
    from src.mlops.gold_standard_eval.feature_builder import KEEP_COLUMNS, FeatureBuilder

    db = await get_async_supabase_client()
    fb = FeatureBuilder(INITIATION)

    df = await fb.load_frame(db, splits=["holdout"])

    assert not df.empty, "load_frame returned empty DataFrame for holdout split"
    assert len(df) > 0, "expected >0 rows"

    # Required columns must be present.
    required = {"patient_id", "journey_start_date", "data_split", INITIATION.label_column}
    required.update(KEEP_COLUMNS)
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in load_frame result: {missing}"

    # Split filter must be respected.
    assert set(df["data_split"].unique()) == {"holdout"}, (
        f"Expected only 'holdout' rows, got: {df['data_split'].unique().tolist()}"
    )

    # Label must be non-null.
    assert df[INITIATION.label_column].isnull().sum() == 0, (
        "label_column has null values"
    )


@pytest.mark.asyncio
async def test_load_frame_holdout_label_mean_in_range() -> None:
    """Label mean for holdout should be ~0.35 (Remibrutinib synthetic cohort)."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.cohort_spec import INITIATION
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    db = await get_async_supabase_client()
    fb = FeatureBuilder(INITIATION)

    df = await fb.load_frame(db, splits=["holdout"])
    mean = df[INITIATION.label_column].mean()

    # Generous tolerance: expect 0.25–0.50 given the synthetic DGP.
    assert 0.25 <= mean <= 0.50, (
        f"label mean {mean:.4f} outside expected range [0.25, 0.50]"
    )


@pytest.mark.asyncio
async def test_load_frame_before_month_filter() -> None:
    """before_month filter narrows the result to dates before the cutoff."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.cohort_spec import INITIATION
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    db = await get_async_supabase_client()
    fb = FeatureBuilder(INITIATION)

    # Load all holdout rows first to confirm there are rows before 2026-06-01.
    all_df = await fb.load_frame(db, splits=["holdout"])
    cutoff = "2026-06-01"

    filtered_df = await fb.load_frame(db, splits=["holdout"], before_month=cutoff)

    if filtered_df.empty:
        pytest.skip(f"No holdout rows with journey_start_date < {cutoff}; skip filter test")

    # Every returned row must satisfy the predicate.
    assert (filtered_df["journey_start_date"] < cutoff).all(), (
        "before_month filter returned rows with journey_start_date >= cutoff"
    )
    # Filtered count must be <= total.
    assert len(filtered_df) <= len(all_df)


@pytest.mark.asyncio
async def test_build_for_split_produces_no_nan_feature_matrix() -> None:
    """build_for_split('holdout') → X has 0 NaN, y matches label column, feature_columns set."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.cohort_spec import INITIATION
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    db = await get_async_supabase_client()
    fb = FeatureBuilder(INITIATION)

    X, y = await fb.build_for_split(db, "holdout")

    # X must be non-empty.
    assert not X.empty, "build_for_split returned empty X"
    assert len(X) > 0

    # X must be NaN-free (the encoding contract guarantees this).
    nan_count = int(X.isnull().sum().sum())
    assert nan_count == 0, f"X has {nan_count} NaN values after encoding"

    # y must be integer, non-null.
    assert y.isnull().sum() == 0, "y (label) has null values"
    assert y.dtype.kind == "i", f"expected integer y, got {y.dtype}"

    # feature_columns must be populated and match X.
    assert fb.feature_columns, "feature_columns is empty after build_for_split"
    assert list(X.columns) == fb.feature_columns, (
        "X.columns does not match feature_columns"
    )

    # Row counts must agree.
    assert len(X) == len(y), "X and y have different row counts"
