"""Tests for the per-brand HCP adoption cohort runner (HCP-T4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    HCP_ADOPTION_COHORT,
    goldstd_experiment_name,
    goldstd_model_name,
)


@pytest.mark.asyncio
async def test_run_calls_run_one_cohort_three_times():
    """run() dispatches _run_one_cohort exactly once per brand."""
    fake_client = object()
    fake_results = {
        brand.lower(): {
            "model": f"hcp_adoption_{brand.lower()}_goldstd_lr_v1",
            "holdout_auc": 0.75,
            "backtest_points": 36,
            "n_train": 800,
            "n_holdout": 200,
        }
        for brand in BRANDS
    }

    # Build a side_effect that returns the per-brand dict in BRANDS order.
    brand_order = list(BRANDS)
    side_effects = [fake_results[b.lower()] for b in brand_order]

    mock_run_one = AsyncMock(side_effect=side_effects)
    mock_resolve = AsyncMock(return_value=fake_client)

    with (
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._run_one_cohort",
            mock_run_one,
        ),
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._resolve_client",
            mock_resolve,
        ),
    ):
        from src.mlops.gold_standard_eval.run_hcp_cohorts import run

        _result = await run(db=fake_client)

    # _resolve_client was awaited with the passed db.
    mock_resolve.assert_awaited_once_with(fake_client)

    # Exactly 3 calls — one per brand.
    assert mock_run_one.await_count == 3

    # Verify each call: spec grain/brand, model_name, experiment_name.
    for idx, brand in enumerate(brand_order):
        actual_call = mock_run_one.call_args_list[idx]
        _client_arg, spec_arg = actual_call.args
        assert _client_arg is fake_client
        assert spec_arg.grain == "hcp"
        assert spec_arg.brand == brand
        assert actual_call.kwargs["model_name"] == goldstd_model_name(HCP_ADOPTION_COHORT, brand)
        assert actual_call.kwargs["experiment_name"] == goldstd_experiment_name(
            HCP_ADOPTION_COHORT, brand
        )


@pytest.mark.asyncio
async def test_run_returns_dict_keyed_by_brand_lower():
    """run() returns a dict with 3 lowercase-brand keys mapping to mocked results."""
    fake_client = object()
    sentinel_results = [
        {
            "model": "m1",
            "holdout_auc": 0.70,
            "backtest_points": 30,
            "n_train": 600,
            "n_holdout": 150,
        },
        {
            "model": "m2",
            "holdout_auc": 0.72,
            "backtest_points": 31,
            "n_train": 610,
            "n_holdout": 151,
        },
        {
            "model": "m3",
            "holdout_auc": 0.74,
            "backtest_points": 32,
            "n_train": 620,
            "n_holdout": 152,
        },
    ]

    mock_run_one = AsyncMock(side_effect=sentinel_results)
    mock_resolve = AsyncMock(return_value=fake_client)

    with (
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._run_one_cohort",
            mock_run_one,
        ),
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._resolve_client",
            mock_resolve,
        ),
    ):
        from src.mlops.gold_standard_eval.run_hcp_cohorts import run

        result = await run(db=fake_client)

    expected_keys = {brand.lower() for brand in BRANDS}
    assert set(result.keys()) == expected_keys

    for idx, brand in enumerate(BRANDS):
        assert result[brand.lower()] is sentinel_results[idx]


@pytest.mark.asyncio
async def test_resolve_client_is_awaited_with_passed_db():
    """run() passes the db argument through to _resolve_client."""
    sentinel_db = object()
    fake_client = object()

    mock_run_one = AsyncMock(
        return_value={
            "model": "x",
            "holdout_auc": 0.5,
            "backtest_points": 1,
            "n_train": 100,
            "n_holdout": 50,
        }
    )
    mock_resolve = AsyncMock(return_value=fake_client)

    with (
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._run_one_cohort",
            mock_run_one,
        ),
        patch(
            "src.mlops.gold_standard_eval.run_hcp_cohorts._resolve_client",
            mock_resolve,
        ),
    ):
        from src.mlops.gold_standard_eval.run_hcp_cohorts import run

        await run(db=sentinel_db)

    mock_resolve.assert_awaited_once_with(sentinel_db)
