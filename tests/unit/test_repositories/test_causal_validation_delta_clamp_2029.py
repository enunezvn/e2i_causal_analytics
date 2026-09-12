"""#2029: every persisted row's delta_percent is clamped to the column bound at the write
boundary, with the exact value kept in details_json.delta_percent_exact.

Lane G (#2007) clamped only the negative-control test inside the runner; the four
perturbation tests compute ``delta_percent`` unclamped, and ``save_suite`` inserts
the whole suite in one call, so one overflowing row dropped every row. The clamp
now lives in ``CausalValidationRepository`` -- once, for every writer.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.repositories.causal_validation import (
    DELTA_PERCENT_COLUMN_MAX,
    CausalValidationRepository,
)


def _suite(delta: float) -> RefutationSuite:
    test = RefutationResult(
        test_name=RefutationTestType.PLACEBO_TREATMENT,
        status=RefutationStatus.PASSED,
        original_effect=1e-8,
        refuted_effect=0.01,
        p_value=0.4,
        delta_percent=delta,
        details={"message": "m", "config": {"n": 2}},
    )
    return RefutationSuite(
        passed=True,
        confidence_score=0.9,
        tests=[test],
        gate_decision=GateDecision.PROCEED,
        treatment_variable="t",
        outcome_variable="y",
        brand="B",
    )


def _row(suite: RefutationSuite) -> dict:
    return CausalValidationRepository()._test_to_row(
        test=suite.tests[0], suite=suite, estimate_id="e", estimate_source="causal_paths"
    )


def test_bound_is_the_numeric_12_4_maximum():
    assert DELTA_PERCENT_COLUMN_MAX == 99999999.9999


def test_row_is_clamped_and_exact_value_kept():
    row = _row(_suite(1e8))  # 0.01 / 1e-8 * 100
    assert row["delta_percent"] == DELTA_PERCENT_COLUMN_MAX
    assert row["details_json"]["delta_percent_exact"] == 1e8


def test_row_below_bound_is_untouched():
    row = _row(_suite(42.5))
    assert row["delta_percent"] == 42.5
    assert "delta_percent_exact" not in row["details_json"]


def test_row_at_the_bound_is_not_marked_clamped():
    row = _row(_suite(DELTA_PERCENT_COLUMN_MAX))
    assert row["delta_percent"] == DELTA_PERCENT_COLUMN_MAX
    assert "delta_percent_exact" not in row["details_json"]


def test_clamp_does_not_mutate_the_result_details():
    suite = _suite(1e8)
    _row(suite)
    assert "delta_percent_exact" not in suite.tests[0].details  # the row gets a copy


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_delta_is_left_alone_not_saturated(value):
    """+inf > MAX is True in Python; saturating it would fabricate a bounded,
    plausible-looking number for a value that is not a measurement. The clamp
    only bounds finite values."""
    row = _row(_suite(value))
    assert isinstance(row["delta_percent"], float)
    assert not math.isfinite(row["delta_percent"])
    assert "delta_percent_exact" not in row["details_json"]


def test_none_delta_stays_none():
    row = _row(_suite(None))  # type: ignore[arg-type]
    assert row["delta_percent"] is None
    assert "delta_percent_exact" not in row["details_json"]


@pytest.mark.asyncio
async def test_save_single_test_clamps_at_the_same_boundary():
    """The single-row writer builds its row separately from ``_test_to_row``;
    it must clamp identically."""
    client = MagicMock()
    insert = client.table.return_value.insert
    insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"validation_id": "v1"}]))
    repo = CausalValidationRepository()
    repo.client = client
    suite = _suite(1e8)
    vid = await repo.save_single_test(
        suite.tests[0], estimate_id="e", gate_decision=GateDecision.PROCEED, confidence_score=0.9
    )
    assert vid == "v1"
    row = insert.call_args[0][0]
    assert row["delta_percent"] == DELTA_PERCENT_COLUMN_MAX
    assert row["details_json"]["delta_percent_exact"] == 1e8
    assert row["test_config"] == {"n": 2}
    assert "delta_percent_exact" not in suite.tests[0].details


@pytest.mark.asyncio
async def test_save_suite_persists_every_row_when_one_overflows():
    """End to end through the bulk writer (spec section 4, "near-zero claim"):
    the suite goes to Supabase in ONE ``insert(rows)`` call, so the overflowing
    row must arrive clamped (exact value kept) and the ordinary row untouched --
    and no row is dropped."""
    overflowing = _suite(1e8).tests[0]  # 0.01 / 1e-8 * 100
    ordinary = RefutationResult(
        test_name=RefutationTestType.DATA_SUBSET,
        status=RefutationStatus.PASSED,
        original_effect=0.10,
        refuted_effect=0.1125,
        p_value=0.3,
        delta_percent=12.5,
        details={"message": "m2", "config": {"n": 3}},
    )
    suite = RefutationSuite(
        passed=True,
        confidence_score=0.9,
        tests=[overflowing, ordinary],
        gate_decision=GateDecision.PROCEED,
        treatment_variable="t",
        outcome_variable="y",
        brand="B",
    )
    client = MagicMock()
    insert = client.table.return_value.insert
    insert.return_value.execute = AsyncMock(
        return_value=MagicMock(data=[{"validation_id": "v1"}, {"validation_id": "v2"}])
    )
    repo = CausalValidationRepository()
    repo.client = client

    ids = await repo.save_suite(suite, estimate_id="e")

    assert ids == ["v1", "v2"]
    client.table.assert_called_once_with("causal_validations")
    rows = insert.call_args[0][0]
    assert len(rows) == 2
    clamped, plain = rows
    assert clamped["test_type"] == RefutationTestType.PLACEBO_TREATMENT.value
    assert clamped["delta_percent"] == DELTA_PERCENT_COLUMN_MAX
    assert clamped["details_json"]["delta_percent_exact"] == 1e8
    assert plain["test_type"] == RefutationTestType.DATA_SUBSET.value
    assert plain["delta_percent"] == 12.5
    assert "delta_percent_exact" not in plain["details_json"]
    assert "delta_percent_exact" not in overflowing.details
