"""Regression tests for #1583 — composer memory contribution must survive
non-JSON-safe payload members.

Measured mechanism (2026-08-13 forced q08 replay, deploy b338d78e): the
executor threads the real cohort ``pandas.DataFrame`` into EVERY step's
``ToolInput.parameters``/``context`` (``_maybe_autopopulate_dataframe``), so
``CompositionResult.model_dump(mode="json")`` in
``ToolComposer._contribute_to_memory`` raises
``PydanticSerializationError: Unable to serialize unknown type:
<class 'pandas.core.frame.DataFrame'>`` BEFORE ``contribute_to_memory`` is
reached — the whole contribution (working cache + episodic + procedural) is
dropped with only a warning.

These tests exercise the public seam (``_contribute_to_memory``) and assert on
the payload handed to ``contribute_to_memory``, not on any helper's internals.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    ComposedResponse,
    CompositionResult,
    DecompositionResult,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    ToolInput,
    ToolOutput,
)

CONTRIBUTION_COUNTS = {
    "episodic_stored": 1,
    "procedural_stored": 1,
    "working_cached": 1,
}


class _Opaque:
    """An object no serializer can encode — stands in for a stray client/handle."""

    def __repr__(self) -> str:  # pragma: no cover - only for debugging output
        return "<_Opaque secret=hunter2>"


def _make_result(
    *,
    step_context: Optional[Dict[str, Any]] = None,
    step_parameters: Optional[Dict[str, Any]] = None,
    tool_result: Optional[Dict[str, Any]] = None,
    supporting_data: Optional[Dict[str, Any]] = None,
) -> CompositionResult:
    """Build a production-shaped CompositionResult with one executed step."""
    sub_question = SubQuestion(
        id="sq1",
        question="What is the Kisqali west-region conversion rate?",
        intent="metric_lookup",
    )
    decomposition = DecompositionResult(
        original_query="Kisqali west conversion rate?",
        sub_questions=[sub_question],
        decomposition_reasoning="single metric lookup",
    )
    plan = ExecutionPlan(
        decomposition=decomposition,
        steps=[
            ExecutionStep(
                step_id="step1",
                sub_question_id="sq1",
                tool_name="cohort_statistics",
                source_agent="cohort_constructor",
            )
        ],
        tool_mappings=[],
        planning_reasoning="one tool suffices",
    )
    started = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 8, 13, 12, 0, 5, tzinfo=timezone.utc)

    trace = ExecutionTrace(plan_id=plan.plan_id, started_at=started, completed_at=completed)
    trace.add_result(
        StepResult(
            step_id="step1",
            sub_question_id="sq1",
            tool_name="cohort_statistics",
            input=ToolInput(
                tool_name="cohort_statistics",
                parameters={"brand": "Kisqali", **(step_parameters or {})},
                context={"session_id": "sess-1", **(step_context or {})},
            ),
            output=ToolOutput(
                tool_name="cohort_statistics",
                success=True,
                result={"conversion_rate": 0.6364, **(tool_result or {})},
                execution_time_ms=5000,
            ),
            status=ExecutionStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
        )
    )

    response = ComposedResponse(
        answer="Kisqali west conversion rate is 63.6%.",
        confidence=0.86,
        supporting_data={"conversion_rate": 0.6364, **(supporting_data or {})},
        citations=["step1"],
        synthesis_reasoning="single tool result",
        timestamp=completed,
    )

    return CompositionResult(
        query="Kisqali west conversion rate?",
        session_id="sess-1",
        decomposition=decomposition,
        plan=plan,
        execution=trace,
        response=response,
        total_duration_ms=5000,
        started_at=started,
        completed_at=completed,
    )


async def _contributed_payload(
    composer: ToolComposer,
    result: CompositionResult,
) -> Dict[str, Any]:
    """Run the contribution seam and return the payload it handed downstream."""
    with patch(
        "src.agents.tool_composer.composer.contribute_to_memory",
        new_callable=AsyncMock,
        return_value=dict(CONTRIBUTION_COUNTS),
    ) as mock_contribute:
        await composer._contribute_to_memory(
            result,
            {"session_id": "sess-1", "brand": "Kisqali", "region": "west"},
        )

    assert mock_contribute.await_count == 1, (
        "memory contribution was dropped entirely — #1583: one unserializable "
        "member must not silently no-op the whole contribution"
    )
    payload = mock_contribute.await_args.kwargs["result"]
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def composer(mock_llm_client, mock_tool_registry) -> ToolComposer:
    return ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        memory_hooks=AsyncMock(),
        enable_memory_contribution=True,
    )


class TestDataFrameMembers:
    """A DataFrame anywhere in the payload must not drop the contribution."""

    @pytest.mark.asyncio
    async def test_dataframe_member_keeps_contribution_and_lands_as_summary(self, composer):
        frame = pd.DataFrame(
            {
                "region": ["west", "east", "central"],
                "trx": [7128, 7495, 4210],
            }
        )
        result = _make_result(
            step_context={"estimation_data": frame},
            step_parameters={"estimation_data": frame},
            supporting_data={"cohort_frame": frame},
        )

        payload = await _contributed_payload(composer, result)

        # The JSON-safe members still land, untouched.
        assert payload["query"] == "Kisqali west conversion rate?"
        assert payload["response"]["answer"] == "Kisqali west conversion rate is 63.6%."
        assert payload["response"]["supporting_data"]["conversion_rate"] == 0.6364
        step = payload["execution"]["step_results"][0]
        assert step["tool_name"] == "cohort_statistics"
        assert step["input"]["parameters"]["brand"] == "Kisqali"
        assert step["output"]["result"]["conversion_rate"] == 0.6364

        # The frame lands as a compact structured summary, at every site.
        for summary in (
            payload["response"]["supporting_data"]["cohort_frame"],
            step["input"]["parameters"]["estimation_data"],
            step["input"]["context"]["estimation_data"],
        ):
            assert isinstance(summary, dict)
            assert summary["__type__"] == "pandas.core.frame.DataFrame"
            assert summary["__summarized__"] is True
            assert summary["shape"] == [3, 2]
            assert summary["columns"] == ["region", "trx"]
            assert summary["dtypes"] == ["object", "int64"]

    @pytest.mark.asyncio
    async def test_duplicate_column_labels_keep_every_dtype(self, composer):
        """A sloppy merge can leave duplicate labels — none may be dropped."""
        frame = pd.DataFrame([[1, "west"], [2, "east"]], columns=["trx", "trx"])
        result = _make_result(step_context={"estimation_data": frame})

        payload = await _contributed_payload(composer, result)

        summary = payload["execution"]["step_results"][0]["input"]["context"]["estimation_data"]
        assert summary["columns"] == ["trx", "trx"]
        assert summary["dtypes"] == ["int64", "object"]

    @pytest.mark.asyncio
    async def test_frame_contents_are_not_dumped_into_the_payload(self, composer):
        """Production frames run to ~37.5k rows — the summary must stay bounded."""
        frame = pd.DataFrame(
            {
                "patient_id": [f"pat_{i}" for i in range(5000)],
                "trx": np.arange(5000),
            }
        )
        result = _make_result(step_context={"estimation_data": frame})

        payload = await _contributed_payload(composer, result)
        serialized = json.dumps(payload)

        assert "pat_4999" not in serialized
        assert len(serialized) < 8000, (
            f"payload grew to {len(serialized)} chars — frame contents must be "
            "summarized, never serialized into the memory contribution"
        )
        summary = payload["execution"]["step_results"][0]["input"]["context"]["estimation_data"]
        assert summary["shape"] == [5000, 2]
        assert summary["columns"] == ["patient_id", "trx"]


class TestJsonSafePayloadUnchanged:
    """No unserializable member => byte-identical to the pre-#1583 behaviour."""

    @pytest.mark.asyncio
    async def test_clean_result_payload_matches_plain_model_dump(self, composer):
        result = _make_result(
            tool_result={"n_patients": 7128, "segments": ["A", "B"]},
            supporting_data={"trend": [1.0, 2.0, 3.0]},
        )

        payload = await _contributed_payload(composer, result)

        # Byte-identical, not merely equal: the encoded JSON must match what
        # the pre-#1583 `model_dump(mode="json")` produced.
        assert json.dumps(payload, sort_keys=True) == json.dumps(
            result.model_dump(mode="json"), sort_keys=True
        )
        assert payload == result.model_dump(mode="json")


class TestNumpyAndPandasScalars:
    """The realistic non-JSON-safe set, not just DataFrame."""

    @pytest.mark.asyncio
    async def test_numpy_scalars_preserved_arrays_and_series_summarized(self, composer):
        result = _make_result(
            tool_result={
                "n_rows": np.int64(37515),
                "converged": np.bool_(True),
                "coefficients": np.array([0.1, 0.25]),
                "monthly_trx": pd.Series([10, 20, 30], name="trx"),
                "index": pd.Index(["west", "east"]),
            }
        )

        payload = await _contributed_payload(composer, result)
        tool_result = payload["execution"]["step_results"][0]["output"]["result"]

        # Scalars are lossless natives (segments.py:_to_native precedent).
        assert tool_result["n_rows"] == 37515
        assert isinstance(tool_result["n_rows"], int)
        assert tool_result["converged"] is True

        # Containers are summarized.
        assert tool_result["coefficients"]["__type__"] == "numpy.ndarray"
        assert tool_result["coefficients"]["shape"] == [2]
        assert tool_result["coefficients"]["dtype"] == "float64"

        assert tool_result["monthly_trx"]["__type__"] == "pandas.core.series.Series"
        assert tool_result["monthly_trx"]["length"] == 3
        assert tool_result["monthly_trx"]["dtype"] == "int64"
        assert tool_result["monthly_trx"]["name"] == "trx"

        assert tool_result["index"]["length"] == 2


class TestGenuinelyUnserializableRemainder:
    """An arbitrary object still warns honestly — but keeps the rest."""

    @pytest.mark.asyncio
    async def test_opaque_object_warns_and_rest_of_payload_lands(self, composer, caplog):
        result = _make_result(
            tool_result={"handle": _Opaque(), "conversion_rate": 0.6364},
            supporting_data={"trend": [1.0, 2.0]},
        )

        with caplog.at_level(logging.WARNING):
            payload = await _contributed_payload(composer, result)

        tool_result = payload["execution"]["step_results"][0]["output"]["result"]
        assert tool_result["conversion_rate"] == 0.6364
        assert payload["response"]["supporting_data"]["trend"] == [1.0, 2.0]

        marker = tool_result["handle"]
        assert marker["__unserializable__"] is True
        assert marker["__type__"].endswith("_Opaque")

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("_Opaque" in message for message in warnings), (
            f"expected an honest warning naming the unserializable type, got {warnings}"
        )
        assert not any("Failed to contribute to memory" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_opaque_object_repr_is_not_leaked_into_memory(self, composer):
        """Memory contributions are persisted — an arbitrary repr can carry secrets."""
        result = _make_result(tool_result={"handle": _Opaque()})

        payload = await _contributed_payload(composer, result)

        assert "hunter2" not in json.dumps(payload)
