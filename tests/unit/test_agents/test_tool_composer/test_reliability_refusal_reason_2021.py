"""#2021 D3: get_tool_reliability's most common refusal reason survives ToolReliability.from_row."""

from src.agents.tool_composer.reliability import ToolReliability

_COUNTS = {
    "n_invoked": 12,
    "n_succeeded": 4,
    "n_refused": 8,
    "n_health_failures": 0,
    "n_health": 4,
    "n_retried": 0,
    "n_synthetic": 0,
}


def test_from_row_carries_the_most_common_refusal_reason():
    row = ToolReliability.from_row(
        {
            "tool_name": "causal_effect_estimator",
            **_COUNTS,
            "most_common_refusal_reason": "non_binary_treatment",
            "n_refused_coded": 3,
        }
    )
    assert row.most_common_refusal_reason == "non_binary_treatment"
    assert row.n_refused_coded == 3


def test_a_row_from_a_database_without_ml_043_reads_as_none():
    row = ToolReliability.from_row({"tool_name": "gap_calculator", **_COUNTS})
    assert row.most_common_refusal_reason is None
    # Unknown, not zero: a pre-043 database cannot say how many refusals were coded.
    assert row.n_refused_coded is None
