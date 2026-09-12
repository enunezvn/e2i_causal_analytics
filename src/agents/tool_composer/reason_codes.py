"""The closed reason-code vocabulary for composable-tool refusals and failures (#2021).

Why a code and not the message. A refusal's human message is written for the person
reading the answer and interpolates the data that caused it — column names, value
reprs, kwargs keys. That makes it useless as an aggregation key (every refusal is its
own bucket) and unsafe to persist: ml/041's recording contract stores structure only,
"no error text is stored" (041 header, §5.5). The code is the aggregation key; the
canonical sentence is the data-free rendering that is safe to store and to show.

Adding a member REQUIRES adding its canonical sentence in the same edit —
``test_every_code_has_a_canonical_sentence`` fails otherwise.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Union


class ReasonCode(str, Enum):
    """Why a step did not produce a result. Closed set; stable wire values."""

    # --- Tool-authored refusals: the inputs are legal but cannot answer the question
    NON_BINARY_TREATMENT = "non_binary_treatment"
    SINGLE_CLASS_TREATMENT = "single_class_treatment"
    NON_BINARY_OUTCOME = "non_binary_outcome"
    SINGLE_CLASS_OUTCOME = "single_class_outcome"
    NO_USABLE_ROWS = "no_usable_rows"
    NO_USABLE_COLUMNS = "no_usable_columns"
    NON_FINITE_INPUT = "non_finite_input"
    INSUFFICIENT_GROUPS = "insufficient_groups"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    COVERAGE_GAP = "coverage_gap"
    UNKNOWN_COLUMN = "unknown_column"
    NON_NUMERIC_COLUMN = "non_numeric_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    AMBIGUOUS_GROUP_LABEL = "ambiguous_group_label"
    MISSING_DATAFRAME = "missing_dataframe"
    CI_OUTSIDE_ESTIMATE = "ci_outside_estimate"
    POINT_ESTIMATE_ONLY = "point_estimate_only"
    UPSTREAM_STEP_FAILED = "upstream_step_failed"
    UNSUPPORTED_REQUEST = "unsupported_request"
    DEGENERATE_DESIGN = "degenerate_design"

    # --- Tool-authored input rejections: the value is not a legal input
    MISSING_REQUIRED_INPUT = "missing_required_input"
    INVALID_INPUT_TYPE = "invalid_input_type"
    INVALID_INPUT_VALUE = "invalid_input_value"
    UNEXPECTED_INPUT = "unexpected_input"

    # --- Executor-assigned: no tool authored these
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    PLAN_DEFECT = "plan_defect"
    REFERENCE_UNRESOLVABLE = "reference_unresolvable"
    DEPENDENCY_UNMET = "dependency_unmet"
    CIRCUIT_OPEN = "circuit_open"
    TOOL_NOT_REGISTERED = "tool_not_registered"


CANONICAL_SENTENCES: Dict[ReasonCode, str] = {
    ReasonCode.NON_BINARY_TREATMENT: "the treatment column is not a binary 0/1 indicator",
    ReasonCode.SINGLE_CLASS_TREATMENT: "the treatment column has only one class, so there is nothing to compare",
    ReasonCode.NON_BINARY_OUTCOME: "the outcome column is not a binary 0/1 indicator",
    ReasonCode.SINGLE_CLASS_OUTCOME: "the outcome column has only one class, so there is nothing to discriminate",
    ReasonCode.NO_USABLE_ROWS: "no rows remained that the analysis could use",
    ReasonCode.NO_USABLE_COLUMNS: "no usable columns of the required kind were present",
    ReasonCode.NON_FINITE_INPUT: "the inputs contained missing or non-finite values where finite numbers are required",
    ReasonCode.INSUFFICIENT_GROUPS: "fewer groups were present than the comparison requires",
    ReasonCode.INSUFFICIENT_SAMPLE: "too few observations were available to support an estimate",
    ReasonCode.COVERAGE_GAP: "the data does not cover everything the question asked about",
    ReasonCode.UNKNOWN_COLUMN: "a column the request named is not present in the data",
    ReasonCode.NON_NUMERIC_COLUMN: "a column the analysis has to average does not hold numbers",
    ReasonCode.AMBIGUOUS_COLUMN: "a column the request named matches more than one column in the data",
    ReasonCode.AMBIGUOUS_GROUP_LABEL: "two or more groups share a label, so a per-group number cannot be attributed to either",
    ReasonCode.MISSING_DATAFRAME: "the real source data was not supplied to the tool",
    ReasonCode.CI_OUTSIDE_ESTIMATE: "the confidence interval is inconsistent with the point estimate",
    ReasonCode.POINT_ESTIMATE_ONLY: "no uncertainty could be computed for the estimate",
    ReasonCode.UPSTREAM_STEP_FAILED: "an earlier step this one depends on did not produce a result",
    ReasonCode.UNSUPPORTED_REQUEST: "the tool cannot answer a question of this shape",
    ReasonCode.DEGENERATE_DESIGN: "the study design the inputs imply is too small to support a valid comparison",
    ReasonCode.MISSING_REQUIRED_INPUT: "a required input was not provided",
    ReasonCode.INVALID_INPUT_TYPE: "an input was of the wrong type",
    ReasonCode.INVALID_INPUT_VALUE: "an input value was outside what the tool accepts",
    ReasonCode.UNEXPECTED_INPUT: "an input was supplied that this tool does not accept",
    ReasonCode.TOOL_ERROR: "the tool failed to complete",
    ReasonCode.TOOL_TIMEOUT: "the tool exceeded its time budget",
    ReasonCode.PLAN_DEFECT: "the plan called this tool incorrectly",
    ReasonCode.REFERENCE_UNRESOLVABLE: "the plan referred to a result that does not exist",
    ReasonCode.DEPENDENCY_UNMET: "a step this one depends on did not run",
    ReasonCode.CIRCUIT_OPEN: "the tool was temporarily withheld after repeated failures",
    ReasonCode.TOOL_NOT_REGISTERED: "the plan named a tool that is not registered",
}

# Bounds on the structured ``details`` payload. It is persisted, so it must stay
# structure: short scalar values, never a sentence smuggled back in as a string.
_MAX_DETAIL_KEYS = 8
_MAX_DETAIL_STR = 64


def canonical_sentence(code: Union[ReasonCode, str, None]) -> str:
    """The data-free sentence for ``code``; the generic one for anything unknown.

    Accepts a raw string because the recorder and the composer read codes back off
    Pydantic models, where the value has already been coerced to ``str``.
    """
    if isinstance(code, ReasonCode):
        return CANONICAL_SENTENCES[code]
    if isinstance(code, str):
        try:
            return CANONICAL_SENTENCES[ReasonCode(code)]
        except ValueError:
            pass
    return CANONICAL_SENTENCES[ReasonCode.TOOL_ERROR]


def validate_details(details: Dict[str, object]) -> Dict[str, object]:
    """Reject a ``details`` payload that is really free text. Returns it unchanged."""
    if len(details) > _MAX_DETAIL_KEYS:
        raise ValueError(f"details carries {len(details)} keys; at most {_MAX_DETAIL_KEYS}")
    for key, value in details.items():
        if isinstance(value, str) and len(value) > _MAX_DETAIL_STR:
            raise ValueError(
                f"details[{key!r}] is {len(value)} characters; details is structure, not prose "
                f"(at most {_MAX_DETAIL_STR})"
            )
    return details
