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

import math
import re
from enum import StrEnum
from typing import Dict, Mapping, Optional, Tuple, Union


class ReasonCode(StrEnum):
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
    UNSUPPORTED_REQUEST = "unsupported_request"
    DEGENERATE_DESIGN = "degenerate_design"
    SIMULATION_INCOMPLETE = "simulation_incomplete"
    EFFECT_NOT_ESTIMABLE = "effect_not_estimable"
    MISSING_REQUIRED_COLUMN = "missing_required_column"
    NO_TREATMENT_CONTRAST = "no_treatment_contrast"
    ESTIMATOR_FAILED = "estimator_failed"

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
    ReasonCode.UNSUPPORTED_REQUEST: "the tool cannot answer a question of this shape",
    ReasonCode.DEGENERATE_DESIGN: "the study design the inputs imply is too small to support a valid comparison",
    ReasonCode.SIMULATION_INCOMPLETE: "the twin simulation did not complete",
    ReasonCode.EFFECT_NOT_ESTIMABLE: "the cohort data cannot support a causal effect estimate for this intervention",
    ReasonCode.MISSING_REQUIRED_COLUMN: "a column the estimate requires is not present in the data",
    ReasonCode.NO_TREATMENT_CONTRAST: "the treatment does not split the rows into a treated and a comparison group",
    ReasonCode.ESTIMATOR_FAILED: "the effect estimator could not produce an estimate from data that passed its checks",
    ReasonCode.MISSING_REQUIRED_INPUT: "a required input was not provided",
    ReasonCode.INVALID_INPUT_TYPE: "an input was of the wrong type",
    ReasonCode.INVALID_INPUT_VALUE: "an input value was outside what the tool accepts",
    ReasonCode.UNEXPECTED_INPUT: "an input was supplied that this tool does not accept",
    ReasonCode.TOOL_ERROR: "the tool failed to complete",
    ReasonCode.TOOL_TIMEOUT: "the tool exceeded its time budget",
    ReasonCode.PLAN_DEFECT: "the plan called this tool incorrectly",
    ReasonCode.REFERENCE_UNRESOLVABLE: "the plan referred to a result that does not exist",
    ReasonCode.DEPENDENCY_UNMET: "a step this one depends on did not produce a result",
    ReasonCode.CIRCUIT_OPEN: "the tool was temporarily withheld after repeated failures",
    ReasonCode.TOOL_NOT_REGISTERED: "the plan named a tool that is not registered",
}

#: Codes that describe what the executor observed. A tool raise site may not use them
#: (the coverage tests enforce it); the constructor's fail-soft TOOL_ERROR is runtime only.
EXECUTOR_ASSIGNED = frozenset(
    {
        ReasonCode.TOOL_ERROR,
        ReasonCode.TOOL_TIMEOUT,
        ReasonCode.PLAN_DEFECT,
        ReasonCode.REFERENCE_UNRESOLVABLE,
        ReasonCode.DEPENDENCY_UNMET,
        ReasonCode.CIRCUIT_OPEN,
        ReasonCode.TOOL_NOT_REGISTERED,
    }
)

#: The only outcome classes a TOOL authors (#2020). The executor sets these two solely in the arm
#: that catches ToolRefusalError / ToolInputError, and since #2021 always with a reason code.
TOOL_AUTHORED_CLASSES = frozenset({"refused", "input_rejected"})

#: The reason code for each twin ``EffectCause`` (#2021 9b), keyed by its string value so the
#: tool composer does not import ``src.digital_twin`` at load time. A reused code is one whose
#: sentence is literally true for the cause. ``test_effect_reason_codes_2021`` pins that every
#: cause is here; ``test_reason_code_coverage_2021`` that every value is a literal tool member.
EFFECT_CAUSE_CODES: Dict[str, ReasonCode] = {
    "intervention_not_identified": ReasonCode.EFFECT_NOT_ESTIMABLE,
    "empty_cohort": ReasonCode.NO_USABLE_ROWS,
    "required_column_missing": ReasonCode.MISSING_REQUIRED_COLUMN,
    "too_few_usable_rows": ReasonCode.INSUFFICIENT_SAMPLE,
    "no_treatment_contrast": ReasonCode.NO_TREATMENT_CONTRAST,
    "target_region_not_covered": ReasonCode.COVERAGE_GAP,
    "estimation_failed": ReasonCode.ESTIMATOR_FAILED,
    "target_inference_failed": ReasonCode.ESTIMATOR_FAILED,
}


def effect_reason_code(cause: object, *, fallback: ReasonCode) -> ReasonCode:
    """The code for a twin effect cause, or ``fallback`` when there is none or this build does
    not know it — so an effect refusal never goes without a code."""
    return fallback if cause is None else EFFECT_CAUSE_CODES.get(str(cause), fallback)


def user_safe_failure_text(
    outcome_class: Optional[str],
    reason_code: Union[ReasonCode, str, None],
    raw: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """What a failed step may say to a user or an LLM prompt (#2020), and what it withheld.

    One rule, shared by the composer's fail-closed answer and the synthesis prompt so the two
    cannot drift:

    * verbatim only when the class is tool-authored, the code is a KNOWN member and the text is
      non-empty (the refusal arm's constructor fails soft to tool_error, so an unknown code on an
      authored class came from a foreign producer);
    * otherwise ``"<canonical sentence> [<code>]"``, where an unknown or absent code renders as
      tool_error in both the sentence and the tag (``reason_code`` is a plain string on the model);
    * ``(None, None)`` when there is no text and no code.

    Returns ``(fragment, withheld)``: ``withheld`` is the raw text the fragment replaced, else
    ``None``. Pure — no prefix and no logging; each caller prefixes, and logs ``withheld``.
    """
    text = str(raw or "").strip()
    if not text and not reason_code:
        return None, None
    known = reason_code in ReasonCode
    if outcome_class in TOOL_AUTHORED_CLASSES and text and known:
        return text, None
    code = str(reason_code) if known else ReasonCode.TOOL_ERROR.value
    return f"{canonical_sentence(code)} [{code}]", text or None


# Bounds on the structured ``details`` payload. It is persisted, so it must stay
# structure: counts, shares and flags under prefixed snake_case keys. No strings at any
# length — a short one still fits a column or brand name, the data this module keeps out
# of the database. Ints stay within JavaScript's safe range (Number.MAX_SAFE_INTEGER,
# inclusive): the admin page reads them.
_MAX_DETAIL_KEYS = 8
_MAX_DETAIL_INT = 2**53 - 1
# At most "share_" (6) + 58 = 64 characters: ml/043's reducer silently drops keys longer than 64.
_DETAIL_KEY = re.compile(r"(n|is|has|share)_[a-z0-9_]{1,58}")


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


def known_sentence(code: Union[ReasonCode, str, None]) -> Optional[str]:
    """The sentence for a code this build knows, else ``None``. For read-side display.

    Unlike :func:`canonical_sentence`, an unknown code does NOT fall back to the generic
    tool-failure sentence: an operator page must not relabel a refusal it cannot read as a tool
    failure. The caller shows the bare code instead.
    """
    if isinstance(code, ReasonCode):
        return CANONICAL_SENTENCES[code]
    if isinstance(code, str):
        try:
            return CANONICAL_SENTENCES[ReasonCode(code)]
        except ValueError:
            return None
    return None


def validate_details(details: Mapping[str, object]) -> Dict[str, object]:
    """Reject a ``details`` payload that is really free text; return a normalized copy.

    The copy holds builtin ``bool`` / ``int`` / ``float`` only: a numpy scalar is unwrapped
    first (without importing numpy) and then checked like any other value. Rejection
    messages carry no values, and name a key only once it has passed the key rule, because
    the error constructor logs them.
    """
    if not isinstance(details, Mapping):
        raise ValueError(f"details is a {type(details).__name__}, not a mapping")
    if len(details) > _MAX_DETAIL_KEYS:
        raise ValueError(f"details carries {len(details)} keys; at most {_MAX_DETAIL_KEYS}")
    normalized: Dict[str, object] = {}
    for key, value in details.items():
        # The key is persisted too, so it gets the same rule as the value.
        if not isinstance(key, str) or not _DETAIL_KEY.fullmatch(key):
            raise ValueError("details key is not an n_/is_/has_/share_ snake_case identifier")
        # A (str, Enum) key renders as its member name under str(); store the builtin value.
        key = str.__str__(key)
        item = getattr(value, "item", None)
        if type(value).__module__ == "numpy" and callable(item):
            try:
                value = item()
            except (TypeError, ValueError):
                raise ValueError(
                    f"details[{key!r}] is a numpy value that is not a scalar"
                ) from None
        # bool is an int subclass, so this admits booleans; str, None and containers fail.
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"details[{key!r}] is a {type(value).__name__}; details holds numbers and "
                "booleans only"
            )
        if isinstance(value, bool):
            normalized[key] = bool(value)
        elif isinstance(value, int):
            if abs(value) > _MAX_DETAIL_INT:
                raise ValueError(f"details[{key!r}] is outside JavaScript's safe integer range")
            normalized[key] = int(value)
        elif not math.isfinite(value):
            raise ValueError(f"details[{key!r}] is not a finite number")
        else:
            normalized[key] = float(value)
    return normalized
