"""#2021: refusals carry a closed reason code and a data-free canonical sentence."""

import pytest

from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.agents.tool_composer.reason_codes import (
    CANONICAL_SENTENCES,
    ReasonCode,
    canonical_sentence,
)


def test_every_code_has_a_canonical_sentence():
    """The catalogue is total: a code without a sentence cannot be displayed or stored."""
    missing = [c.value for c in ReasonCode if c not in CANONICAL_SENTENCES]
    assert missing == [], f"codes with no canonical sentence: {missing}"


def test_canonical_sentences_carry_no_interpolation():
    """The sentence is persisted and shown; it must never be a format string."""
    for code, sentence in CANONICAL_SENTENCES.items():
        assert "{" not in sentence and "}" not in sentence, code
        assert "%s" not in sentence, code
        assert sentence and sentence[0].islower(), code
        assert len(sentence) <= 160, code


def test_refusal_carries_code_and_details():
    err = ToolRefusalError(
        "carries 4 distinct non-null values, including [0, 1, 2, 3].",
        reason_code=ReasonCode.NON_BINARY_TREATMENT,
        details={"n_distinct": 4},
    )
    assert err.reason_code is ReasonCode.NON_BINARY_TREATMENT
    assert err.details == {"n_distinct": 4}
    assert err.canonical_sentence == canonical_sentence(ReasonCode.NON_BINARY_TREATMENT)
    assert "4 distinct" in str(err), "the human message is preserved unchanged"


def test_input_error_carries_code():
    err = ToolInputError(
        "expected_effect must not be None",
        reason_code=ReasonCode.MISSING_REQUIRED_INPUT,
    )
    assert err.reason_code is ReasonCode.MISSING_REQUIRED_INPUT
    assert isinstance(err, ValueError)


def test_code_is_required():
    """A refusal without a code is the defect #2021 exists to remove."""
    with pytest.raises(TypeError):
        ToolRefusalError("no code given")  # type: ignore[call-arg]


def test_details_must_be_structure_only():
    """details is persisted; it may not smuggle free text back in."""
    with pytest.raises(ValueError, match="details"):
        ToolRefusalError(
            "x",
            reason_code=ReasonCode.NO_USABLE_ROWS,
            details={"message": "a sentence that is really free text " * 5},
        )


@pytest.mark.parametrize(
    "details",
    [
        pytest.param({"brand": "Kisqali"}, id="short-string"),
        pytest.param({"column": ""}, id="empty-string"),
        pytest.param({"n_rows": None}, id="none"),
        pytest.param({"cols": ["free text" * 5]}, id="list"),
        pytest.param({"scope": {"n": 1}}, id="dict"),
        pytest.param({"ratio": float("nan")}, id="nan"),
        pytest.param({"ratio": float("inf")}, id="inf"),
        pytest.param({"ratio": float("-inf")}, id="neg-inf"),
        pytest.param({"NRows": 1}, id="camel-key"),
        pytest.param({"1_rows": 1}, id="digit-first-key"),
        pytest.param({"n rows": 1}, id="space-key"),
        pytest.param({"": 1}, id="empty-key"),
        pytest.param({1: 1}, id="non-string-key"),
    ],
)
def test_details_values_are_finite_numbers_or_booleans_under_snake_case_keys(details):
    """L1: a string of any length fits a column or brand name — the data the lane keeps
    out of the database — so details carries numbers and booleans only."""
    with pytest.raises(ValueError, match="details"):
        ToolRefusalError("x", reason_code=ReasonCode.NO_USABLE_ROWS, details=details)


def test_details_accepts_ints_finite_floats_and_booleans():
    details = {"n_rows": 0, "share": 0.25, "is_binary": False, "n_2": -3}
    err = ToolRefusalError("x", reason_code=ReasonCode.NO_USABLE_ROWS, details=details)
    assert err.details == details


def test_details_key_count_is_still_bounded():
    with pytest.raises(ValueError, match="details"):
        ToolRefusalError(
            "x",
            reason_code=ReasonCode.NO_USABLE_ROWS,
            details={f"k_{i}": i for i in range(9)},
        )


# --- R1 recodes: each pin fails on the code the site carried before the review


def test_segment_ranker_without_an_effect_map_is_a_missing_input_not_a_failed_upstream():
    """The executor already short-circuits a step whose upstream FAILED
    (``dependency_unmet``), so this guard fires on a wrong-shape or literal input."""
    from src.agents.tool_composer.tool_registrations import segment_ranker

    with pytest.raises(ToolRefusalError) as caught:
        segment_ranker(cate_results={})
    assert caught.value.reason_code is ReasonCode.MISSING_REQUIRED_INPUT


def test_roi_estimator_without_a_gap_is_a_missing_input_not_a_failed_upstream():
    from src.agents.tool_composer.tool_registrations import roi_estimator

    with pytest.raises(ToolRefusalError) as caught:
        roi_estimator(gap_analysis={}, investment=1.0)
    assert caught.value.reason_code is ReasonCode.MISSING_REQUIRED_INPUT


@pytest.mark.parametrize(
    ("value", "code"),
    [
        pytest.param("a", ReasonCode.INVALID_INPUT_TYPE, id="string"),
        pytest.param(True, ReasonCode.INVALID_INPUT_TYPE, id="bool"),
        pytest.param([0.2], ReasonCode.INVALID_INPUT_TYPE, id="list"),
        pytest.param(float("nan"), ReasonCode.NON_FINITE_INPUT, id="nan"),
        pytest.param(float("inf"), ReasonCode.NON_FINITE_INPUT, id="inf"),
    ],
)
def test_power_number_tells_a_wrong_type_from_a_non_finite_number(value, code):
    from src.agents.tool_composer.tool_registrations import _power_number

    with pytest.raises(ToolInputError) as caught:
        _power_number("x", value)
    assert caught.value.reason_code is code
    assert str(caught.value) == (
        f"power_calculator: x must be a finite number; got {value!r}. No sample "
        "size can be computed from it."
    ), "the message is unchanged by the split"


def test_canonical_sentence_accepts_a_raw_string_code():
    """The recorder and composer read codes back off models as plain strings."""
    assert canonical_sentence("non_binary_treatment") == canonical_sentence(
        ReasonCode.NON_BINARY_TREATMENT
    )
    assert canonical_sentence("not_a_real_code") == canonical_sentence(ReasonCode.TOOL_ERROR)


def test_a_coded_error_survives_pickle_and_deepcopy():
    """Making reason_code keyword-only must not cost picklability.

    ``BaseException.__reduce__`` replays ``self.args`` POSITIONALLY, so a
    keyword-only required argument turns any pickle or deepcopy of the error into
    a confusing ``TypeError: missing 1 required keyword-only argument`` far from
    the raise site. These errors crossed a process boundary fine before #2021 and
    must keep doing so — pytest-xdist, multiprocessing and any task queue reduce
    exceptions this way.
    """
    import copy
    import pickle

    err = ToolRefusalError(
        "gap_calculator: refusing",
        reason_code=ReasonCode.NO_USABLE_ROWS,
        details={"n_rows": 0},
    )
    for clone in (pickle.loads(pickle.dumps(err)), copy.deepcopy(err)):
        assert type(clone) is ToolRefusalError
        assert str(clone) == "gap_calculator: refusing"
        assert clone.reason_code is ReasonCode.NO_USABLE_ROWS
        assert clone.details == {"n_rows": 0}
        assert isinstance(clone, RuntimeError)
