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
