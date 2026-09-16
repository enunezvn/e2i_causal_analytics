"""#2021: refusals carry a closed reason code and a data-free canonical sentence."""

import copy
import logging
import pickle
import re
from collections.abc import Mapping
from enum import Enum, StrEnum

import pytest

from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.agents.tool_composer.reason_codes import (
    CANONICAL_SENTENCES,
    EXECUTOR_ASSIGNED,
    ReasonCode,
    canonical_sentence,
    known_sentence,
    validate_details,
)

_ERRORS_LOGGER = "src.agents.tool_composer.errors"


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


def test_the_dependency_unmet_sentence_does_not_claim_the_upstream_never_ran():
    """The executor skips a step whose bound upstream is in ``failed_step_ids``, which holds
    every step whose output is not a success: refused, error and timeout steps RAN."""
    assert (
        canonical_sentence(ReasonCode.DEPENDENCY_UNMET)
        == "a step this one depends on did not produce a result"
    )


# The wire values as of the #2021 lane. Once codes are persisted, removing or renaming a
# member orphans stored rows and needs a data migration; adding one does not.
_WIRE_VALUES_2021 = frozenset(
    {
        "non_binary_treatment",
        "single_class_treatment",
        "non_binary_outcome",
        "single_class_outcome",
        "no_usable_rows",
        "no_usable_columns",
        "non_finite_input",
        "insufficient_groups",
        "insufficient_sample",
        "coverage_gap",
        "unknown_column",
        "non_numeric_column",
        "ambiguous_column",
        "ambiguous_group_label",
        "missing_dataframe",
        "ci_outside_estimate",
        "point_estimate_only",
        "unsupported_request",
        "degenerate_design",
        "simulation_incomplete",
        "effect_not_estimable",
        "missing_required_column",
        "no_treatment_contrast",
        "estimator_failed",
        "missing_required_input",
        "invalid_input_type",
        "invalid_input_value",
        "unexpected_input",
        "tool_error",
        "tool_timeout",
        "plan_defect",
        "reference_unresolvable",
        "dependency_unmet",
        "circuit_open",
        "tool_not_registered",
    }
)


def test_wire_values_may_grow_but_never_shrink_or_rename():
    """Removing or renaming a member needs a data migration once codes are persisted.

    upstream_step_failed removed 2026-09-12 before first deploy; never persisted.
    """
    assert len(_WIRE_VALUES_2021) == 35
    current = {c.value for c in ReasonCode}
    assert current >= _WIRE_VALUES_2021, sorted(_WIRE_VALUES_2021 - current)
    for member in ReasonCode:
        assert member.value == member.name.lower(), member


def test_the_twin_effect_causes_have_their_own_codes_9b():
    """#2021 9b: three counterfactual_simulator causes had no existing code whose sentence is
    true for them. They are tool-authored codes, never executor-assigned."""
    sentences = {code.value: sentence for code, sentence in CANONICAL_SENTENCES.items()}
    expected = {
        "missing_required_column": "a column the estimate requires is not present in the data",
        "no_treatment_contrast": (
            "the treatment does not split the rows into a treated and a comparison group"
        ),
        "estimator_failed": (
            "the effect estimator could not produce an estimate from data that passed its checks"
        ),
    }
    assert {value: sentences.get(value) for value in expected} == expected
    assert not {c.value for c in EXECUTOR_ASSIGNED} & set(expected)


def test_a_code_formats_as_its_wire_value():
    """A ``(str, Enum)`` formats as ``ReasonCode.NO_USABLE_ROWS`` on 3.12; the wire value
    is what a log line, an f-string or a database parameter must carry."""
    assert f"{ReasonCode.NO_USABLE_ROWS}" == "no_usable_rows"
    assert str(ReasonCode.NO_USABLE_ROWS) == "no_usable_rows"


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


# --- details: the validator is strict ----------------------------------------------


def test_details_must_be_structure_only():
    """details is persisted; it may not smuggle free text back in."""
    with pytest.raises(ValueError, match="details"):
        validate_details({"n_message": "a sentence that is really free text " * 5})


@pytest.mark.parametrize(
    "details",
    [
        pytest.param({"n_brand": "Kisqali"}, id="short-string"),
        pytest.param({"n_column": ""}, id="empty-string"),
        pytest.param({"n_rows": None}, id="none"),
        pytest.param({"n_cols": ["free text" * 5]}, id="list"),
        pytest.param({"n_scope": {"n": 1}}, id="dict"),
        pytest.param({"share_x": float("nan")}, id="nan"),
        pytest.param({"share_x": float("inf")}, id="inf"),
        pytest.param({"share_x": float("-inf")}, id="neg-inf"),
        pytest.param({"n_rows": 2**53}, id="int-beyond-js-safe"),
        pytest.param({"n_rows": -(2**53)}, id="negative-int-beyond-js-safe"),
        pytest.param({"rows": 1}, id="no-prefix-key"),
        pytest.param({"count_rows": 1}, id="unknown-prefix-key"),
        pytest.param({"n_": 1}, id="bare-prefix-key"),
        pytest.param({"NRows": 1}, id="camel-key"),
        pytest.param({"n_Rows": 1}, id="upper-after-prefix"),
        pytest.param({"n rows": 1}, id="space-key"),
        pytest.param({"": 1}, id="empty-key"),
        pytest.param({1: 1}, id="non-string-key"),
        pytest.param([("n_rows", 1)], id="not-a-mapping"),
    ],
)
def test_details_values_are_finite_numbers_or_booleans_under_prefixed_keys(details):
    """L1/M5/M7: a string of any length fits a column or brand name — the data the lane
    keeps out of the database — so details carries numbers and booleans only, ints within
    JavaScript's safe range (the admin page reads them), under ``n_``/``is_``/``has_``/
    ``share_`` keys."""
    with pytest.raises(ValueError, match="details"):
        validate_details(details)


def test_details_accepts_ints_finite_floats_and_booleans():
    details = {"n_rows": 0, "share_kept": 0.25, "is_binary": False, "has_gaps": True, "n_2": -3}
    out = validate_details(details)
    assert out == details
    assert out is not details, "the validator returns a new dict"
    # JavaScript's Number.MAX_SAFE_INTEGER is 2**53 - 1; that bound is inclusive.
    assert validate_details({"n_rows": 2**53 - 1}) == {"n_rows": 2**53 - 1}
    assert validate_details({"n_rows": -(2**53 - 1)}) == {"n_rows": -(2**53 - 1)}


def test_details_key_count_is_still_bounded():
    with pytest.raises(ValueError, match="details"):
        validate_details({f"n_{i}": i for i in range(9)})


def test_details_key_length_matches_the_043_reducer():
    """ml/043's reducer keeps only keys matching ``_DETAIL_KEY``, anchored, and so at most 64
    characters, and drops the rest silently; the Python rule must refuse them first."""
    from src.agents.tool_composer.reason_codes import _DETAIL_KEY

    prefixes, max_tail = re.fullmatch(
        r"\(([a-z|]+)\)_\[a-z0-9_\]\{1,(\d+)\}", _DETAIL_KEY.pattern
    ).groups()
    longest = max(prefixes.split("|"), key=len) + "_" + "x" * int(max_tail)
    assert len(longest) == 64
    assert validate_details({longest: 1}) == {longest: 1}
    with pytest.raises(ValueError, match="details"):
        validate_details({longest + "x": 1})


def test_numpy_scalars_are_normalized_to_builtins():
    """A count is often ``int(mask.sum())`` — or, without the cast, an ``np.int64`` that
    neither JSON nor a database driver takes."""
    np = pytest.importorskip("numpy")
    out = validate_details(
        {"n_rows": np.int64(3), "share_kept": np.float64(0.5), "is_binary": np.bool_(True)}
    )
    assert out == {"n_rows": 3, "share_kept": 0.5, "is_binary": True}
    assert type(out["n_rows"]) is int
    assert type(out["share_kept"]) is float
    assert type(out["is_binary"]) is bool


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("nan", id="float64-nan"),
        pytest.param("str", id="numpy-str"),
        pytest.param("array", id="multi-element-array"),
    ],
)
def test_numpy_values_are_checked_after_normalization(value):
    np = pytest.importorskip("numpy")
    concrete = {
        "nan": np.float64("nan"),
        "str": np.str_("Kisqali"),
        "array": np.array([1, 2]),
    }[value]
    with pytest.raises(ValueError, match="details"):
        validate_details({"n_x": concrete})


# --- details and codes: the constructor fails SOFT (I1) ------------------------------


def test_a_refusal_with_bad_details_still_refuses(caplog):
    """A raise from the constructor would escape the executor's refusal arm into its
    generic ``except Exception``: retried, charged to the circuit breaker, and the
    refusal message lost (#1600). The refusal is kept; only the details are dropped."""
    with caplog.at_level(logging.ERROR, logger=_ERRORS_LOGGER):
        err = ToolRefusalError(
            "gap_calculator: refusing",
            reason_code=ReasonCode.NO_USABLE_ROWS,
            details={"n_brand": "Kisqali-secret"},
        )
    assert str(err) == "gap_calculator: refusing"
    assert err.reason_code is ReasonCode.NO_USABLE_ROWS
    assert err.details == {}
    records = [r for r in caplog.records if r.name == _ERRORS_LOGGER]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "no_usable_rows" in records[0].getMessage()
    assert "Kisqali-secret" not in caplog.text, "the offending value may be data"


def test_an_unknown_code_is_recorded_as_a_tool_error_not_raised(caplog):
    bad_code = "not_a_real_code_" * 6  # 96 characters
    with caplog.at_level(logging.ERROR, logger=_ERRORS_LOGGER):
        err = ToolInputError("bad input", reason_code=bad_code)  # type: ignore[arg-type]
    assert str(err) == "bad input"
    assert err.reason_code is ReasonCode.TOOL_ERROR
    assert isinstance(err, ValueError)
    records = [r for r in caplog.records if r.name == _ERRORS_LOGGER]
    assert [r.levelno for r in records] == [logging.ERROR]
    # The container log is where raw text belongs (#2020); it names the bad code, clipped.
    assert repr(bad_code[:64]) in records[0].getMessage()
    assert bad_code not in caplog.text


def test_a_valid_code_string_is_still_accepted():
    err = ToolRefusalError("x", reason_code="no_usable_rows")  # type: ignore[arg-type]
    assert err.reason_code is ReasonCode.NO_USABLE_ROWS


class _StrRaises:
    def __str__(self):
        raise RuntimeError("no str")


class _StrAndReprRaise(_StrRaises):
    def __repr__(self):
        raise RuntimeError("no repr")


@pytest.mark.parametrize(
    ("code", "printable"),
    [
        pytest.param(_StrRaises(), True, id="str-raises"),
        pytest.param(_StrAndReprRaise(), False, id="str-and-repr-raise"),
    ],
)
def test_an_unprintable_code_still_refuses(caplog, code, printable):
    """N1: the unknown-code handler formats the bad value; that must not raise either.
    A non-str code is logged as its clipped repr, or as a placeholder when that raises."""
    with caplog.at_level(logging.ERROR, logger=_ERRORS_LOGGER):
        err = ToolRefusalError("kept", reason_code=code)  # type: ignore[arg-type]
    assert str(err) == "kept"
    assert err.reason_code is ReasonCode.TOOL_ERROR
    expected = repr(code)[:64] if printable else "<unprintable>"
    assert repr(expected) in caplog.text


class _LenRaises(Mapping):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        raise TypeError("no len")


class _FakeNumpyScalar:
    def item(self):
        raise RuntimeError("item failed")


_FakeNumpyScalar.__module__ = "numpy"


@pytest.mark.parametrize(
    "details",
    [
        pytest.param(_LenRaises(), id="mapping-len-raises"),
        pytest.param({"n_rows": _FakeNumpyScalar()}, id="numpy-item-raises-runtimeerror"),
    ],
)
def test_details_that_raise_something_other_than_valueerror_still_refuse(caplog, details):
    """N1: the constructor must not raise whatever the validator trips over."""
    assert type(_FakeNumpyScalar()).__module__ == "numpy"
    with caplog.at_level(logging.ERROR, logger=_ERRORS_LOGGER):
        err = ToolRefusalError("kept", reason_code=ReasonCode.NO_USABLE_ROWS, details=details)
    assert str(err) == "kept"
    assert err.reason_code is ReasonCode.NO_USABLE_ROWS
    assert err.details == {}
    assert [r.levelno for r in caplog.records if r.name == _ERRORS_LOGGER] == [logging.ERROR]


class _StrEnumKey(StrEnum):
    N_ROWS = "n_rows"


class _StrMixinKey(str, Enum):
    N_ROWS = "n_rows"


@pytest.mark.parametrize(
    "key", [_StrEnumKey.N_ROWS, _StrMixinKey.N_ROWS], ids=["StrEnum", "str-Enum"]
)
def test_detail_keys_are_stored_as_plain_str(key):
    """N3: a ``(str, Enum)`` key renders as its member name under ``str()``; the stored key
    is the builtin string of its value."""
    out = validate_details({key: 1})
    (stored,) = out
    assert type(stored) is str
    assert stored == "n_rows"


def test_the_stored_details_are_the_normalized_copy():
    np = pytest.importorskip("numpy")
    supplied = {"n_rows": np.int64(4)}
    err = ToolRefusalError("x", reason_code=ReasonCode.NO_USABLE_ROWS, details=supplied)
    assert err.details == {"n_rows": 4}
    assert type(err.details["n_rows"]) is int
    assert err.details is not supplied


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
    ("gap", "code"),
    [
        pytest.param("0.4", ReasonCode.INVALID_INPUT_TYPE, id="string"),
        pytest.param(None, ReasonCode.INVALID_INPUT_TYPE, id="none"),
        pytest.param([0.4], ReasonCode.INVALID_INPUT_TYPE, id="list"),
        pytest.param(float("nan"), ReasonCode.NON_FINITE_INPUT, id="nan"),
        pytest.param(float("-inf"), ReasonCode.NON_FINITE_INPUT, id="neg-inf"),
    ],
)
def test_roi_estimator_tells_a_wrong_type_gap_from_a_non_finite_one(gap, code):
    """M11: the same split as ``_power_number``, with the message unchanged."""
    from src.agents.tool_composer.tool_registrations import roi_estimator

    with pytest.raises(ToolRefusalError) as caught:
        roi_estimator(gap_analysis={"gap": gap}, investment=1.0)
    assert caught.value.reason_code is code
    assert str(caught.value) == f"roi_estimator: gap value is not a finite number (got {gap!r})."


def test_roi_estimator_still_accepts_a_boolean_gap():
    """``True`` passed the isinstance check before the split, and must still."""
    from src.agents.tool_composer.tool_registrations import roi_estimator

    out = roi_estimator(gap_analysis={"gap": True}, investment=1.0)
    assert out.estimated_roi == pytest.approx(1.0)


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

    ``BaseException.__reduce__`` replays ``self.args`` POSITIONALLY, so a required
    keyword-only argument turns any pickle or deepcopy of the error into a
    ``TypeError``. Notes and any other instance attributes must survive too.
    """
    err = ToolRefusalError(
        "gap_calculator: refusing",
        reason_code=ReasonCode.NO_USABLE_ROWS,
        details={"n_rows": 0},
    )
    err.add_note("n")
    for clone in (pickle.loads(pickle.dumps(err)), copy.deepcopy(err)):
        assert type(clone) is ToolRefusalError
        assert str(clone) == "gap_calculator: refusing"
        assert clone.reason_code is ReasonCode.NO_USABLE_ROWS
        assert clone.details == {"n_rows": 0}
        assert clone.__notes__ == ["n"]
        assert isinstance(clone, RuntimeError)


def test_known_sentence_renders_only_codes_this_build_knows():
    """Read-side rendering: an unknown code must not be relabelled as a tool failure."""
    assert (
        known_sentence("non_binary_treatment")
        == CANONICAL_SENTENCES[ReasonCode.NON_BINARY_TREATMENT]
    )
    assert known_sentence(ReasonCode.COVERAGE_GAP) == CANONICAL_SENTENCES[ReasonCode.COVERAGE_GAP]
    assert known_sentence(None) is None
    assert known_sentence("a_code_this_build_does_not_know") is None
