"""#2020 Task 4b (owner decision D5): a coded refusal carries only text its tool authored.

The fail-closed answer shows a tool-authored refusal verbatim when its reason code is known
(Task 4). That is sound only if no coded refusal carries library text, and several did: each
interpolated the exception its handler caught — an ``AttributeError``-shaped repr of whatever
object was passed as a frame, pandas' "Could not convert string 'hi' to numeric" (which quotes a
data VALUE), "agg function failed [how->mean,dtype->object]", econml / sklearn internals, and
Python's "int too large to convert to float". ``TwinEffectEstimator`` did the same without an
exception object, from causalml's ``UpliftResult.error_message``.

Each site now keeps every authored word, logs the library text at WARNING with the exception,
and keeps the original as ``__cause__``. An AST guard over ``tool_registrations.py`` and every
``src/digital_twin/effect`` module keeps it that way; its short allowlist names the handlers
whose caught exceptions are themselves authored, each with the reason.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.agents.tool_composer.reason_codes import ReasonCode
from src.causal_engine.errors import EstimationError
from src.causal_engine.uplift import UpliftRandomForest
from src.causal_engine.uplift.base import UpliftResult
from src.digital_twin.effect.cohort_causal_estimator import (
    CohortCausalEstimator,
    estimate_cohort_effect,
)
from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable
from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import CohortEffectDataProvider, SyntheticEffectDataProvider
from src.digital_twin.models.simulation_models import InterventionConfig
from src.digital_twin.simulation_engine import SimulationEngine
from tests.unit.ast_guards import CaughtInterpolation, caught_exception_interpolations
from tests.unit.test_agents.test_tool_composer.test_counterfactual_simulator_2015 import (
    _cohort as _region_cohort,
)
from tests.unit.test_agents.test_tool_composer.test_counterfactual_simulator_2015 import (
    _population as _region_population,
)
from tests.unit.test_digital_twin.effect.test_cohort_causal_estimator import (
    _make_confounded_cohort,
)
from tests.unit.test_digital_twin.test_engine_real_effect import _population as _numeric_population

SENTINEL = "LIBTEXT_SENTINEL"
REPO_ROOT = Path(__file__).resolve().parents[4]

# Messages that are now authored end to end, pinned whole.
REFUTATION_NOT_A_FRAME = (
    "refutation_runner: supplied data is not a DataFrame. Refusing to fabricate refutation results."
)
SENSITIVITY_NOT_A_FRAME = (
    "sensitivity_analyzer: supplied data is not a DataFrame. Refusing to fabricate the "
    "sensitivity benchmark inputs."
)
POWER_UNREPRESENTABLE = (
    "power_calculator: the requested design is outside the range the calculation can "
    "represent; check the effect size, alpha, power and the design's rate or cluster size."
)
COHORT_NOT_ESTIMABLE = (
    "cohort causal estimation failed for 'engagement_score': the causal forest could not "
    "estimate an effect on this cohort."
)
TARGET_REGION_NO_INTERVAL = (
    "target-region inference failed for 'engagement_score': the causal forest could not "
    "compute an interval on the targeted rows."
)
UPLIFT_NOT_FITTED = (
    "TwinEffectEstimator: the uplift model could not be fitted on the training frame."
)


def _assert_authored(
    err: BaseException,
    caplog,
    *,
    library_text: str,
    exact: Optional[str] = None,
    kept: Sequence[str] = (),
) -> None:
    message = str(err)
    assert library_text not in message, message
    if exact is not None:
        assert message == exact
    for words in kept:
        assert words in message, (words, message)
    assert library_text in caplog.text


# ---------------------------------------------------------------------------
# Frames that are not frames
# ---------------------------------------------------------------------------


class _UnreadableColumns:
    def __iter__(self):
        raise RuntimeError(SENTINEL)


class _NotAFrame:
    """Passes ``_extract_dataframe_from_kwargs``' duck-typed check (``columns`` + ``__len__``)
    and fails when its columns are read — the input that reaches the ``not a DataFrame`` wrap."""

    columns = _UnreadableColumns()

    def __len__(self) -> int:
        return 3


def test_refutation_runner_non_frame_text_is_logged_not_refused(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr.refutation_runner(treatment="t", outcome="y", estimation_data=_NotAFrame())
    _assert_authored(caught.value, caplog, exact=REFUTATION_NOT_A_FRAME, library_text=SENTINEL)
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT_TYPE
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_sensitivity_non_frame_text_is_logged_not_refused(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr.sensitivity_analyzer(
                ate=0.1,
                ci_lower=0.05,
                ci_upper=0.15,
                treatment="t",
                outcome="y",
                estimation_data=_NotAFrame(),
            )
    _assert_authored(caught.value, caplog, exact=SENSITIVITY_NOT_A_FRAME, library_text=SENTINEL)
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT_TYPE
    assert isinstance(caught.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Non-numeric columns: pandas' text quotes the data
# ---------------------------------------------------------------------------


def test_sensitivity_unusable_column_text_is_logged_not_refused(caplog):
    rng = np.random.default_rng(2020)
    frame = pd.DataFrame(
        {"t": rng.integers(0, 2, 200), "y": [SENTINEL] * 200, "c": rng.normal(size=200)}
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr.sensitivity_analyzer(
                ate=0.1,
                ci_lower=0.05,
                ci_upper=0.15,
                treatment="t",
                outcome="y",
                confounders=["c"],
                estimation_data=frame,
            )
    _assert_authored(
        caught.value,
        caplog,
        kept=[
            "sensitivity benchmark inputs could not be computed from the frame",
            "treatment='t' / outcome='y'",
            "refusing to serve an unstandardized reading in its place.",
        ],
        library_text=SENTINEL,
    )
    assert "could not convert" not in str(caught.value)
    assert caught.value.reason_code is ReasonCode.NO_USABLE_COLUMNS
    assert isinstance(caught.value.__cause__, ValueError)


def test_cate_non_numeric_outcome_value_is_logged_not_refused(caplog):
    frame = pd.DataFrame(
        {"t": [0, 1] * 50, "y": [SENTINEL] * 100, "seg": ["a", "a", "b", "b"] * 25}
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr.cate_analyzer(treatment="t", outcome="y", segments=["seg"], estimation_data=frame)
    _assert_authored(
        caught.value,
        caplog,
        kept=[
            "cate_analyzer: outcome column 'y' is not numeric (dtype=object)",
            "for 'seg'='a'",
            "Refusing to fabricate a treatment effect from a non-numeric outcome.",
        ],
        library_text=SENTINEL,
    )
    assert "Could not convert" not in str(caught.value)
    assert caught.value.reason_code is ReasonCode.NON_NUMERIC_COLUMN
    assert isinstance(caught.value.__cause__, TypeError)


def test_gap_non_numeric_metric_text_is_logged_not_refused(caplog):
    frame = pd.DataFrame({"brand": ["a", "b"] * 5, "market_share": ["high"] * 10})
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr.gap_calculator(
                metric="market_share", entity_type="brand", entities=[], estimation_data=frame
            )
    _assert_authored(
        caught.value,
        caplog,
        kept=[
            "gap_calculator: metric column 'market_share' is not numeric (dtype=object)",
            "per-'brand' group mean",
            "Refusing to fabricate a comparison from a non-numeric metric.",
        ],
        library_text="agg function failed",
    )
    assert caught.value.reason_code is ReasonCode.NON_NUMERIC_COLUMN
    assert isinstance(caught.value.__cause__, TypeError)


# ---------------------------------------------------------------------------
# power_calculator: PowerCalculationError is authored, ArithmeticError is not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "library_text"),
    [
        # Python's own text: ``math.isfinite`` on an integer too large for a float.
        (
            {"effect_size": 0.5, "design": "cluster", "icc": 0.05, "cluster_size": 10**400},
            "int too large to convert to float",
        ),
        # The library's OverflowError for an unrepresentable size.
        ({"effect_size": 1e-200}, "the per-arm sample size is not a finite number"),
    ],
    ids=["python-overflow", "library-overflow"],
)
def test_power_overflow_text_is_logged_not_refused(caplog, kwargs, library_text):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolInputError) as caught:
            tr.power_calculator(**kwargs)
    _assert_authored(caught.value, caplog, exact=POWER_UNREPRESENTABLE, library_text=library_text)
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT_VALUE
    assert isinstance(caught.value.__cause__, OverflowError)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("effect_size", {"effect_size": 10**400}),
        ("alpha", {"effect_size": 0.5, "alpha": 10**400}),
        ("power", {"effect_size": 0.5, "power": 10**400}),
    ],
    ids=["effect_size", "alpha", "power"],
)
def test_power_input_too_large_for_a_float_is_refused_with_a_code(caplog, name, kwargs):
    # #2021: these escaped as a bare OverflowError from ``_power_number``, outside the refusal
    # ``try`` -- uncoded, so retried and charged to the circuit breaker, with Python's text.
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolInputError) as caught:
            tr.power_calculator(**kwargs)
    _assert_authored(
        caught.value,
        caplog,
        exact=(
            f"power_calculator: {name} is too large to be represented as a finite number. No "
            "sample size can be computed from it."
        ),
        library_text="int too large",
    )
    assert name in str(caught.value)
    assert "0" * 20 not in str(caught.value)
    assert caught.value.reason_code is ReasonCode.NON_FINITE_INPUT
    assert isinstance(caught.value.__cause__, OverflowError)


def test_power_calculation_error_keeps_its_authored_text():
    with pytest.raises(ToolInputError) as caught:
        tr.power_calculator(effect_size=0.5, outcome_type="binary", baseline_rate=0.9)
    assert str(caught.value).startswith("power_calculator: Treatment rate (p2=")
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT_VALUE


# ---------------------------------------------------------------------------
# Digital-twin estimators
# ---------------------------------------------------------------------------


class _FitRaises:
    def __init__(self, **_kwargs):
        pass

    def fit(self, *_args, **_kwargs):
        raise RuntimeError(SENTINEL)


class _TargetIntervalRaises:
    """A forest whose cohort-wide fit succeeds and whose second, target-region
    ``ate_interval`` call raises."""

    def __init__(self, **_kwargs):
        self.intervals = 0

    def fit(self, *_args, **_kwargs):
        return self

    def effect(self, x):
        return np.full(len(x), 0.2)

    def ate_interval(self, _x, alpha=0.05):
        self.intervals += 1
        if self.intervals == 2:
            raise RuntimeError(SENTINEL)
        return (0.1, 0.3)


def test_cohort_fit_failure_text_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr("econml.dml.CausalForestDML", _FitRaises)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(EffectDataUnavailable) as caught:
            estimate_cohort_effect(_make_confounded_cohort(n_per_region=100), "engagement_score")
    _assert_authored(caught.value, caplog, exact=COHORT_NOT_ESTIMABLE, library_text=SENTINEL)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == SENTINEL
    # #2021 9b: the cause and its counts ride along; neither reads the library exception.
    assert caught.value.cause is EffectCause.ESTIMATION_FAILED
    assert caught.value.details == {"n_usable_rows": 400, "is_target_inference": False}


def test_cohort_target_region_failure_text_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr("econml.dml.CausalForestDML", _TargetIntervalRaises)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(EffectDataUnavailable) as caught:
            estimate_cohort_effect(
                _make_confounded_cohort(n_per_region=100),
                "engagement_score",
                target_regions=["west"],
            )
    _assert_authored(caught.value, caplog, exact=TARGET_REGION_NO_INTERVAL, library_text=SENTINEL)
    assert str(caught.value.__cause__) == SENTINEL
    assert caught.value.cause is EffectCause.TARGET_INFERENCE_FAILED
    assert caught.value.details == {
        "n_usable_rows": 400,
        "is_target_inference": True,
        "n_target_rows": 100,
    }


def test_targeted_effect_passes_the_authored_estimator_text_through(monkeypatch, caplog):
    monkeypatch.setattr("econml.dml.CausalForestDML", _TargetIntervalRaises)
    provider = CohortEffectDataProvider(_region_cohort())
    frame = provider.get_training_frame("email_campaign", brand="Kisqali", twin_type="hcp")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ToolRefusalError) as caught:
            tr._targeted_effect(frame, ["west"])
    _assert_authored(
        caught.value,
        caplog,
        kept=["counterfactual_simulator: target-region inference failed", "No effect is returned."],
        library_text=SENTINEL,
    )
    # #2021 9b: the estimator's cause maps to its own code, and its counts ride along.
    assert caught.value.reason_code is ReasonCode.ESTIMATOR_FAILED
    assert caught.value.details["is_target_inference"] is True


def _failed_uplift(self, *_args, **_kwargs) -> UpliftResult:
    # ``BaseUpliftModel.estimate`` turns any exception into ``error_message=str(e)``.
    return UpliftResult(model_type=self.model_type, success=False, error_message=SENTINEL)


def _synthetic_frame_provider() -> SyntheticEffectDataProvider:
    return SyntheticEffectDataProvider(n=300, true_ate=0.2, seed=42)


def test_uplift_fit_failure_text_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr(UpliftRandomForest, "estimate", _failed_uplift)
    frame = _synthetic_frame_provider().get_training_frame(
        "email_campaign", brand="Remibrutinib", twin_type="hcp"
    )
    estimator = TwinEffectEstimator(n_estimators=25, max_depth=3, min_training_samples=100)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(EstimationError) as caught:
            estimator.estimate(frame, frame.df[frame.confounders])
    _assert_authored(caught.value, caplog, exact=UPLIFT_NOT_FITTED, library_text=SENTINEL)


def _uplift_engine(monkeypatch, brand: str):
    monkeypatch.setattr(UpliftRandomForest, "estimate", _failed_uplift)
    provider = _synthetic_frame_provider()
    engine = SimulationEngine(
        population=_numeric_population(),
        effect_provider=provider,
        effect_estimator=TwinEffectEstimator(
            n_estimators=25, max_depth=3, min_training_samples=100
        ),
    )
    return engine, provider.get_training_frame("email_campaign", brand=brand, twin_type="hcp")


def _cohort_engine(monkeypatch, brand: str):
    monkeypatch.setattr("econml.dml.CausalForestDML", _FitRaises)
    provider = CohortEffectDataProvider(_region_cohort())
    engine = SimulationEngine(
        population=_region_population(),
        effect_provider=provider,
        effect_estimator=CohortCausalEstimator(),
    )
    return engine, provider.get_training_frame("email_campaign", brand=brand, twin_type="hcp")


@pytest.mark.parametrize(
    ("build", "brand", "code", "cause", "details"),
    [
        # EstimationError names no cause, and its diagnostic details may hold text: never read.
        (_uplift_engine, "Remibrutinib", ReasonCode.SIMULATION_INCOMPLETE, None, {}),
        # #2021 9b (Part B): the engine keeps the estimator's cause and counts.
        (
            _cohort_engine,
            "Kisqali",
            ReasonCode.ESTIMATOR_FAILED,
            "estimation_failed",
            {"n_usable_rows": 1200, "is_target_inference": False},
        ),
    ],
    ids=["uplift", "cohort"],
)
def test_failed_simulation_refusal_carries_no_estimator_library_text(
    monkeypatch, caplog, build, brand, code, cause, details
):
    engine, frame = build(monkeypatch, brand)
    with caplog.at_level(logging.WARNING):
        result = engine.simulate(
            InterventionConfig(intervention_type="email_campaign"), use_cache=False
        )
        assert result.status.value == "failed"
        with pytest.raises(ToolRefusalError) as caught:
            tr._simulation_results(
                result,
                brand=brand,
                intervention_type="email_campaign",
                frame=frame,
                targeted=None,
            )
    _assert_authored(
        caught.value,
        caplog,
        kept=["did not complete: Effect estimation failed: ", "No effect is returned."],
        library_text=SENTINEL,
    )
    assert caught.value.reason_code is code
    assert caught.value.details == details
    assert result.error_cause == cause
    assert result.error_details == details


# ---------------------------------------------------------------------------
# AST guard: no coded refusal interpolates a caught exception
# (the helper's own self-tests are in tests/unit/test_ast_guards.py)
# ---------------------------------------------------------------------------

CODED_REFUSALS = frozenset(
    {"ToolRefusalError", "ToolInputError", "EffectDataUnavailable", "EstimationError"}
)

AllowKey = Tuple[str, Optional[str], Tuple[str, ...]]

# Keyed by (file, enclosing function, caught types) — never by line. Function names repeat across
# the guarded files (``estimate`` is defined twice), so the file is part of the key. Each handler
# here catches an exception whose text is authored, so passing it through keeps the refusal
# authored; an entry covers exactly one handler, so a second one of the same shape is flagged.
ALLOWED: Dict[AllowKey, str] = {
    ("tool_registrations.py", "sensitivity_analyzer", ("ValueError",)): (
        "evalue.classify on already-derived floats raises only evalue's own ValueErrors "
        "(_finite / _orient / _validate_outcome_std / the CI and covariates_measured checks)"
    ),
    ("tool_registrations.py", "_point_only_sensitivity", ("ValueError",)): (
        "point_e_value / joint_confounding_benchmark / measured_confounding_benchmark on derived "
        "floats raise only evalue's own ValueErrors; e_value_from_rr returns before sqrt when "
        "r <= 1 and rr_from_smd's math.exp can raise only OverflowError"
    ),
    ("tool_registrations.py", "_targeted_effect", ("EffectDataUnavailable",)): (
        "estimate_cohort_effect raises EffectDataUnavailable with authored text only; its econml "
        "wraps log the library error instead (pinned above). The same raise also reads exc.cause "
        "(a closed EffectCause, mapped to a code) and exc.details (counts and flags, re-validated "
        "by ToolRefusalError), neither of which carries text (#2021 9b)"
    ),
    ("tool_registrations.py", "power_calculator", ("PowerCalculationError",)): (
        "power_analysis_lib raises PowerCalculationError with authored text, pinned by "
        "test_power_calculator_2015's match=reason; ArithmeticError has its own clause"
    ),
}


def _guarded_files() -> List[Path]:
    effect = REPO_ROOT / "src" / "digital_twin" / "effect"
    return [REPO_ROOT / "src" / "agents" / "tool_composer" / "tool_registrations.py"] + sorted(
        effect.glob("*.py")
    )


def _key(file_name: str, site: CaughtInterpolation) -> AllowKey:
    return (file_name, site.function, site.caught_types)


def _unallowed(
    found: Dict[str, List[CaughtInterpolation]],
) -> Dict[str, List[CaughtInterpolation]]:
    return {
        name: [site for site in sites if _key(name, site) not in ALLOWED]
        for name, sites in found.items()
    }


def _allowlist_match_counts(found: Dict[str, List[CaughtInterpolation]]) -> Dict[AllowKey, int]:
    """How many real sites each entry matches: 0 is a stale entry that would silently allow a
    future site, 2 or more lets a second handler through under the first one's reason."""
    counts = Counter(_key(name, site) for name, sites in found.items() for site in sites)
    return {key: counts[key] for key in ALLOWED}


def _scan(file_name: str, source: str) -> Dict[str, List[CaughtInterpolation]]:
    return {
        file_name: caught_exception_interpolations(
            ast.parse(textwrap.dedent(source)), CODED_REFUSALS
        )
    }


def test_no_coded_refusal_interpolates_a_caught_exception():
    files = _guarded_files()
    assert {"estimator.py", "cohort_causal_estimator.py"} <= {path.name for path in files}
    found = {
        path.name: caught_exception_interpolations(ast.parse(path.read_text()), CODED_REFUSALS)
        for path in files
    }
    assert _unallowed(found) == {name: [] for name in found}
    assert _allowlist_match_counts(found) == dict.fromkeys(ALLOWED, 1)


def test_allowlist_flags_a_second_handler_under_an_allowlisted_key():
    found = _scan(
        "tool_registrations.py",
        """\
        def sensitivity_analyzer():
            try:
                pass
            except ValueError as exc:
                raise ToolRefusalError(f"sensitivity_analyzer refused its inputs: {exc}")
            try:
                pass
            except ValueError as exc:
                raise ToolRefusalError(f"pandas said: {exc}")
        """,
    )
    assert _unallowed(found) == {"tool_registrations.py": []}
    key = ("tool_registrations.py", "sensitivity_analyzer", ("ValueError",))
    assert _allowlist_match_counts(found)[key] == 2


def test_allowlist_does_not_admit_the_unsplit_power_handler():
    found = _scan(
        "tool_registrations.py",
        """\
        def power_calculator():
            try:
                pass
            except (PowerCalculationError, ArithmeticError) as exc:
                raise ToolInputError(f"power_calculator: {exc}")
        """,
    )
    assert _unallowed(found) == {
        "tool_registrations.py": [
            CaughtInterpolation(
                5,
                "power_calculator",
                ("PowerCalculationError", "ArithmeticError"),
                "ToolInputError",
                "exc",
            )
        ]
    }


def test_allowlist_is_keyed_by_caught_type_not_function_alone():
    source = """\
        def sensitivity_analyzer():
            try:
                pass
            except ValueError as exc:
                raise ToolRefusalError(f"refused: {exc}")
            try:
                pass
            except Exception as exc:
                raise ToolRefusalError(f"refused: {exc}")
        """
    found = _scan("tool_registrations.py", source)
    assert [site.line for site in found["tool_registrations.py"]] == [5, 9]
    assert [site.line for site in _unallowed(found)["tool_registrations.py"]] == [9]
    # The same function and caught type in another guarded file is not the allowlisted handler.
    assert [site.line for site in _unallowed(_scan("estimator.py", source))["estimator.py"]] == [
        5,
        9,
    ]
