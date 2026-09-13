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
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.digital_twin.effect.errors import EffectDataUnavailable
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


def _assert_authored(err: BaseException, caplog, *, kept: List[str], library_text: str) -> None:
    message = str(err)
    assert library_text not in message, message
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
    _assert_authored(
        caught.value,
        caplog,
        kept=[
            "refutation_runner: supplied data is not a DataFrame",
            "Refusing to fabricate refutation results.",
        ],
        library_text=SENTINEL,
    )
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
    _assert_authored(
        caught.value,
        caplog,
        kept=[
            "sensitivity_analyzer: supplied data is not a DataFrame",
            "Refusing to fabricate the sensitivity benchmark inputs.",
        ],
        library_text=SENTINEL,
    )
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
    _assert_authored(
        caught.value,
        caplog,
        kept=["power_calculator: ", "outside the range the calculation can represent"],
        library_text=library_text,
    )
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT_VALUE
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
    _assert_authored(
        caught.value,
        caplog,
        kept=["cohort causal estimation failed for 'engagement_score'"],
        library_text=SENTINEL,
    )
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == SENTINEL


def test_cohort_target_region_failure_text_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr("econml.dml.CausalForestDML", _TargetIntervalRaises)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(EffectDataUnavailable) as caught:
            estimate_cohort_effect(
                _make_confounded_cohort(n_per_region=100),
                "engagement_score",
                target_regions=["west"],
            )
    _assert_authored(
        caught.value,
        caplog,
        kept=["target-region inference failed for 'engagement_score'"],
        library_text=SENTINEL,
    )
    assert str(caught.value.__cause__) == SENTINEL


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
    assert caught.value.reason_code is ReasonCode.EFFECT_NOT_ESTIMABLE


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
    _assert_authored(
        caught.value,
        caplog,
        kept=["TwinEffectEstimator: uplift fit failed"],
        library_text=SENTINEL,
    )


def _uplift_engine(monkeypatch):
    monkeypatch.setattr(UpliftRandomForest, "estimate", _failed_uplift)
    provider = _synthetic_frame_provider()
    engine = SimulationEngine(
        population=_numeric_population(),
        effect_provider=provider,
        effect_estimator=TwinEffectEstimator(
            n_estimators=25, max_depth=3, min_training_samples=100
        ),
    )
    frame = provider.get_training_frame("email_campaign", brand="Remibrutinib", twin_type="hcp")
    return engine, frame


def _cohort_engine(monkeypatch):
    monkeypatch.setattr("econml.dml.CausalForestDML", _FitRaises)
    provider = CohortEffectDataProvider(_region_cohort())
    engine = SimulationEngine(
        population=_region_population(),
        effect_provider=provider,
        effect_estimator=CohortCausalEstimator(),
    )
    return engine, provider.get_training_frame("email_campaign", brand="Kisqali", twin_type="hcp")


@pytest.mark.parametrize("build", [_uplift_engine, _cohort_engine], ids=["uplift", "cohort"])
def test_failed_simulation_refusal_carries_no_estimator_library_text(monkeypatch, caplog, build):
    engine, frame = build(monkeypatch)
    with caplog.at_level(logging.WARNING):
        result = engine.simulate(
            InterventionConfig(intervention_type="email_campaign"), use_cache=False
        )
        assert result.status.value == "failed"
        with pytest.raises(ToolRefusalError) as caught:
            tr._simulation_results(
                result,
                brand="Kisqali",
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
    assert caught.value.reason_code is ReasonCode.SIMULATION_INCOMPLETE


# ---------------------------------------------------------------------------
# AST guard: no coded refusal interpolates a caught exception
# ---------------------------------------------------------------------------

CODED_REFUSALS = frozenset(
    {"ToolRefusalError", "ToolInputError", "EffectDataUnavailable", "EstimationError"}
)

# Keyed by (enclosing function, caught types) — never by line. Each handler here catches an
# exception whose text is authored, so passing it through keeps the refusal authored.
ALLOWED: Dict[Tuple[str, Tuple[str, ...]], str] = {
    ("sensitivity_analyzer", ("ValueError",)): (
        "evalue.classify on already-derived floats raises only evalue's own ValueErrors "
        "(_finite / _orient / _validate_outcome_std / the CI and covariates_measured checks)"
    ),
    ("_point_only_sensitivity", ("ValueError",)): (
        "point_e_value / joint_confounding_benchmark / measured_confounding_benchmark on derived "
        "floats raise only evalue's own ValueErrors; e_value_from_rr returns before sqrt when "
        "r <= 1 and rr_from_smd's math.exp can raise only OverflowError"
    ),
    ("_targeted_effect", ("EffectDataUnavailable",)): (
        "estimate_cohort_effect raises EffectDataUnavailable with authored text only; its econml "
        "wraps log the library error instead (pinned above)"
    ),
    ("power_calculator", ("PowerCalculationError",)): (
        "power_analysis_lib raises PowerCalculationError with authored text, pinned by "
        "test_power_calculator_2015's match=reason; ArithmeticError has its own clause"
    ),
}


def _guarded_files() -> List[Path]:
    effect = REPO_ROOT / "src" / "digital_twin" / "effect"
    return [REPO_ROOT / "src" / "agents" / "tool_composer" / "tool_registrations.py"] + sorted(
        effect.glob("*.py")
    )


def _unallowed(found: List[CaughtInterpolation]) -> List[CaughtInterpolation]:
    return [site for site in found if (site.function, site.caught_types) not in ALLOWED]


def test_no_coded_refusal_interpolates_a_caught_exception():
    files = _guarded_files()
    assert {"estimator.py", "cohort_causal_estimator.py"} <= {path.name for path in files}
    found = {
        path.name: caught_exception_interpolations(ast.parse(path.read_text()), CODED_REFUSALS)
        for path in files
    }
    assert {name: _unallowed(sites) for name, sites in found.items()} == {
        name: [] for name in found
    }
    sites = [site for per_file in found.values() for site in per_file]
    # A function on the allowlist holds allowlisted handlers only.
    allowed_functions = {function for function, _caught in ALLOWED}
    assert [
        site
        for site in sites
        if site.function in allowed_functions and (site.function, site.caught_types) not in ALLOWED
    ] == []
    # Every entry still names a real handler; a stale one would silently allow a future site.
    assert set(ALLOWED) <= {(site.function, site.caught_types) for site in sites}


def test_allowlist_does_not_admit_the_unsplit_power_handler():
    source = textwrap.dedent(
        """\
        def power_calculator():
            try:
                pass
            except (PowerCalculationError, ArithmeticError) as exc:
                raise ToolInputError(f"power_calculator: {exc}")
        """
    )
    assert _unallowed(caught_exception_interpolations(ast.parse(source), CODED_REFUSALS)) == [
        CaughtInterpolation(
            5,
            "power_calculator",
            ("PowerCalculationError", "ArithmeticError"),
            "ToolInputError",
            "exc",
        )
    ]


def test_allowlist_is_keyed_by_caught_type_not_function_alone():
    source = textwrap.dedent(
        """\
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
    )
    found = caught_exception_interpolations(ast.parse(source), CODED_REFUSALS)
    assert [site.line for site in found] == [5, 9]
    assert [site.line for site in _unallowed(found)] == [9]


# ---------------------------------------------------------------------------
# Guard self-tests: each form the helper follows, flagged and allowed
# ---------------------------------------------------------------------------


def _in_handler(body: str) -> str:
    return "try:\n    pass\nexcept Exception as e:\n" + textwrap.indent(
        textwrap.dedent(body), "    "
    )


def _flagged(body: str) -> List[Tuple[int, str, str]]:
    found = caught_exception_interpolations(ast.parse(_in_handler(body)), CODED_REFUSALS)
    return [(site.line, site.raised, site.name) for site in found]


@pytest.mark.parametrize(
    ("body", "line"),
    [
        ('msg = "failed: "\nmsg += str(e)\nraise ToolRefusalError(msg)\n', 6),
        ("print((d := str(e)))\nraise ToolRefusalError(d)\n", 5),
        ("if (d := str(e)):\n    raise ToolRefusalError(d)\n", 5),
        ("for arg in e.args:\n    raise ToolRefusalError(arg)\n", 5),
        ('err = ToolRefusalError(f"{e}")\nraise err\n', 5),
        ('err = ToolRefusalError(f"{e}")\nalias = err\nraise alias\n', 6),
        ('def build():\n    return f"{e}"\nraise ToolRefusalError(build())\n', 6),
        ('def build():\n    raise ToolRefusalError(f"{e}")\n', 5),
        ("with open(str(e)) as handle:\n    raise ToolRefusalError(handle)\n", 5),
        ('msg = str(e)\nif cond:\n    msg = "fixed"\nraise ToolRefusalError(msg)\n', 7),
        ("while cond:\n    raise ToolRefusalError(msg)\n    msg = str(e)\n", 5),
    ],
    ids=[
        "augassign",
        "walrus",
        "walrus-in-test",
        "for-over-args",
        "prebuilt-instance",
        "prebuilt-alias",
        "nested-def-called",
        "nested-def-raises",
        "with-as",
        "rebound-on-one-branch-only",
        "loop-carries-taint-back",
    ],
)
def test_helper_flags_derived_forms(body, line):
    raised = "ToolRefusalError"
    assert _flagged(body) == [(line, raised, "e")]


@pytest.mark.parametrize(
    "body",
    [
        'msg = "failed"\nmsg += "!"\nraise ToolRefusalError(msg) from e\n',
        'if (d := "fixed"):\n    raise ToolRefusalError(d)\n',
        'for arg in ("a", "b"):\n    raise ToolRefusalError(arg)\n',
        'err = ToolRefusalError("fixed")\nraise err from e\n',
        'err = ValueError(f"{e}")\nraise err\n',
        'def build():\n    return "fixed"\nraise ToolRefusalError(build())\n',
        'def build(e):\n    return str(e)\nraise ToolRefusalError(build("fixed"))\n',
        "raise ToolRefusalError(type(e).__name__)\n",
        "raise ToolRefusalError(e.__class__.__name__)\n",
        'raise ToolRefusalError(f"{type(e)}")\n',
        'e = "fixed"\nraise ToolRefusalError(f"{e}")\n',
        'msg = str(e)\nmsg = "fixed"\nraise ToolRefusalError(msg)\n',
        'msg = {}\nmsg["k"] = 1\nraise ToolRefusalError(msg)\n',
    ],
    ids=[
        "augassign-untainted",
        "walrus-untainted",
        "for-over-literal",
        "prebuilt-authored",
        "prebuilt-other-class",
        "nested-def-untainted",
        "nested-def-shadowing-param",
        "type-name",
        "class-name",
        "type-repr",
        "handler-name-rebound",
        "derived-name-rebound",
        "subscript-store-untainted",
    ],
)
def test_helper_allows_what_does_not_render_the_exception(body):
    assert _flagged(body) == []


def test_helper_reports_a_nested_handler_once_under_its_own_clause():
    source = textwrap.dedent(
        """\
        def tool():
            try:
                pass
            except Exception as e:
                try:
                    pass
                except (ValueError, errors.EffectDataUnavailable) as e:
                    raise ToolRefusalError(f"{e}")
        """
    )
    assert caught_exception_interpolations(ast.parse(source), CODED_REFUSALS) == [
        CaughtInterpolation(
            8, "tool", ("ValueError", "EffectDataUnavailable"), "ToolRefusalError", "e"
        )
    ]


def test_helper_carries_a_nested_handlers_bindings_to_the_outer_scan():
    body = "try:\n    pass\nexcept ValueError:\n    msg = str(e)\nraise ToolRefusalError(msg)\n"
    assert _flagged(body) == [(8, "ToolRefusalError", "e")]


def test_helper_names_the_enclosing_function_and_caught_types():
    source = textwrap.dedent(
        """\
        try:
            pass
        except RuntimeError as e:
            raise ToolRefusalError(str(e))

        async def outer():
            try:
                pass
            except (KeyError, module.LookupFailure) as exc:
                raise EstimationError(repr(exc))
        """
    )
    assert caught_exception_interpolations(ast.parse(source), CODED_REFUSALS) == [
        CaughtInterpolation(4, None, ("RuntimeError",), "ToolRefusalError", "e"),
        CaughtInterpolation(10, "outer", ("KeyError", "LookupFailure"), "EstimationError", "exc"),
    ]
