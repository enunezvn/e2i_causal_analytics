"""#2021 9b: each twin effect cause maps to one reason code, and its details stay storable.

``counterfactual_simulator`` refused every twin effect failure as ``effect_not_estimable`` or
``simulation_incomplete``. The loader, estimator and provider now name a cause
(``EffectCause``); ``tool_registrations._EFFECT_CAUSE_CODES`` maps it to a code. The map is
keyed by string value so the tool module never imports the twin package at load time.

These tests live in the tool composer tree because they need ``ReasonCode`` and
``validate_details``, which importing from the digital-twin tree would load (~564 MB).
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError
from src.agents.tool_composer.reason_codes import EXECUTOR_ASSIGNED, ReasonCode, validate_details
from src.digital_twin.effect.cohort_causal_estimator import estimate_cohort_effect
from src.digital_twin.effect.cohort_loader import assess_cohort_frame
from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable
from src.digital_twin.effect.provider import CohortEffectDataProvider
from tests.unit.test_agents.test_tool_composer.test_counterfactual_simulator_2015 import _cohort
from tests.unit.test_agents.test_tool_composer.test_refusal_text_is_authored_2020 import (
    _FitRaises,
    _TargetIntervalRaises,
)

_ERRORS_LOGGER = "src.agents.tool_composer.errors"


def test_every_cause_is_mapped_and_nothing_else_is():
    """No unmapped cause (it would silently get the fallback) and no stale key."""
    assert {c.value for c in EffectCause} == set(tr._EFFECT_CAUSE_CODES)


def test_the_owner_approved_mapping():
    """Owner decision 2026-09-13: reuse a code whose sentence is literally true for the cause."""
    assert tr._EFFECT_CAUSE_CODES == {
        "intervention_not_identified": ReasonCode.EFFECT_NOT_ESTIMABLE,
        "empty_cohort": ReasonCode.NO_USABLE_ROWS,
        "required_column_missing": ReasonCode.MISSING_REQUIRED_COLUMN,
        "too_few_usable_rows": ReasonCode.INSUFFICIENT_SAMPLE,
        "no_treatment_contrast": ReasonCode.NO_TREATMENT_CONTRAST,
        "target_region_not_covered": ReasonCode.COVERAGE_GAP,
        "estimation_failed": ReasonCode.ESTIMATOR_FAILED,
        "target_inference_failed": ReasonCode.ESTIMATOR_FAILED,
    }


def test_no_cause_maps_to_an_executor_assigned_code():
    assert not set(tr._EFFECT_CAUSE_CODES.values()) & EXECUTOR_ASSIGNED


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        pytest.param(None, ReasonCode.SIMULATION_INCOMPLETE, id="no-cause"),
        pytest.param("a_future_cause", ReasonCode.SIMULATION_INCOMPLETE, id="unknown-cause"),
        pytest.param(EffectCause.EMPTY_COHORT, ReasonCode.NO_USABLE_ROWS, id="member"),
        pytest.param("empty_cohort", ReasonCode.NO_USABLE_ROWS, id="string-value"),
    ],
)
def test_an_absent_or_unknown_cause_gets_the_fallback(cause, expected):
    """A cause this build does not know still refuses with a code, never without one."""
    assert tr._effect_reason_code(cause, fallback=ReasonCode.SIMULATION_INCOMPLETE) is expected


def _loader_details():
    cohort = _cohort()
    no_treatment = cohort.copy()
    no_treatment["email_campaign_count"] = float("nan")
    frames = [
        pd.DataFrame(),
        cohort.drop(columns="conversion_rate"),
        cohort.head(100),
        no_treatment,
    ]
    for frame in frames:
        usability = assess_cohort_frame(frame, "email_campaign")
        assert usability.cause is not None
        yield usability.cause, dict(usability.details)


def _estimator_details(monkeypatch):
    cohort = _cohort()
    constant = cohort.copy()
    constant["email_campaign_count"] = 3.0
    frames = [
        cohort.drop(columns="email_campaign_count"),
        cohort.drop(columns="region"),
        cohort.drop(columns="market_share"),
        cohort.head(20),
        constant,
    ]
    for frame in frames:
        with pytest.raises(EffectDataUnavailable) as caught:
            estimate_cohort_effect(frame, "email_campaign_count")
        yield caught.value.cause, caught.value.details
    # "atlantis" is refused as not covered before the targeted interval is computed, so that
    # case yields a coverage_gap payload; "west" below reaches the interval and its failure.
    for forest, targets in ((_FitRaises, []), (_TargetIntervalRaises, ["atlantis"])):
        monkeypatch.setattr("econml.dml.CausalForestDML", forest)
        with pytest.raises(EffectDataUnavailable) as caught:
            estimate_cohort_effect(cohort, "email_campaign_count", target_regions=targets)
        yield caught.value.cause, caught.value.details
    monkeypatch.setattr("econml.dml.CausalForestDML", _TargetIntervalRaises)
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(cohort, "email_campaign_count", target_regions=["west"])
    yield caught.value.cause, caught.value.details
    with pytest.raises(EffectDataUnavailable) as caught:
        CohortEffectDataProvider(cohort.drop(columns="email_campaign_count")).get_training_frame(
            "email_campaign", brand="Kisqali", twin_type="hcp"
        )
    yield caught.value.cause, caught.value.details


def test_every_loader_and_estimator_details_payload_is_storable(monkeypatch, caplog):
    """A payload ``validate_details`` rejects is dropped by the refusal with an ERROR log: the
    refusal would stand, but its counts would be lost."""
    payloads = list(_loader_details()) + list(_estimator_details(monkeypatch))
    assert len(payloads) == 13
    # Every cause a tool-path refusal can carry details for; intervention_not_identified has none.
    assert {cause for cause, _ in payloads} == set(EffectCause) - {
        EffectCause.INTERVENTION_NOT_IDENTIFIED
    }
    caplog.set_level(logging.ERROR, logger=_ERRORS_LOGGER)
    for _, details in payloads:
        assert details, "every tool-path cause carries its counts"
        assert validate_details(details) == details
        refusal = ToolRefusalError("m", reason_code=ReasonCode.NO_USABLE_ROWS, details=details)
        assert refusal.details == details
    assert not [r for r in caplog.records if r.name == _ERRORS_LOGGER]
