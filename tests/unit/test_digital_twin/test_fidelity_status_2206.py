"""#2206: a NULL model fidelity must read as UNVALIDATED, never as "passed".

The engine's gate was ``if score and score < 0.7`` — a ``None`` (no experiment
outcome has ever been compared against the model) fell through as "no warning",
and so did a real ``0.0``. ``classify_fidelity`` is the one rule every surface
(engine result, /simulate response, /models listing, chat tool) derives from.
"""

from __future__ import annotations

import pytest

from src.digital_twin.models.simulation_models import (
    FIDELITY_WARNING_THRESHOLD,
    FidelityStatus,
    classify_fidelity,
)


def test_null_score_is_unvalidated_and_warns() -> None:
    status, warning, reason = classify_fidelity(None)
    assert status is FidelityStatus.UNVALIDATED
    assert warning is True
    assert reason is not None and "unvalidated" in reason.lower()
    assert "compared" in reason.lower()


def test_below_threshold_is_explicit() -> None:
    status, warning, reason = classify_fidelity(0.55)
    assert status is FidelityStatus.BELOW_THRESHOLD
    assert warning is True
    assert reason is not None and "0.55" in reason and "below threshold" in reason


@pytest.mark.parametrize("score", [FIDELITY_WARNING_THRESHOLD, 0.85, 1.0])
def test_at_or_above_threshold_is_validated_without_warning(score: float) -> None:
    status, warning, reason = classify_fidelity(score)
    assert status is FidelityStatus.VALIDATED
    assert warning is False
    assert reason is None


def test_a_real_zero_is_a_score_not_a_null() -> None:
    """The old falsy check treated 0.0 like None; a measured 0.0 is the worst fidelity."""
    status, warning, _ = classify_fidelity(0.0)
    assert status is FidelityStatus.BELOW_THRESHOLD
    assert warning is True


def test_threshold_is_the_engine_gate_value() -> None:
    assert FIDELITY_WARNING_THRESHOLD == 0.7
