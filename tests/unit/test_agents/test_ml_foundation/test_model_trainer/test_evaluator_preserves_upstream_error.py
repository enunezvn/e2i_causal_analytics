"""#773 cluster 4 prerequisite: evaluate_model must NOT mask the TRUE training error.

``graph.py`` wires ``train_model -> evaluate_model`` unconditionally. When
``train_model`` fails it returns ``{"error": ..., "error_type":
"training_failed" | "instantiation_failed" | "unsupported_algorithm" | ...,
"training_status": "failed"}`` and (by definition) NO ``trained_model``.
``evaluate_model`` then hit its ``trained_model is None`` guard and emitted

    {"error": "No trained model available for evaluation",
     "error_type": "missing_trained_model"}

OVERWRITING the true error in the merged LangGraph state — so every nightly
failure since 2026-06-08 surfaced as the same uninformative
``RuntimeError: Training error (missing_trained_model): ...`` (agent.py:384),
masking the actual root cause out of CI logs.

Fix: the same F2 skip-on-upstream-error idiom the graph already uses in
``augment_training_data`` and ``learning_curve`` — when ``state["error"]`` is
already set, ``evaluate_model`` emits nothing, letting the true
``error``/``error_type`` flow to the caller (downstream conditionals route to
END on the pre-existing error regardless).
"""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

pytestmark = pytest.mark.asyncio


async def test_preserves_upstream_training_failed_error():
    """State enters with error_type='training_failed' (and no trained model):
    after the node, the EFFECTIVE state error_type must still be
    'training_failed' — not the masking 'missing_trained_model'."""
    state = {
        "error": "Model training failed: ValueError: boom",
        "error_type": "training_failed",
        "training_status": "failed",
        # No trained_model — exactly what a train_model failure leaves behind.
        "problem_type": "binary_classification",
    }

    result = await evaluate_model(state)

    # LangGraph merges the node's partial update into the state; the merged
    # view is what agent.py:384 reads to raise "Training error (<type>)".
    merged = {**state, **result}
    assert merged["error_type"] == "training_failed", (
        "evaluate_model overwrote the true upstream error_type "
        f"({merged['error_type']!r}) — the nightly mask this PR removes"
    )
    assert merged["error"] == "Model training failed: ValueError: boom"


async def test_preserves_other_upstream_error_types():
    """Same guarantee for the other train_model failure modes the mask hid."""
    for true_type in ("instantiation_failed", "unsupported_algorithm", "missing_training_data"):
        state = {
            "error": f"upstream: {true_type}",
            "error_type": true_type,
        }
        result = await evaluate_model(state)
        merged = {**state, **result}
        assert merged["error_type"] == true_type, (
            f"true error_type {true_type!r} was masked as {merged['error_type']!r}"
        )


async def test_still_reports_missing_trained_model_without_upstream_error():
    """Genuinely-missing model with NO upstream error: the evaluator's own
    honest error must still fire (this is the pre-existing contract pinned by
    test_evaluator.py::test_error_when_no_trained_model)."""
    state = {
        "problem_type": "binary_classification",
        "X_test_preprocessed": np.random.rand(10, 3),
        "test_data": {"y": np.random.randint(0, 2, 10)},
    }

    result = await evaluate_model(state)

    assert result["error_type"] == "missing_trained_model"
