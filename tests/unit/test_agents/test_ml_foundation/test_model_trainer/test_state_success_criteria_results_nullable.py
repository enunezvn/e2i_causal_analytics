"""#617: ``success_criteria_results`` must accept ``None`` values.

The v3 adaptive evaluator intentionally records ``met=None`` for criteria that
were skipped or whose metric was NaN (Option C audit contract — see
``evaluator.py`` ``_check_success_criteria``). The field was typed
``Optional[Dict[str, bool]]``, so when LangGraph coerced the evaluate_model node
return into the ``ModelTrainerState`` Pydantic schema it raised
``ValidationError: success_criteria_results.<metric> Input should be a valid
boolean [input_value=None]``, surfaced as ``RuntimeError('Model training
workflow failed')`` — aborting the run before the artifact was written and
failing all four ``test_adaptive_criteria_e2e`` tests once the Feast gate was
unblocked. Widening the value type to ``Optional[bool]`` fixes it.
"""

from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_trainer.state import ModelTrainerState


@pytest.mark.unit
def test_success_criteria_results_accepts_none_values() -> None:
    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        success_criteria_results={"minimum_auc": None, "minimum_recall": True},
    )
    assert state.success_criteria_results == {"minimum_auc": None, "minimum_recall": True}
