"""#2027 part A: the CausalML stage payload already states its estimand — pin it.

CausalML left the ATE consensus (its presence forced ``_apply_consensus`` off
inverse-variance weighting; see ``test_consensus_causalml_excluded_2027.py``), but the
sequential stage still serves ``effect_estimate`` = mean model-predicted uplift. What
keeps that number from reading as a sampling-validated ATE is what the stage ALREADY
carries: ``ci_lower`` / ``ci_upper`` / ``p_value`` are None (the only interval
available is a dispersion), ``additional_results["data_provenance"]`` is the
executor's honesty marker, and ``additional_results["estimand"]`` names the estimand
in the #2106 vocabulary (``"ate"`` on a genuinely binary outcome). No new key, no
change to the ``estimand`` string.

Same harness as ``test_causal_causalml_estimand_2067.py``: a REAL
``CausalMLExecutor.execute`` run with only the forest fit stood in for.
"""

from __future__ import annotations

from src.api.schemas.causal import AnalysisStatus
from src.causal_engine.uplift.base import PROVENANCE_MODEL_PREDICTED_UPLIFT
from tests.unit.test_api.test_causal_causalml_estimand_2067 import _run_causalml, _stage


def test_causalml_stage_states_its_estimand_without_a_sampling_interval() -> None:
    output, state = _run_causalml([0.0, 1.0], reps=30)

    stage = _stage(output, state)

    assert stage.status == AnalysisStatus.COMPLETED
    assert isinstance(stage.effect_estimate, float)
    assert stage.ci_lower is None
    assert stage.ci_upper is None
    assert stage.p_value is None
    assert stage.additional_results["data_provenance"] == PROVENANCE_MODEL_PREDICTED_UPLIFT
    assert stage.additional_results["estimand"] == "ate"
    assert "identified_estimand" not in stage.additional_results
