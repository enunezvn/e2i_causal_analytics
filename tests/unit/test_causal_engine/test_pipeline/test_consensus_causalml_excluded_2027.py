"""#2027 part A: CausalML is not a member of the ATE consensus.

``_apply_consensus`` weights by inverse variance only when EVERY contributing
library has a sampling SE (an all-or-nothing gate, by design). CausalML can never
supply one post-#2014 — its ``ate_ci_*`` is ``std(predicted uplift) / sqrt(n)``, a
dispersion — so its mere presence in ``_collect_ate_estimates`` forced DoWhy and
EconML off precision weighting and onto the incommensurable confidence scale, where
DoWhy's hardcoded 1.0 dominates. Measured on planted truth (2026-09-12, binary
outcome where CausalML itself recovers fine): consensus error 0.048 with CausalML in
the consensus vs 0.028 without.

The fix removes CausalML from the ATE track. Its uplift channel (``uplift_summary``:
auuc / qini / ate, targeting) and its sequential stage payload are untouched.

Real ``_aggregate_results`` / ``_aggregate_parallel_results`` over executor-shaped
result payloads placed in state (the same shape ``test_aggregation_4lib.py`` uses),
extended with the two fields the SE gate reads: DoWhy's ``standard_error`` and an
EconML estimator from ``ECONML_SAMPLING_INTERVAL_ESTIMATORS``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.causal_engine.pipeline.orchestrator import PipelineOrchestrator
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.router import CausalLibrary, LibraryRouter
from src.causal_engine.pipeline.sequential import (
    _Z_95,
    SequentialPipeline,
    _collect_ate_estimates,
)
from src.causal_engine.pipeline.state import (
    LibraryExecutionResult,
    PipelineInput,
    PipelineOutput,
    PipelineState,
)
from tests.unit.test_causal_engine.test_pipeline.test_aggregation_4lib import (
    _le_result,
    _minimal_pipeline_state,
    _real_causalml_payload,
    _real_dowhy_payload,
    _real_econml_payload,
)

DOWHY_EFFECT = 0.15
DOWHY_SE = 0.02
ECONML_EFFECT = 0.17
ECONML_SE = 0.05


class _ConcreteOrchestrator(PipelineOrchestrator):
    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        routing_decision = await self.route(input_data["query"])
        state = self._create_initial_state(input_data, routing_decision)
        return self._create_output(state)


def _dowhy_payload_with_se() -> dict[str, Any]:
    payload = _real_dowhy_payload()
    payload["standard_error"] = DOWHY_SE  # the HC1 SE of DoWhy's OLS fit (#2014)
    return payload


def _econml_sampling_payload() -> dict[str, Any]:
    payload = _real_econml_payload()
    half = _Z_95 * ECONML_SE
    payload["estimator"] = "linear_dml"  # in ECONML_SAMPLING_INTERVAL_ESTIMATORS
    payload["ate_ci_lower"] = ECONML_EFFECT - half
    payload["ate_ci_upper"] = ECONML_EFFECT + half
    return payload


def _three_library_state() -> PipelineState:
    """DoWhy + EconML with sampling SEs, plus a SUCCESSFUL CausalML run whose ``ate``
    reached ``uplift_summary`` through the orchestrator's extraction path."""
    state = _minimal_pipeline_state()
    state["dowhy_result"] = _le_result("dowhy", _dowhy_payload_with_se(), confidence=1.0)
    state["causal_effect"] = DOWHY_EFFECT
    state["econml_result"] = _le_result("econml", _econml_sampling_payload(), confidence=0.8)
    state["overall_ate"] = ECONML_EFFECT
    state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.6)
    state = _ConcreteOrchestrator()._update_state_with_result(
        state, CausalLibrary.CAUSALML, cast(LibraryExecutionResult, state["causalml_result"])
    )
    assert state["uplift_summary"] is not None and state["uplift_summary"]["ate"] == 0.18
    return state


def _inverse_variance_of_dowhy_and_econml() -> float:
    w_d, w_e = 1.0 / DOWHY_SE**2, 1.0 / ECONML_SE**2
    return (DOWHY_EFFECT * w_d + ECONML_EFFECT * w_e) / (w_d + w_e)


def test_sequential_consensus_is_inverse_variance_over_dowhy_and_econml() -> None:
    state = _three_library_state()

    updated = SequentialPipeline(router=LibraryRouter())._aggregate_results(state)

    assert updated["consensus_weighting"] == "inverse_variance"
    assert updated["consensus_effect"] == pytest.approx(
        _inverse_variance_of_dowhy_and_econml(), abs=1e-12
    )


def test_parallel_consensus_is_inverse_variance_over_dowhy_and_econml() -> None:
    state = _three_library_state()

    updated = ParallelPipeline(router=LibraryRouter())._aggregate_parallel_results(state)

    assert updated["consensus_weighting"] == "inverse_variance"
    assert updated["consensus_effect"] == pytest.approx(
        _inverse_variance_of_dowhy_and_econml(), abs=1e-12
    )


def test_collect_ate_estimates_has_no_causalml_triple() -> None:
    state = _three_library_state()

    effects = _collect_ate_estimates(state)

    assert [lib for lib, _, _ in effects] == ["dowhy", "econml"]


def test_uplift_channel_survives_aggregation_unchanged() -> None:
    state = _three_library_state()
    before = dict(cast(dict, state["uplift_summary"]))

    updated = SequentialPipeline(router=LibraryRouter())._aggregate_results(state)

    assert updated["uplift_summary"] == before
    assert updated["uplift_summary"]["auuc"] == 0.72
    assert updated["uplift_summary"]["qini"] == 0.55
    assert updated["uplift_summary"]["ate"] == 0.18
    agreement = updated["library_agreement"] or {}
    assert set(agreement) == {"dowhy_econml"}
