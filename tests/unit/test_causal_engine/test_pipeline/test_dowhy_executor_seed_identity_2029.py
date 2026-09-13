"""#2029 (lane 1b): the DoWhy pipeline executor seeds its refutation from the pair identity.

The executor is the second production caller of ``RefutationRunner.run_all_tests``
(the tool-composer ``refutation_runner`` registration reaches it with
``run_refutation=True``, i.e. the copilot path that shows verdicts to leaders).
It passes no ``estimate_id``, so without ``seed_identity`` that path stayed
UNSEEDED (``random_state: None``) while the causal-impact node was already seeded
from ``brand|treatment|outcome``. The pipeline state carries no brand, so the
identity is ``|treatment|outcome`` -- the same value ``seed_identity_for`` gives
the node for a brand-less pair.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationRunner,
    RefutationSuite,
    seed_identity_for,
)
from tests.unit.test_causal_engine.test_pipeline.test_executor_dowhy_refutation import (
    _build_real_dataframe,
    _build_state,
)


@pytest.mark.asyncio
async def test_dowhy_executor_passes_pair_seed_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The executor's ``run_all_tests`` call carries ``seed_identity`` for its pair."""
    recorded: Dict[str, Any] = {}

    def _record(self: RefutationRunner, *args: Any, **kwargs: Any) -> RefutationSuite:
        recorded.update(kwargs)
        return RefutationSuite(
            passed=True, confidence_score=1.0, tests=[], gate_decision=GateDecision.PROCEED
        )

    monkeypatch.setattr(RefutationRunner, "run_all_tests", _record)

    df = _build_real_dataframe()
    state = _build_state(df=df, confounders=["confounder_a"], run_refutation=True)
    result = await DoWhyExecutor().execute(state, state["config"])

    assert result["success"] is True, f"error={result['error']!r}"
    assert recorded, "run_all_tests was not reached with run_refutation=True"
    assert recorded["treatment"] == "treatment"
    assert recorded["outcome"] == "outcome"
    expected = seed_identity_for(brand=None, treatment="treatment", outcome="outcome")
    assert expected == "|treatment|outcome"
    assert recorded["seed_identity"] == expected
    # No estimate id is invented on this path: the pair identity IS the seed source.
    assert recorded.get("estimate_id") is None
