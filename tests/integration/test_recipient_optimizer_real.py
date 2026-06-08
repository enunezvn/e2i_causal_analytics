"""Shard 09 integration: a real per-recipient optimization run (live LM).

Faithful end-to-end check that the recipient producer optimizes a recipient's
signature on golden seeds and saves a placeholder-safe PromptBundle. Skipped
without ANTHROPIC_API_KEY. A full GEPA run is slow (minutes) — this is for
manual/CI verification, not the default fast path.
"""

from __future__ import annotations

import os

import pytest

# Gated behind an explicit opt-in: a real GEPA recipient run blocks for minutes in a
# thread pool pytest-timeout cannot interrupt, hanging CI's --timeout=60 shard to the
# job limit (#504 lesson). CI sets ANTHROPIC_API_KEY, so require E2I_RUN_REAL_LLM_E2E=1.
pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1" or not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires E2I_RUN_REAL_LLM_E2E=1 + live Anthropic LM (slow real GEPA run)",
)


@pytest.mark.asyncio
async def test_optimize_and_save_experiment_monitor(tmp_path, monkeypatch):
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle
    from src.agents.feedback_learner.recipient_optimizer import optimize_and_save_recipient
    from tests.unit.test_agents.test_feedback_learner._recipient_seed_fixtures import (
        default_example_provider,
    )

    monkeypatch.chdir(tmp_path)
    # Drive GEPA off the relocated golden-seed fixture so this faithful real-LM run
    # validates the compile/materialize mechanism without depending on the live
    # signals table being populated. Production reads real emitted signals instead.
    path = await optimize_and_save_recipient(
        "experiment_monitor",
        example_provider=default_example_provider("experiment_monitor"),
        budget="light",
    )
    # If optimization yields instructions, a placeholder-safe bundle is saved.
    if path:
        bundle = load_prompt_bundle("experiment_monitor")
        assert bundle is not None
        # Every saved template must still .format() (placeholders preserved).
        srm = bundle["templates"].get("srm_template", "")
        if srm:
            srm.format(
                experiment_name="E",
                chi_squared=1.0,
                p_value=0.01,
                expected_ratio="50/50",
                actual_counts="1/1",
            )
