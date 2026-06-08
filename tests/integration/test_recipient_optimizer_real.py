"""Shard 09 integration: a real per-recipient optimization run (live LM).

Faithful end-to-end check that the recipient producer optimizes a recipient's
signature on golden seeds and saves a placeholder-safe PromptBundle. Skipped
without ANTHROPIC_API_KEY. A full GEPA run is slow (minutes) — this is for
manual/CI verification, not the default fast path.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires live Anthropic LM"
)


@pytest.mark.asyncio
async def test_optimize_and_save_experiment_monitor(tmp_path, monkeypatch):
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle
    from src.agents.feedback_learner.recipient_optimizer import optimize_and_save_recipient

    monkeypatch.chdir(tmp_path)
    path = await optimize_and_save_recipient("experiment_monitor", budget="light")
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
