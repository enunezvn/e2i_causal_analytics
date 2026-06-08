"""Shard 08 integration: the scheduled task runs the loop end-to-end (force=True)."""

from __future__ import annotations

import os

import pytest

# Gated behind an explicit opt-in (NOT just the API key): this runs real GEPA
# optimization which blocks for minutes inside a thread pool that pytest-timeout's
# thread method cannot interrupt, so it would hang CI's --timeout=60 integration
# shard to the job limit (the #504 lesson). CI sets ANTHROPIC_API_KEY/SUPABASE_URL,
# so key-gating alone is not enough — require E2I_RUN_REAL_LLM_E2E=1. Run manually:
#   E2I_RUN_REAL_LLM_E2E=1 pytest tests/integration/test_dspy_optimization_task_real.py
pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1"
    or not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("SUPABASE_URL")),
    reason="requires E2I_RUN_REAL_LLM_E2E=1 + live LM + DB (slow real GEPA run)",
)


def test_force_run_executes_pipeline():
    from src.tasks.dspy_optimization_tasks import run_dspy_prompt_optimization

    # .apply() runs the task body synchronously in-process.
    result = run_dspy_prompt_optimization.apply(kwargs={"force": True, "budget": "light"}).get()
    assert result["status"] in {"completed", "skipped"}
    if result["status"] == "completed":
        assert "optimization" in result
