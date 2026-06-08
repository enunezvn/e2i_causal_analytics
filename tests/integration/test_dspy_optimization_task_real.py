"""Shard 08 integration: the scheduled task runs the loop end-to-end (force=True)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("SUPABASE_URL")),
    reason="requires live LM + DB",
)


def test_force_run_executes_pipeline():
    from src.tasks.dspy_optimization_tasks import run_dspy_prompt_optimization

    # .apply() runs the task body synchronously in-process.
    result = run_dspy_prompt_optimization.apply(kwargs={"force": True, "budget": "light"}).get()
    assert result["status"] in {"completed", "skipped"}
    if result["status"] == "completed":
        assert "optimization" in result
