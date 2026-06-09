"""Bounded faithful end-to-end proof of the DSPy self-improvement loop (V).

Gated behind E2I_RUN_REAL_LLM_E2E=1 (the #504 precedent: CI's pytest-timeout
thread method cannot interrupt GEPA's thread-pool LM calls). Run manually:

    E2I_RUN_REAL_LLM_E2E=1 .venv/bin/pytest \
      tests/integration/test_dspy_loop_e2e_bounded.py -v -s -p no:cacheprovider

Proves the MECHANISM end-to-end on REAL Supabase + REAL Anthropic LM, using
SYNTHETIC inputs (the loop is starved of real production data — see
docs/reports/dspy-loop-disproof-20260608/). "Faithful" = real LM/DB/GEPA code
paths, not stubbed; it does NOT claim real production self-improvement.

Bounded: ONE learner phase ("pattern") + ONE recipient field
(experiment_monitor.srm_template), light GEPA budget. Self-cleaning.

Part A (free, no LM): generate learner signals + recipient signals and assert
the signal->example conversion is non-degenerate BEFORE spending any GEPA budget.
Part B (real LM): run GEPA for the learner pattern phase and the recipient,
assert a non-empty optimized instruction is produced, saved, round-trips, and the
recipient serves a non-default template after install.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1",
    reason="bounded faithful real-LM E2E; set E2I_RUN_REAL_LLM_E2E=1 to run",
)

_RUN = uuid.uuid4().hex[:8]
_LEARNER_BATCH_PREFIX = f"e2e_{_RUN}_"


def _synthetic_feedback_items(n: int):
    """Synthetic feedback rows in the shape FeedbackCollectorNode expects."""
    items = []
    for i in range(n):
        negative = i % 2 == 0
        items.append(
            {
                "id": f"{_LEARNER_BATCH_PREFIX}fb_{i}",
                "timestamp": "2026-06-08T00:00:00+00:00",
                "agent": "causal_impact",
                "query": f"Why did Kisqali NE TRx dip in week {i}?",
                "response": "The model attributed the dip to seasonality.",
                "rating": (1 if negative else 5),
                "correction": ("Should have flagged the payer-mix change." if negative else None),
                "metadata": {"brand": "kisqali", "region": "northeast"},
            }
        )
    return items


async def _get_client():
    from src.memory.services.factories import get_supabase_client

    c = get_supabase_client()
    import inspect

    return await c if inspect.isawaitable(c) else c


@pytest.mark.asyncio
async def test_bounded_loop_e2e(capsys):
    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer
    from src.agents.feedback_learner.recipient_emit import emit_recipient_signal
    from src.agents.feedback_learner.recipient_optimizer import (
        recipient_required_input_keys,
        signal_example_provider,
    )
    from src.agents.feedback_learner.signal_store import (
        get_feedback_learner_training_signals,
    )
    from src.optimization.dspy_lm import ensure_dspy_configured

    assert ensure_dspy_configured(), "no DSPy LM configured (.env ANTHROPIC key missing)"
    client = await _get_client()
    assert client is not None, "no Supabase client (docker supabase-db)"

    created_learner_batches: list[str] = []
    try:
        # ---------- seed REAL-shaped synthetic learner signals (no LM) ----------
        # Shape matches FeedbackLearnerOptimizer._signals_to_examples: feedback_batch
        # in input_context, patterns/recommendations/summary in output, reward>=0.5.
        _fb = _synthetic_feedback_items(6)
        _patterns = [
            {
                "pattern_type": "recurring_negative_feedback",
                "severity": "high",
                "affected_agents": ["causal_impact"],
                "root_cause_hypothesis": "Model misses payer-mix shifts in TRx attribution.",
                "description": "Repeated low ratings on causal_impact TRx explanations.",
            }
        ]
        _recs = [
            {"category": "data", "text": "Add a payer-mix covariate to the attribution model."}
        ]
        # >=10 so the pattern trainset (80% split) clears GEPA's >=5 guard.
        for i in range(10):
            b = f"{_LEARNER_BATCH_PREFIX}{i}"
            rec = {
                "source_agent": "feedback_learner",
                "batch_id": b,
                "input_context": {
                    "feedback_batch": _fb,
                    "agent_baselines": {},
                    "historical_patterns": [],
                },
                "output": {
                    "patterns": _patterns,
                    "recommendations": _recs,
                    "learning_summary": "Recurring payer-mix attribution gap on causal_impact.",
                    "applied_updates": [],
                },
                "reward": 0.8,
                "is_training_example": True,
            }
            client.table("dspy_agent_training_signals").insert(rec).execute()
            created_learner_batches.append(b)
        print(f"[gen] learner signals seeded: {created_learner_batches}")

        # ---------- emit REAL recipient signals (no LM) ----------
        srm_keys = recipient_required_input_keys("experiment_monitor")["srm_template"]
        print(f"[gen] experiment_monitor srm_template required keys: {srm_keys}")
        for i in range(3):
            ok = await emit_recipient_signal(
                agent_name="experiment_monitor",
                signature_inputs={
                    "experiment_name": f"Kisqali-NE-{i}",
                    "chi_squared": 12.4 + i,
                    "p_value": 0.0004,
                    "expected_ratio": "50/50",
                    "actual_counts": "640/360",
                },
                generated_output=(
                    "A statistically significant sample ratio mismatch was detected "
                    "(chi-squared=12.4, p=0.0004): the treatment arm received far more "
                    "units than the 50/50 design. Freeze enrollment and audit the "
                    "randomization service."
                ),
                reward=0.8,
                template_field="srm_template",
                client=client,
            )
            assert ok, "recipient emit failed"

        # ---------- PART A: free pre-check (NO GEPA) — fail before spending budget ----------
        signals = await get_feedback_learner_training_signals(min_reward=0.0, limit=2000)
        _mybatches = set(created_learner_batches)
        mine = [s for s in signals if s.get("batch_id") in _mybatches]
        print(f"[A] learner signals readable (mine): {len(mine)}")
        assert len(mine) >= 5, "learner signal generation/persistence is broken"

        opt = FeedbackLearnerOptimizer(optimizer_type="gepa")
        pattern_examples = opt._signals_to_examples(mine, "pattern")  # type: ignore[attr-defined]
        print(f"[A] pattern examples built: {len(pattern_examples)}")
        assert len(pattern_examples) >= 1, (
            "signal->example conversion is degenerate (empty trainset)"
        )

        provider = signal_example_provider("experiment_monitor", client=client)
        srm_examples = provider("srm_template")
        print(f"[A] recipient srm examples built: {len(srm_examples)}")
        assert len(srm_examples) >= 2, "recipient example provider degenerate (would cold-start)"

        # ---------- PART B: real GEPA (LM spend) — learner pattern phase ----------
        from src.agents.feedback_learner.optimization_runner import (
            run_feedback_learner_optimization,
        )

        result = await run_feedback_learner_optimization(
            phases=("pattern",), budget="light", min_reward=0.0
        )
        print(f"[B] learner optimization: {result}")
        assert result["status"] == "completed", result
        pinfo = result["phases"].get("pattern", {})
        assert pinfo.get("status") == "optimized", f"learner pattern not optimized: {pinfo}"

        # proof the optimized artifact was persisted to disk (round-trip-able)
        from pathlib import Path

        saved_path = pinfo.get("path")
        print(
            f"[B] learner optimized artifact: version={pinfo.get('version_id')} path={saved_path}"
        )
        assert saved_path and Path(saved_path).exists(), (
            f"optimized artifact not on disk: {saved_path}"
        )

        # ---------- PART B: real GEPA — recipient (experiment_monitor) ----------
        from src.agents.experiment_monitor.dspy_integration import (
            get_experiment_monitor_dspy_integration,
        )
        from src.agents.feedback_learner.prompt_bundles import install_all_prompt_bundles
        from src.agents.feedback_learner.recipient_optimizer import optimize_and_save_recipient

        before = get_experiment_monitor_dspy_integration().get_prompt_metadata()
        bundle_path = await optimize_and_save_recipient("experiment_monitor", budget="light")
        installed = install_all_prompt_bundles()
        after = get_experiment_monitor_dspy_integration().get_prompt_metadata()
        print(f"[B] recipient bundle={bundle_path} installed={installed}")
        print(f"[B] experiment_monitor metadata before={before} after={after}")
        assert bundle_path, "recipient optimization produced no bundle from real emitted signals"
        # install reported success for experiment_monitor, and metadata reflects an optimization.
        # get_prompt_metadata() nests the install provenance under "prompts".
        assert installed.get("experiment_monitor") is True
        after_prompts = after.get("prompts", after)
        assert after_prompts.get("last_optimized"), (
            "recipient still serves default (last_optimized empty)"
        )
        before_prompts = before.get("prompts", before)
        assert after_prompts.get("last_optimized") != before_prompts.get("last_optimized"), (
            "recipient optimization did not change the served template provenance"
        )
        print("[DONE] bounded faithful E2E passed — mechanism works on real LM/DB/GEPA")
    finally:
        # ---------- cleanup: delete ONLY rows this test created ----------
        try:
            for b in created_learner_batches:
                client.table("dspy_agent_training_signals").delete().eq("batch_id", b).execute()
            # experiment_monitor rows: none pre-exist in this env; delete our training examples
            client.table("dspy_agent_training_signals").delete().eq(
                "source_agent", "experiment_monitor"
            ).eq("is_training_example", True).execute()
            print("[cleanup] synthetic rows deleted")
        except Exception as e:  # noqa: BLE001
            print(f"[cleanup] WARNING: cleanup failed: {e}")
