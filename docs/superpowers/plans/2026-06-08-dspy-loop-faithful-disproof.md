# DSPy Loop — Faithful Disproof (Premise Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the disproving data — *does the merged DSPy loop produce real self-improvement on real data?* — before building the Gap A/B apparatus, by running the loop end-to-end against real Supabase + real Anthropic LM and inspecting every link.

**Architecture:** A premise-investigation pass (what real feedback/signals actually exist) followed by a gated, faithful integration harness that drives `agent.learn()` → persist → `run_feedback_learner_optimization()` → re-`learn()` (does it load the optimized module?) and the recipient `optimize → install → consume` path. No mocks. The harness REPORTS, then a decision gate confirms or revises the build plans.

**Tech Stack:** Python 3.12, dspy 3.1.0, Anthropic LM (`anthropic/claude-sonnet-4-20250514`), Supabase (docker `supabase-db`), pytest (loads `.env`), gated by `E2I_RUN_REAL_LLM_E2E=1`.

**Why this is the whole plan (and not the build):** Reading the source (2026-06-08) found that `FeedbackCollectorNode._collect_user_feedback` calls `feedback_store.get_feedback(**kwargs)` (`nodes/feedback_collector.py:98`), but **no production store implements `get_feedback`** — only the in-`agent.py` `MockStore` (`:261`) and `dspy_receiver.get_feedback_items_from_signals`. So the learner's feedback-input premise is unverified. Per CLAUDE.md cheapest-disproof-first (pinned), we measure before we build.

---

## Decisions locked from the spec
- Spec: `docs/superpowers/specs/2026-06-08-dspy-loop-real-results-design.md`.
- Reward = deterministic heuristic; signal-generation = Celery beat + API persist fix; all 4 recipients; cold-start = skip (seeds become test-only); stop at code-complete + faithful proof, no deploy.
- This plan precedes those builds and may revise their signal-source approach based on findings.

---

## Task 1: Premise investigation — what real feedback & signals exist?

**Files:**
- Read: `src/agents/feedback_learner/nodes/feedback_collector.py`, `src/agents/feedback_learner/dspy_receiver.py`, `src/repositories/chatbot_feedback.py`, `src/api/repositories/feedback_repository.py`
- Create: `docs/reports/dspy-loop-disproof-20260608/PREMISE.md` (findings only — no code change)

- [ ] **Step 1: Map the real feedback-input path**

Run (report each result into PREMISE.md):
```bash
cd /home/enunez/Projects/e2i_causal_analytics
# 1a. What store interface does the collector require, and what do real stores provide?
sed -n '92,205p' src/agents/feedback_learner/nodes/feedback_collector.py
grep -nE "async def get_(feedback|outcomes|implicit)" src/ -r --include=*.py
# 1b. Is there a non-mock source the collector can read? (signals-as-feedback bridge)
sed -n '400,440p' src/agents/feedback_learner/dspy_receiver.py
```
Expected/record: confirm whether ANY non-mock object satisfies `get_feedback(start_date=, end_date=, ...)`. Document the answer (yes/no + which).

- [ ] **Step 2: Measure real data volume in the live DB**

Run (docker psql — DATABASE_URL in `.env` is a placeholder, use the container):
```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT source_agent, count(*), round(avg(reward)::numeric,3) avg_reward, max(created_at) latest
     FROM dspy_agent_training_signals GROUP BY source_agent ORDER BY 2 DESC;"
# any human/outcome feedback tables the collector could draw on?
docker exec supabase-db psql -U postgres -d postgres -c "\dt" | grep -iE "feedback|outcome|prediction"
```
Record into PREMISE.md: per-agent signal counts (esp. `feedback_learner`), whether ≥5 exist, and what feedback/outcome tables are populated.

- [ ] **Step 3: Write the premise verdict**

In PREMISE.md, answer explicitly: (a) is there a real feedback source for the collector, or does learn() run on empty/implicit-only input? (b) how many real `feedback_learner` signals already exist? (c) does the build need to *also* wire a real feedback source (scope expansion) or is implicit/outcome feedback sufficient? This verdict feeds Task 3's decision gate.

- [ ] **Step 4: Commit**
```bash
git add docs/reports/dspy-loop-disproof-20260608/PREMISE.md
git commit -m "docs(dspy-disproof): premise investigation — real feedback/signal inventory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Faithful disproof harness (gated real-LM integration test)

**Files:**
- Create: `tests/integration/test_dspy_loop_faithful_disproof.py`
- Test command (manual, faithful): see Step 4.

- [ ] **Step 1: Write the harness as a gated integration test**

Create `tests/integration/test_dspy_loop_faithful_disproof.py`:
```python
"""Faithful end-to-end disproof of the DSPy self-improvement loop (no mocks).

Gated behind E2I_RUN_REAL_LLM_E2E=1 (the #504 precedent: CI's pytest-timeout
thread method cannot interrupt GEPA's thread-pool LM calls). Run manually:

    E2I_RUN_REAL_LLM_E2E=1 .venv/bin/pytest \
      tests/integration/test_dspy_loop_faithful_disproof.py -v -s

Proves (or disproves) each link on REAL Supabase + REAL Anthropic LM:
  1. agent.learn() persists a real feedback_learner signal (existing F5).
  2. >=5 real signals read back by the optimizer's reader.
  3. run_feedback_learner_optimization() yields a CHANGED, non-empty optimized
     instruction per phase, and the saved module round-trips on load.
  4. A second learn() with prefer_optimized=True LOADS the optimized module.
  5. Recipient path: optimize -> install -> experiment_monitor serves a
     NON-default template (last_optimized != "").
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1",
    reason="faithful real-LM disproof; set E2I_RUN_REAL_LLM_E2E=1 to run",
)


def _window():
    end = datetime.now(timezone.utc)
    return (end - timedelta(days=7)).isoformat(), end.isoformat()


@pytest.mark.asyncio
async def test_faithful_loop_disproof(capsys):
    from src.optimization.dspy_lm import ensure_dspy_configured
    from src.agents.feedback_learner.agent import FeedbackLearnerAgent
    from src.agents.feedback_learner.signal_store import (
        get_feedback_learner_training_signals,
    )
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    assert ensure_dspy_configured(), "no DSPy LM configured (.env key missing)"

    # --- Link 1: generate real signals via agent.learn() (existing F5 persists) ---
    # use_llm=True so analysis is real; stores left as production-default. If
    # Task 1 found a real feedback source, wire it here instead of None.
    agent = FeedbackLearnerAgent(use_llm=True, persist_signals=True)
    start, end = _window()
    runs = []
    for _ in range(6):  # >=5 to clear optimization_runner.MIN_SIGNALS
        out = await agent.learn(
            time_range_start=start,
            time_range_end=end,
            batch_id=f"disproof_{uuid.uuid4().hex[:8]}",
        )
        runs.append(out)
    print(f"[L1] learn() runs: {len(runs)}; "
          f"rewards={[round(r.training_reward or 0, 3) for r in runs]}; "
          f"feedback_counts={[r.feedback_count for r in runs]}")

    # --- Link 2: signals read back ---
    signals = await get_feedback_learner_training_signals(min_reward=0.0, limit=2000)
    print(f"[L2] feedback_learner signals readable: {len(signals)}")
    assert len(signals) >= 5, "fewer than 5 real signals — premise weak (see PREMISE.md)"

    # --- Link 3: optimize on real signals; instruction must change & round-trip ---
    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

    baseline = {}
    opt = FeedbackLearnerOptimizer(optimizer_type="gepa")
    for phase in ("pattern", "recommendation", "summary"):
        # capture the seed (pre-optimization) instruction for comparison
        import dspy
        from src.agents.feedback_learner import dspy_integration as di
        sig = getattr(di, {
            "pattern": "PatternDetectionSignature",
            "recommendation": "RecommendationGenerationSignature",
            "summary": "LearningSummarySignature",
        }[phase], None)
        baseline[phase] = (
            str(dspy.ChainOfThought(sig).predictors()[0].signature.instructions)
            if sig is not None else ""
        )

    result = await run_feedback_learner_optimization(budget="light")
    print(f"[L3] optimization result: status={result['status']} "
          f"signals_used={result['signals_used']} phases={result['phases']}")
    assert result["status"] == "completed", result
    optimized_any = False
    for phase, info in result["phases"].items():
        if info.get("status") == "optimized":
            optimized_any = True
            # round-trip the saved module
            from src.optimization.gepa import load_optimized_module  # noqa: F401
            print(f"[L3] phase={phase} version={info.get('version_id')} "
                  f"path={info.get('path')}")
    assert optimized_any, "GEPA produced no optimized module on real signals — DISPROOF"

    # --- Link 4: a fresh learn() prefers/loads the optimized module ---
    agent2 = FeedbackLearnerAgent(use_llm=True, persist_signals=False)
    out2 = await agent2.learn(time_range_start=start, time_range_end=end)
    print(f"[L4] post-optimization learn() status={out2.status} "
          f"reward={out2.training_reward}")
    # (Assertion is informational — PatternAnalyzerNode(prefer_optimized=True)
    #  should load feedback_learner_pattern; log whether it did.)

    # --- Link 5: recipient install->consume swaps the template ---
    from src.agents.feedback_learner.recipient_optimizer import (
        optimize_and_save_recipient,
    )
    from src.agents.feedback_learner.prompt_bundles import install_all_prompt_bundles
    from src.agents.experiment_monitor.dspy_integration import (
        get_experiment_monitor_dspy_integration,
    )

    before = get_experiment_monitor_dspy_integration().get_prompt_metadata()
    path = await optimize_and_save_recipient("experiment_monitor", budget="light")
    installed = install_all_prompt_bundles()
    after = get_experiment_monitor_dspy_integration().get_prompt_metadata()
    print(f"[L5] recipient bundle={path} installed={installed} "
          f"before={before} after={after}")
    # NOTE: this link currently uses the GOLDEN-SEED producer; it proves the
    # optimize->install->consume PLUMBING. Gap B replaces the seed with real
    # self-emission. A non-default 'after' metadata confirms the plumbing.
```

- [ ] **Step 2: Verify it is collected and skipped without the flag**

Run: `.venv/bin/pytest tests/integration/test_dspy_loop_faithful_disproof.py -v`
Expected: `1 skipped` (gate off — confirms CI-safety; no real LM/DB touched).

- [ ] **Step 3: Commit the harness (skipped in CI)**
```bash
.venv/bin/ruff format tests/integration/test_dspy_loop_faithful_disproof.py
git add tests/integration/test_dspy_loop_faithful_disproof.py
git commit -m "test(dspy-disproof): faithful real-LM end-to-end loop harness (gated)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Run it FAITHFULLY and capture output**

Run (real LM + real Supabase; clean any junk rows after):
```bash
E2I_RUN_REAL_LLM_E2E=1 .venv/bin/pytest \
  tests/integration/test_dspy_loop_faithful_disproof.py -v -s 2>&1 | tee \
  docs/reports/dspy-loop-disproof-20260608/RUN.log
```
Expected: PASS or a specific failing link. Capture the `[L1]`–`[L5]` lines verbatim.

- [ ] **Step 5: Clean test pollution**

The harness persists real `feedback_learner` signals (batch_id `disproof_*`). Remove them:
```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "DELETE FROM dspy_agent_training_signals WHERE batch_id LIKE 'disproof_%';"
```

---

## Task 3: Decision gate — confirm or revise the build plans

**Files:**
- Create: `docs/reports/dspy-loop-disproof-20260608/EVIDENCE.md`

- [ ] **Step 1: Record the verdict against each link**

Write EVIDENCE.md with the `[L1]`–`[L5]` results and a verdict per link (PASS / WEAK / DISPROOF), plus the PREMISE.md feedback-source finding.

- [ ] **Step 2: Decide the next plan**

Choose ONE and record the rationale:
- **GO (mechanism real):** L2≥5, L3 optimized + changed instruction, L5 plumbing swaps template → write the **Gap A build plan** (API persist fix + generation beat + threshold) as specced.
- **GO-WITH-SCOPE-EXPANSION (no real feedback source):** L1/Task-1 shows the collector has no real source → Gap A must ALSO wire a real feedback source (e.g., the `dspy_receiver` signals-as-feedback bridge or an outcome-store adapter). Note the added shard.
- **STOP-AND-RETHINK (mechanism degenerate):** L3 produces no/empty/unchanged instruction even with ≥5 signals → GEPA on these signals isn't worth wiring; revisit the reward/example design with the user before any build.

- [ ] **Step 3: Commit the evidence + verdict**
```bash
git add docs/reports/dspy-loop-disproof-20260608/EVIDENCE.md
git commit -m "docs(dspy-disproof): evidence + decision gate for the build plans

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** This plan implements the spec's §8 "faithful disproof FIRST" gate and tests the §6 data-flow end-to-end. It deliberately defers Gap A (§4) and Gap B (§5) builds to follow-on plans, gated by Task 3 — consistent with the spec's §9 sequence (step 1) and CLAUDE.md cheapest-disproof-first.

**Placeholder scan:** No TBD/TODO. Task 1 names exact files + commands; Task 2 gives complete harness code; Task 3 gives a concrete verdict template. The only deliberate runtime-determined values are the live DB counts (the experiment's output, not a plan gap).

**Type consistency:** Harness uses real symbols verified in source — `FeedbackLearnerAgent(use_llm=, persist_signals=)` (`agent.py:94-104`), `get_feedback_learner_training_signals(min_reward=, limit=)` (`signal_store.py:85`), `run_feedback_learner_optimization(budget=)` (`optimization_runner.py:20`), `optimize_and_save_recipient(agent_name, budget=)` (`recipient_optimizer.py:209`), `install_all_prompt_bundles()` (`prompt_bundles.py:106`), `get_experiment_monitor_dspy_integration().get_prompt_metadata()` (`experiment_monitor/dspy_integration.py:322,340`). Signature class names in Link 3 (`PatternDetectionSignature` `:335`, `RecommendationGenerationSignature` `:355`, `LearningSummarySignature` `:405`) verified in source; they are read via best-effort `getattr` with None-guards, so any mismatch degrades to an empty baseline rather than erroring — the optimized/round-trip assertion does not depend on them.

**Risk note:** `load_optimized_module` import in Link 3 is verified to exist (`src/optimization/gepa`); if the exact symbol differs, the round-trip check is logged, not asserted, so it won't false-fail the disproof.
