# Feedback-Learning Honest Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the /feedback-learning page with real learning artifacts by fixing the silently-ignored `auto_apply` flag, replaying the 30 golden questions through the real chat pipeline, and fixing the structurally-starved lookback window.

**Architecture:** One code PR (Tasks 1–6): thread `auto_apply` from `RunLearningRequest` → `FeedbackLearnerState` → `KnowledgeUpdaterNode` (fail-closed propose-only default), keep `update_effectiveness` honest when application is withheld, add a replay CLI script, widen the frontend quick-cycle window to 30 days with honest empty states. Then two prod operations (Tasks 7–8): a backfill learning cycle over the real June 14–July 7 signal window, and a golden-set replay (smoke first) followed by a second cycle. Spec: `docs/superpowers/specs/2026-07-15-feedback-learning-honest-demo-design.md`.

**Tech Stack:** Python 3.12 / FastAPI / LangGraph (backend), React + TypeScript + vitest (frontend), pytest, curl + psql via `docker exec supabase-db` (ops). Prod == dev == this host.

**Critical context for the implementing engineer:**

- The learner must NEVER consume synthetic rows. Do not touch the `is_synthetic=false` filter in `src/repositories/learning_signals_feedback.py`.
- `KnowledgeUpdaterNode.execute` currently applies every proposed update unconditionally (`src/agents/feedback_learner/nodes/knowledge_updater.py:50`) even though the API request carries `auto_apply=False`. Both the sync and async API paths funnel through the single `_execute_learning_cycle` state-construction site (`src/api/routes/feedback.py:1392`) — that is the only threading point. The mock fallback near line 1713 never runs the graph and needs no change.
- CI lint gate = `ruff check` AND `ruff format --check` on every touched file. Ruff is pinned 0.14.10 (`.venv/bin/ruff`).
- Do NOT run whole-tree mypy or pytest on this box — scope to touched files; CI is the arbiter.
- Verify `git branch --show-current` immediately before EVERY commit (parallel sessions can switch the checkout).
- Merge with `--merge` (never squash). Before any GitHub op: `git config --global http.https://github.com.proxy ""`.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/agents/feedback_learner/state.py` | Modify (~line 168) | Add `auto_apply` input field to `FeedbackLearnerState` |
| `src/agents/feedback_learner/nodes/knowledge_updater.py` | Modify (lines 48–53, 182–199) | Gate the apply loop on `auto_apply`; honest summary wording |
| `src/agents/feedback_learner/graph.py` | Modify (~line 316) | `update_effectiveness=None` when application was withheld |
| `src/api/routes/feedback.py` | Modify (~line 1402) | Thread `request.auto_apply` into `initial_state` |
| `scripts/replay_golden_set.py` | Create | CLI: play 30 golden questions through `POST /api/copilotkit/chat` |
| `frontend/src/api/feedback.ts` | Modify (~line 386) | Quick-cycle lookback 7d → 30d |
| `frontend/src/pages/FeedbackLearning.tsx` | Modify (~lines 676, 757) | Honest empty-state copy on Patterns / Updates tabs |
| `tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py` | Modify | New gate tests + `auto_apply: True` on existing apply-path tests |
| `tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py` | Modify | Withheld-apply → `update_effectiveness is None` |
| `tests/unit/test_api/test_routes/test_feedback.py` | Modify | Route threads `auto_apply` into graph state |
| `tests/integration/test_feedback_learner_knowledge_stores_realdb.py` | Modify (~line 162) | Add `auto_apply: True` to keep exercising the real apply path |
| `tests/unit/test_scripts/test_replay_golden_set.py` | Create | Payload shape, JWT sub decode, dry-run sends nothing, dataset sanity |
| `frontend/src/api/feedback.test.ts` | Create | 30-day window + sync mode + `auto_apply: false` |
| `frontend/src/pages/FeedbackLearning.test.tsx` | Modify | Empty-state copy renders |

All test directories above already run in CI (`tests/unit/test_agents/` via the sharded agent lanes, `tests/unit/test_api/` + `tests/unit/test_scripts/` via the unit lane, frontend via vitest).

---

## Task 0: Branch setup

- [ ] **Step 0.1: Create the feature branch from up-to-date main**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git config --global http.https://github.com.proxy ""
git checkout main && git pull --ff-only
git checkout -b feat/feedback-learning-honest-demo
git branch --show-current   # MUST print: feat/feedback-learning-honest-demo
```

---

## Task 1: Gate `KnowledgeUpdaterNode` on `auto_apply`

**Files:**
- Modify: `src/agents/feedback_learner/state.py` (~line 168, INPUT section)
- Modify: `src/agents/feedback_learner/nodes/knowledge_updater.py` (lines 48–53 and `_generate_summary`)
- Test: `tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py`

Fixtures used below (`base_state`, `mock_knowledge_stores`, `state_with_recommendations`) already exist in `tests/unit/test_agents/test_feedback_learner/conftest.py`. `mock_knowledge_stores` is a dict of `AsyncMock(update=AsyncMock())` keyed by `baseline`/`agent_config`/`prompt`/`threshold`; its `update` return value is truthy, so an applied update counts.

- [ ] **Step 1.1: Write the failing tests**

Add to the `TestKnowledgeUpdaterNode` class in `tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py` (after `test_updates_applied_to_stores`, ~line 220):

```python
    @pytest.mark.asyncio
    async def test_auto_apply_absent_withholds_apply(self, base_state, mock_knowledge_stores):
        """Fail-closed default: no auto_apply in state -> propose only, never touch stores."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New value",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_knowledge_stores)

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert len(result["proposed_updates"]) == 1
        assert result["applied_updates"] == []
        mock_knowledge_stores["baseline"].update.assert_not_called()
        assert "awaiting manual apply" in result["learning_summary"]

    @pytest.mark.asyncio
    async def test_auto_apply_false_withholds_apply(self, base_state, mock_knowledge_stores):
        """Explicit auto_apply=False (what the UI sends) -> propose only."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New value",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
            "auto_apply": False,
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_knowledge_stores)

        result = await node.execute(state)

        assert result["applied_updates"] == []
        mock_knowledge_stores["baseline"].update.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_apply_true_applies(self, base_state, mock_knowledge_stores):
        """Explicit opt-in preserves the original apply behavior."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New value",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
            "auto_apply": True,
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_knowledge_stores)

        result = await node.execute(state)

        assert len(result["applied_updates"]) == 1
        mock_knowledge_stores["baseline"].update.assert_called()
        assert "Applied 1 of 1" in result["learning_summary"]
```

- [ ] **Step 1.2: Update the three existing apply-path tests to opt in**

These tests exercise the apply path and must now do so explicitly. In `tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py`, add `"auto_apply": True,` to the `state = {...}` dict of each of:

- `test_updates_applied_to_stores` (~line 193) — asserts `len(applied_updates) >= 1`
- `test_store_update_failure_handled` (~line 223) — store raises; asserts 0 applied (with the gate off this would pass vacuously)
- `test_missing_store_type_handled` (~line 353) — same reasoning

Add the key right after `"status": "updating",` in each dict. Do NOT touch `test_execute_with_recommendations`, `test_execute_without_stores`, or `test_execute_empty_recommendations` — their assertions (`is not None`, `== 0` without stores, `== []`) hold under the propose-only default.

- [ ] **Step 1.3: Run the tests to verify the new ones fail**

```bash
.venv/bin/pytest tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py -n0 -q
```

Expected: `test_auto_apply_absent_withholds_apply` and `test_auto_apply_false_withholds_apply` FAIL (updates get applied; `update.assert_not_called()` raises). `test_auto_apply_true_applies` and the three edited tests PASS (current behavior applies unconditionally).

- [ ] **Step 1.4: Add the state field**

In `src/agents/feedback_learner/state.py`, inside `FeedbackLearnerState` (line 162), the INPUT section ends with `focus_agents: NotRequired[List[str]]`. Add directly below it:

```python
    # When absent/False (fail-closed default), the knowledge updater only
    # PROPOSES updates — a human applies them via the API's manual apply
    # endpoint. True = apply immediately (original pre-gate behavior).
    auto_apply: NotRequired[bool]
```

- [ ] **Step 1.5: Gate the apply loop**

In `src/agents/feedback_learner/nodes/knowledge_updater.py`, replace lines 48–53:

```python
            # Apply updates (with validation)
            applied = []
            for update in proposed_updates:
                success = await self._apply_update(update)
                if success:
                    applied.append(update["update_id"])
```

with:

```python
            # Apply only on explicit opt-in. Fail-closed: absent/False means
            # the cycle PROPOSES updates and a human applies them via the
            # API's manual apply endpoint — the request's auto_apply flag was
            # previously ignored here, silently applying every update.
            auto_apply = bool(state.get("auto_apply", False))
            applied: List[str] = []
            if auto_apply:
                for update in proposed_updates:
                    success = await self._apply_update(update)
                    if success:
                        applied.append(update["update_id"])
```

(`List` is already imported in this module.)

- [ ] **Step 1.6: Make the summary wording honest**

In the same file, `_generate_summary` builds a `parts` list whose last element is `f"Applied {len(applied)} of {len(proposed)} proposed updates."` (~line 198). With the gate off, "Applied 0 of N" reads as failure. Replace that single list element so the block becomes:

```python
        parts = [
            "Learning cycle complete.",
            f"Processed {feedback_count} feedback items.",
            f"Detected {pattern_count} patterns.",
            f"Generated {rec_count} recommendations.",
            (
                f"Applied {len(applied)} of {len(proposed)} proposed updates."
                if bool(state.get("auto_apply", False))
                else f"Proposed {len(proposed)} updates; awaiting manual apply (auto_apply=false)."
            ),
        ]
```

- [ ] **Step 1.7: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py -n0 -q
```

Expected: ALL PASS.

- [ ] **Step 1.8: Update the realdb integration test's opt-in**

`tests/integration/test_feedback_learner_knowledge_stores_realdb.py` (~line 162) executes the node directly and asserts all four updates applied + a positive `update_effectiveness`. Change:

```python
        out = await node.execute(
            {"status": "running", "learning_recommendations": recommendations}  # type: ignore[arg-type]
        )
```

to:

```python
        out = await node.execute(
            # auto_apply: this test verifies the REAL apply path end-to-end.
            {"status": "running", "learning_recommendations": recommendations, "auto_apply": True}  # type: ignore[arg-type]
        )
```

Do not run this integration test locally (needs the real DB lane); CI covers it.

- [ ] **Step 1.9: Commit**

```bash
git branch --show-current   # feat/feedback-learning-honest-demo
git add src/agents/feedback_learner/state.py src/agents/feedback_learner/nodes/knowledge_updater.py tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py tests/integration/test_feedback_learner_knowledge_stores_realdb.py
git commit -m "fix(feedback-learner): honor auto_apply — propose-only unless explicitly opted in

KnowledgeUpdaterNode applied every proposed update unconditionally even
though RunLearningRequest.auto_apply defaults to False and the UI sends
false. Gate the apply loop on the new FeedbackLearnerState.auto_apply
field (fail-closed) and report 'awaiting manual apply' instead of
'Applied 0 of N'.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Honest `update_effectiveness` when application is withheld

**Files:**
- Modify: `src/agents/feedback_learner/graph.py` (~line 316)
- Test: `tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py`

Context: `_finalize_training_signal` computes `update_effectiveness = applied/proposed` when `update_backend_wired and _proposed` (graph.py:316). With `auto_apply=False`, `applied` is structurally 0 even with a wired backend — 0/N would fabricate "completely ineffective" when application was merely withheld. Same honesty class as the existing F15 `None`.

- [ ] **Step 2.1: Write the failing test**

Add to `tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py` (a new class at the end; the file's `_make_state` and `_run` helpers are defined at the top):

```python
class TestUpdateEffectivenessWithheldApply:
    """auto_apply=False makes applied_updates structurally empty — 0/N would be
    a fabricated 'ineffective'; the honest value is None (unmeasurable)."""

    def test_none_when_apply_withheld_despite_wired_backend(self) -> None:
        state = _make_state()
        state["update_backend_wired"] = True
        state["proposed_updates"] = [
            {
                "update_id": "U1",
                "knowledge_type": "baseline",
                "key": "agent1",
                "old_value": None,
                "new_value": "v",
                "justification": "j",
                "effective_date": "2026-07-15T00:00:00+00:00",
            }
        ]
        state["applied_updates"] = []
        # auto_apply absent -> withheld (fail-closed default)
        result = _run(_finalize_training_signal(state))
        assert result["training_signal"].update_effectiveness is None

    def test_measured_when_auto_apply_true(self) -> None:
        state = _make_state()
        state["update_backend_wired"] = True
        state["auto_apply"] = True
        state["proposed_updates"] = [
            {
                "update_id": "U1",
                "knowledge_type": "baseline",
                "key": "agent1",
                "old_value": None,
                "new_value": "v",
                "justification": "j",
                "effective_date": "2026-07-15T00:00:00+00:00",
            }
        ]
        state["applied_updates"] = ["U1"]
        result = _run(_finalize_training_signal(state))
        assert result["training_signal"].update_effectiveness == 1.0
```

- [ ] **Step 2.2: Run to verify the first test fails**

```bash
.venv/bin/pytest tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py -n0 -q
```

Expected: `test_none_when_apply_withheld_despite_wired_backend` FAILS (gets `0.0`, expects `None`). `test_measured_when_auto_apply_true` PASSES already.

- [ ] **Step 2.3: Implement**

In `src/agents/feedback_learner/graph.py` (~line 315), change:

```python
    if state.get("update_backend_wired") and _proposed:
        update_effectiveness = len(applied_updates) / len(_proposed)
    else:
        update_effectiveness = None
```

to:

```python
    # ... AND application was actually attempted: with auto_apply=False the
    # updater withholds every apply, so applied/proposed would fabricate a
    # 0.0 "ineffective" when effectiveness is simply unmeasurable this cycle.
    if state.get("update_backend_wired") and _proposed and state.get("auto_apply"):
        update_effectiveness = len(applied_updates) / len(_proposed)
    else:
        update_effectiveness = None
```

(Keep the existing F15 comment block above it intact; append the new comment lines to it.)

- [ ] **Step 2.4: Run tests to verify pass**

```bash
.venv/bin/pytest tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py -n0 -q
```

Expected: ALL PASS.

- [ ] **Step 2.5: Commit**

```bash
git branch --show-current
git add src/agents/feedback_learner/graph.py tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py
git commit -m "fix(feedback-learner): update_effectiveness=None when apply withheld

With auto_apply=false applied_updates is structurally empty, so
applied/proposed would report a fabricated 0.0 'ineffective'. Extend the
F15 honesty gate: effectiveness is only measurable when application was
actually attempted.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Thread `auto_apply` through the API route

**Files:**
- Modify: `src/api/routes/feedback.py` (~line 1402)
- Test: `tests/unit/test_api/test_routes/test_feedback.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/unit/test_api/test_routes/test_feedback.py` (the file already imports `patch` and `AsyncMock`; mirror the mocking pattern of `test_execute_learning_cycle_applied_updates_consistent_with_count` at ~line 270):

```python
@pytest.mark.asyncio
async def test_execute_learning_cycle_threads_auto_apply_into_state():
    """The request's auto_apply flag must reach the graph's initial state.

    It was silently dropped: RunLearningRequest.auto_apply existed but
    _execute_learning_cycle never put it in initial_state, so
    KnowledgeUpdaterNode applied every update regardless of the request.
    """
    from src.api.routes.feedback import RunLearningRequest, _execute_learning_cycle

    result_state = {
        "status": "completed",
        "detected_patterns": [],
        "learning_recommendations": [],
        "priority_improvements": [],
        "proposed_updates": [],
        "applied_updates": [],
        "learning_summary": "ok",
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "errors": [],
        "warnings": [],
    }

    for flag in (False, True):
        fake_graph = AsyncMock()
        fake_graph.ainvoke.return_value = result_state
        request = RunLearningRequest(auto_apply=flag)
        with patch(
            "src.agents.feedback_learner.graph.build_feedback_learner_graph",
            return_value=fake_graph,
        ):
            with patch(
                "src.agents.feedback_learner.agent.build_production_feedback_stores",
                new=AsyncMock(return_value=(None, None, None)),
            ):
                await _execute_learning_cycle(request)
        initial_state = fake_graph.ainvoke.call_args.args[0]
        assert initial_state["auto_apply"] is flag
```

- [ ] **Step 3.2: Run to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_api/test_routes/test_feedback.py::test_execute_learning_cycle_threads_auto_apply_into_state -n0 -q
```

Expected: FAIL with `KeyError: 'auto_apply'`.

- [ ] **Step 3.3: Implement**

In `src/api/routes/feedback.py`, the `initial_state` dict in `_execute_learning_cycle` (lines 1392–1403) currently ends with:

```python
                "focus_agents": request.focus_agents or [],
                "status": "pending",
                "errors": [],
                "warnings": [],
```

Add one line after `"focus_agents": ...`:

```python
                "focus_agents": request.focus_agents or [],
                # Previously dropped here — the updater then applied every
                # update regardless of the request (see KnowledgeUpdaterNode).
                "auto_apply": request.auto_apply,
                "status": "pending",
                "errors": [],
                "warnings": [],
```

- [ ] **Step 3.4: Run the route tests**

```bash
.venv/bin/pytest tests/unit/test_api/test_routes/test_feedback.py tests/unit/test_api/test_routes/test_feedback_sync_persists.py tests/unit/test_api/test_routes/test_feedback_persistence.py tests/unit/test_api/test_routes/test_feedback_positive_ratio.py -n0 -q
```

Expected: ALL PASS (the #837 test mocks the graph, so threading an extra key does not affect it).

- [ ] **Step 3.5: Commit**

```bash
git branch --show-current
git add src/api/routes/feedback.py tests/unit/test_api/test_routes/test_feedback.py
git commit -m "fix(api): thread RunLearningRequest.auto_apply into learner graph state

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Frontend — 30-day quick-cycle window + honest empty states

**Files:**
- Modify: `frontend/src/api/feedback.ts` (~line 386)
- Modify: `frontend/src/pages/FeedbackLearning.tsx` (~lines 676, 757)
- Create: `frontend/src/api/feedback.test.ts`
- Modify: `frontend/src/pages/FeedbackLearning.test.tsx`

- [ ] **Step 4.1: Write the failing API test**

Create `frontend/src/api/feedback.test.ts`:

```typescript
/**
 * quickLearningCycle — lookback window regression tests.
 *
 * The 7-day window structurally missed all real signals older than a week
 * (signal accrual is bursty); the window is now 30 days. Same defect class
 * as the Fabhalta gap-floor bug (#1237): "no results" ≠ "correctly no results".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/lib/api-client', () => ({
  get: vi.fn(),
  post: vi.fn().mockResolvedValue({}),
}));

import { post } from '@/lib/api-client';
import { quickLearningCycle } from './feedback';

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

describe('quickLearningCycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('requests a 30-day lookback, sync mode, auto_apply=false', async () => {
    const before = Date.now();
    await quickLearningCycle();
    const after = Date.now();

    expect(post).toHaveBeenCalledTimes(1);
    const [path, body, opts] = (post as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(path).toBe('/feedback/learn');
    expect(body.auto_apply).toBe(false);
    expect(body.min_feedback_count).toBe(5);
    expect(body.pattern_threshold).toBe(0.1);
    expect(opts.params.async_mode).toBe(false);

    const start = new Date(body.time_range_start).getTime();
    expect(start).toBeGreaterThanOrEqual(before - THIRTY_DAYS_MS - 2000);
    expect(start).toBeLessThanOrEqual(after - THIRTY_DAYS_MS + 2000);
  });

  it('passes focus agents through', async () => {
    await quickLearningCycle(['causal_impact']);
    const [, body] = (post as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(body.focus_agents).toEqual(['causal_impact']);
  });
});
```

- [ ] **Step 4.2: Run to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/api/feedback.test.ts
```

Expected: FAIL on the window assertion (start is ~7 days ago, not ~30).

- [ ] **Step 4.3: Widen the window**

In `frontend/src/api/feedback.ts`, replace the body of `quickLearningCycle` (~lines 386–403):

```typescript
export async function quickLearningCycle(
  focusAgents?: string[]
): Promise<LearningResponse> {
  // 30-day window: real feedback (chat thumbs + cognitive reward signals)
  // accrues per active chat day and is bursty — a short lookback (backend
  // default 24h; previously 7d here) can structurally miss every signal
  // and report "No feedback items collected" forever.
  const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
  return runLearningCycle(
    {
      time_range_start: thirtyDaysAgo,
      focus_agents: focusAgents,
      min_feedback_count: 5,
      pattern_threshold: 0.1,
      auto_apply: false,
    },
    false // Run synchronously for quick analysis
  );
}
```

Also update the function's JSDoc line "Simplified interface for processing feedback from the last 7 days." → "Simplified interface for processing feedback from the last 30 days."

- [ ] **Step 4.4: Run to verify pass**

```bash
npx vitest run src/api/feedback.test.ts
```

Expected: PASS.

- [ ] **Step 4.5: Write the failing empty-state page test**

Add to `frontend/src/pages/FeedbackLearning.test.tsx`, inside the existing `describe('FeedbackLearning — F-002 empty state', ...)` block (reuse the hook mocks exactly as the existing test in that block does — copy its `beforeEach`-style mock setup lines):

```typescript
  it('explains WHY the patterns/updates tabs can be empty (window-bounded cycles)', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 0 },
      refetch: vi.fn().mockResolvedValue({}),
    });
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });
    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false, data: undefined, error: null });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByRole('tab', { name: /^Patterns$/i }));
    expect(screen.getByText(/bounded lookback window/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /Knowledge Updates/i }));
    expect(screen.getByText(/wait here for manual review/i)).toBeInTheDocument();
  });
```

Note: the page uses Radix Tabs (`TabsTrigger value="patterns"` at FeedbackLearning.tsx:485). If `fireEvent.click` does not switch the tab in jsdom, use `fireEvent.mouseDown(...)` followed by `fireEvent.click(...)` on the same trigger — Radix activates triggers on pointer-down.

- [ ] **Step 4.6: Run to verify it fails**

```bash
npx vitest run src/pages/FeedbackLearning.test.tsx
```

Expected: new test FAILS (`bounded lookback window` not found); existing tests PASS.

- [ ] **Step 4.7: Implement the empty-state copy**

In `frontend/src/pages/FeedbackLearning.tsx`, the Patterns tab empty state (~line 676) is:

```tsx
              ) : (
                <div className="flex items-center justify-center gap-2 py-8 text-[var(--color-muted-foreground)]">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  No patterns detected
                </div>
              )}
```

Replace it with:

```tsx
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-8 text-center text-[var(--color-muted-foreground)]">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    No patterns detected in the analyzed window
                  </div>
                  <p className="max-w-md text-xs">
                    Feedback signals accrue per chat turn, and each learning cycle scans a
                    bounded lookback window — an empty tab can also mean the last cycle&apos;s
                    window contained no feedback (see cycle warnings above).
                  </p>
                </div>
              )}
```

The Knowledge Updates tab empty state (~line 757) is the identical block with the text `No knowledge updates available`. Replace it with:

```tsx
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-8 text-center text-[var(--color-muted-foreground)]">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    No knowledge updates proposed
                  </div>
                  <p className="max-w-md text-xs">
                    Updates are generated from patterns detected by a learning cycle and
                    wait here for manual review and apply.
                  </p>
                </div>
              )}
```

- [ ] **Step 4.8: Run the page tests to verify pass**

```bash
npx vitest run src/pages/FeedbackLearning.test.tsx src/api/feedback.test.ts
```

Expected: ALL PASS.

- [ ] **Step 4.9: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git branch --show-current
git add frontend/src/api/feedback.ts frontend/src/api/feedback.test.ts frontend/src/pages/FeedbackLearning.tsx frontend/src/pages/FeedbackLearning.test.tsx
git commit -m "fix(feedback-learning): 30d quick-cycle window + honest empty-state copy

The hardcoded 7d lookback structurally missed all signals older than a
week (accrual is bursty — same defect class as the #1237 gap floor).
Empty Patterns/Updates tabs now explain the window-bounded mechanics
instead of a bare no-data line.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Golden-set replay script

**Files:**
- Create: `scripts/replay_golden_set.py`
- Test: `tests/unit/test_scripts/test_replay_golden_set.py`

Context for the engineer:
- Endpoint: `POST {api_base}/copilotkit/chat` (`src/api/routes/copilotkit.py:4197`, auth `require_viewer`). The server derives identity from the JWT; a body `user_id` that disagrees with the token's `sub` is rejected 403 (`_resolve_chat_identity`) — so the script decodes `sub` from the token and sends exactly that.
- Auth-mint pattern mirrors `scripts/sync_goldstd_serving.py:_admin_token` (GoTrue password grant with the anon key).
- Dataset: `get_default_evaluation_dataset()` from `src/rag/evaluation.py` — 30 `EvaluationSample` pydantic models; only `.query` is used. This script must NOT run RAGAS metrics (manual-only, incident #504).
- `load_dotenv()` at module top (repo convention for CLI scripts; `.env` carries `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `E2I_ADMIN_PASSWORD`).

- [ ] **Step 5.1: Write the failing tests**

Create `tests/unit/test_scripts/test_replay_golden_set.py`:

```python
"""Unit tests for scripts/replay_golden_set.py.

Network-free: payload construction, JWT sub decoding, dry-run discipline,
and golden-dataset sanity. The live path is exercised manually (smoke run
--limit 2 against prod) per the plan's Task 8.
"""

from __future__ import annotations

import base64
import json

from scripts.replay_golden_set import build_chat_payload, jwt_sub, main


def _fake_jwt(sub: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).decode().rstrip("=")
    return f"{header}.{payload}."


def test_jwt_sub_decodes_unpadded_payload():
    assert jwt_sub(_fake_jwt("user-123")) == "user-123"


def test_build_chat_payload_shape():
    p = build_chat_payload("What drives Kisqali TRx?", "u1", "goldset-replay-20260715-q01")
    assert p == {
        "query": "What drives Kisqali TRx?",
        "user_id": "u1",
        "session_id": "goldset-replay-20260715-q01",
    }


def test_dry_run_sends_nothing(monkeypatch, capsys):
    import scripts.replay_golden_set as mod

    def _boom(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("network I/O attempted during --dry-run")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _boom)
    rc = main(["--dry-run", "--limit", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("goldset-replay-") >= 3
    assert "3 questions" in out


def test_golden_dataset_has_30_nonempty_queries():
    """Guards accidental dataset edits — the replay banks on 30 real questions."""
    from src.rag.evaluation import get_default_evaluation_dataset

    samples = get_default_evaluation_dataset()
    assert len(samples) == 30
    assert all(s.query.strip() for s in samples)
```

- [ ] **Step 5.2: Run to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_scripts/test_replay_golden_set.py -n0 -q
```

Expected: FAIL at import (`ModuleNotFoundError: No module named 'scripts.replay_golden_set'`).

- [ ] **Step 5.3: Write the script**

Create `scripts/replay_golden_set.py`:

> **Superseded (2026-08-05, #1485) — do not copy this listing.**
> The sender contract below is the original 2-tuple `(ok, detail)` that this
> July plan implemented, preserved here as the record of what was built.
> #1485 changed `send_chat` / `send_cognitive` to return `(ok, detail, body)`
> so the response body — the generated answer and the really-retrieved
> contexts — reaches the real-pipeline RAGAS judge; discarding it was the
> defect. Both the `def send_chat(...) -> Tuple[bool, str]` signature and the
> `ok, detail = send_chat(...)` unpack further down this listing are therefore
> out of date. `scripts/replay_golden_set.py` is canonical.

```python
#!/usr/bin/env python3
"""Replay the RAGAS golden QA set through the real chat pipeline.

Purpose (spec: docs/superpowers/specs/2026-07-15-feedback-learning-honest-demo-design.md)
------------------------------------------------------------------------------------------
Sends each of the 30 curated golden questions
(``src.rag.evaluation.get_default_evaluation_dataset``) through the REAL
non-streaming chat endpoint (``POST /api/copilotkit/chat`` -> ``run_chatbot``
-> cognitive pipeline). Every completed turn makes the cognitive workflow
write ~3 genuine reward signals (agent / investigator / summarizer) to
``learning_signals`` with ``is_synthetic=false`` — the real system grading
its real answers. A feedback-learning cycle over the replay window then has
honest material for the /feedback-learning page.

This does NOT run RAGAS metric evaluation (manual-only per incident #504);
it only reuses the dataset's questions.

Provenance: session ids are ``goldset-replay-<YYYYMMDD>-q<NN>`` so replay
turns stay identifiable in chat history. Verify signals landed after a run:

    docker exec supabase-db psql -U postgres -d postgres -c \\
      "SELECT count(*) FROM learning_signals \\
       WHERE created_at > now() - interval '2 hours' AND is_synthetic = false;"

Usage:
    .venv/bin/python scripts/replay_golden_set.py --dry-run
    .venv/bin/python scripts/replay_golden_set.py --limit 2   # smoke, then verify
    .venv/bin/python scripts/replay_golden_set.py             # full 30
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replay_golden_set")


def mint_token() -> str:
    """Mint a JWT via the GoTrue password grant (mirrors sync_goldstd_serving)."""
    su = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    email = os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local")
    pw = os.environ["E2I_ADMIN_PASSWORD"]
    body = json.dumps({"email": email, "password": pw}).encode()
    req = urllib.request.Request(
        f"{su}/auth/v1/token?grant_type=password",
        data=body,
        headers={"apikey": anon, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())["access_token"]


def jwt_sub(token: str) -> str:
    """Decode the JWT payload's ``sub`` claim (the authoritative user id).

    The chat endpoint rejects (403) any body ``user_id`` that disagrees with
    the token identity (``_resolve_chat_identity``), so we send exactly the
    token's subject.
    """
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    return str(payload["sub"])


def build_chat_payload(query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Body for POST /api/copilotkit/chat (``ChatRequest``)."""
    return {"query": query, "user_id": user_id, "session_id": session_id}


def send_chat(
    api_base: str, token: str, payload: Dict[str, Any], timeout: int
) -> Tuple[bool, str]:
    """POST one chat turn; fail-soft — returns (ok, detail), never raises."""
    req = urllib.request.Request(
        f"{api_base}/copilotkit/chat",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            ok = bool(body.get("success")) and bool(body.get("response"))
            return ok, f"agent={body.get('agent_name')} len={len(body.get('response') or '')}"
    except HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode()[:200]}"
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay golden QA questions through the real chat pipeline."
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("E2I_API_BASE", "https://eznomics.site/api"),
        help="API base URL (default: E2I_API_BASE or https://eznomics.site/api)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Send only the first N questions (smoke run)"
    )
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds between turns")
    parser.add_argument("--timeout", type=int, default=300, help="Per-turn HTTP timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions; send nothing")
    args = parser.parse_args(argv)

    from src.rag.evaluation import get_default_evaluation_dataset

    samples = get_default_evaluation_dataset()
    if args.limit is not None:
        samples = samples[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if args.dry_run:
        for i, sample in enumerate(samples, 1):
            print(f"[dry-run] goldset-replay-{stamp}-q{i:02d}: {sample.query}")
        print(f"[dry-run] {len(samples)} questions -> {args.api_base}/copilotkit/chat")
        return 0

    token = mint_token()
    user_id = jwt_sub(token)
    sent, failed = 0, 0
    for i, sample in enumerate(samples, 1):
        session_id = f"goldset-replay-{stamp}-q{i:02d}"
        ok, detail = send_chat(
            args.api_base, token, build_chat_payload(sample.query, user_id, session_id), args.timeout
        )
        if ok:
            sent += 1
            logger.info("[%d/%d] %s OK %s", i, len(samples), session_id, detail)
        else:
            failed += 1
            logger.warning("[%d/%d] %s FAILED %s", i, len(samples), session_id, detail)
        if i < len(samples):
            time.sleep(args.sleep)

    print(f"replay complete: {sent} ok, {failed} failed of {len(samples)}")
    print(
        "verify: docker exec supabase-db psql -U postgres -d postgres -c "
        "\"SELECT count(*) FROM learning_signals WHERE created_at > now() - "
        "interval '2 hours' AND is_synthetic = false;\""
    )
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.4: Run the tests to verify pass**

```bash
.venv/bin/pytest tests/unit/test_scripts/test_replay_golden_set.py -n0 -q
```

Expected: 4 PASS. (`test_dry_run_sends_nothing` also proves `--dry-run` needs no env vars and no network.)

- [ ] **Step 5.5: Local dry-run sanity check**

```bash
.venv/bin/python scripts/replay_golden_set.py --dry-run | tail -5
```

Expected: last lines list `goldset-replay-<today>-q28..q30` questions and `[dry-run] 30 questions -> https://eznomics.site/api/copilotkit/chat`.

- [ ] **Step 5.6: Commit**

```bash
git branch --show-current
git add scripts/replay_golden_set.py tests/unit/test_scripts/test_replay_golden_set.py
git commit -m "feat(scripts): golden-set replay through the real chat pipeline

Plays the 30 curated golden questions through POST /api/copilotkit/chat
(run_chatbot cognitive path) so SignalCollector writes genuine reward
signals for feedback-learning cycles. Provenance session ids
goldset-replay-<date>-qNN; --dry-run/--limit for cheap disproof first.
Does NOT run RAGAS metrics (manual-only, #504).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Quality gates and PR

- [ ] **Step 6.1: Ruff (check AND format) on every touched Python file**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
TOUCHED="src/agents/feedback_learner/state.py src/agents/feedback_learner/nodes/knowledge_updater.py src/agents/feedback_learner/graph.py src/api/routes/feedback.py scripts/replay_golden_set.py tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py tests/unit/test_api/test_routes/test_feedback.py tests/integration/test_feedback_learner_knowledge_stores_realdb.py tests/unit/test_scripts/test_replay_golden_set.py"
.venv/bin/ruff check $TOUCHED
.venv/bin/ruff format --check $TOUCHED
```

Expected: both clean. If `format --check` flags files, run `.venv/bin/ruff format <file>` and re-verify tests still pass.

- [ ] **Step 6.2: Scoped mypy (touched src files only — never whole-tree on this box)**

```bash
.venv/bin/mypy --config-file pyproject.toml src/agents/feedback_learner/nodes/knowledge_updater.py src/agents/feedback_learner/graph.py src/api/routes/feedback.py scripts/replay_golden_set.py
```

Expected: no NEW errors in the touched files (pre-existing errors elsewhere are CI's arbiter).

- [ ] **Step 6.3: Full targeted backend test sweep**

```bash
.venv/bin/pytest tests/unit/test_agents/test_feedback_learner/test_knowledge_updater.py tests/unit/test_agents/test_feedback_learner/test_graph_finalize_training_signal.py tests/unit/test_api/test_routes/test_feedback.py tests/unit/test_api/test_routes/test_feedback_sync_persists.py tests/unit/test_scripts/test_replay_golden_set.py -n0 -q
```

Expected: ALL PASS.

- [ ] **Step 6.4: Frontend test + lint sweep**

```bash
cd frontend
npx vitest run src/api/feedback.test.ts src/pages/FeedbackLearning.test.tsx
npx eslint src/api/feedback.ts src/api/feedback.test.ts src/pages/FeedbackLearning.tsx src/pages/FeedbackLearning.test.tsx
cd ..
```

Expected: tests pass, eslint clean.

- [ ] **Step 6.5: Push and open the PR**

```bash
git config --global http.https://github.com.proxy ""
git branch --show-current   # feat/feedback-learning-honest-demo
git push -u origin feat/feedback-learning-honest-demo
gh pr create \
  --title "fix(feedback-learning): honor auto_apply + golden-set replay + 30d window" \
  --body "$(cat <<'EOF'
## Summary
Spec: docs/superpowers/specs/2026-07-15-feedback-learning-honest-demo-design.md

- **auto_apply gate (latent bug):** `RunLearningRequest.auto_apply` (default false, UI sends false) was never threaded into graph state — `KnowledgeUpdaterNode` applied every proposed update to real knowledge stores unconditionally. Now fail-closed propose-only; a human applies via the page's Apply button. `update_effectiveness` honestly `None` when application was withheld (F15 extension).
- **Golden-set replay script:** `scripts/replay_golden_set.py` plays the 30 golden questions through the real `POST /api/copilotkit/chat` pipeline so `SignalCollector` writes genuine reward signals (`goldset-replay-<date>-qNN` provenance). No RAGAS metrics (manual-only, #504).
- **Frontend:** quick-cycle lookback 7d → 30d (7d structurally missed all real signals, June 14–July 7); Patterns/Updates empty states explain window-bounded mechanics.

Synthetic-exclusion guardrail in `LearningSignalsFeedbackStore` untouched. Copilot signal wiring is follow-up #1240.

## Test plan
- [ ] Unit: updater gate (absent/false withholds, true applies), finalize effectiveness None-when-withheld, route threads flag into initial_state
- [ ] Unit: replay payload/JWT-sub/dry-run/dataset sanity
- [ ] Frontend: 30d window body assertions; empty-state copy renders
- [ ] Post-merge ops (plan Tasks 7–8): backfill cycle over 2026-06-14→07-08, replay smoke `--limit 2` → full 30 → second cycle → live-verify page

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6.6: Wait for CI, then merge (never squash)**

```bash
gh pr checks --watch
```

All required checks green, then:

```bash
gh pr merge --merge
```

Known path-filter caveat: this PR touches `src/agents/...`, so the required "Tier 1-5 agent harness" WILL trigger (unlike the api/frontend-only PRs that needed sanctioned `--admin`). If an UNSTABLE merge-state appears, plain `--merge` retry; if a required check is path-skipped and blocking, rebase onto main first — `--admin` only with the user's explicit say-so.

---

## Task 7 (operation): Backfill learning cycle over real history

Run AFTER the PR is merged and deployed. Deploy auto-triggers on push to main; confirm convergence first.

- [ ] **Step 7.1: Confirm the deploy converged on the merge SHA**

```bash
git config --global http.https://github.com.proxy ""
gh run list --workflow=deploy.yml --limit 3
git log origin/main -1 --format=%H
```

Expected: newest deploy run `completed/success` for the merge commit SHA. If an older-SHA deploy is still running, let the newer one converge — never rerun an older deploy.

- [ ] **Step 7.2: Mint an operator token**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
set -a; source .env; set +a
TOKEN=$(curl -s -X POST "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d "{\"email\":\"${E2I_ADMIN_EMAIL:-admin@e2i.local}\",\"password\":\"$E2I_ADMIN_PASSWORD\"}" | jq -r .access_token)
[ -n "$TOKEN" ] && [ "$TOKEN" != "null" ] && echo "token OK" || echo "TOKEN MINT FAILED - STOP"
```

- [ ] **Step 7.3: Run the backfill cycle (sync) over the real signal window**

```bash
curl -s --max-time 600 -X POST "https://eznomics.site/api/feedback/learn?async_mode=false" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"time_range_start":"2026-06-14T00:00:00Z","time_range_end":"2026-07-08T00:00:00Z","min_feedback_count":5,"pattern_threshold":0.1,"auto_apply":false}' | jq .
```

Expected: `status` completed; feedback items processed ≈ 126 (123 graded `learning_signals` + thumbs); patterns likely FEW (real rewards avg 0.80–0.92 — "healthy system" is the true story); `updates_applied` 0 with any proposed updates in `proposed` status. A 403 here means the admin user lacks the operator role — STOP and report; do not work around auth.

- [ ] **Step 7.4: Verify persistence in the DB**

```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT * FROM feedback_learning_batches ORDER BY created_at DESC LIMIT 1;"
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT count(*) FROM feedback_patterns;"
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT count(*), count(*) FILTER (WHERE status = 'applied') FROM feedback_knowledge_updates;"
```

Expected: newest batch matches the curl response (completed, real item count); zero updates in `applied` status from this cycle. (If a column in the first query differs, check `src/api/repositories/feedback_repository.py` for the actual schema rather than guessing.)

- [ ] **Step 7.5: Verify the page**

Load https://eznomics.site/feedback-learning — Overview shows the completed batch with real counts; Patterns/Updates tabs show whatever the cycle honestly found. Also confirm the deployed bundle carries the new empty-state copy (code-split page — grep its lazy chunk, not index):

```bash
INDEX=$(curl -s https://eznomics.site/ | grep -o 'assets/index-[^"]*\.js' | head -1)
LAZY=$(curl -s "https://eznomics.site/$INDEX" | grep -o 'assets/FeedbackLearning-[^"]*\.js' | head -1)
echo "lazy chunk: $LAZY"
curl -s "https://eznomics.site/$LAZY" | grep -c "bounded lookback window"
```

Expected: count ≥ 1.

---

## Task 8 (operation): Golden-set replay + second cycle

- [ ] **Step 8.1: Smoke run — cheapest disproof BEFORE spending 30 turns**

The core assumption: a `POST /api/copilotkit/chat` turn on prod causes `SignalCollector` to write graded `learning_signals` rows (it did for the 123 historical signals, but signal-writing stopped July 7 — confirm the path still works TODAY before the full run).

```bash
cd /home/enunez/Projects/e2i_causal_analytics
.venv/bin/python scripts/replay_golden_set.py --limit 2
```

Expected: `replay complete: 2 ok, 0 failed of 2` (each turn may take 30–120 s).

- [ ] **Step 8.2: Verify signals landed (STOP if zero)**

```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT created_at, signal_details->>'type' AS type, signal_details->>'reward' AS reward, \
          signal_details->'metadata'->>'conversation_id' AS conv \
   FROM learning_signals \
   WHERE created_at > now() - interval '30 minutes' AND is_synthetic = false \
   ORDER BY created_at DESC LIMIT 12;"
```

Expected: ~6 fresh rows (~3 per turn: agent/investigator/summarizer) with numeric rewards. Note whether `conv` carries the `goldset-replay-` prefix (session id propagates to the cognitive `conversation_id`) — if it does, record it; if not, provenance falls back to the time window and chat-history session ids. **If ZERO rows: STOP.** The chat→signal path is broken on prod (this is exactly why signals stopped accruing after July 7 on some path) — report the finding; do not run the remaining 28 questions.

- [ ] **Step 8.3: Full replay (~30–60 min wall clock)**

```bash
.venv/bin/python scripts/replay_golden_set.py 2>&1 | tee /tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/671d5ede-2293-44f5-959d-a679f1a8b8d6/scratchpad/goldset_replay_$(date +%Y%m%d).log
```

Expected: `replay complete: N ok, M failed of 30` with N ≥ 25 (fail-soft tolerates stragglers). Then re-run the Step 8.2 query with `interval '2 hours'` — expect ~3×N graded rows.

- [ ] **Step 8.4: Second learning cycle over the replay window**

Re-mint the token if more than ~1 h has passed (Step 7.2), then:

```bash
TODAY=$(date -u +%Y-%m-%d)
curl -s --max-time 600 -X POST "https://eznomics.site/api/feedback/learn?async_mode=false" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"time_range_start\":\"${TODAY}T00:00:00Z\",\"min_feedback_count\":5,\"pattern_threshold\":0.1,\"auto_apply\":false}" | jq .
```

Expected: completed; items ≈ 3×N; harder golden questions should yield genuinely low rewards → real patterns and proposed updates (all `proposed`, zero auto-applied).

- [ ] **Step 8.5: Live-verify the demo flow end-to-end**

1. https://eznomics.site/feedback-learning shows BOTH batches (backfill + replay) with real counts.
2. Patterns tab lists the replay cycle's patterns (or the honest empty state with its explanation).
3. Updates tab lists proposed updates with working Apply/Rollback buttons (do not click Apply — that is the user's demo moment; applying is a real write to agent knowledge stores).
4. DB cross-check: `feedback_knowledge_updates` rows from today are all `status='proposed'`.
5. Report the final state to the user: batch counts, pattern counts, update counts, and the provenance finding from Step 8.2.

---

## Out of scope (do not touch)

- `LearningSignalsFeedbackStore`'s `is_synthetic=false` / `dspy_signal` filters — the honesty guardrail.
- Copilot/frontend chat signal wiring — issue #1240.
- RAGAS metric evaluation — manual-only (incident #504).
- Any synthetic backfill of signals, patterns, or updates.
