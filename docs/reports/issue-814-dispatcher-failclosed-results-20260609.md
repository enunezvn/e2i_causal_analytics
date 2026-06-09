# Issue #814 — Dispatcher fail-closed: results

**Date:** 2026-06-09
**Branch:** `fix/tc-814-dispatcher-mock-failclosed` (off `main` `ed66d6b0`)
**Scope:** make `DispatcherNode` fail closed for a missing registry entry instead of fabricating analytics values, while preserving the legitimate test-only mock scaffold. No deploy, no migrations.

## Problem
`DispatcherNode._mock_agent_execution` returned canned narratives with fabricated values (`ATE=0.12`, `$2.5M ROI`, `2x higher response rate`) whenever a routed `agent_name` was absent from the registry. Reachable in a **degraded / partial-registry** misconfiguration (an agent that fails to instantiate is dropped at `factory.py`, then a query routed to it surfaced a fabricated `success=True` result). The mock is, however, a **legitimate test-only scaffold** — used by unit tests that exercise routing/timeout/fallback without instantiating real agents — so it must be *preserved*, not deleted.

## Reasoning (REASON-BEFORE-RULES)
The mock is documented test support ("used by unit tests that exercise routing without instantiating real agents"), so DELETE is wrong. The fix gates it behind an explicit, default-off flag: production fails closed; tests opt in. This closes the harm (#814) without removing requested functionality.

## What changed
1. **`DispatcherNode.__init__`** — new keyword-only `allow_mock: bool = False`.
2. **`_dispatch_agent`** — when `agent_name not in self.agents`: if `allow_mock` return the canned `_mock_agent_execution`; **else FAIL CLOSED** with `AgentResult(success=False, result=None, error="Agent '<name>' is not available …")` + a `logger.warning`. `_dispatch_fallback` routes through the same guard, so a missing *fallback* agent also fails closed.
3. **`_mock_agent_execution`** docstring marked TEST-ONLY; **`dispatch_to_agents`** (registry-less graph else-branch) documents + yields fail-closed.
4. **`factory.create_agent_registry`** — tracks enabled-but-dropped agents and logs a loud **PARTIAL-registry WARNING** naming them (option 3 — operator visibility of the precondition; still degraded-mode, does not raise). A `None`-returning constructor is now also treated as a drop.
5. **`cognitive.get_orchestrator`** docstring corrected (no longer claims a silent mock fallback).

## Tests (no mocking of dispatcher logic)
- NEW `test_dispatcher_failclosed.py` (5): default fails closed; partial registry does not fabricate (faithful degraded repro); `allow_mock=True` preserves the scaffold; registry-less module fn fails closed; registered agent unaffected.
- NEW `test_factory_partial_registry.py` (3): partial registry logs the warning + names the dropped agent; full registry emits none; `None` instance treated as a drop.
- Migrated `test_dispatcher.py` (dispatch-mechanics tests opt into `allow_mock=True`; registry-less module-fn test now asserts fail-closed) and `test_dispatcher_method_map.py` (the old "keeps mock path" test now asserts fail-closed).
- **52 touched-file tests green**; changed source mypy-clean (single-file); ruff clean.

## Codex (gpt-5.5) ralph-loop → fixed point
iter-1 REJECT (2 findings, both verified against source and fixed: a missed `test_dispatcher_method_map.py` test still asserting the old mock-success contract; a stale `cognitive.py` docstring) → iter-2 **ACCEPT-WITH-NOTES, zero findings**. Codex confirmed no `allow_mock=True` in `src`, fallback + registry-less node fail closed, and the RAG paths (`causal_rag.py`, `cognitive_rag_dspy.py`) do not use `DispatcherNode`.

## Honest notes
- Normal production was never reachable by the mock (the main path passes a populated registry); this hardens the **degraded** path and adds operator visibility for the precondition.
- The mock scaffold is retained, intentionally, for dispatch-mechanics unit tests — reachable only via `DispatcherNode(allow_mock=True)`.
