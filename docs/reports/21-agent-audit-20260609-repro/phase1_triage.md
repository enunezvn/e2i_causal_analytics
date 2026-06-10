# Phase 1 Triage — candidate findings (UNVERIFIED — pending Phase 2 source verification)

Screen ran against `feat/dspy-loop-real-results`. Branch is 7 behind main, 0 ahead.
Only `orchestrator/dispatcher.py` + `tool_composer/*` differ from main → re-verify those vs `main`.
Other 19 agents: HEAD == main. #814/#816 fail-closed fix is UNMERGED (live mock on main too).

## Clean
- **scope_definer** — CLEAR. Computed spec agent (no LLM); hardcoded thresholds are literature-cited deployment gates, not fake analytics. Refute held.

## USER-FACING / HARMFUL-NOW (top priority)
- **health_score** — SILENT-MOCK. Composer outputs invariant 100.0/A because all 4 component inputs are mock-pinned to 1.0 (`component_health.py:83 _create_mock_status`; None-store→1.0 at model/pipeline/agent_health.py). REST `/components,/models,/pipelines,/agents` serve hardcoded mock tagged `data_provenance='measured'` (`health_score.py:480,521,898+,940+`).
- **gap_analyzer** — HARMFUL-NOW. Real path raises `KeyError('region')` (`benchmark_store.py:187-229` vs `gap_detector.py:558-562`), swallowed (`gap_detector.py:222`); formatter upgrades `failed`→`completed` (`formatter.py:98`) and emits "No significant performance gaps" (`formatter.py:260-264`) as HTTP 200 (`gaps.py:664-666`).

## SILENT-MOCK / fabricated (prod-reachable)
- **observability_connector** — SILENT-MOCK. `_get_span_repository()` TypeErrors on bad `client=` kwarg, swallowed; `aggregate_metrics` always returns `_get_mock_spans` fabricated metrics, unmarked. Not yet consumed → HARMFUL-on-exposure.
- **model_deployer** — FAIL-SIMULATED. `register_model/promote_stage` emit `success=True`+`version=1` constants when MLflow fails (default `model_uri='simulated://model'` guarantees failure); fabricated rollback uuid+URL; `_store_to_database` short-circuits (`agent.py:375`) → no DB rows despite `status='completed'`.
- **model_selector** — DEGENERATE. `MLDataLoader.execute_query` does not exist → AttributeError silenced (type:ignore+bare except) → 40% of selection score is a frozen constant history table.
- **experiment_designer** — UNMARKED-SILENT-MOCK. `context_loader.MockKnowledgeStore` unmarked, non-flag-gated, reachable every prod run; seeds design LLM prompt with fabricated org context.
- **drift_monitor** — HARMFUL. `structural_drift` node orphaned both boundaries (`agent.py:54-97` omits DAG fields; `alert_aggregator.py:142-165` drops structural results) → critical DAG drift surfaces as `drift_score=0.0` / 'NO DRIFT'.
- **orchestrator** — HARMFUL (live on main). Mock reached unconditionally when routed agent absent from registry (no `allow_mock` guard; `_mock_agent_execution`). Partial registry reachable via `create_agent_registry(fail_on_import_error=False)`.

## WIRED-BUT-UNREACHABLE via orchestrator (fail-closed, feature dead via chat)
- **causal_impact** — orchestrator route raises ValueError (causal inputs never supplied); degraded-registry serves ATE=0.12. Real compute only via direct/harness call.
- **heterogeneous_optimizer** — orchestrator supplies none of 6 required fields → ValueError fail-closed; works only via `/segments` REST + tier0 harness.
- **prediction_synthesizer** — dispatch TypeError (`synthesize()` needs entity_id/prediction_target never supplied; no input_model coercion). Ensemble math real but dead via prod route.
- **resource_optimizer** — orchestrator `optimize(**agent_input)` TypeError (positional args, no **kwargs); works only via REST route.

## DEGENERATE-EMPTY / starved substrate
- **experiment_monitor** — every DB node fails at `await get_supabase_client()` (await-on-sync-client TypeError) in `health_checker.py:45`, `srm_detector.py:49`, `interim_analyzer.py:47`, `fidelity_checker.py:41` → terminal always 'Experiments checked: 0', no alerts.
- **feedback_learner** — `feedback_store` unpopulated all prod paths; `knowledge_stores={}` pins `update_effectiveness=0.0` (dead reward). Matches known "loop starved" memory.

## CRASH on specific reachable path
- **data_preparer** — Supabase entity-split path crashes (`DataFrame.append` removed in pandas 2.x, `data_loader.py:196`). + OOM SPIKE confirmed (5.9 GB precedent).
- **cohort_constructor** — `tier0_integration.py:24 from cohort_constructor import` ModuleNotFoundError (non-existent top-level pkg) on data_source path; iterrows OOM at 100K scale.

## Laundered success (verify vs main — #813 may fix)
- **tool_composer** — `success=True` when 0/N tools succeed (`composer.py:331-349`, `composition_models.py:250-252`); confidence~0.8 from prompt with no anti-fab guard. MAIN differs (#813 "honest tools — real compute or fail-close").

## Mostly-clean with caveat
- **feature_analyzer** — prod LLM path real; silently degrades to np.random SHAP background when X_sample absent (`shap_computer.py:328-335`, documented fallback, low reachability); V4.4 causal branch dead/unreachable.
- **model_trainer** — real sklearn metrics (no mock); D4 OOM-RISK (unconditional 5-fold CV + permutation + bootstrap, no in-agent loky cap; relies on operator `LOKY_MAX_CPU_COUNT=1`).
