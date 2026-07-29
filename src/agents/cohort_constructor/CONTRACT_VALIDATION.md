# CohortConstructor Agent - Contract Validation Report

**Agent**: Cohort Constructor
**Tier**: 0 (ML Foundation)
**Agent Type**: Standard (tool-heavy, SLA-bound, **no LLM**)
**Contract**: `.claude/contracts/tier0/cohort_constructor.md`
**Pipeline Position**: `scope_definer → cohort_constructor → data_preparer`

**Date**: 2026-07-29
**Version**: 1.0
**Status**: ✅ Pipeline API implemented · 🔒 Chat dispatch fails closed by design

> Produced from the 2026-07-29 contract-review findings (GitHub issue #1344). Every claim below was verified against the code at the cited `file:line` locations.

---

## ⚠️ Chat Routability — Pipeline-Only

**`cohort_constructor` can NEVER be a valid gold routing target for a chat query.** It is driven exclusively by the ML pipeline via structured study parameters, not by free-text. The rich request/response contract in the tier-0 doc describes the **pipeline API**, not a chat interface.

| Fact | Evidence |
|------|----------|
| Real entry point is `run(patient_df, brand, indication, config, ...)` / `run_sync(...)`, both requiring a real patient `DataFrame` + brand/indication config | `agent.py:231-266` (`run`), `agent.py:516-543` (`run_sync`) |
| **Absent** from `INTENT_TO_AGENTS` (no intent routes here; the chat form routes to `cohort_profiler`) | `src/agents/orchestrator/nodes/router.py:44-185`, esp. `176-184` |
| **Deliberately NOT** in `AGENT_METHOD_MAP` (Tier 1–5 conversational contract only) | `src/agents/orchestrator/_agent_method_map.py:63-69` |
| IS a member of `VALID_AGENTS`, so a chat route *could* name it | `src/api/routes/chatbot_dspy.py:680-693` |
| Any chat dispatch **fails closed**: resolver always returns `NeedsStructuredInput(missing=('patient_df','brand'))` with an actionable "use the ML cohort pipeline" message, fabricating nothing (#814) | `src/agents/orchestrator/nodes/dispatcher.py:1184-1213`; registered in `INPUT_RESOLVERS` at `dispatcher.py:1248` |

The resolver runs *before* the method lookup, so the raw "registered but has no method 'analyze'" registry error never leaks to a user. It self-activates into real execution the moment the ML pipeline supplies the structured inputs.

---

## Contract Compliance Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Pipeline entry point** | ✅ | `run` / `run_sync` accept `(patient_df, brand, indication, config)` |
| **Input Contract** | ✅ | Structured study params (`scope_spec`, `cohort_config`, patient source) — pipeline-supplied |
| **Output Contract** | ✅ | `cohort_spec`, `eligible_patients`, `eligibility_stats`, `summary`, handoff |
| **State Contract** | ✅ | `CohortConstructorState` (see `state.py`) |
| **Graph pipeline** | ✅ | 4 nodes + error handler, conditional edges (`graph.py:88-142`) |
| **No LLM** | ✅ | Pure tool/compute; deterministic criterion application |
| **SLA** | ✅ | <120s per 100K patients (`constants.py:83-109`, `TOTAL_EXECUTION_MS = 120_000`) |
| **Chat dispatch** | 🔒 | Fails closed by design — not a chat routing target |
| **Error handling** | ✅ | `CC_001`–`CC_007` codes; error_handler node terminal branch |

---

## Pipeline Contract (LangGraph, no LLM)

**Implementation**: `src/agents/cohort_constructor/graph.py:88-142`

A single LangGraph dispatch with a linear happy path and a shared error-handler branch. No LLM is involved at any node.

```
validate_config → apply_criteria → validate_temporal → generate_metadata → END
        │                │                 │                    │
        └────────────────┴─────────────────┴────────────────────┘
                         (any node failure)
                                  ↓
                          error_handler → END
```

### Node Table

| Order | Node | Responsibility | Key State Outputs |
|-------|------|----------------|-------------------|
| 1 | `validate_config` | Validate brand/indication, operators, temporal params, required fields | `config_valid`, `config_validation_errors`, `required_fields`, `supported_operators` |
| 2 | `apply_criteria` | Apply AND inclusion / AND-NOT exclusion criteria; per-criterion removed/remaining log | `eligible_patients`, `eligible_patient_count`, `exclusion_rate`, `eligibility_log` |
| 3 | `validate_temporal` | Enforce lookback/followup eligibility windows | `temporal_validation_passed`, `lookback_failures`, `followup_failures`, `temporally_eligible_patients` |
| 4 | `generate_metadata` | Build `cohort_spec`, `config_hash` (SHA256), summary, `data_preparer` handoff | `cohort_spec`, `cohort_id`, `config_hash`, `summary_report`, `handoff_data` |
| — | `error_handler` | Terminal branch on any node failure; emits structured errors and `status="failed"` | `errors`, `warnings`, `status` |

**Conditional edges**: each of the four main nodes routes through `_should_continue` → `{"continue": <next>, "error_handler": "error_handler"}`; `error_handler` always ends (`graph.py:101-139`).

**Execution modes** (`agent.py:83-97, 262-269`): graph mode (default, LangGraph workflow) or direct mode (`_run_direct`, faster standalone). `run_sync` always uses direct mode.

---

## Input Contract (pipeline-supplied)

**Contract**: `.claude/contracts/tier0/cohort_constructor.md` — Input Contract section.

The ML flow supplies structured study parameters from `scope_definer`:

- `scope_spec` — `experiment_id`, `problem_type`, `target_variable`, `brand`, `indication`
- `cohort_config` (optional; pre-built config used if omitted) — `inclusion_criteria` (AND), `exclusion_criteria` (AND-NOT), `lookback_days`, `followup_days`, `index_date_field`, `required_fields`
- `patient_data_source` / `patient_df` — the real patient dataset

**Supported brands**: Remibrutinib/CSU, Fabhalta/PNH, Kisqali/HR+HER2- (`constants.py` `SUPPORTED_BRANDS`; pre-built configs in `configs.py`). The `run`/`run_sync` signature raises `ValueError` if neither `brand` nor `config` is provided (`agent.py:259-261`).

---

## Output Contract

**Contract**: `.claude/contracts/tier0/cohort_constructor.md` — Output Contract section. Required keys: `cohort_spec`, `eligible_patients`, `eligibility_stats`, `summary`.

- **Cohort spec** — `cohort_id`, criteria applied, temporal requirements, `version`, `config_hash` (SHA256 reproducibility), `status`
- **Eligible population** — `patient_ids`, `count`
- **Eligibility statistics** — `total_input_patients`, `eligible_patient_count`, `exclusion_rate`, per-criterion `eligibility_log`, `temporal_validation`
- **Handoff** — `cohort_id`, `eligible_patient_ids`, full `cohort_spec`, `quality_checks_required` (for `data_preparer`)

Return type of `run`/`run_sync`: `Tuple[pd.DataFrame, CohortExecutionResult]` (`agent.py:240, 522`).

---

## Error Handling

**Contract error codes**: `CC_001` (invalid config) · `CC_002` (missing required fields) · `CC_003` (empty cohort) · `CC_004` (insufficient temporal data) · `CC_005` (unsupported operator) · `CC_006` (unsupported brand) · `CC_007` (SLA timeout). Codes and recovery messages in `constants.py:38-76`.

Any node failure routes to `error_handler`, which terminates the graph with structured `errors`, `warnings`, and `status="failed"`.

---

## Performance / SLA

**Target**: <120s for 100K patients (`constants.py:83-109`).

- `TOTAL_EXECUTION_MS = 120_000`
- `PATIENTS_PER_SECOND = 833` (100,000 / 120s)
- Size-based sub-thresholds defined in `SLAThreshold`.

Agent metadata (`constants.py:193-196`): `tier=0`, `type="standard"`, `sla_seconds=120`.

---

## Test Coverage

Unit tests live in `tests/unit/test_agents/test_cohort_constructor/`:

| File | Focus |
|------|-------|
| `test_agent.py` | Agent orchestration, run/run_sync entry points |
| `test_constructor.py` | Core `CohortConstructor` criterion logic |
| `test_configs.py` | Pre-built brand configs (Remibrutinib/Fabhalta/Kisqali) |
| `test_types.py` | `CohortConfig` / result TypedDicts |
| `test_memory_wiring_883b.py` | Memory-hook wiring (#883 PR B) |
| `test_observability.py` | MLflow/Opik tracing |
| `test_tier0_integration_import_f17.py` | Tier-0 import integration |

**Chat fail-closed behavior** is covered by `tests/unit/test_agents/test_orchestrator/test_dispatcher_cohort_constructor_resolver.py`:
- `test_cohort_constructor_registered_in_input_resolvers`
- `test_resolver_always_fails_closed_without_fabricating`
- `test_dispatch_fails_closed_not_raw_registry_error`

---

## Verification Discrepancies

1. The tier-0 contract's "Validation Tests" section cites `tests/integration/test_cohort_constructor_contracts.py`, which **does not exist**. The actual coverage is the unit-test set above. The contract's test list is aspirational, not current.
2. `dispatcher.py:1760` contains a `cohort_constructor` mock response (`eligible_count: 150`, etc.) — this lives inside `_mock_agent_execution`, reachable **only** when the dispatcher is constructed with `allow_mock=True` (dev/test). Production default is `allow_mock=False`, which fails closed on a missing/unresolvable agent (`dispatcher.py:1660-1673`). Not a production path.

---

## Conclusion

`cohort_constructor` is a **Tier-0 ML-pipeline agent with no LLM**. Its documented request/response contract is the pipeline API — reached via `run(patient_df, brand, indication, config)` / `run_sync`, driven by structured study parameters from `scope_definer`, producing an eligible cohort + audit trail for `data_preparer`. It is **not** chat-routable: absent from `INTENT_TO_AGENTS` and `AGENT_METHOD_MAP`, and although present in `VALID_AGENTS`, any chat dispatch fails closed via `_resolve_cohort_constructor_input` with an actionable "use the ML cohort pipeline" message and no fabrication. The chat form of a cohort question is owned by the `cohort_profiler` companion.
