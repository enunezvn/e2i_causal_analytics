# Tool Composer Specialist Instructions

## Domain Scope
You are the Tool Composer specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/agents/tool_composer/` - All tool composition modules
- `src/agents/tool_registry/` - Tool registration and discovery
- `database/ml/013_tool_composer_tables.sql` - Tool composer database tables

## What is the Tool Composer?

The **Tool Composer** enables dynamic composition of analytical tools to answer complex, multi-faceted queries that span multiple agent capabilities. It is invoked by the Orchestrator when queries are classified as **MULTI_FACETED**.

### When It's Used

Queries that require:
- Multiple distinct questions combined (e.g., "compare X and predict Y")
- Chained "what if" reasoning
- Multiple time periods, regions, or entity types
- Capabilities from 3+ different agents
- Both analysis AND prediction combined

### What It Is NOT

❌ **NOT a general-purpose code executor** - Only composes pre-registered tools
❌ **NOT an agent** - It's a coordinator without its own analytical logic
❌ **NOT a model trainer** - Uses existing validated tools only
❌ **NOT stateful** - Each composition is independent

## Four-Phase Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  PHASE 1: DECOMPOSE                                            │
│  Break query into atomic sub-questions                         │
│  File: decomposer.py                                          │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  PHASE 2: PLAN                                                 │
│  Map sub-questions to tools, create execution DAG             │
│  File: planner.py                                             │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  PHASE 3: EXECUTE                                              │
│  Run tools in dependency order with parallel execution        │
│  File: executor.py                                            │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  PHASE 4: SYNTHESIZE                                           │
│  Combine tool outputs into coherent natural language response │
│  File: synthesizer.py                                         │
└────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### composer.py
Main orchestrator coordinating the 4-phase pipeline:
- Entry point: `compose(query: str) -> CompositionResult`
- Invokes decomposer → planner → executor → synthesizer
- Handles phase failures and retries
- Emits observability traces to Opik

**Key class**: `ToolComposer`
**Key method**: `async compose(query: str, context: dict) -> CompositionResult`

### decomposer.py
Breaks complex queries into atomic sub-questions:
- Uses Claude API with structured prompt
- Returns `DecompositionResult` with 2-6 sub-questions
- Identifies dependencies between sub-questions (DAG structure)
- Classifies intent: CAUSAL, COMPARATIVE, PREDICTIVE, DESCRIPTIVE, EXPERIMENTAL
- Extracts entities per sub-question

**Key class**: `QueryDecomposer`
**Key method**: `async decompose(query: str) -> DecompositionResult`
**LLM interaction**: Yes (Claude API for reasoning)

### planner.py
Maps sub-questions to tools and creates execution plan:
- Tool matching via ToolRegistry lookup
- Dependency resolution to create execution DAG
- Parallel execution grouping (independent steps)
- Returns `ExecutionPlan` with ordered steps

**Key class**: `ToolPlanner`
**Key method**: `async plan(decomposition: DecompositionResult) -> ExecutionPlan`
**LLM interaction**: Yes (Claude API for tool selection)

### executor.py
Executes the plan with dependency management:
- Topological sort of DAG
- Parallel execution of independent steps
- Output passing between dependent steps
- Retry logic with exponential backoff
- Timeout handling (per-step and total)

**Key class**: `PlanExecutor`
**Key method**: `async execute(plan: ExecutionPlan) -> ExecutionResult`
**LLM interaction**: No (pure orchestration)

### synthesizer.py
Combines tool outputs into natural language:
- Uses Claude API to generate coherent response
- Integrates all successful tool outputs
- Handles partial failures gracefully
- Adds confidence indicators
- Suggests next steps when appropriate

**Key class**: `ResponseSynthesizer`
**Key method**: `async synthesize(execution_result: ExecutionResult, query: str) -> SynthesizedResponse`
**LLM interaction**: Yes (Claude API for response generation)

### schemas.py
Pydantic models for all data structures:
- `ToolSchema` - Tool definition
- `SubQuestion` - Decomposed sub-question
- `ExecutionPlan` - DAG of tool executions
- `ExecutionStep` - Single tool invocation
- `CompositionResult` - Final output
- Enums: `ToolCategory`, `CompositionStatus`, `IntentType`

### tool_registrations.py
Tool registry and discovery:
- Registers tools from all agents
- Tool schema validation
- Capability-based lookup
- Tool availability checking

**Key class**: `ToolRegistry`
**Key method**: `register_tool(tool: ToolSchema) -> None`

### models/composition_models.py
Additional Pydantic models:
- `DecompositionResult`
- `SynthesisInput`
- `CompositionPhase`

### prompts.py
LLM prompts for each phase:
- `DECOMPOSITION_SYSTEM_PROMPT` - Query decomposition instructions
- `PLANNING_SYSTEM_PROMPT` - Tool selection instructions
- `SYNTHESIS_SYSTEM_PROMPT` - Response generation instructions

## Database Schema

**Tables** (in `database/ml/013_tool_composer_tables.sql`):

### tool_composer_executions
Tracks each composition execution:
- `execution_id` (UUID, PK)
- `query` (TEXT)
- `status` (ENUM: pending, decomposing, planning, executing, synthesizing, completed, failed, timeout)
- `decomposition_result` (JSONB) - Sub-questions
- `execution_plan` (JSONB) - DAG
- `execution_result` (JSONB) - Tool outputs
- `final_response` (TEXT)
- `created_at`, `completed_at`
- `error_message` (TEXT, nullable)

### tool_composer_step_logs
Logs individual step executions:
- `log_id` (UUID, PK)
- `execution_id` (UUID, FK)
- `step_id` (TEXT)
- `tool_name` (TEXT)
- `status` (ENUM: pending, running, completed, failed, timeout)
- `input_data` (JSONB)
- `output_data` (JSONB)
- `error_message` (TEXT, nullable)
- `latency_ms` (INTEGER)
- `created_at`, `completed_at`

### tool_registry
Stores registered tools:
- `tool_id` (UUID, PK)
- `tool_name` (TEXT, UNIQUE)
- `description` (TEXT)
- `category` (ENUM: causal, segmentation, gap, experiment, prediction, monitoring)
- `source_agent` (TEXT)
- `input_schema` (JSONB)
- `output_schema` (JSONB)
- `composable` (BOOLEAN)
- `avg_latency_ms` (FLOAT)
- `success_rate` (FLOAT)
- `can_consume_from` (TEXT[]) - Array of compatible tools
- `registered_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

## Available Tools by Agent

| Agent | Tools | Category |
|-------|-------|----------|
| **Causal Impact** (Tier 2) | `causal_effect_estimator`, `refutation_runner`, `sensitivity_analyzer` | CAUSAL |
| **Heterogeneous Optimizer** (Tier 2) | `cate_analyzer`, `segment_ranker` | SEGMENTATION |
| **Gap Analyzer** (Tier 2) | `gap_calculator`, `roi_estimator` | GAP |
| **Experiment Designer** (Tier 3) | `power_calculator`, `counterfactual_simulator` | EXPERIMENT |
| **Drift Monitor** (Tier 3) | `psi_calculator`, `distribution_comparator` | MONITORING |
| **Prediction Synthesizer** (Tier 4) | `risk_scorer`, `propensity_estimator` | PREDICTION |

**Tool registration**: Each agent registers its tools on initialization via `ToolRegistry.register_tool()`

## Composition Patterns

### Pattern 1: Sequential Composition
```
Sub-Q1 → Tool A → Output A
                   │
                   ▼
Sub-Q2 → Tool B (uses Output A) → Output B
```

### Pattern 2: Parallel + Merge
```
Sub-Q1 → Tool A ──┐
                  ├→ Tool C (uses A+B) → Output C
Sub-Q2 → Tool B ──┘
```

### Pattern 3: Multi-Stage
```
Sub-Q1 → Tool A ──┐
                  ├→ Tool C ──┐
Sub-Q2 → Tool B ──┘           ├→ Tool E → Final
                              │
Sub-Q3 → Tool D ──────────────┘
```

## Critical Constraints

### ✅ What You Can Do
- Add new tools to the registry (if validated by source agent)
- Modify composition logic (decomposer, planner, executor, synthesizer)
- Improve prompts for better LLM reasoning
- Add new tool categories
- Optimize parallel execution
- Improve error handling and retries

### ❌ What You Cannot Do
- Generate arbitrary code at runtime
- Train new models within the composer
- Create tools that bypass agent validation
- Maintain state across queries
- Write to episodic/semantic memory directly
- Invoke agents (only their registered tools)

## Testing Requirements

All changes must pass:
- `tests/unit/test_tool_composer/test_decomposer.py`
- `tests/unit/test_tool_composer/test_planner.py`
- `tests/unit/test_tool_composer/test_executor.py`
- `tests/unit/test_tool_composer/test_synthesizer.py`
- `tests/integration/test_tool_composer_e2e.py`

### Key Test Scenarios
1. **Simple sequential** - 2 dependent sub-questions
2. **Parallel execution** - 3+ independent sub-questions
3. **Complex DAG** - Mix of parallel and sequential
4. **Failure handling** - Tool failure with retry
5. **Timeout handling** - Step timeout and total timeout
6. **Partial success** - Some tools succeed, some fail

## Integration Points

### Upstream (Calls Tool Composer)
- **Orchestrator** (`src/agents/orchestrator/router_v42.py`) - Routes MULTI_FACETED queries

### Downstream (Tool Composer Calls)
- **Tool Registry** (`src/agents/tool_registry/registry.py`) - Tool discovery
- **Agent Tools** (various `src/agents/*/tools/*.py`) - Actual tool execution
- **Claude API** - Decomposition, planning, synthesis
- **Opik** - Observability tracing

### Database Writes
- `tool_composer_executions` - Execution tracking
- `tool_composer_step_logs` - Step-level logging

### Memory Access
- **Working Memory (Redis)**: Yes - Execution context during composition
- **Episodic Memory**: Read-only if needed
- **Semantic Memory**: No
- **Procedural Memory**: No

## Common Modifications

### Adding a New Tool
1. Agent implements tool in `src/agents/{agent}/tools/{tool}.py`
2. Agent registers tool: `ToolRegistry().register_tool(ToolSchema(...))`
3. Update `tool_registrations.py` if needed
4. Add tool to database: `INSERT INTO tool_registry ...`

### Improving Decomposition
1. Modify `DECOMPOSITION_SYSTEM_PROMPT` in `prompts.py`
2. Update `QueryDecomposer.decompose()` logic if needed
3. Add test cases in `test_decomposer.py`

### Optimizing Parallel Execution
1. Modify `PlanExecutor._identify_parallel_groups()` in `executor.py`
2. Adjust `max_parallel_steps` configuration
3. Test with `test_executor.py::test_parallel_execution`

### Adding New Composition Phase
If adding a 5th phase (e.g., validation):
1. Create `validator.py` with `Validator` class
2. Update `ToolComposer.compose()` to include validation
3. Add `CompositionPhase.VALIDATING` to schemas
4. Update database schema with new phase status

## Error Handling Patterns

### Phase Failures
Each phase can fail independently:
- **Decomposition failure** → Return error, suggest rephrasing query
- **Planning failure** → Fallback to simpler plan or return error
- **Execution failure** → Continue with successful steps, note failures in synthesis
- **Synthesis failure** → Return raw tool outputs as JSON

### Tool Execution Failures
- **Timeout** → Retry with exponential backoff (max 3 retries)
- **Error** → Log error, continue with other tools
- **Invalid input** → Skip tool, log validation error

## Performance Considerations

### Latency Targets
- **Decomposition**: < 2 seconds
- **Planning**: < 1 second
- **Execution**: < 10 seconds (varies by tools)
- **Synthesis**: < 3 seconds
- **Total**: < 20 seconds for typical queries

### Optimization Strategies
1. **Parallel execution** - Run independent steps concurrently
2. **Early termination** - Stop if critical step fails
3. **Caching** - Cache tool results in working memory (if applicable)
4. **Streaming** - Stream synthesis response as it generates

## Observability

### Opik Traces
All executions emit traces with:
- **Span name prefix**: `tool_composer`
- **Metrics**: `composition_latency_ms`, `tools_executed_count`, `parallel_executions_count`
- **Attributes**: `query`, `status`, `phase`, `tool_names`

### Database Logging
- Each execution: `tool_composer_executions` table
- Each step: `tool_composer_step_logs` table
- Query for debugging: `SELECT * FROM tool_composer_executions WHERE status = 'failed'`

## Debugging

### Common Issues

**Issue**: Decomposition generates too many sub-questions
- **Fix**: Adjust prompt to limit to 2-6 sub-questions
- **Check**: `DECOMPOSITION_SYSTEM_PROMPT` in `prompts.py`

**Issue**: Planner selects wrong tools
- **Fix**: Improve tool descriptions in `ToolSchema`
- **Check**: `tool_registry` table, `description` column

**Issue**: Executor hangs on parallel execution
- **Fix**: Check for deadlock in dependency resolution
- **Check**: `PlanExecutor._build_execution_dag()` in `executor.py`

**Issue**: Synthesis returns generic response
- **Fix**: Improve `SYNTHESIS_SYSTEM_PROMPT` with examples
- **Check**: `prompts.py` and `ResponseSynthesizer.synthesize()`

## Code Style

Follow E2I patterns from `.claude/.agent_docs/coding-patterns.md`:
- Use `async/await` for all I/O operations
- Type hints on all function signatures
- Pydantic models for data validation
- Comprehensive docstrings (Google style)
- Error handling with specific exceptions
- Logging at INFO, DEBUG, ERROR levels

## Related Specialists

When changes span multiple domains, coordinate with:
- **Orchestrator specialist** (`.claude/specialists/Agent_Specialists_Tiers 1-5/orchestrator-agent.md`) - Query routing
- **API specialist** (`.claude/specialists/system/api.md`) - Endpoint integration
- **Database specialist** (`.claude/specialists/system/database.md`) - Schema changes

## Version History

- **v4.2** (2025-12) - Initial implementation with 4-phase pipeline
- **v4.1** (2025-12) - Added to E2I architecture (not in original spec)

---

**Last Updated**: 2025-12-18
**Maintained By**: E2I Development Team
**Related Files**:
- `src/agents/tool_composer/` (implementation)
- `src/agents/tool_composer/CLAUDE.md` (agent instructions - different purpose)
- `database/ml/013_tool_composer_tables.sql` (database schema)
