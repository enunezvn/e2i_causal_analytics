# Tier 5 Self-Improvement Agents: Implementation Fixes Plan

**Created**: 2026-01-24
**Status**: Planning
**Estimated Phases**: 8 phases across 4 workstreams
**Context Window Strategy**: Each phase designed for single-session completion

---

## Overview

This plan addresses all issues and gaps identified in the Tier 5 agents evaluation:

| Category | Issues | Phases |
|----------|--------|--------|
| Critical Issues | 2 | Phases 1-2 |
| Moderate Issues | 3 | Phases 3-4 |
| Test Coverage Gaps | 4 | Phases 5-6 |
| Documentation | 2 | Phases 7-8 |

---

## Phase 1: Smart LLM Mode Selection

**Goal**: Replace static `use_llm=False` default with intelligent mode selection
**Scope**: Explainer agent only
**Files**: 3-4 files modified

### 1.1 Problem Statement
Currently `ExplainerAgent` defaults to `use_llm=False` (`agent.py:92`), requiring explicit opt-in for advanced reasoning. Complex analyses benefit from LLM but simple ones don't need it.

### 1.2 Solution Design
Implement auto-detection based on input complexity:

```python
# New complexity scoring in agent.py
def _should_use_llm(self, analysis_results: List[Dict], query: str) -> bool:
    """Auto-detect if LLM reasoning is beneficial."""
    # Score based on:
    # - Number of analysis results (>3 = complex)
    # - Query length/complexity (contains "why", "explain", "compare")
    # - User expertise level (executive needs simpler explanations)
    # - Presence of causal discovery data
    complexity_score = self._compute_complexity(analysis_results, query)
    return complexity_score > self.llm_threshold
```

### 1.3 Tasks
- [ ] **1.3.1** Add `_compute_complexity()` method to `ExplainerAgent`
- [ ] **1.3.2** Add `llm_threshold` config parameter (default: 0.5)
- [ ] **1.3.3** Add `auto_llm` mode option (default: True, can be overridden)
- [ ] **1.3.4** Update `explain()` to call complexity check when `use_llm=None`
- [ ] **1.3.5** Add unit tests for complexity scoring
- [ ] **1.3.6** Add integration test for auto-selection behavior

### 1.4 Files to Modify
```
src/agents/explainer/agent.py           # Add complexity scoring
src/agents/explainer/config.py          # Add llm_threshold config (new file)
tests/unit/test_agents/test_explainer/test_agent.py  # Add tests
tests/unit/test_agents/test_explainer/test_complexity.py  # New test file
```

### 1.5 Acceptance Criteria
- [ ] Simple queries (1 result, short query) use deterministic mode
- [ ] Complex queries (>3 results, "why/explain" keywords) auto-enable LLM
- [ ] `use_llm=True/False` explicit setting overrides auto-detection
- [ ] Tests pass with >90% coverage on new code

### 1.6 Estimated Effort
- Implementation: ~150 lines of code
- Tests: ~200 lines
- Context window usage: ~30% of capacity

---

## Phase 2: Feedback Learner Scheduler Implementation

**Goal**: Implement missing async scheduler for batch feedback processing
**Scope**: Feedback Learner agent
**Files**: 2-3 new files, 1-2 modified

### 2.1 Problem Statement
Specialist documentation references `feedback_learner/scheduler.py` for async batch scheduling, but this module doesn't exist. Feedback learning should run on a schedule, not just on-demand.

### 2.2 Solution Design

```python
# New: src/agents/feedback_learner/scheduler.py
class FeedbackLearnerScheduler:
    """Async scheduler for feedback learning cycles."""

    def __init__(
        self,
        agent: FeedbackLearnerAgent,
        schedule: str = "0 */6 * * *",  # Every 6 hours
        batch_size: int = 1000,
        min_feedback_threshold: int = 10,
    ):
        self.agent = agent
        self.schedule = schedule
        self.batch_size = batch_size
        self.min_threshold = min_feedback_threshold
        self._running = False

    async def start(self) -> None:
        """Start the scheduler loop."""

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""

    async def run_cycle(self) -> FeedbackLearnerOutput:
        """Run a single learning cycle."""

    async def check_pending_feedback(self) -> int:
        """Check if enough feedback accumulated to trigger learning."""
```

### 2.3 Tasks
- [ ] **2.3.1** Create `scheduler.py` with `FeedbackLearnerScheduler` class
- [ ] **2.3.2** Add APScheduler or asyncio-based scheduling logic
- [ ] **2.3.3** Add `min_feedback_threshold` check before running cycles
- [ ] **2.3.4** Add graceful shutdown handling
- [ ] **2.3.5** Export scheduler from `__init__.py`
- [ ] **2.3.6** Add unit tests for scheduler lifecycle
- [ ] **2.3.7** Add integration test with mock feedback store

### 2.4 Files to Create/Modify
```
src/agents/feedback_learner/scheduler.py      # NEW - Main scheduler
src/agents/feedback_learner/__init__.py       # Export scheduler
config/feedback_learner.yaml                  # Add scheduler config section
tests/unit/test_agents/test_feedback_learner/test_scheduler.py  # NEW
```

### 2.5 Acceptance Criteria
- [ ] Scheduler starts/stops cleanly
- [ ] Learning cycles run on configured schedule
- [ ] Minimum threshold prevents empty batches
- [ ] Graceful shutdown completes in-progress cycles
- [ ] Tests cover lifecycle, threshold, and error scenarios

### 2.6 Estimated Effort
- Implementation: ~200 lines
- Tests: ~250 lines
- Context window usage: ~35% of capacity

---

## Phase 3: GEPA Orchestration Integration

**Goal**: Clarify and implement explicit GEPA optimization trigger from Feedback Learner
**Scope**: Feedback Learner + GEPA integration
**Files**: 3-4 files modified

### 3.1 Problem Statement
Feedback Learner collects DSPy training signals (`agent.py:198-205`) but GEPA optimization runs independently. There's no explicit trigger condition or handoff protocol.

### 3.2 Solution Design

```python
# Add to feedback_learner/dspy_integration.py
class GEPAOptimizationTrigger:
    """Determines when to trigger GEPA optimization."""

    def __init__(
        self,
        min_signals: int = 100,
        min_reward_delta: float = 0.05,
        cooldown_hours: int = 24,
    ):
        self.min_signals = min_signals
        self.min_reward_delta = min_reward_delta
        self.cooldown_hours = cooldown_hours

    def should_trigger(
        self,
        signal_count: int,
        current_reward: float,
        last_optimization: datetime,
    ) -> Tuple[bool, str]:
        """Check if GEPA optimization should be triggered."""
        # Returns (should_trigger, reason)
```

### 3.3 Tasks
- [ ] **3.3.1** Add `GEPAOptimizationTrigger` class to `dspy_integration.py`
- [ ] **3.3.2** Add trigger check to `KnowledgeUpdaterNode` after updates
- [ ] **3.3.3** Add `trigger_gepa_optimization()` method to agent
- [ ] **3.3.4** Create handoff format for GEPA module
- [ ] **3.3.5** Add MLflow logging for trigger decisions
- [ ] **3.3.6** Add unit tests for trigger conditions
- [ ] **3.3.7** Document trigger conditions in specialist docs

### 3.4 Files to Modify
```
src/agents/feedback_learner/dspy_integration.py   # Add trigger logic
src/agents/feedback_learner/nodes/knowledge_updater.py  # Call trigger
src/agents/feedback_learner/agent.py              # Add trigger method
tests/unit/test_agents/test_feedback_learner/test_gepa_trigger.py  # NEW
```

### 3.5 Acceptance Criteria
- [ ] Trigger conditions are explicit and configurable
- [ ] Trigger decisions are logged to MLflow
- [ ] Handoff format matches GEPA module expectations
- [ ] Cooldown prevents excessive optimization runs
- [ ] Tests verify all trigger conditions

### 3.6 Estimated Effort
- Implementation: ~180 lines
- Tests: ~200 lines
- Context window usage: ~30% of capacity

---

## Phase 4: Discovery Feedback Loop Completion (V4.4)

**Goal**: Complete integration between Feedback Learner and causal discovery runner
**Scope**: Feedback Learner + discovery integration
**Files**: 2-3 files modified, 1 new

### 4.1 Problem Statement
State fields for discovery feedback (`discovered_dag_adjacency`, `discovery_gate_decision`, etc.) are defined but integration with causal discovery runner is partially visible.

### 4.2 Solution Design

```python
# New: src/agents/feedback_learner/nodes/discovery_feedback_node.py
class DiscoveryFeedbackNode:
    """Process feedback specific to causal discovery results."""

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """
        Process discovery-specific feedback:
        1. Collect gate override feedback
        2. Track algorithm accuracy by discovery type
        3. Generate parameter recommendations
        4. Update discovery configuration store
        """
```

### 4.3 Tasks
- [ ] **4.3.1** Create `discovery_feedback_node.py` with specialized node
- [ ] **4.3.2** Add discovery feedback collection to `FeedbackCollectorNode`
- [ ] **4.3.3** Add algorithm accuracy tracking to pattern analyzer
- [ ] **4.3.4** Add discovery-specific recommendations to learning extractor
- [ ] **4.3.5** Update graph to conditionally include discovery node
- [ ] **4.3.6** Add unit tests for discovery feedback processing
- [ ] **4.3.7** Add integration test with mock discovery results

### 4.4 Files to Create/Modify
```
src/agents/feedback_learner/nodes/discovery_feedback_node.py  # NEW
src/agents/feedback_learner/nodes/feedback_collector.py       # Add discovery source
src/agents/feedback_learner/graph.py                          # Conditional node
tests/unit/test_agents/test_feedback_learner/test_discovery_feedback.py  # NEW
```

### 4.5 Acceptance Criteria
- [ ] Discovery feedback is collected and processed
- [ ] Algorithm accuracy is tracked per discovery type
- [ ] Parameter recommendations are generated
- [ ] Node is only activated when discovery feedback present
- [ ] Tests cover discovery-specific scenarios

### 4.6 Estimated Effort
- Implementation: ~250 lines
- Tests: ~200 lines
- Context window usage: ~35% of capacity

---

## Phase 5: Concurrency & Race Condition Tests

**Goal**: Add tests for concurrent feedback batch processing
**Scope**: Test suite only
**Files**: 2 new test files

### 5.1 Problem Statement
No tests verify behavior when multiple feedback batches are processed concurrently, risking race conditions in knowledge updates.

### 5.2 Solution Design

```python
# New test file structure
class TestConcurrentFeedbackProcessing:
    """Test concurrent batch processing."""

    async def test_parallel_batch_processing(self):
        """Multiple batches can process in parallel."""

    async def test_knowledge_update_locking(self):
        """Knowledge updates use proper locking."""

    async def test_pattern_detection_isolation(self):
        """Pattern detection is isolated per batch."""

    async def test_signal_aggregation_thread_safety(self):
        """DSPy signal aggregation is thread-safe."""
```

### 5.3 Tasks
- [ ] **5.3.1** Create `test_concurrency.py` for Feedback Learner
- [ ] **5.3.2** Add parallel batch processing test with asyncio.gather
- [ ] **5.3.3** Add knowledge update locking test
- [ ] **5.3.4** Add pattern detection isolation test
- [ ] **5.3.5** Add DSPy signal aggregation thread-safety test
- [ ] **5.3.6** Add stress test with 10 concurrent batches
- [ ] **5.3.7** Document any discovered race conditions

### 5.4 Files to Create
```
tests/unit/test_agents/test_feedback_learner/test_concurrency.py  # NEW
tests/integration/test_feedback_learner_concurrency.py            # NEW
```

### 5.5 Acceptance Criteria
- [ ] Concurrent batch processing works without data corruption
- [ ] Any race conditions are identified and documented
- [ ] Fixes for discovered issues are tracked (separate phase)
- [ ] Tests are marked with appropriate pytest markers

### 5.6 Estimated Effort
- Tests: ~400 lines
- Context window usage: ~25% of capacity

---

## Phase 6: Memory Failure & LLM Timeout Tests

**Goal**: Add tests for graceful degradation scenarios
**Scope**: Test suite for both agents
**Files**: 2-3 new test files

### 6.1 Problem Statement
No tests verify behavior when Redis/Supabase are unavailable or when LLM calls timeout.

### 6.2 Solution Design

```python
# Test scenarios for graceful degradation
class TestMemoryFailureScenarios:
    """Test behavior when memory backends are unavailable."""

    async def test_redis_unavailable_fallback(self):
        """Agent continues with reduced functionality when Redis down."""

    async def test_supabase_unavailable_fallback(self):
        """Agent continues when Supabase unavailable."""

    async def test_all_memory_unavailable(self):
        """Agent returns valid response even with no memory."""

class TestLLMTimeoutRecovery:
    """Test LLM timeout and recovery behavior."""

    async def test_llm_timeout_fallback_to_deterministic(self):
        """LLM timeout triggers deterministic fallback."""

    async def test_llm_partial_response_handling(self):
        """Partial LLM response is handled gracefully."""
```

### 6.3 Tasks
- [ ] **6.3.1** Create `test_memory_failures.py` for Explainer
- [ ] **6.3.2** Create `test_memory_failures.py` for Feedback Learner
- [ ] **6.3.3** Add Redis unavailable test with mock connection error
- [ ] **6.3.4** Add Supabase unavailable test
- [ ] **6.3.5** Add combined failure test
- [ ] **6.3.6** Add LLM timeout test (mock 30s timeout)
- [ ] **6.3.7** Add LLM partial response test
- [ ] **6.3.8** Verify warnings are logged appropriately

### 6.4 Files to Create
```
tests/unit/test_agents/test_explainer/test_memory_failures.py       # NEW
tests/unit/test_agents/test_feedback_learner/test_memory_failures.py  # NEW
tests/unit/test_agents/test_explainer/test_llm_timeout.py           # NEW
```

### 6.5 Acceptance Criteria
- [ ] Agents return valid (degraded) responses on memory failure
- [ ] LLM timeout triggers deterministic fallback within 35 seconds
- [ ] Appropriate warnings are logged for each failure mode
- [ ] No unhandled exceptions in any failure scenario

### 6.6 Estimated Effort
- Tests: ~450 lines
- Context window usage: ~30% of capacity

---

## Phase 7: Specialist Documentation Updates

**Goal**: Align specialist documentation with actual implementation
**Scope**: Documentation only
**Files**: 2 specialist docs

### 7.1 Problem Statement
- Scheduler is referenced but wasn't implemented (now Phase 2 fixes this)
- LLM model selection not documented
- GEPA trigger conditions not documented

### 7.2 Tasks
- [ ] **7.2.1** Update `explainer.md` with auto-LLM selection logic
- [ ] **7.2.2** Update `explainer.md` with model recommendations (Sonnet for reasoning)
- [ ] **7.2.3** Update `feedback-learner.md` with scheduler documentation
- [ ] **7.2.4** Update `feedback-learner.md` with GEPA trigger conditions
- [ ] **7.2.5** Add discovery feedback loop documentation (V4.4 completion)
- [ ] **7.2.6** Add configuration reference tables
- [ ] **7.2.7** Review and remove any aspirational features not implemented

### 7.3 Files to Modify
```
.claude/specialists/Agent_Specialists_Tiers 1-5/explainer.md
.claude/specialists/Agent_Specialists_Tiers 1-5/feedback-learner.md
```

### 7.4 Acceptance Criteria
- [ ] All documented features exist in codebase
- [ ] Configuration options are fully documented
- [ ] Model recommendations are explicit
- [ ] GEPA integration is clearly explained

### 7.5 Estimated Effort
- Documentation: ~300 lines of updates
- Context window usage: ~20% of capacity

---

## Phase 8: Operational Guide Creation

**Goal**: Create deployment and operations guide for Tier 5 agents
**Scope**: New documentation
**Files**: 1 new file

### 8.1 Problem Statement
No operational documentation exists for deploying, monitoring, and scaling Tier 5 agents.

### 8.2 Solution Design

```markdown
# Tier 5 Agents Operations Guide

## Deployment Checklist
- [ ] Memory backends (Redis, Supabase, FalkorDB) accessible
- [ ] LLM API keys configured
- [ ] MLflow tracking server running
- [ ] GEPA optimization module deployed

## Monitoring
- Key metrics to track
- Alert thresholds
- Dashboard setup

## Scaling Considerations
- Horizontal scaling for Feedback Learner
- Rate limiting for LLM calls
- Memory backend connection pooling

## Troubleshooting
- Common issues and resolutions
```

### 8.3 Tasks
- [ ] **8.3.1** Create `docs/operations/tier5-agents-guide.md`
- [ ] **8.3.2** Add deployment checklist section
- [ ] **8.3.3** Add monitoring metrics and thresholds
- [ ] **8.3.4** Add scaling recommendations
- [ ] **8.3.5** Add troubleshooting section with common issues
- [ ] **8.3.6** Add configuration reference
- [ ] **8.3.7** Review with DevOps specialist context

### 8.4 Files to Create
```
docs/operations/tier5-agents-guide.md  # NEW
```

### 8.5 Acceptance Criteria
- [ ] Guide covers full deployment lifecycle
- [ ] Monitoring metrics are specific and measurable
- [ ] Scaling guidance includes resource estimates
- [ ] Troubleshooting covers scenarios from Phase 6 tests

### 8.6 Estimated Effort
- Documentation: ~400 lines
- Context window usage: ~25% of capacity

---

## Implementation Schedule

### Workstream A: Core Fixes (Phases 1-2)
| Phase | Dependency | Priority |
|-------|------------|----------|
| Phase 1: Smart LLM Mode | None | P0 - Critical |
| Phase 2: Scheduler | None | P0 - Critical |

### Workstream B: Integration (Phases 3-4)
| Phase | Dependency | Priority |
|-------|------------|----------|
| Phase 3: GEPA Orchestration | Phase 2 | P1 - High |
| Phase 4: Discovery Feedback | None | P1 - High |

### Workstream C: Testing (Phases 5-6)
| Phase | Dependency | Priority |
|-------|------------|----------|
| Phase 5: Concurrency Tests | Phase 2 | P2 - Medium |
| Phase 6: Failure Tests | Phase 1 | P2 - Medium |

### Workstream D: Documentation (Phases 7-8)
| Phase | Dependency | Priority |
|-------|------------|----------|
| Phase 7: Specialist Docs | Phases 1-4 | P3 - Low |
| Phase 8: Operations Guide | Phase 6 | P3 - Low |

---

## Parallel Execution Opportunities

These phases can be worked on in parallel:
- **Parallel Set 1**: Phase 1 + Phase 2 (no dependencies)
- **Parallel Set 2**: Phase 3 + Phase 4 (after Set 1)
- **Parallel Set 3**: Phase 5 + Phase 6 (after relevant core fixes)
- **Parallel Set 4**: Phase 7 + Phase 8 (after implementation complete)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM complexity scoring inaccurate | Use conservative thresholds; add override option |
| Scheduler adds operational complexity | Make scheduler opt-in; document thoroughly |
| GEPA trigger too aggressive | Default to conservative thresholds; add cooldown |
| Tests find race conditions | Document findings; create Phase 5B for fixes |

---

## Success Metrics

After all phases complete:
- [ ] 100% of documented features are implemented
- [ ] Test coverage for Tier 5 agents > 85%
- [ ] All failure scenarios have graceful degradation
- [ ] GEPA optimization is explicitly triggered
- [ ] Operational documentation enables self-service deployment

---

## Appendix: Context Window Budget

Each phase designed for single-session completion:

| Phase | Est. Context Usage | Files | Lines Changed |
|-------|-------------------|-------|---------------|
| Phase 1 | 30% | 4 | ~350 |
| Phase 2 | 35% | 4 | ~450 |
| Phase 3 | 30% | 4 | ~380 |
| Phase 4 | 35% | 4 | ~450 |
| Phase 5 | 25% | 2 | ~400 |
| Phase 6 | 30% | 3 | ~450 |
| Phase 7 | 20% | 2 | ~300 |
| Phase 8 | 25% | 1 | ~400 |

**Total New/Modified Code**: ~3,180 lines across 24 files
