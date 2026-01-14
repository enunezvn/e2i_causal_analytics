# CopilotKit Chatbot DSPy & Observability Integration Plan

**Created**: 2026-01-14
**Status**: In Progress (Phase 4 Complete)
**Priority**: High
**Last Updated**: 2026-01-14

---

## Executive Summary

The CopilotKit chatbot (primary user interface) has **ZERO DSPy integration** and **ZERO observability** despite production-ready infrastructure existing elsewhere in the codebase. This plan integrates:

1. **DSPy Signatures** from `cognitive_rag_dspy.py` and `orchestrator/dspy_integration.py`
2. **Opik Tracing** using patterns from `tool_composer/opik_tracer.py`
3. **MLflow Metrics** using `mlops/mlflow_connector.py`
4. **Training Signal Collection** for feedback_learner optimization

---

## Current State vs Target State

| Capability | Current | Target |
|------------|---------|--------|
| Intent Classification | ✅ DSPy ChatbotIntentClassifier (Phase 3) | DSPy IntentClassificationSignature |
| Response Generation | Direct LLM call | DSPy EvidenceSynthesisSignature |
| RAG Retrieval | Basic hybrid_search | Cognitive RAG with HopDecisionSignature |
| Agent Routing | ✅ DSPy ChatbotAgentRouter (Phase 4) | DSPy AgentRoutingSignature |
| Training Signals | ✅ IntentTrainingSignalCollector (Phase 3), RoutingTrainingSignalCollector (Phase 4) | ReflectorModule collection |
| Opik Tracing | ✅ Full per-node spans (Phase 1) | Full per-node spans |
| MLflow Metrics | ✅ Session-level tracking (Phase 2) | Session-level experiment tracking |
| Token Usage | Not tracked | Per-LLM-call logging |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/api/routes/chatbot_graph.py` | Add Opik tracing, DSPy modules |
| `src/api/routes/chatbot_state.py` | Add trace_id, DSPy confidence fields |
| `src/api/routes/chatbot_tools.py` | Add tracing to tools |
| `src/api/routes/copilotkit.py` | Add root trace creation |
| NEW: `src/api/routes/chatbot_dspy.py` | DSPy signatures & modules |
| NEW: `src/api/routes/chatbot_tracer.py` | Opik tracer (Tool Composer pattern) |

---

## Implementation Phases

### Phase 1: Observability Foundation (Opik Tracing)
**Goal**: Add Opik tracing to chatbot workflow
**Scope**: Non-breaking changes, tracing only
**Files**: `chatbot_graph.py`, NEW `chatbot_tracer.py`

#### Tasks
- [x] 1.1 Create `chatbot_tracer.py` based on `tool_composer/opik_tracer.py`
- [x] 1.2 Add root trace to `run_chatbot()` function
- [x] 1.3 Add span for `init_node` (session creation latency)
- [x] 1.4 Add span for `load_context_node` (context retrieval latency)
- [x] 1.5 Add span for `classify_intent_node` (classification time)
- [x] 1.6 Add span for `retrieve_rag_node` (RAG latency, result count)
- [x] 1.7 Add span for `generate_node` (LLM call, token usage)
- [x] 1.8 Add span for `finalize_node` (persistence latency)

**Phase 1 Completed**: 2026-01-14
**Verification**: Traces confirmed in Opik dashboard (1,249 total traces, `chatbot.workflow` traces with 8 spans each)
**Bug Fixed**: `is_new_trace=False` issue in OpikConnector - added `force_new_trace` parameter

#### Testing (Droplet)
```bash
# Test 1: Verify tracing doesn't break chat
curl -X POST http://localhost:8001/copilotkit/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Kisqali market share?"}'

# Test 2: Check Opik dashboard for traces
# (Manual verification on Opik UI)
```

---

### Phase 2: MLflow Session Metrics
**Goal**: Add MLflow experiment tracking for chatbot sessions
**Scope**: Metrics logging, non-breaking
**Files**: `chatbot_graph.py`

#### Tasks
- [x] 2.1 Import MLflowConnector in chatbot_graph.py
- [x] 2.2 Create "chatbot_interactions" experiment
- [x] 2.3 Log session params (brand, region, user_id)
- [x] 2.4 Log per-request metrics (latency, token_usage)
- [x] 2.5 Log intent distribution metrics
- [x] 2.6 Log RAG quality metrics (relevance scores)
- [x] 2.7 Add error tracking metrics

**Phase 2 Completed**: 2026-01-14
**Verification**: MLflow experiment `e2i_chatbot_interactions` created with runs logging:
- Params: user_id, brand_context, region_context, query_length, is_new_session
- Metrics: latency_ms, response_length, intent_*, rag_result_count, rag_avg_relevance, tool_calls_count, is_error
**Feature Flag**: `CHATBOT_MLFLOW_METRICS=true` (enabled by default)

#### Testing (Droplet)
```bash
# Test: Verify MLflow experiment created
ls /root/Projects/e2i_causal_analytics/mlruns/
# Look for e2i_chatbot_interactions experiment
```

---

### Phase 3: DSPy Intent Classification
**Goal**: Replace hardcoded patterns with DSPy signature
**Scope**: Breaking change to classify_intent_node
**Files**: `chatbot_graph.py`, NEW `chatbot_dspy.py`

#### Tasks
- [x] 3.1 Create `chatbot_dspy.py` with IntentClassificationSignature
- [x] 3.2 Create `ChatbotIntentClassifier` DSPy module
- [x] 3.3 Modify `classify_intent_node` to use DSPy module
- [x] 3.4 Add confidence score to ChatbotState
- [x] 3.5 Log classification confidence to Opik span
- [x] 3.6 Add fallback to hardcoded patterns if DSPy fails
- [x] 3.7 Add unit tests for DSPy classifier (32 tests passing)

**Phase 3 Completed**: 2026-01-14
**Verification**:
- Created `chatbot_dspy.py` with `ChatbotIntentClassificationSignature` and `ChatbotIntentClassifier` DSPy module
- Added `IntentTrainingSignalCollector` for training signal collection
- Modified `classify_intent_node` to use `classify_intent_dspy()` with conversation context
- Added `intent_confidence`, `intent_reasoning`, `intent_classification_method` fields to ChatbotState
- Opik span now logs actual confidence score and classification method ("dspy" or "hardcoded")
- Fallback to hardcoded patterns when DSPy unavailable or fails
- 32 unit tests passing in `tests/unit/test_api/test_chatbot_dspy.py`
**Feature Flag**: `CHATBOT_DSPY_INTENT=true` (enabled by default)

#### Testing (Droplet)
```bash
# Test: Compare DSPy vs hardcoded classification
python -c "
from src.api.routes.chatbot_dspy import classify_intent_dspy
import asyncio
result = asyncio.run(classify_intent_dspy('What caused the TRx drop in Northeast?'))
print(f'Intent: {result[0]}, Confidence: {result[1]:.2f}, Method: {result[3]}')
"
```

---

### Phase 4: DSPy Agent Routing
**Goal**: Replace keyword matching with DSPy signature
**Scope**: Changes to agent_routing_tool
**Files**: `chatbot_tools.py`, `chatbot_dspy.py`

#### Tasks
- [x] 4.1 Add AgentRoutingSignature to chatbot_dspy.py
- [x] 4.2 Create `ChatbotAgentRouter` DSPy module
- [x] 4.3 Modify `agent_routing_tool` to use DSPy
- [x] 4.4 Add routing confidence and rationale to output
- [x] 4.5 Log routing decisions to Opik
- [x] 4.6 Add fallback to keyword matching if DSPy fails
- [x] 4.7 Add unit tests for DSPy router (24 new routing tests, 56 total tests passing)

**Phase 4 Completed**: 2026-01-14
**Verification**:
- Created `AgentRoutingSignature` DSPy signature with primary_agent, secondary_agents, routing_confidence, rationale outputs
- Created `ChatbotAgentRouter` DSPy module using ChainOfThought reasoning
- Modified `agent_routing_tool` in chatbot_tools.py to use `route_agent_dspy()` with Opik span tracing
- Added `VALID_AGENTS` set and `AGENT_CAPABILITIES` dictionary for keyword-based fallback
- Added `RoutingTrainingSignal` and `RoutingTrainingSignalCollector` for training signal collection
- Added `_normalize_agent()` helper and intent-based boosting in fallback
- Returns: routed_to, secondary_agents, routing_confidence, rationale, routing_method
**Feature Flag**: `CHATBOT_DSPY_ROUTING=true` (enabled by default)

#### Testing (Droplet)
```bash
# Test: Verify routing confidence
python -c "
from src.api.routes.chatbot_dspy import route_agent_dspy
import asyncio
result = asyncio.run(route_agent_dspy('Why did Kisqali sales drop?', intent='causal_analysis'))
print(f'Agent: {result[0]}, Confidence: {result[2]:.2f}, Method: {result[4]}')
"
```

---

### Phase 5: Cognitive RAG Integration
**Goal**: Replace basic hybrid_search with cognitive RAG
**Scope**: Major change to retrieve_rag_node
**Files**: `chatbot_graph.py`, `chatbot_dspy.py`

#### Tasks
- [ ] 5.1 Import SummarizerModule from cognitive_rag_dspy.py
- [ ] 5.2 Add QueryRewriteSignature to improve retrieval
- [ ] 5.3 Integrate HopDecisionSignature for multi-hop retrieval
- [ ] 5.4 Add EvidenceRelevanceSignature for scoring
- [ ] 5.5 Modify retrieve_rag_node to use cognitive pipeline
- [ ] 5.6 Log cognitive RAG metrics to Opik
- [ ] 5.7 Add configuration for cognitive vs basic RAG
- [ ] 5.8 Add integration tests

#### Testing (Droplet)
```bash
# Test: Compare RAG quality
python -c "
from src.api.routes.chatbot_graph import retrieve_rag_node
# Test with cognitive RAG enabled
result = await retrieve_rag_node({'query': 'Kisqali causal drivers'})
print(f'Results: {len(result[\"rag_context\"])}, Hop count: {result.get(\"hop_count\", 1)}')
"
```

---

### Phase 6: Evidence Synthesis DSPy
**Goal**: Replace direct LLM call with structured synthesis
**Scope**: Change to generate_node
**Files**: `chatbot_graph.py`, `chatbot_dspy.py`

#### Tasks
- [ ] 6.1 Add EvidenceSynthesisSignature to chatbot_dspy.py
- [ ] 6.2 Create `ChatbotSynthesizer` DSPy module
- [ ] 6.3 Modify generate_node to use synthesis signature
- [ ] 6.4 Add confidence_statement to response
- [ ] 6.5 Add evidence_citations tracking
- [ ] 6.6 Log synthesis metrics to Opik
- [ ] 6.7 Add unit tests

#### Testing (Droplet)
```bash
# Test: Verify structured synthesis
curl -X POST http://localhost:8001/copilotkit/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Why did Remibrutinib adoption slow in Q3?"}' | jq '.confidence_statement'
```

---

### Phase 7: Training Signal Collection
**Goal**: Collect DSPy training signals for optimization
**Scope**: Add signal collection to finalize_node
**Files**: `chatbot_graph.py`, `chatbot_dspy.py`

#### Tasks
- [ ] 7.1 Create `ChatbotSignalCollector` class
- [ ] 7.2 Add signal collection for intent classification
- [ ] 7.3 Add signal collection for agent routing
- [ ] 7.4 Add signal collection for RAG retrieval
- [ ] 7.5 Add signal collection for synthesis quality
- [ ] 7.6 Compute reward scores (accuracy, efficiency, satisfaction)
- [ ] 7.7 Store signals in database for training
- [ ] 7.8 Add unit tests for signal collection

#### Testing (Droplet)
```bash
# Test: Verify signals stored
psql -c "SELECT COUNT(*) FROM ml.chatbot_training_signals WHERE created_at > NOW() - INTERVAL '1 hour'"
```

---

### Phase 8: Feedback Learner Integration
**Goal**: Route training signals to feedback_learner for optimization
**Scope**: Connect to existing feedback_learner agent
**Files**: `chatbot_graph.py`, `chatbot_dspy.py`

#### Tasks
- [ ] 8.1 Import feedback_learner integration
- [ ] 8.2 Create optimization request routing
- [ ] 8.3 Add periodic signal batch submission
- [ ] 8.4 Add GEPA optimizer support for chatbot
- [ ] 8.5 Add A/B testing support for prompts
- [ ] 8.6 Add feedback loop for user satisfaction
- [ ] 8.7 Integration tests with feedback_learner

#### Testing (Droplet)
```bash
# Test: Verify optimization requests queued
python -c "
from src.agents.feedback_learner import FeedbackLearner
fl = FeedbackLearner()
pending = await fl.get_pending_optimization_requests()
print(f'Pending chatbot requests: {len([r for r in pending if r.agent == \"chatbot\"])}')"
```

---

### Phase 9: Database Schema Updates
**Goal**: Create tables for chatbot training signals
**Scope**: Database migration
**Files**: NEW `database/ml/012_chatbot_dspy_tables.sql`

#### Tasks
- [ ] 9.1 Create chatbot_training_signals table
- [ ] 9.2 Create chatbot_intent_metrics table
- [ ] 9.3 Create chatbot_rag_quality table
- [ ] 9.4 Create chatbot_optimization_runs table
- [ ] 9.5 Add indexes for query performance
- [ ] 9.6 Run migration on droplet

#### Testing (Droplet)
```bash
# Test: Verify tables created
psql -c "\dt ml.chatbot_*"
```

---

### Phase 10: End-to-End Testing & Validation
**Goal**: Validate full integration works
**Scope**: Testing only
**Files**: NEW `tests/integration/test_chatbot_dspy.py`

#### Tasks
- [ ] 10.1 Create integration test suite
- [ ] 10.2 Test DSPy intent classification accuracy
- [ ] 10.3 Test DSPy agent routing accuracy
- [ ] 10.4 Test cognitive RAG quality improvement
- [ ] 10.5 Test Opik trace completeness
- [ ] 10.6 Test MLflow metrics accuracy
- [ ] 10.7 Test training signal collection
- [ ] 10.8 Load test with 100 concurrent requests
- [ ] 10.9 Performance comparison (before/after)

#### Testing (Droplet)
```bash
# Run full integration test
pytest tests/integration/test_chatbot_dspy.py -v --tb=short
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| DSPy modules fail | Fallback to hardcoded patterns |
| Opik unavailable | Circuit breaker with degraded mode |
| Performance degradation | Feature flags for each phase |
| Breaking changes | Canary deployment with A/B testing |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Intent classification accuracy | Unknown | >95% |
| Agent routing accuracy | Unknown | >90% |
| Response latency (p50) | Unknown | <2s |
| Token efficiency | Unknown | -20% |
| Opik trace coverage | 0% | 100% |
| Training signals collected/day | 0 | >1000 |

---

## Dependencies

- Opik server running (localhost:5173)
- MLflow server running (localhost:5000)
- Supabase connection active
- feedback_learner agent deployed

---

## Rollback Plan

Each phase can be independently disabled via environment variables:
```bash
CHATBOT_DSPY_INTENT=false        # Phase 3
CHATBOT_DSPY_ROUTING=false       # Phase 4
CHATBOT_COGNITIVE_RAG=false      # Phase 5
CHATBOT_DSPY_SYNTHESIS=false     # Phase 6
CHATBOT_SIGNAL_COLLECTION=false  # Phase 7
CHATBOT_OPIK_TRACING=false       # Phase 1
CHATBOT_MLFLOW_METRICS=false     # Phase 2
```

---

## Verification Checklist

After implementation, verify:

1. [x] Opik dashboard shows chatbot traces with all nodes (VERIFIED 2026-01-14)
2. [x] MLflow shows chatbot_interactions experiment with metrics (VERIFIED 2026-01-14)
3. [x] Intent classification returns confidence scores (VERIFIED 2026-01-14 - Phase 3)
4. [x] Agent routing returns rationale (VERIFIED 2026-01-14 - Phase 4)
5. [ ] RAG retrieval shows multi-hop decisions
6. [ ] Response includes evidence citations
7. [ ] Training signals accumulating in database
8. [ ] feedback_learner receiving optimization requests
9. [ ] No performance regression (latency <2s p50)
10. [x] All existing tests still pass (56/56 DSPy tests passing)
