# E2I V4.2 Component Update List
## Tool Composer & Orchestrator Classifier Integration

**Version:** 4.2.0  
**Date:** 2024-12-17  
**Impact Level:** Medium-High (New capability, minimal breaking changes)

---

## Executive Summary

Adding Tool Composer introduces a new routing pattern for multi-faceted queries. This requires updates across 7 categories of components, affecting approximately 25-30 files. The changes are **additive** (no breaking changes to existing functionality) but require coordination across layers.

---

## 1. DATABASE LAYER

### 1.1 New Migration Required
| File | Action | Priority |
|------|--------|----------|
| `migrations/011_tool_composer_tables.sql` | **CREATE** | P0 |

**Details:**
- 4 new ENUMs
- 6 new tables
- 4 new views
- 3 new functions
- Seed data for 14 tools and dependencies

### 1.2 Existing Files to Update
| File | Change | Priority |
|------|--------|----------|
| `migrations/README.md` | Add migration 011 documentation | P2 |
| `scripts/setup_db.py` | Add migration 011 to execution order | P1 |

---

## 2. CONFIGURATION LAYER

### 2.1 Updated Files
| File | Changes | Priority |
|------|---------|----------|
| `config/domain_vocabulary.yaml` | **REPLACE** with v4.2.0 (see deliverable) | P0 |
| `config/agent_config.yaml` | Add Tool Composer configuration section | P1 |

### 2.2 New Files
| File | Purpose | Priority |
|------|---------|----------|
| `config/tool_composer_config.yaml` | Tool Composer settings (timeouts, thresholds) | P1 |
| `config/classifier_config.yaml` | Orchestrator classifier settings | P1 |

**tool_composer_config.yaml contents:**
```yaml
tool_composer:
  max_composition_time_ms: 30000
  max_tools_per_composition: 10
  max_dependency_depth: 4
  parallel_execution: true
  
  phases:
    decompose:
      timeout_ms: 5000
      use_llm: true
    plan:
      timeout_ms: 3000
      use_episodic_memory: true
    execute:
      timeout_ms: 20000
      retry_failed_tools: true
      max_retries: 2
    synthesize:
      timeout_ms: 5000
      use_llm: true

  fallback:
    on_timeout: single_agent_fallback
    on_failure: clarification_request
```

**classifier_config.yaml contents:**
```yaml
classifier:
  confidence_threshold: 0.5
  multi_domain_threshold: 2
  word_count_threshold: 25
  llm_layer_enabled: true
  llm_model: claude-3-5-haiku-20241022
  
  stage_weights:
    intent_keywords: 0.5
    structural_features: 0.2
    entity_features: 0.2
    temporal_features: 0.1
```

---

## 3. SOURCE CODE - NEW FILES

### 3.1 Orchestrator Classifier Module
| File | Purpose | Priority |
|------|---------|----------|
| `src/agents/orchestrator/classifier/__init__.py` | Module exports | P0 |
| `src/agents/orchestrator/classifier/schemas.py` | Pydantic models | P0 |
| `src/agents/orchestrator/classifier/feature_extractor.py` | Stage 1 | P0 |
| `src/agents/orchestrator/classifier/domain_mapper.py` | Stage 2 | P0 |
| `src/agents/orchestrator/classifier/dependency_detector.py` | Stage 3 | P0 |
| `src/agents/orchestrator/classifier/pattern_selector.py` | Stage 4 | P0 |
| `src/agents/orchestrator/classifier/pipeline.py` | Main pipeline | P0 |
| `src/agents/orchestrator/classifier/prompts.py` | LLM prompts | P1 |

### 3.2 Tool Composer Module
| File | Purpose | Priority |
|------|---------|----------|
| `src/agents/tool_composer/__init__.py` | Module exports | P0 |
| `src/agents/tool_composer/schemas.py` | Pydantic models | P0 |
| `src/agents/tool_composer/composer.py` | Main composer | P0 |
| `src/agents/tool_composer/decomposer.py` | Phase 1 | P0 |
| `src/agents/tool_composer/planner.py` | Phase 2 | P0 |
| `src/agents/tool_composer/executor.py` | Phase 3 | P0 |
| `src/agents/tool_composer/synthesizer.py` | Phase 4 | P0 |
| `src/agents/tool_composer/tool_registry.py` | Tool management | P0 |
| `src/agents/tool_composer/prompts.py` | LLM prompts | P1 |

### 3.3 Memory Clients (if not existing)
| File | Purpose | Priority |
|------|---------|----------|
| `src/memory/composer_memory.py` | Composer-specific memory operations | P1 |

---

## 4. SOURCE CODE - EXISTING FILES TO UPDATE

### 4.1 Orchestrator Agent
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/orchestrator/agent.py` | Integrate classifier pipeline, add Tool Composer routing | P0 |
| `src/agents/orchestrator/router.py` | **CREATE** or update routing logic | P0 |
| `src/agents/orchestrator/prompts.py` | Add routing decision prompts | P1 |

**Key changes to agent.py:**
```python
# Before: Simple intent → agent mapping
# After: 4-stage classification → pattern-based routing

async def process(self, query: str, context: dict) -> Response:
    # NEW: Run classification pipeline
    classification = await self.classifier.classify(query, context)
    
    # NEW: Route based on pattern
    return await self.router.route(query, classification, context)
```

### 4.2 Base Agent
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/base_agent.py` | Add `expose_tools()` method for tool registration | P1 |

**New method:**
```python
def expose_tools(self) -> list[ToolSchema]:
    """Override in subclasses to expose tools for composition."""
    return []
```

### 4.3 Causal Analytics Agents (Tool Exposure)
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/causal/causal_impact/agent.py` | Implement `expose_tools()` | P1 |
| `src/agents/causal/heterogeneous_optimizer/agent.py` | Implement `expose_tools()` | P1 |
| `src/agents/causal/gap_analyzer/agent.py` | Implement `expose_tools()` | P1 |
| `src/agents/causal/experiment_designer/agent.py` | Implement `expose_tools()` | P1 |

**Example implementation:**
```python
# In causal_impact/agent.py
def expose_tools(self) -> list[ToolSchema]:
    return [
        ToolSchema(
            name="causal_effect_estimator",
            fn=self.tools.estimate_causal_effect,
            # ... schema details
        ),
        ToolSchema(
            name="refutation_runner",
            fn=self.tools.run_refutation,
        ),
        # ...
    ]
```

### 4.4 Prediction & Monitoring Agents (Tool Exposure)
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/prediction/prediction_synthesizer/agent.py` | Implement `expose_tools()` | P1 |
| `src/agents/monitoring/drift_monitor/agent.py` | Implement `expose_tools()` | P1 |

### 4.5 Agent Registry
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/registry.py` | Register Tool Composer, populate ToolRegistry on startup | P0 |

**Key changes:**
```python
class AgentRegistry:
    def __init__(self):
        self.agents = {}
        self.tool_registry = ToolRegistry()  # NEW
    
    def register_agent(self, agent):
        self.agents[agent.name] = agent
        # NEW: Auto-register exposed tools
        for tool in agent.expose_tools():
            self.tool_registry.register(tool)
```

### 4.6 NLP Layer
| File | Changes | Priority |
|------|---------|----------|
| `src/nlp/intent_classifier.py` | Add MULTI_FACETED intent type | P1 |
| `src/nlp/models/intent_models.py` | Add new intent enum value | P1 |
| `src/nlp/entity_extractor.py` | Update to use v4.2 vocabulary patterns | P2 |

### 4.7 Memory Clients
| File | Changes | Priority |
|------|---------|----------|
| `src/memory/supabase_client.py` | Add composer_episodes, classification_logs methods | P1 |
| `src/memory/redis_client.py` | Add composer state methods | P1 |
| `src/memory/falkordb_client.py` | Add tool capability graph queries | P2 |

**New methods for supabase_client.py:**
```python
async def store_composition_episode(self, episode: ComposerEpisode) -> str
async def find_similar_compositions(self, embedding: list[float], limit: int = 5) -> list[ComposerEpisode]
async def store_classification_log(self, log: ClassificationLog) -> str
async def update_classification_feedback(self, classification_id: str, was_correct: bool, correct_pattern: str = None)
```

### 4.8 Feedback Learner Integration
| File | Changes | Priority |
|------|---------|----------|
| `src/agents/self_improvement/feedback_learner/agent.py` | Handle CompositionCompleteEvent | P2 |

---

## 5. API LAYER

### 5.1 New Endpoints
| File | Endpoints | Priority |
|------|-----------|----------|
| `src/api/routes/composer.py` | **CREATE** | P1 |

**New endpoints:**
```
GET  /api/v1/composer/episodes              # List past compositions
GET  /api/v1/composer/episodes/{id}         # Get composition details
GET  /api/v1/composer/active                # Get active compositions
POST /api/v1/composer/feedback/{id}         # Submit feedback
GET  /api/v1/composer/tools                 # List available tools
GET  /api/v1/composer/tools/{name}/performance  # Tool metrics
```

### 5.2 New Endpoints (Classifier)
| File | Endpoints | Priority |
|------|-----------|----------|
| `src/api/routes/classifier.py` | **CREATE** | P2 |

**New endpoints:**
```
GET  /api/v1/classifier/logs                # Classification audit logs
POST /api/v1/classifier/feedback/{id}       # Submit classification feedback
GET  /api/v1/classifier/accuracy            # Accuracy metrics
POST /api/v1/classifier/test                # Test classification (dev only)
```

### 5.3 Existing Endpoints to Update
| File | Changes | Priority |
|------|---------|----------|
| `src/api/routes/chat.py` | Add composition_id to response when Tool Composer used | P1 |
| `src/api/routes/__init__.py` | Register new routers | P1 |

---

## 6. FRONTEND LAYER

### 6.1 New Components
| File | Purpose | Priority |
|------|---------|----------|
| `frontend/src/components/CompositionProgress.tsx` | Show composition phases/progress | P1 |
| `frontend/src/components/ToolChainVisualization.tsx` | Visualize tool execution DAG | P2 |
| `frontend/src/components/ClassificationBadge.tsx` | Show routing pattern used | P2 |

### 6.2 Existing Components to Update
| File | Changes | Priority |
|------|---------|----------|
| `frontend/src/components/ChatMessage.tsx` | Display composition metadata | P1 |
| `frontend/src/components/ResponseMetadata.tsx` | Show tool chain if Tool Composer used | P2 |

### 6.3 Types
| File | Changes | Priority |
|------|---------|----------|
| `frontend/src/types/agent.ts` | Add ToolComposer to agent types | P1 |
| `frontend/src/types/chat.ts` | Add composition_id, routing_pattern to response | P1 |
| `frontend/src/types/composer.ts` | **CREATE** - Composition types | P1 |

**New types (composer.ts):**
```typescript
export type RoutingPattern = 
  | 'SINGLE_AGENT' 
  | 'PARALLEL_DELEGATION' 
  | 'TOOL_COMPOSER' 
  | 'CLARIFICATION_NEEDED';

export type CompositionStatus = 
  | 'PENDING' 
  | 'DECOMPOSING' 
  | 'PLANNING' 
  | 'EXECUTING' 
  | 'SYNTHESIZING' 
  | 'COMPLETED' 
  | 'FAILED';

export interface CompositionStep {
  stepId: string;
  toolName: string;
  status: CompositionStatus;
  latencyMs?: number;
  error?: string;
}

export interface CompositionMetadata {
  compositionId: string;
  routingPattern: RoutingPattern;
  subQuestionCount: number;
  steps: CompositionStep[];
  totalLatencyMs: number;
}
```

### 6.4 Dashboard Updates
| File | Changes | Priority |
|------|---------|----------|
| `frontend/src/pages/Dashboard.tsx` | Add Tool Composer metrics section | P2 |

---

## 7. TESTING LAYER

### 7.1 New Test Files
| File | Purpose | Priority |
|------|---------|----------|
| `tests/unit/test_classifier/test_feature_extractor.py` | Stage 1 tests | P0 |
| `tests/unit/test_classifier/test_domain_mapper.py` | Stage 2 tests | P0 |
| `tests/unit/test_classifier/test_dependency_detector.py` | Stage 3 tests | P0 |
| `tests/unit/test_classifier/test_pattern_selector.py` | Stage 4 tests | P0 |
| `tests/unit/test_classifier/test_pipeline.py` | Pipeline integration | P0 |
| `tests/unit/test_tool_composer/test_decomposer.py` | Phase 1 tests | P0 |
| `tests/unit/test_tool_composer/test_planner.py` | Phase 2 tests | P0 |
| `tests/unit/test_tool_composer/test_executor.py` | Phase 3 tests | P0 |
| `tests/unit/test_tool_composer/test_synthesizer.py` | Phase 4 tests | P0 |
| `tests/unit/test_tool_composer/test_composer.py` | Composer integration | P0 |
| `tests/unit/test_tool_composer/test_tool_registry.py` | Registry tests | P1 |

### 7.2 Integration Tests
| File | Purpose | Priority |
|------|---------|----------|
| `tests/integration/test_multi_faceted_queries.py` | End-to-end multi-domain queries | P0 |
| `tests/integration/test_tool_composition_flow.py` | Full composition pipeline | P0 |
| `tests/integration/test_classifier_routing.py` | Classifier → router flow | P1 |

### 7.3 Existing Tests to Update
| File | Changes | Priority |
|------|---------|----------|
| `tests/integration/test_agent_coordination.py` | Add Tool Composer coordination tests | P1 |
| `tests/unit/test_agents/test_orchestrator.py` | Update for new routing logic | P1 |
| `tests/conftest.py` | Add Tool Composer fixtures | P1 |

---

## 8. DOCUMENTATION

### 8.1 New Documentation
| File | Purpose | Priority |
|------|---------|----------|
| `docs/tool_composer.md` | Tool Composer architecture & usage | P1 |
| `docs/classifier.md` | Orchestrator classifier documentation | P1 |
| `docs/api/composer_endpoints.md` | API documentation | P2 |

### 8.2 Existing Documentation to Update
| File | Changes | Priority |
|------|---------|----------|
| `docs/architecture.md` | Add Tool Composer to architecture diagram | P1 |
| `docs/agents.md` | Document Tool Composer as special agent type | P1 |
| `README.md` | Update feature list | P2 |
| `CHANGELOG.md` | Add v4.2.0 entry | P1 |

---

## 9. SCRIPTS & TOOLS

### 9.1 New Scripts
| File | Purpose | Priority |
|------|---------|----------|
| `scripts/register_tools.py` | Manually register/update tools in registry | P2 |
| `scripts/analyze_classifications.py` | Analyze classifier performance | P2 |
| `scripts/benchmark_composer.py` | Benchmark Tool Composer performance | P2 |

### 9.2 Existing Scripts to Update
| File | Changes | Priority |
|------|---------|----------|
| `scripts/seed_data.py` | Add tool registry seed data | P1 |

---

## 10. DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION DEPENDENCY ORDER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: Foundation (Week 1)                                               │
│  ─────────────────────────────                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ Migration    │────▶│ Config       │────▶│ Schemas      │                │
│  │ 011          │     │ Files        │     │ (Pydantic)   │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
│  PHASE 2: Core Logic (Week 2)                                               │
│  ────────────────────────────                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ Classifier   │────▶│ Tool         │────▶│ Orchestrator │                │
│  │ Pipeline     │     │ Composer     │     │ Integration  │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
│  PHASE 3: Agent Integration (Week 3)                                        │
│  ───────────────────────────────────                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ Tool         │────▶│ Agent        │────▶│ Memory       │                │
│  │ Exposure     │     │ Registry     │     │ Integration  │                │
│  │ (6 agents)   │     │ Update       │     │              │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
│  PHASE 4: API & Frontend (Week 4)                                           │
│  ────────────────────────────────                                           │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ API          │────▶│ Frontend     │────▶│ Dashboard    │                │
│  │ Endpoints    │     │ Components   │     │ Updates      │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
│  PHASE 5: Testing & Docs (Week 5)                                           │
│  ────────────────────────────────                                           │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ Unit Tests   │────▶│ Integration  │────▶│ Documentation│                │
│  │              │     │ Tests        │     │              │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. SUMMARY STATISTICS

| Category | New Files | Updated Files | Total |
|----------|-----------|---------------|-------|
| Database | 1 | 2 | 3 |
| Configuration | 3 | 1 | 4 |
| Source Code (New) | 18 | - | 18 |
| Source Code (Update) | - | 15 | 15 |
| API | 2 | 2 | 4 |
| Frontend | 4 | 5 | 9 |
| Tests | 13 | 3 | 16 |
| Documentation | 3 | 4 | 7 |
| Scripts | 3 | 1 | 4 |
| **TOTAL** | **47** | **33** | **80** |

---

## 12. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Classifier accuracy issues | Medium | High | Extensive test coverage, feedback loop |
| Tool execution timeouts | Medium | Medium | Configurable timeouts, fallback to single agent |
| Memory pressure from episodic storage | Low | Medium | Retention policies, vector index optimization |
| Agent tool exposure breaks existing functionality | Low | High | Feature flag for gradual rollout |
| Frontend complexity increase | Medium | Low | Phased UI rollout, basic first |

---

## 13. ROLLOUT STRATEGY

**Phase 1: Shadow Mode (Week 1-2)**
- Deploy classifier but log decisions only
- Continue using existing single-agent routing
- Collect accuracy data

**Phase 2: Opt-In (Week 3-4)**
- Enable Tool Composer for specific query patterns
- Monitor performance metrics
- Gather user feedback

**Phase 3: Full Rollout (Week 5+)**
- Enable for all multi-faceted queries
- Remove feature flags
- Publish documentation

---

## 14. CHECKLIST

### Pre-Implementation
- [ ] Review and approve SQL migration
- [ ] Review and approve domain_vocabulary.yaml v4.2.0
- [ ] Create feature branch
- [ ] Set up test environment

### Implementation
- [ ] Run migration 011
- [ ] Deploy config files
- [ ] Implement classifier pipeline
- [ ] Implement Tool Composer
- [ ] Update Orchestrator
- [ ] Implement tool exposure in agents
- [ ] Update API layer
- [ ] Update frontend

### Post-Implementation
- [ ] Run full test suite
- [ ] Performance benchmarking
- [ ] Documentation review
- [ ] Security review
- [ ] Stakeholder demo
