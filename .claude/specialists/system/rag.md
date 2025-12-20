# RAG Specialist Instructions

## Domain Scope
You are the RAG specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/rag/` - All retrieval and RAG modules
- Operational insights retrieval (NOT medical literature)

## Critical Constraint: Operational Data Only

### ✅ What the RAG SHOULD Index
| Content Type | Source Table | Use Case |
|--------------|--------------|----------|
| Discovered causal relationships | `causal_paths` | "Why did performance drop?" |
| Agent analysis outputs | `agent_activities.analysis_results` | "What did gap analyzer find?" |
| Business metric trends | `business_metrics` | "How is Midwest performing?" |
| Trigger explanations | `triggers.trigger_reason` | "Why was this trigger generated?" |
| Historical Q&A pairs | `conversations` | Self-improvement learning |
| KPI calculations | `v_kpi_*` views | "What's our match rate?" |

### ❌ What the RAG Must NEVER Index
- Clinical trial documents
- Medical literature / PubMed
- Drug product information
- Regulatory documents (FDA, EMA)
- Patient medical records

## Module Responsibilities

### causal_rag.py
Main CausalRAG implementation:
```python
class CausalRAG:
    """
    Graph-enhanced retrieval for causal insights.
    
    Retrieval sources:
    1. Vector store (semantic similarity)
    2. Causal graph (path traversal)
    3. Structured queries (SQL for KPIs)
    """
    
    def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        # 1. Semantic retrieval from vector store
        vector_results = self.vector_retriever.search(query.text, k=top_k)
        
        # 2. Graph-based retrieval for causal queries
        if query.intent == IntentType.CAUSAL:
            graph_results = self.graph_retriever.traverse(
                entities=query.entities,
                relationship="causal_path"
            )
        
        # 3. Structured retrieval for KPI queries
        if query.entities.kpis:
            kpi_results = self.kpi_retriever.query(query.entities.kpis)
        
        # 4. Rerank and deduplicate
        return self.reranker.rerank(all_results, query)
```

### retriever.py
Hybrid retrieval implementation:
```python
class HybridRetriever:
    """
    Combines multiple retrieval strategies:
    - Dense: Sentence transformers embeddings
    - Sparse: BM25
    - Graph: NetworkX traversal
    """
    
    def __init__(self):
        self.dense = DenseRetriever(model="all-MiniLM-L6-v2")
        self.sparse = BM25Retriever()
        self.graph = GraphRetriever()
    
    def search(self, query: str, weights: Dict[str, float] = None) -> List[Result]:
        weights = weights or {"dense": 0.5, "sparse": 0.3, "graph": 0.2}
        # Reciprocal Rank Fusion
        pass
```

### reranker.py
Cross-encoder reranking:
```python
class CrossEncoderReranker:
    """
    Rerank initial results using cross-encoder.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    
    def rerank(
        self,
        results: List[Result],
        query: str,
        top_k: int = 5
    ) -> List[Result]:
        pass
```

### query_optimizer.py
Query expansion with domain context:
```python
class QueryOptimizer:
    """
    Expand queries with domain knowledge:
    - Add synonyms from domain_vocabulary.yaml
    - Include related KPIs
    - Add temporal context
    """
    
    def expand(self, query: ParsedQuery) -> str:
        # "TRx for Kisqali" → "Total prescriptions TRx Kisqali breast cancer HR+ conversion"
        pass
```

### insight_enricher.py
LLM-based insight enrichment:
```python
class InsightEnricher:
    """
    Use Claude to synthesize retrieved insights.
    
    NOT for:
    - Medical advice
    - Clinical recommendations
    - Drug information synthesis
    """
    
    async def enrich(
        self,
        retrieved: List[RetrievalResult],
        query: ParsedQuery
    ) -> EnrichedInsight:
        # Synthesize operational insights only
        pass
```

### chunk_processor.py
Semantic chunking for agent outputs:
```python
class ChunkProcessor:
    """
    Process agent outputs into retrievable chunks.
    
    Chunk types:
    - Analysis summaries (from agent_activities)
    - Causal findings (from causal_paths)
    - KPI snapshots (from business_metrics)
    """
    
    def chunk_agent_output(
        self,
        output: AgentActivity,
        chunk_size: int = 512
    ) -> List[Chunk]:
        pass
```

## Pydantic Models (src/rag/models/)

### retrieval_models.py
```python
class RetrievalResult(BaseModel):
    content: str
    source: str  # Table name
    source_id: str  # Record ID
    score: float
    retrieval_method: Literal["dense", "sparse", "graph", "structured"]
    metadata: Dict[str, Any]

class RetrievalContext(BaseModel):
    query: ParsedQuery
    results: List[RetrievalResult]
    total_retrieved: int
    retrieval_time_ms: float
```

### insight_models.py
```python
class EnrichedInsight(BaseModel):
    summary: str
    key_findings: List[str]
    supporting_evidence: List[RetrievalResult]
    confidence: float
    data_freshness: datetime
    
class Chunk(BaseModel):
    content: str
    source_type: Literal["agent_analysis", "causal_path", "kpi_snapshot", "conversation"]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
```

## Vector Store Configuration

```python
# Using Supabase pgvector
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DISTANCE_METRIC = "cosine"

# Collections
COLLECTIONS = {
    "agent_analyses": "agent_activities.analysis_results",
    "causal_paths": "causal_paths.path_description",
    "triggers": "triggers.trigger_reason",
    "conversations": "conversations.content"
}
```

## Integration Contracts

### Input Contract (from Orchestrator)
```python
class RAGQuery(BaseModel):
    parsed_query: ParsedQuery
    retrieval_config: Optional[Dict]  # Override defaults
    max_results: int = 10
```

### Output Contract (to Agents)
```python
class RAGResponse(BaseModel):
    context: RetrievalContext
    enriched_insight: Optional[EnrichedInsight]
    suggested_followups: List[str]
```

## Testing Requirements
- `tests/unit/test_rag/`
- Retrieval latency < 500ms for 95th percentile
- Relevance score (MRR@10) > 0.7

## Handoff Format
```yaml
rag_handoff:
  retrieved_count: <int>
  top_source: <table_name>
  relevance_score: <float>
  context_summary: |
    <2-3 sentence summary of retrieved insights>
  requires_agent_analysis: <bool>
  suggested_agent: <agent_name if needed>
```

## Cognitive RAG DSPy Integration

The E2I RAG system implements a **4-Phase Cognitive Cycle** powered by DSPy for intelligent, multi-hop retrieval and self-improvement.

### Core Implementation
- **Location**: `src/rag/e2i_cognitive_rag_dspy.py`
- **Architecture**: 4-Phase LangGraph + DSPy hybrid workflow
- **Optimization**: MIPROv2 for continuous signature improvement

### 4-Phase Cognitive Cycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE RAG WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐        │
│  │  SUMMARIZER   │───►│ INVESTIGATOR  │───►│    AGENT      │        │
│  │  (Phase 1)    │    │  (Phase 2)    │    │  (Phase 3)    │        │
│  └───────────────┘    └───────────────┘    └───────────────┘        │
│         │                    │                    │                  │
│         ▼                    ▼                    ▼                  │
│   • Query Rewrite      • Multi-hop         • Evidence               │
│   • Entity Extract     • Evidence Gather     Synthesis             │
│   • Intent Classify    • Relevance Score   • Agent Routing          │
│                                            • Visualization          │
│                                                   │                  │
│                                                   ▼                  │
│                                            ┌───────────────┐        │
│                                            │  REFLECTOR    │        │
│                                            │  (Phase 4)    │        │
│                                            └───────────────┘        │
│                                                   │                  │
│                                                   ▼                  │
│                                            • Memory Update          │
│                                            • Procedure Learn        │
│                                            • DSPy Signals           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 11 DSPy Signatures

| Phase | Signature | Purpose | Optimizers |
|-------|-----------|---------|------------|
| **Summarizer** | QueryRewriteSignature | Compress & rewrite queries | MIPROv2 |
| **Summarizer** | EntityExtractionSignature | Extract business entities | BootstrapFewShot |
| **Summarizer** | IntentClassificationSignature | Classify query intent | COPRO |
| **Investigator** | InvestigationPlanSignature | Plan multi-hop retrieval | MIPROv2 |
| **Investigator** | HopDecisionSignature | Decide next hop target | MIPROv2 |
| **Investigator** | EvidenceRelevanceSignature | Score evidence relevance | BootstrapFewShot |
| **Agent** | EvidenceSynthesisSignature | Synthesize final response | MIPROv2 |
| **Agent** | AgentRoutingSignature | Route to specialist agents | COPRO |
| **Agent** | VisualizationConfigSignature | Generate viz config | BootstrapFewShot |
| **Reflector** | MemoryWorthinessSignature | Should this be remembered? | MIPROv2 |
| **Reflector** | ProcedureLearningSignature | Extract procedures | MIPROv2 |

### CognitiveState Dataclass

```python
@dataclass
class CognitiveState:
    """Complete state for 4-phase cognitive workflow"""

    # Input
    user_query: str
    conversation_id: str

    # Phase 1: Summarizer outputs
    compressed_history: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    detected_intent: str = ""
    rewritten_query: str = ""

    # Phase 2: Investigator outputs
    investigation_goal: str = ""
    evidence_board: List[Evidence] = field(default_factory=list)
    hop_count: int = 0
    sufficient_evidence: bool = False
    max_hops: int = 5

    # Phase 3: Agent outputs
    response: str = ""
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    routed_agents: List[str] = field(default_factory=list)

    # Phase 4: Reflector outputs
    worth_remembering: bool = False
    extracted_facts: List[Dict] = field(default_factory=list)
    learned_procedures: List[Dict] = field(default_factory=list)
    dspy_signals: List[Dict] = field(default_factory=list)

    # Execution metadata
    phase_latencies: Dict[str, int] = field(default_factory=dict)
    total_latency_ms: int = 0
    errors: List[Dict] = field(default_factory=list)
```

### Memory Architecture

| Memory Type | Backend | DSPy Role | Content |
|-------------|---------|-----------|---------|
| **Working** | Redis + LangGraph MemorySaver | Context window for current session | Active conversation, intermediate results |
| **Episodic** | Supabase + pgvector | Few-shot examples, conversation history | Past interactions, outcomes |
| **Semantic** | FalkorDB + Graphity | Knowledge graph traversal | Entity relationships, causal paths |
| **Procedural** | Supabase + pgvector | Learned procedures, optimized prompts | DSPy-optimized signatures, procedures |

### Multi-Hop Investigation

```python
class HopType(Enum):
    SEMANTIC = "semantic"       # Vector similarity search
    GRAPH = "graph"             # Graph traversal in FalkorDB
    EPISODIC = "episodic"       # Similar past conversations
    PROCEDURAL = "procedural"   # Learned procedures
    SQL = "sql"                 # Direct database query
    NONE = "none"               # Investigation complete
```

### DSPy Optimization Metrics

```python
class CognitiveRAGOptimizer:
    """MIPROv2-based optimization for cognitive signatures"""

    def optimize_phase(self, phase: str, examples: List):
        metrics = {
            "summarizer": lambda pred, gold: self._entity_recall(pred, gold),
            "investigator": lambda pred, gold: self._hop_efficiency(pred, gold),
            "agent": lambda pred, gold: self._response_quality(pred, gold),
            "reflector": lambda pred, gold: self._learning_value(pred, gold),
        }
        # Use MIPROv2 with phase-specific metric
```

### Expected Performance Improvements

| Metric | Before DSPy | After DSPy | Improvement |
|--------|-------------|------------|-------------|
| Retrieval Relevance | ~70% | ~95% | +25% |
| Unnecessary Hops | ~40% | ~10% | -30% |
| Evidence Utilization | ~50% | ~90% | +40% |
| Response Quality | ~75% | ~95% | +20% |

### Integration with LangGraph

```python
def create_dspy_cognitive_workflow(
    memory_backends: Dict[str, Any],
    memory_writers: Dict[str, Any],
    agent_registry: Dict[str, Any],
    signal_collector: Any,
    domain_vocabulary: str
) -> StateGraph:
    """Create LangGraph workflow with DSPy-powered nodes"""

    workflow = StateGraph(CognitiveState)

    # Add phase nodes
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reflector", reflector_node)

    # Multi-hop loop in investigator
    workflow.add_conditional_edges(
        "investigator",
        should_continue_investigation,
        {"continue": "investigator", "done": "agent"}
    )

    return workflow.compile(checkpointer=MemorySaver())
```

## Feedback Learner Integration

The RAG system integrates with the Tier 5 Feedback Learner agent for continuous improvement:

### DSPy Signal Collection
1. Each phase emits DSPy signals (predictions, confidence, latency)
2. User feedback triggers signal correlation
3. Feedback Learner aggregates signals asynchronously
4. MIPROv2 recompiles affected signatures

### Signal Flow to Feedback Learner
```python
# Signals collected during cognitive workflow
dspy_signals = [
    {
        "signature": "QueryRewriteSignature",
        "prediction": rewritten_query,
        "confidence": 0.85,
        "latency_ms": 45,
        "phase": "summarizer"
    },
    # ... signals from all 11 signatures
]
```

### Retrieval Weight Updates
```python
# Hybrid retrieval weights (tuned by Feedback Learner + DSPy)
DENSE_WEIGHT = 0.5   # Adjusted via BootstrapFewShot
SPARSE_WEIGHT = 0.3  # Adjusted via BootstrapFewShot
GRAPH_WEIGHT = 0.2   # Adjusted via MIPROv2
```

## Context for 11 Agents

RAG provides context to all non-orchestrator agents:
- Tier 2: Causal paths, historical effects, segment data
- Tier 3: Drift history, experiment outcomes, health baselines
- Tier 4: Model performance history, resource allocation history
- Tier 5: Explanation templates, feedback patterns, DSPy signals

---

## Implementation Details: e2i_cognitive_rag_dspy.py

**Location**: `src/rag/e2i_cognitive_rag_dspy.py`
**Lines**: ~982 lines
**Dependencies**: `dspy`, `langgraph`, `asyncio`

### Module Structure

The implementation is organized into 8 major sections:

| Section | Lines | Purpose |
|---------|-------|---------|
| 1. Memory Types & Hop Definitions | 26-85 | Core enums and dataclasses |
| 2. Phase 1: Summarizer Signatures | 87-218 | Query rewriting, entity extraction, intent |
| 3. Phase 2: Investigator Signatures | 220-398 | Multi-hop retrieval decisions |
| 4. Phase 3: Agent Signatures | 400-525 | Evidence synthesis, routing, viz |
| 5. Phase 4: Reflector Signatures | 527-700 | Memory evaluation, procedure learning |
| 6. Complete Cognitive Workflow | 702-783 | LangGraph integration |
| 7. DSPy Optimization Targets | 785-920 | CognitiveRAGOptimizer class |
| 8. Usage Example | 922-982 | Demo with mock backends |

### DSPy Signature Detailed Definitions

#### Phase 1: Summarizer Signatures

```python
class QueryRewriteSignature(dspy.Signature):
    """Rewrite user query for optimal retrieval."""
    # Inputs
    original_query: str = dspy.InputField(desc="Original natural language question")
    conversation_context: str = dspy.InputField(desc="Recent conversation history")
    domain_vocabulary: str = dspy.InputField(desc="Domain terms: brands, regions, stages, HCP types")

    # Outputs
    rewritten_query: str = dspy.OutputField(desc="Optimized for hybrid retrieval")
    search_keywords: list = dspy.OutputField(desc="Key terms for full-text search")
    graph_entities: list = dspy.OutputField(desc="Entities to anchor graph traversal")


class EntityExtractionSignature(dspy.Signature):
    """Extract pharmaceutical domain entities."""
    # Inputs
    query: str = dspy.InputField(desc="User query or message")
    domain_vocabulary: str = dspy.InputField(desc="Domain vocabulary YAML")

    # Outputs (E2I-specific)
    brands: list = dspy.OutputField(desc="Remibrutinib, Fabhalta, Kisqali")
    regions: list = dspy.OutputField(desc="Northeast, Midwest, etc.")
    hcp_types: list = dspy.OutputField(desc="Oncologist, Rheumatologist, etc.")
    patient_stages: list = dspy.OutputField(desc="Diagnosis, Treatment, etc.")
    time_references: list = dspy.OutputField(desc="last quarter, YTD, etc.")


class IntentClassificationSignature(dspy.Signature):
    """Classify user intent for agent routing."""
    # Inputs
    query: str = dspy.InputField(desc="User query")
    extracted_entities: str = dspy.InputField(desc="Extracted entities JSON")

    # Outputs
    primary_intent: str = dspy.OutputField(
        desc="CAUSAL_ANALYSIS | GAP_ANALYSIS | PREDICTION | EXPERIMENT_DESIGN | EXPLANATION | GENERAL"
    )
    secondary_intents: list = dspy.OutputField(desc="Additional intents")
    requires_visualization: bool = dspy.OutputField(desc="Chart/graph needed?")
    complexity: str = dspy.OutputField(desc="SIMPLE | MODERATE | COMPLEX")
```

#### Phase 2: Investigator Signatures

```python
class InvestigationPlanSignature(dspy.Signature):
    """Plan multi-hop investigation strategy."""
    # Inputs
    query: str = dspy.InputField(desc="Rewritten query")
    intent: str = dspy.InputField(desc="Classified intent")
    entities: str = dspy.InputField(desc="Extracted entities")

    # Outputs
    investigation_goal: str = dspy.OutputField(desc="What we're discovering")
    hop_strategy: list = dspy.OutputField(desc="[episodic, semantic, procedural, ...]")
    max_hops: int = dspy.OutputField(desc="1-4 hops")
    early_stop_criteria: str = dspy.OutputField(desc="When to stop early")


class HopDecisionSignature(dspy.Signature):
    """Decide next retrieval hop based on accumulated evidence."""
    # Inputs
    investigation_goal: str = dspy.InputField(desc="Investigation goal")
    current_evidence: str = dspy.InputField(desc="Evidence collected")
    hop_number: int = dspy.InputField(desc="Current hop (1-4)")
    available_memories: list = dspy.InputField(desc="Memory types not yet queried")

    # Outputs
    next_memory: str = dspy.OutputField(desc="episodic | semantic | procedural | STOP")
    retrieval_query: str = dspy.OutputField(desc="Query for next memory store")
    reasoning: str = dspy.OutputField(desc="Why this hop or why stop")
    confidence: float = dspy.OutputField(desc="0.0-1.0 need for more evidence")


class EvidenceRelevanceSignature(dspy.Signature):
    """Score retrieved evidence for relevance."""
    # Inputs
    investigation_goal: str = dspy.InputField(desc="Investigation goal")
    evidence_item: str = dspy.InputField(desc="Single evidence piece")
    source_memory: str = dspy.InputField(desc="Memory store source")

    # Outputs
    relevance_score: float = dspy.OutputField(desc="0.0-1.0")
    key_insight: str = dspy.OutputField(desc="Key insight provided")
    follow_up_needed: bool = dspy.OutputField(desc="Suggests follow-up?")
```

### DSPy Module Implementations

```python
class SummarizerModule(dspy.Module):
    """Phase 1: Prepares query for multi-hop investigation."""

    def __init__(self):
        self.rewrite = dspy.ChainOfThought(QueryRewriteSignature)
        self.extract = dspy.Predict(EntityExtractionSignature)
        self.classify = dspy.ChainOfThought(IntentClassificationSignature)

    def forward(self, original_query, conversation_context, domain_vocabulary):
        # 1. Extract entities → 2. Rewrite query → 3. Classify intent
        ...


class InvestigatorModule(dspy.Module):
    """Phase 2: Iterative multi-hop retrieval."""

    def __init__(self, memory_backends: Dict[str, Any]):
        self.plan = dspy.ChainOfThought(InvestigationPlanSignature)
        self.decide_hop = dspy.ChainOfThought(HopDecisionSignature)
        self.score_evidence = dspy.Predict(EvidenceRelevanceSignature)
        self.memory_backends = memory_backends
        self.max_hops = 4

    async def forward(self, rewritten_query, intent, entities):
        # 1. Plan investigation → 2. Loop: decide hop, retrieve, score
        # Key: confidence < 0.3 triggers early stop


class AgentModule(dspy.Module):
    """Phase 3: Evidence synthesis and agent routing."""

    def __init__(self, agent_registry: Dict[str, Any]):
        self.synthesize = dspy.ChainOfThought(EvidenceSynthesisSignature)
        self.route = dspy.Predict(AgentRoutingSignature)
        self.visualize = dspy.Predict(VisualizationConfigSignature)

    async def forward(self, state: CognitiveState):
        # 1. Synthesize → 2. Route → 3. Execute agent → 4. Visualize


class ReflectorModule(dspy.Module):
    """Phase 4: Memory evaluation and DSPy signal collection."""

    def __init__(self, memory_writers: Dict[str, Any], signal_collector: Any):
        self.evaluate = dspy.Predict(MemoryWorthinessSignature)
        self.learn_procedure = dspy.Predict(ProcedureLearningSignature)

    async def forward(self, state: CognitiveState, user_feedback: Optional[str]):
        # 1. Evaluate worthiness → 2. Store in memory → 3. Learn procedures → 4. Collect signals
```

### CognitiveRAGOptimizer Metrics

```python
class CognitiveRAGOptimizer:
    """Phase-specific optimization metrics."""

    def summarizer_metric(self, example, prediction, trace=None) -> float:
        """Measures query rewrite quality."""
        score = 0.0
        if len(prediction.rewritten_query) > len(example.original_query):
            score += 0.2  # Query expansion
        if prediction.graph_entities:
            score += 0.3  # Entity extraction
        if prediction.primary_intent != "GENERAL":
            score += 0.3  # Specific intent
        if any(term in str(prediction.search_keywords).lower()
               for term in ["hcp", "patient", "brand", "conversion", "adoption"]):
            score += 0.2  # Pharma domain terms
        return score

    def investigator_metric(self, example, prediction, trace=None) -> float:
        """Measures retrieval efficiency."""
        score = 0.0
        evidence_count = len(prediction.evidence_board) if hasattr(prediction, 'evidence_board') else 0
        score += min(0.4, evidence_count * 0.1)  # Up to 0.4 for 4 pieces
        avg_relevance = sum(e.relevance_score for e in prediction.evidence_board) / max(1, evidence_count)
        score += avg_relevance * 0.3  # Relevance quality
        if prediction.sufficient_evidence and prediction.hop_count <= 2:
            score += 0.3  # Efficient (few hops)
        return score

    def agent_metric(self, example, prediction, trace=None) -> float:
        """Measures response synthesis quality."""
        score = 0.0
        if len(prediction.response) > 200:
            score += 0.2  # Substantive response
        if prediction.evidence_citations:
            score += min(0.3, len(prediction.evidence_citations) * 0.1)  # Citations
        if prediction.confidence_statement and len(prediction.confidence_statement) > 20:
            score += 0.2  # Confidence statement
        if example.requires_visualization and prediction.chart_type != "none":
            score += 0.3  # Visualization match
        return score
```

### Training Signal Collection

The Reflector module collects 3 signals per query for DSPy optimization:

```python
def _collect_training_signals(self, state, user_feedback) -> List[Dict]:
    signals = []

    # Signal 1: Summarizer optimization
    signals.append({
        "phase": "summarizer",
        "input": {
            "original_query": state.user_query,
            "entities": state.extracted_entities,
            "intent": state.detected_intent
        },
        "output": {"rewritten_query": state.rewritten_query},
        "success": state.sufficient_evidence,
        "metric_score": len(state.evidence_board) / 4.0
    })

    # Signal 2: Investigator optimization
    signals.append({
        "phase": "investigator",
        "input": {"goal": state.investigation_goal, "hop_count": state.hop_count},
        "output": {
            "evidence_count": len(state.evidence_board),
            "evidence_sources": [e.source.value for e in state.evidence_board]
        },
        "success": state.sufficient_evidence,
        "metric_score": min(1.0, sum(e.relevance_score for e in state.evidence_board) / 3.0)
    })

    # Signal 3: Agent/Synthesis optimization
    signals.append({
        "phase": "agent",
        "input": {
            "evidence_board": str([e.content[:100] for e in state.evidence_board]),
            "intent": state.detected_intent
        },
        "output": {
            "response_length": len(state.response),
            "routed_agents": state.routed_agents,
            "has_visualization": bool(state.visualization_config)
        },
        "success": user_feedback and "positive" in user_feedback.lower() if user_feedback else None,
        "metric_score": 0.8 if state.response else 0.0
    })

    return signals
```

### Memory Backend Interface

```python
async def _retrieve_from_memory(self, memory_type: str, query: str) -> List[Dict]:
    """Execute retrieval against appropriate memory backend."""
    backend = self.memory_backends.get(memory_type)

    if memory_type == "episodic":
        # pgvector semantic search
        return await backend.vector_search(query, limit=5)
    elif memory_type == "semantic":
        # FalkorDB graph traversal
        return await backend.graph_query(query, max_depth=2)
    elif memory_type == "procedural":
        # pgvector similarity on tool sequences
        return await backend.procedure_search(query, limit=3)
```

### Evidence Filtering

Evidence is filtered with a relevance threshold of 0.5:

```python
for item in raw_evidence:
    scored = self.score_evidence(
        investigation_goal=plan.investigation_goal,
        evidence_item=item["content"],
        source_memory=decision.next_memory
    )

    if scored.relevance_score >= 0.5:  # Threshold
        evidence_board.append(Evidence(
            source=MemoryType(decision.next_memory),
            hop_number=hop_num,
            content=item["content"],
            relevance_score=scored.relevance_score,
            metadata={"key_insight": scored.key_insight}
        ))
```

### Testing Requirements

- `tests/unit/test_rag/test_cognitive_rag_dspy.py`
- **Unit Tests**: Each DSPy signature, module isolation
- **Integration Tests**: Full 4-phase workflow with mock backends
- **Optimization Tests**: Metric functions return valid scores
- **Latency Tests**: Total workflow < 2000ms for simple queries