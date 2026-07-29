# Integration Contracts

**Purpose**: Define system-level integration contracts for the E2I Causal Analytics platform, ensuring consistent communication across all layers and components.

**Version**: 1.2
**Last Updated**: 2025-12-31
**Owner**: E2I Development Team

---

## Overview

This contract defines integration points and data flow contracts across the entire E2I Causal Analytics system:

- **18 agents** across 6 tiers
- **4 system layers**: NLP, Causal Engine, RAG, API
- **5 MLOps tools**: MLflow, Opik, Great Expectations, Feast, BentoML
- **3 data stores**: PostgreSQL (Supabase), Vector DB, Graph DB
- **Frontend-Backend integration**: React ↔ FastAPI

This ensures all components communicate predictably and errors propagate correctly.

---

## System Architecture Integration Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER                                  │
│  React + TypeScript + Redux Toolkit                                     │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │ REST API (JSON)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                      │
│  FastAPI + Pydantic Validation                                          │
└──────────┬──────────────────┬──────────────────┬────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
│   NLP LAYER      │  │ ORCHESTRATOR │  │  SYSTEM AGENTS   │
│  Query parsing   │  │   (Tier 1)   │  │  (Health, etc.)  │
└──────┬───────────┘  └──────┬───────┘  └──────────────────┘
       │                     │
       │                     ▼
       │         ┌───────────────────────────┐
       │         │    AGENT EXECUTION        │
       │         │  (Tiers 0, 2-5: 17 agents)│
       │         └────┬─────────────┬────────┘
       │              │             │
       ▼              ▼             ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  RAG SYSTEM │  │   MEMORY    │  │   CAUSAL    │
│  5 indexes  │  │   4 types   │  │   ENGINE    │
└─────────────┘  └─────────────┘  └─────────────┘
       │              │             │
       └──────────────┴─────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │       DATA LAYER             │
       │  PostgreSQL + Vector + Graph │
       └──────────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │      MLOPS LAYER             │
       │  MLflow, Opik, Feast, etc.   │
       └──────────────────────────────┘
```

---

## Frontend-Backend Integration Contract

### API Request Format

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class QueryRequest(BaseModel):
    """
    Standard API request format from frontend.

    Endpoint: POST /api/v1/query
    """

    # === QUERY ===
    query: str = Field(..., min_length=1, max_length=2000, description="User query")

    # === SESSION ===
    session_id: Optional[str] = Field(
        None,
        description="Session ID (auto-generated if not provided)",
        regex=r"^sess_[a-z0-9]{16}$"
    )

    # === PREFERENCES ===
    user_expertise: Literal["executive", "analyst", "data_scientist", "developer"] = Field(
        default="analyst",
        description="User expertise level (affects explanation depth)"
    )

    output_format: Literal["narrative", "structured", "visual", "mixed"] = Field(
        default="narrative",
        description="Preferred output format"
    )

    # === CONSTRAINTS ===
    max_response_time_seconds: Optional[float] = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Maximum acceptable response time"
    )

    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Query priority"
    )

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Recent conversation history for context"
    )

    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional filters (brands, regions, time periods)"
    )

    class Config:
        extra = "forbid"
```

### API Response Format

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class QueryResponse(BaseModel):
    """
    Standard API response format to frontend.

    Returned by: POST /api/v1/query
    """

    # === IDENTIFICATION ===
    query_id: str = Field(..., description="Unique query identifier")
    session_id: str = Field(..., description="Session identifier")

    # === STATUS ===
    status: Literal["completed", "partial", "failed", "timeout"] = Field(
        ...,
        description="Query processing status"
    )

    # === PRIMARY RESPONSE ===
    response: str = Field(..., description="Primary response text")

    response_format: Literal["narrative", "structured", "visual", "mixed"] = Field(
        ...,
        description="Format of response"
    )

    # === INSIGHTS ===
    insights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted insights from analysis"
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings (max 5)"
    )

    # === VISUALIZATIONS ===
    visualizations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Visualization specifications (Vega-Lite format)"
    )

    # === METADATA ===
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")

    agents_used: List[str] = Field(
        default_factory=list,
        description="Agents that contributed to response"
    )

    execution_time_ms: int = Field(..., description="Total execution time")

    tokens_used: Optional[int] = Field(None, description="Total LLM tokens")

    # === FOLLOW-UP ===
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )

    related_queries: List[str] = Field(
        default_factory=list,
        description="Related historical queries"
    )

    # === ERRORS (if status != "completed") ===
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings"
    )

    # === PROVENANCE ===
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )

    class Config:
        extra = "allow"
```

---

## NLP Layer → Orchestrator Contract

### Parsed Query Structure

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class ParsedQuery(BaseModel):
    """
    Output from NLP layer, input to Orchestrator.

    NLP pipeline: tokenization → entity extraction → intent classification
    """

    # === IDENTIFICATION ===
    query_id: str = Field(..., description="Query identifier")
    original_query: str = Field(..., description="Original user query")

    # === INTENT CLASSIFICATION ===
    primary_intent: Literal[
        "causal",
        "exploratory",
        "comparative",
        "trend",
        "what_if",
        "ml_training",
        "model_deployment",
        "monitoring"
    ] = Field(..., description="Primary query intent")

    intent_confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")

    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Additional detected intents"
    )

    # === ENTITY EXTRACTION ===
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities by type"
    )
    # Example:
    # {
    #   "brands": ["Kisqali", "Fabhalta"],
    #   "kpis": ["NRx", "TRx"],
    #   "regions": ["Midwest", "Northeast"],
    #   "time_periods": ["Q1 2024"],
    #   "hcp_segments": ["high-volume", "academic"]
    # }

    # === QUERY STRUCTURE ===
    query_type: Literal["simple", "compound", "multi_part"] = Field(
        default="simple",
        description="Query structural complexity"
    )

    dependencies: List[str] = Field(
        default_factory=list,
        description="Query part dependencies (for multi-part queries)"
    )

    # === CONSTRAINTS ===
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted filters and constraints"
    )

    # === AMBIGUITY ===
    ambiguous_terms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Terms needing clarification"
    )

    requires_clarification: bool = Field(
        default=False,
        description="Whether query needs user clarification"
    )

    # === METADATA ===
    parse_time_ms: int = Field(..., description="NLP processing time")

    models_used: List[str] = Field(
        default_factory=list,
        description="NLP models used"
    )
```

### NLP → Orchestrator Handoff

```yaml
# Example NLP → Orchestrator handoff
nlp_to_orchestrator:
  query_id: qry_abc123
  original_query: "What caused the drop in NRx for Kisqali in Midwest?"

  # Intent
  primary_intent: causal
  intent_confidence: 0.92
  secondary_intents: []

  # Entities
  entities:
    brands: ["Kisqali"]
    kpis: ["NRx"]
    regions: ["Midwest"]
    time_periods: ["recent"]  # Implicit
    direction: ["drop"]

  # Structure
  query_type: simple
  dependencies: []

  # Filters
  filters:
    brand: "Kisqali"
    kpi: "NRx"
    region: "Midwest"
    time_range: "last_90_days"

  # Ambiguity
  ambiguous_terms: []
  requires_clarification: false

  # Metadata
  parse_time_ms: 340
  models_used: ["intent_classifier_v3", "entity_extractor_v2"]
```

---

## Orchestrator → RAG Contract

### RAG Retrieval Request

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class RAGRetrievalRequest(BaseModel):
    """
    Request structure for RAG retrieval.

    Used by agents to retrieve relevant context.
    """

    # === QUERY ===
    query: str = Field(..., description="Query text")
    query_embedding: Optional[List[float]] = Field(
        None,
        description="Pre-computed query embedding (optional)"
    )

    # === INDEXES ===
    indexes: List[Literal[
        "causal_paths",
        "agent_activities",
        "business_metrics",
        "triggers",
        "conversations"
    ]] = Field(
        default=["causal_paths", "agent_activities"],
        description="RAG indexes to search"
    )

    # === RETRIEVAL STRATEGY ===
    retrieval_mode: Literal["hybrid", "dense", "sparse", "graph"] = Field(
        default="hybrid",
        description="Retrieval strategy"
    )

    # === FILTERS ===
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters"
    )
    # Example: {"brand": "Kisqali", "kpi": "NRx"}

    # === LIMITS ===
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results per index")

    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )

    # === RERANKING ===
    enable_reranking: bool = Field(
        default=True,
        description="Whether to rerank results"
    )

    reranker_model: Optional[str] = Field(
        default="cross-encoder-v1",
        description="Reranker model to use"
    )
```

### RAG Retrieval Response

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RAGRetrievalResponse(BaseModel):
    """
    Response structure from RAG retrieval.
    """

    # === RESULTS ===
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved results"
    )
    # Each result:
    # {
    #   "content": str,
    #   "metadata": Dict[str, Any],
    #   "similarity": float,
    #   "index": str,
    #   "chunk_id": str
    # }

    # === STATISTICS ===
    total_retrieved: int = Field(..., description="Total results retrieved")

    results_by_index: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per index"
    )

    retrieval_time_ms: int = Field(..., description="Retrieval time")

    # === QUALITY METRICS ===
    avg_similarity: float = Field(..., ge=0.0, le=1.0, description="Average similarity")

    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well query is covered"
    )

    # === RECOMMENDATIONS ===
    sufficient_context: bool = Field(
        ...,
        description="Whether sufficient context was retrieved"
    )

    suggested_refinements: List[str] = Field(
        default_factory=list,
        description="Suggestions to improve retrieval"
    )
```

---

## Agent → Memory Contract

### Memory Types

E2I uses 4 memory types based on cognitive science:

| Memory Type | Purpose | Storage | Lifespan |
|-------------|---------|---------|----------|
| **Working Memory** | Session-scoped temporary context | Redis/In-memory | Session duration |
| **Episodic Memory** | Historical query-response pairs | Vector DB | Permanent |
| **Procedural Memory** | Successful patterns and workflows | Vector DB | Permanent |
| **Semantic Memory** | Domain knowledge graph | Neo4j Graph DB | Permanent |

### Memory Operations Contract

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class MemoryWriteRequest(BaseModel):
    """
    Request to write to memory system.
    """

    # === MEMORY TYPE ===
    memory_type: Literal["working", "episodic", "procedural", "semantic"] = Field(
        ...,
        description="Type of memory to write"
    )

    # === IDENTIFICATION ===
    session_id: str = Field(..., description="Session identifier")
    agent_name: str = Field(..., description="Agent writing memory")

    # === DATA ===
    data: Dict[str, Any] = Field(..., description="Data to store")

    # === METADATA ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    # === OPTIONS ===
    ttl_seconds: Optional[int] = Field(
        None,
        description="Time-to-live (working memory only)"
    )

    importance: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Memory importance (affects retention)"
    )

class MemoryReadRequest(BaseModel):
    """
    Request to read from memory system.
    """

    # === MEMORY TYPE ===
    memory_type: Literal["working", "episodic", "procedural", "semantic"] = Field(
        ...,
        description="Type of memory to read"
    )

    # === QUERY ===
    query: Optional[str] = Field(
        None,
        description="Query for semantic search (episodic/procedural/semantic)"
    )

    session_id: Optional[str] = Field(
        None,
        description="Session ID (working memory only)"
    )

    # === FILTERS ===
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters"
    )

    # === LIMITS ===
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")

    min_relevance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance threshold"
    )

class MemoryResponse(BaseModel):
    """
    Response from memory system.
    """

    # === RESULTS ===
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved memory items"
    )

    # === STATISTICS ===
    total_retrieved: int = Field(..., description="Total items retrieved")

    retrieval_time_ms: int = Field(..., description="Retrieval time")

    # === QUALITY ===
    avg_relevance: float = Field(..., ge=0.0, le=1.0, description="Average relevance")

    cache_hit: bool = Field(default=False, description="Whether result was cached")
```

---

## Agent → Causal Engine Contract

### Causal Analysis Request

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class CausalAnalysisRequest(BaseModel):
    """
    Request for causal inference from Causal Engine.

    Used by: causal_impact, experiment_designer, heterogeneous_optimizer
    """

    # === CAUSAL QUERY ===
    treatment_var: str = Field(..., description="Treatment variable name")
    outcome_var: str = Field(..., description="Outcome variable name")
    confounders: List[str] = Field(..., description="Confounder variables")

    # === DATA ===
    data_source: str = Field(..., description="Data source identifier")

    sample_size: Optional[int] = Field(
        None,
        description="Sample size (if known)"
    )

    # === METHOD ===
    estimation_method: str = Field(
        default="backdoor.econml.dml.CausalForestDML",
        description="Causal estimation method"
    )

    # === VALIDATION ===
    refutation_tests: List[str] = Field(
        default=["random_common_cause", "placebo_treatment", "data_subset"],
        description="Refutation tests to run"
    )

    # === SENSITIVITY ===
    sensitivity_analysis: bool = Field(
        default=True,
        description="Whether to run sensitivity analysis"
    )

    # === SEGMENTATION (for CATE) ===
    effect_modifiers: Optional[List[str]] = Field(
        None,
        description="Variables for heterogeneous effect estimation"
    )

    # === OPTIONS ===
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )

    bootstrap_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Bootstrap samples for inference"
    )

    # === DISCOVERY (V4.4+) ===
    auto_discover_graph: bool = Field(
        default=False,
        description="Enable automatic DAG structure learning"
    )

    discovery_method: Optional[Literal["ges", "pc", "ensemble"]] = Field(
        None,
        description="Discovery method: 'ges' (Greedy Equivalence Search), 'pc' (Peter-Clark), 'ensemble' (both with voting)"
    )

    discovery_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Discovery configuration: ensemble_threshold (0.5), alpha (0.05), max_iterations (1000)"
    )

class CausalAnalysisResponse(BaseModel):
    """
    Response from Causal Engine.
    """

    # === EFFECT ESTIMATE ===
    ate: float = Field(..., description="Average Treatment Effect")

    confidence_interval: tuple[float, float] = Field(
        ...,
        description="Confidence interval"
    )

    p_value: Optional[float] = Field(None, description="P-value (if applicable)")

    standard_error: float = Field(..., description="Standard error")

    # === HETEROGENEOUS EFFECTS (if requested) ===
    cate: Optional[Dict[str, float]] = Field(
        None,
        description="Conditional Average Treatment Effects by segment"
    )

    # === VALIDATION ===
    refutation_results: Dict[str, bool] = Field(
        default_factory=dict,
        description="Refutation test results (test_name: passed)"
    )

    all_refutations_passed: bool = Field(..., description="Overall refutation status")

    # === SENSITIVITY ===
    sensitivity_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensitivity analysis results"
    )

    # === DIAGNOSTICS ===
    overlap_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Propensity score overlap quality"
    )

    balance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Covariate balance metrics"
    )

    # === METADATA ===
    method_used: str = Field(..., description="Estimation method used")

    sample_size: int = Field(..., description="Effective sample size")

    computation_time_ms: int = Field(..., description="Computation time")

    # === DISCOVERY RESULTS (V4.4+) ===
    discovery_performed: bool = Field(
        default=False,
        description="Whether auto-discovery was performed"
    )

    discovered_graph: Optional[Dict[str, Any]] = Field(
        None,
        description="Discovered DAG structure: {nodes: List[str], edges: List[Tuple[str, str]], edge_confidences: Dict}"
    )

    discovery_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall discovery confidence score"
    )

    discovery_gate_decision: Optional[Literal["accept", "augment", "review", "reject"]] = Field(
        None,
        description="Gate decision: accept (>=0.8), augment (0.5-0.8), review (0.3-0.5), reject (<0.3)"
    )

    discovery_algorithms_used: List[str] = Field(
        default_factory=list,
        description="Algorithms used: ['ges', 'pc']"
    )

    augmented_edges: List[tuple[str, str]] = Field(
        default_factory=list,
        description="High-confidence discovered edges added to manual DAG (if gate=augment)"
    )
```

---

## Causal Engine Discovery Contract (V4.4)

This section defines the V4.4 causal discovery integration interfaces for the Discovery Runner, DriverRanker, and DiscoveryGate components.

### Discovery Runner Interface

The Discovery Runner orchestrates ensemble-based causal structure discovery using multiple algorithms.

```python
from typing import Dict, Any, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import networkx as nx

class DiscoveryAlgorithm(str, Enum):
    """Supported causal discovery algorithms."""
    GES = "ges"      # Greedy Equivalence Search (score-based)
    PC = "pc"        # Peter-Clark (constraint-based)
    FCI = "fci"      # Fast Causal Inference (handles latent confounders)


@dataclass
class DiscoveryConfig:
    """
    Configuration for causal structure discovery.

    Used by: Feature Analyzer, Causal Impact, Orchestrator
    """

    # === ALGORITHM SELECTION ===
    algorithms: List[DiscoveryAlgorithm] = None  # Default: [GES, PC]
    ensemble_threshold: float = 0.5  # Minimum agreement for edge inclusion

    # === STATISTICAL PARAMETERS ===
    alpha: float = 0.05  # Significance level for independence tests
    max_iterations: int = 1000  # Max iterations for score-based methods

    # === CONSTRAINTS ===
    forbidden_edges: Optional[List[Tuple[str, str]]] = None
    required_edges: Optional[List[Tuple[str, str]]] = None

    # === PERFORMANCE ===
    max_conditioning_set_size: Optional[int] = None  # Limit conditioning set size
    timeout_seconds: int = 300  # Per-algorithm timeout


@dataclass
class AlgorithmResult:
    """
    Result from a single discovery algorithm.
    """

    algorithm: DiscoveryAlgorithm
    adjacency_matrix: List[List[int]]  # Binary adjacency matrix
    edge_list: List[Tuple[str, str]]  # (source, target) pairs
    edge_confidences: Dict[str, float]  # "source->target": confidence
    edge_types: Dict[str, str]  # "source->target": DIRECTED|BIDIRECTED|UNDIRECTED
    score: Optional[float]  # BIC score for score-based methods
    execution_time_ms: int
    converged: bool


@dataclass
class DiscoveryResult:
    """
    Complete result from ensemble causal discovery.

    Output from DiscoveryRunner.run()
    """

    # === ENSEMBLE DAG ===
    ensemble_dag: nx.DiGraph  # Final consensus DAG
    adjacency_matrix: List[List[int]]  # Binary adjacency matrix
    node_names: List[str]  # Variable names in matrix order
    edge_list: List[Tuple[str, str]]  # Final edges

    # === EDGE METADATA ===
    edge_confidences: Dict[str, float]  # Agreement scores per edge
    edge_types: Dict[str, str]  # DIRECTED, BIDIRECTED, UNDIRECTED

    # === ALGORITHM RESULTS ===
    algorithm_results: List[AlgorithmResult]  # Per-algorithm details
    algorithms_used: List[str]

    # === GATE DECISION ===
    gate_decision: Literal["accept", "augment", "review", "reject"]
    gate_confidence: float
    high_confidence_edges: List[Tuple[str, str]]  # Edges with >0.8 agreement

    # === METADATA ===
    total_execution_time_ms: int
    discovery_timestamp: str


# Discovery Runner API
async def run_discovery(
    data: pd.DataFrame,
    config: DiscoveryConfig,
    target_variable: Optional[str] = None,
) -> DiscoveryResult:
    """
    Run ensemble causal discovery on data.

    Args:
        data: DataFrame with variables as columns
        config: Discovery configuration
        target_variable: Optional target to ensure paths are identified

    Returns:
        DiscoveryResult with ensemble DAG and algorithm details
    """
    ...
```

### DriverRanker Interface

The DriverRanker combines causal structure with SHAP importance to identify true causal drivers.

```python
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
import numpy as np


@dataclass
class DriverRankingResult:
    """
    Result from causal driver ranking.

    Combines causal structure with SHAP importance to identify
    true causal drivers vs spurious correlations.
    """

    # === RANKINGS ===
    causal_rankings: List[Tuple[str, float]]  # (feature, causal_score) sorted
    shap_rankings: List[Tuple[str, float]]  # (feature, shap_importance) sorted
    combined_rankings: List[Tuple[str, float]]  # Weighted combination

    # === DRIVER CATEGORIES ===
    true_causal_drivers: List[str]  # High causal + high SHAP
    predictive_only: List[str]  # High SHAP but no causal path
    confounded_features: List[str]  # May be spurious

    # === CORRELATIONS ===
    rank_correlation: float  # Spearman correlation between rankings
    divergent_features: List[Dict[str, Any]]  # Features where rankings differ significantly
    # Each: {"feature": str, "causal_rank": int, "shap_rank": int, "divergence": float}

    # === CAUSAL PATHS ===
    direct_causes: List[str]  # Direct parents of target
    indirect_causes: List[str]  # Ancestors with indirect paths
    path_lengths: Dict[str, int]  # Feature -> shortest path to target

    # === METADATA ===
    target_variable: str
    features_analyzed: int
    algorithm_used: str


# DriverRanker API
def rank_drivers(
    dag: nx.DiGraph,
    target: str,
    shap_values: np.ndarray,
    feature_names: List[str],
    weighting: float = 0.5,  # Weight for causal vs SHAP
) -> DriverRankingResult:
    """
    Rank features as causal drivers combining structure and importance.

    Args:
        dag: Discovered or provided causal DAG
        target: Target variable name
        shap_values: SHAP importance values (shape: n_samples x n_features)
        feature_names: Feature names matching SHAP columns
        weighting: Weight for causal score (1-weighting for SHAP)

    Returns:
        DriverRankingResult with rankings and categorization
    """
    ...
```

### DiscoveryGate Interface

The DiscoveryGate evaluates discovery quality and determines whether to accept, augment, review, or reject results.

```python
from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass


class GateDecision(str, Enum):
    """Discovery gate decisions."""
    ACCEPT = "accept"    # Confidence >= 0.8, use discovered DAG directly
    AUGMENT = "augment"  # Confidence 0.5-0.8, add high-confidence edges to manual DAG
    REVIEW = "review"    # Confidence 0.3-0.5, flag for human review
    REJECT = "reject"    # Confidence < 0.3, use manual DAG only


@dataclass
class GateEvaluation:
    """
    Result from discovery gate evaluation.

    Determines how discovered structure should be used.
    """

    # === DECISION ===
    decision: GateDecision
    confidence: float  # Overall confidence score (0.0 to 1.0)

    # === EDGE ANALYSIS ===
    high_confidence_edges: List[Tuple[str, str]]  # Edges with >0.8 agreement
    low_confidence_edges: List[Tuple[str, str]]  # Edges with <0.5 agreement
    conflicting_edges: List[Tuple[str, str]]  # Edges where algorithms disagree

    # === VALIDATION ===
    expected_edges_found: int  # If expected_edges provided
    expected_edges_missing: int
    unexpected_edges: int

    # === QUALITY METRICS ===
    algorithm_agreement: float  # Average pairwise agreement
    structure_stability: float  # Sensitivity to bootstrap samples
    acyclicity_score: float  # 1.0 if DAG, <1.0 if cycles detected

    # === RECOMMENDATIONS ===
    recommendation: str  # Human-readable recommendation
    edges_to_add: List[Tuple[str, str]]  # If decision is AUGMENT
    edges_to_review: List[Tuple[str, str]]  # Edges needing human review


# DiscoveryGate API
def evaluate_discovery(
    result: DiscoveryResult,
    expected_edges: Optional[List[Tuple[str, str]]] = None,
    required_paths: Optional[List[Tuple[str, str]]] = None,
) -> GateEvaluation:
    """
    Evaluate discovery result quality and determine action.

    Args:
        result: DiscoveryResult from run_discovery()
        expected_edges: Optional known edges to validate against
        required_paths: Optional required paths (e.g., treatment->outcome)

    Returns:
        GateEvaluation with decision and recommendations
    """
    ...
```

### Structural Drift Detection Contract (V4.4)

The Structural Drift Detector compares causal DAG structures between baseline and current periods.

```python
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass


@dataclass
class StructuralDriftInput:
    """
    Input for structural drift detection.

    Compares baseline vs current causal DAG structures.
    """

    # === BASELINE DAG ===
    baseline_adjacency: List[List[int]]  # Binary adjacency matrix
    baseline_edge_types: Dict[str, str]  # "A->B": DIRECTED|BIDIRECTED

    # === CURRENT DAG ===
    current_adjacency: List[List[int]]  # Binary adjacency matrix
    current_edge_types: Dict[str, str]  # "A->B": DIRECTED|BIDIRECTED

    # === NODE MAPPING ===
    node_names: List[str]  # Shared variable names


@dataclass
class StructuralDriftResult:
    """
    Result from structural drift detection.

    Output from StructuralDriftNode.execute()
    """

    # === DRIFT DETECTION ===
    detected: bool  # Whether significant drift was found
    drift_score: float  # Percentage of edges changed (0.0 to 1.0)

    # === EDGE CHANGES ===
    added_edges: List[str]  # Format: "source->target"
    removed_edges: List[str]  # Format: "source->target"
    stable_edges: List[str]  # Unchanged edges

    # === EDGE TYPE CHANGES ===
    edge_type_changes: List[Dict[str, Any]]
    # Each: {"edge": "A->B", "from": "DIRECTED", "to": "BIDIRECTED"}

    # === SEVERITY ===
    severity: Literal["none", "low", "medium", "high", "critical"]
    # none: <5%, low: 5-10%, medium: 10-20%, high: 20-30%, critical: >30%

    # === CRITICAL PATH ANALYSIS ===
    critical_path_broken: bool  # If treatment->outcome path affected
    affected_paths: List[List[str]]  # Paths that changed

    # === RECOMMENDATIONS ===
    recommendation: str  # Action recommendation based on severity
```

---

## Agent → MLOps Tools Integration

### MLflow Integration Contract

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class MLflowExperimentRequest(BaseModel):
    """
    Request to create/update MLflow experiment.

    Used by: model_trainer, feature_analyzer, model_deployer
    """

    # === EXPERIMENT ===
    experiment_name: str = Field(..., description="Experiment name")

    experiment_id: Optional[str] = Field(
        None,
        description="Experiment ID (if continuing)"
    )

    # === RUN ===
    run_name: str = Field(..., description="Run name")

    # === PARAMETERS ===
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters"
    )

    # === METRICS ===
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )

    # === ARTIFACTS ===
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Artifact paths (name: local_path)"
    )

    # === TAGS ===
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata tags"
    )

    # === MODEL REGISTRATION ===
    register_model: bool = Field(
        default=False,
        description="Whether to register model"
    )

    model_name: Optional[str] = Field(
        None,
        description="Model registry name"
    )

class OpikTraceRequest(BaseModel):
    """
    Request to create Opik trace for observability.

    Used by: ALL agents via observability_connector
    """

    # === TRACE ===
    trace_id: str = Field(..., description="Unique trace identifier")

    parent_trace_id: Optional[str] = Field(
        None,
        description="Parent trace (for nested spans)"
    )

    # === SPAN ===
    span_name: str = Field(..., description="Span name (agent name)")

    span_type: Literal["agent", "tool", "llm", "retrieval"] = Field(
        default="agent",
        description="Span type"
    )

    # === DATA ===
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data"
    )

    output_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data"
    )

    # === METRICS ===
    duration_ms: int = Field(..., description="Span duration")

    tokens_used: Optional[int] = Field(None, description="LLM tokens (if applicable)")

    cost_dollars: Optional[float] = Field(None, description="Estimated cost")

    # === STATUS ===
    status: Literal["success", "error", "timeout"] = Field(..., description="Span status")

    error_message: Optional[str] = Field(None, description="Error message (if failed)")

    # === METADATA ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
```

---

## Error Propagation and Recovery

### Global Error Handling Strategy

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class SystemError(BaseModel):
    """
    Standard system-level error structure.
    """

    # === IDENTIFICATION ===
    error_id: str = Field(..., description="Unique error identifier")

    timestamp: str = Field(..., description="ISO timestamp")

    # === CLASSIFICATION ===
    error_type: Literal[
        "validation_error",
        "timeout_error",
        "resource_error",
        "database_error",
        "llm_error",
        "computation_error",
        "integration_error",
        "unknown_error"
    ] = Field(..., description="Error type")

    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Error severity"
    )

    # === SOURCE ===
    component: str = Field(..., description="Component where error occurred")

    layer: Literal["frontend", "api", "nlp", "orchestrator", "agent", "rag", "memory", "database", "mlops"] = Field(
        ...,
        description="System layer"
    )

    agent: Optional[str] = Field(None, description="Agent name (if applicable)")

    # === DETAILS ===
    message: str = Field(..., description="Human-readable error message")

    technical_details: Optional[str] = Field(
        None,
        description="Technical error details"
    )

    stack_trace: Optional[str] = Field(
        None,
        description="Stack trace (dev mode only)"
    )

    # === RECOVERY ===
    recoverable: bool = Field(..., description="Whether error is recoverable")

    retry_recommended: bool = Field(
        default=False,
        description="Whether retry is recommended"
    )

    fallback_available: bool = Field(
        default=False,
        description="Whether fallback exists"
    )

    # === CONTEXT ===
    query_id: Optional[str] = Field(None, description="Query ID")

    session_id: Optional[str] = Field(None, description="Session ID")

    request_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Request data (sanitized)"
    )

class ErrorRecoveryPolicy(BaseModel):
    """
    Policy for error recovery at system level.
    """

    # === RETRY STRATEGY ===
    enable_retry: bool = Field(default=True, description="Enable automatic retry")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial retry delay (exponential backoff)"
    )

    retryable_errors: List[str] = Field(
        default=[
            "timeout_error",
            "resource_error",
            "database_error",
            "llm_error"
        ],
        description="Error types that can be retried"
    )

    # === FALLBACK STRATEGY ===
    enable_fallback: bool = Field(
        default=True,
        description="Enable fallback mechanisms"
    )

    fallback_mode: Literal["graceful_degradation", "alternative_path", "cached_response"] = Field(
        default="graceful_degradation",
        description="Fallback mode"
    )

    # === CIRCUIT BREAKER ===
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )

    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Failures before circuit opens"
    )

    recovery_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Time before circuit half-opens"
    )

    # === NOTIFICATION ===
    notify_on_critical: bool = Field(
        default=True,
        description="Send notifications for critical errors"
    )

    notification_channels: List[str] = Field(
        default=["slack", "email"],
        description="Notification channels"
    )
```

### Error Recovery Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ERROR RECOVERY FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Error Occurs                                                        │
│       │                                                              │
│       ├─► Classify error (type, severity, component)                │
│       │                                                              │
│       ├─► Check if recoverable                                      │
│       │        │                                                     │
│       │        ├─► YES: Attempt recovery                            │
│       │        │      │                                              │
│       │        │      ├─► Retry? ──► Exponential backoff            │
│       │        │      │                                              │
│       │        │      ├─► Fallback? ──┬─► Graceful degradation      │
│       │        │      │                ├─► Alternative path          │
│       │        │      │                └─► Cached response           │
│       │        │      │                                              │
│       │        │      └─► Circuit breaker check                     │
│       │        │                                                     │
│       │        └─► NO: Fail gracefully                              │
│       │               │                                              │
│       │               ├─► Log error with full context               │
│       │               ├─► Notify if critical                        │
│       │               └─► Return user-friendly error                │
│       │                                                              │
│       └─► Update system health metrics                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cognitive RAG Integration Contract

The Cognitive RAG system implements a **4-Phase Cognitive Cycle** powered by DSPy. This section defines the integration contracts for the cognitive workflow.

### CognitiveState Dataclass

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class MemoryType(str, Enum):
    """Types of memory in cognitive architecture."""
    WORKING = "working"       # Redis + LangGraph MemorySaver
    EPISODIC = "episodic"     # Supabase + pgvector
    SEMANTIC = "semantic"     # FalkorDB + Graphity
    PROCEDURAL = "procedural" # Supabase + pgvector


class HopType(str, Enum):
    """Multi-hop retrieval strategies in Investigator phase."""
    SEMANTIC = "semantic"       # Vector similarity search
    GRAPH = "graph"             # Graph traversal in FalkorDB
    EPISODIC = "episodic"       # Similar past conversations
    PROCEDURAL = "procedural"   # Learned procedures
    SQL = "sql"                 # Direct database query
    NONE = "none"               # Investigation complete


@dataclass
class Evidence:
    """Single piece of evidence gathered during investigation."""
    content: str
    source: str
    source_id: str
    hop_type: HopType
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CognitiveState:
    """
    Complete state for 4-phase cognitive workflow.

    Flows through: Summarizer → Investigator → Agent → Reflector
    """

    # === INPUT ===
    user_query: str
    conversation_id: str

    # === PHASE 1: SUMMARIZER OUTPUTS ===
    compressed_history: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    detected_intent: str = ""
    rewritten_query: str = ""

    # === PHASE 2: INVESTIGATOR OUTPUTS ===
    investigation_goal: str = ""
    evidence_board: List[Evidence] = field(default_factory=list)
    hop_count: int = 0
    sufficient_evidence: bool = False
    max_hops: int = 5

    # === PHASE 3: AGENT OUTPUTS ===
    response: str = ""
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    routed_agents: List[str] = field(default_factory=list)

    # === PHASE 4: REFLECTOR OUTPUTS ===
    worth_remembering: bool = False
    extracted_facts: List[Dict] = field(default_factory=list)
    learned_procedures: List[Dict] = field(default_factory=list)
    dspy_signals: List[Dict] = field(default_factory=list)

    # === EXECUTION METADATA ===
    phase_latencies: Dict[str, int] = field(default_factory=dict)
    total_latency_ms: int = 0
    errors: List[Dict] = field(default_factory=list)
```

### Memory Backend Contracts

```python
from typing import Dict, Any, List, Optional, Protocol
from pydantic import BaseModel, Field

class MemoryBackend(Protocol):
    """Protocol for cognitive memory backends."""

    async def read(
        self,
        query: str,
        memory_type: MemoryType,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Read from memory."""
        ...

    async def write(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Dict[str, Any]
    ) -> str:
        """Write to memory, return ID."""
        ...


class MemoryBackendConfig(BaseModel):
    """Configuration for memory backends."""

    working: Dict[str, Any] = Field(
        default={
            "backend": "redis",
            "ttl_seconds": 3600,
            "max_items": 100
        },
        description="Working memory config (Redis + LangGraph)"
    )

    episodic: Dict[str, Any] = Field(
        default={
            "backend": "supabase_pgvector",
            "embedding_dim": 384,
            "distance_metric": "cosine"
        },
        description="Episodic memory config (Supabase pgvector)"
    )

    semantic: Dict[str, Any] = Field(
        default={
            "backend": "falkordb",
            "graph_name": "e2i_knowledge",
            "use_graphity": True
        },
        description="Semantic memory config (FalkorDB Graphity)"
    )

    procedural: Dict[str, Any] = Field(
        default={
            "backend": "supabase_pgvector",
            "embedding_model": "all-MiniLM-L6-v2",
            "collection": "procedures"
        },
        description="Procedural memory config (Supabase pgvector)"
    )
```

### DSPy Signature Contracts

```python
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

class DSPySignatureContract(BaseModel):
    """Contract for DSPy signature definition."""

    signature_name: str = Field(..., description="Signature class name")
    phase: Literal["summarizer", "investigator", "agent", "reflector"] = Field(
        ...,
        description="Cognitive phase"
    )

    # Input/Output specifications
    input_fields: List[Dict[str, str]] = Field(
        ...,
        description="Input field definitions"
    )
    output_fields: List[Dict[str, str]] = Field(
        ...,
        description="Output field definitions"
    )

    # Optimization
    optimizer: Literal["MIPROv2", "BootstrapFewShot", "COPRO"] = Field(
        ...,
        description="DSPy optimizer to use"
    )

    # Performance targets
    latency_budget_ms: int = Field(..., description="Max latency in ms")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")


# 11 DSPy Signatures Registry
COGNITIVE_SIGNATURES: Dict[str, DSPySignatureContract] = {
    # Phase 1: Summarizer
    "QueryRewriteSignature": DSPySignatureContract(
        signature_name="QueryRewriteSignature",
        phase="summarizer",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "history", "type": "str"}
        ],
        output_fields=[
            {"name": "rewritten_query", "type": "str"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=100,
        min_confidence=0.8
    ),
    "EntityExtractionSignature": DSPySignatureContract(
        signature_name="EntityExtractionSignature",
        phase="summarizer",
        input_fields=[
            {"name": "query", "type": "str"}
        ],
        output_fields=[
            {"name": "entities", "type": "List[str]"},
            {"name": "entity_types", "type": "Dict[str, str]"}
        ],
        optimizer="BootstrapFewShot",
        latency_budget_ms=80,
        min_confidence=0.85
    ),
    "IntentClassificationSignature": DSPySignatureContract(
        signature_name="IntentClassificationSignature",
        phase="summarizer",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "entities", "type": "List[str]"}
        ],
        output_fields=[
            {"name": "intent", "type": "str"},
            {"name": "confidence", "type": "float"}
        ],
        optimizer="COPRO",
        latency_budget_ms=60,
        min_confidence=0.8
    ),

    # Phase 2: Investigator
    "InvestigationPlanSignature": DSPySignatureContract(
        signature_name="InvestigationPlanSignature",
        phase="investigator",
        input_fields=[
            {"name": "rewritten_query", "type": "str"},
            {"name": "detected_intent", "type": "str"},
            {"name": "entities", "type": "List[str]"}
        ],
        output_fields=[
            {"name": "investigation_goal", "type": "str"},
            {"name": "initial_hop", "type": "str"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=120,
        min_confidence=0.75
    ),
    "HopDecisionSignature": DSPySignatureContract(
        signature_name="HopDecisionSignature",
        phase="investigator",
        input_fields=[
            {"name": "goal", "type": "str"},
            {"name": "evidence_so_far", "type": "List[Evidence]"},
            {"name": "hop_count", "type": "int"}
        ],
        output_fields=[
            {"name": "next_hop", "type": "HopType"},
            {"name": "hop_query", "type": "str"},
            {"name": "sufficient", "type": "bool"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=80,
        min_confidence=0.7
    ),
    "EvidenceRelevanceSignature": DSPySignatureContract(
        signature_name="EvidenceRelevanceSignature",
        phase="investigator",
        input_fields=[
            {"name": "evidence", "type": "Evidence"},
            {"name": "goal", "type": "str"}
        ],
        output_fields=[
            {"name": "relevance_score", "type": "float"},
            {"name": "keep", "type": "bool"}
        ],
        optimizer="BootstrapFewShot",
        latency_budget_ms=40,
        min_confidence=0.8
    ),

    # Phase 3: Agent
    "EvidenceSynthesisSignature": DSPySignatureContract(
        signature_name="EvidenceSynthesisSignature",
        phase="agent",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "evidence_board", "type": "List[Evidence]"},
            {"name": "user_expertise", "type": "str"}
        ],
        output_fields=[
            {"name": "response", "type": "str"},
            {"name": "confidence", "type": "float"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=500,
        min_confidence=0.75
    ),
    "AgentRoutingSignature": DSPySignatureContract(
        signature_name="AgentRoutingSignature",
        phase="agent",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "intent", "type": "str"},
            {"name": "available_agents", "type": "List[str]"}
        ],
        output_fields=[
            {"name": "routed_agents", "type": "List[str]"},
            {"name": "routing_rationale", "type": "str"}
        ],
        optimizer="COPRO",
        latency_budget_ms=60,
        min_confidence=0.85
    ),
    "VisualizationConfigSignature": DSPySignatureContract(
        signature_name="VisualizationConfigSignature",
        phase="agent",
        input_fields=[
            {"name": "response", "type": "str"},
            {"name": "data_types", "type": "List[str]"}
        ],
        output_fields=[
            {"name": "viz_type", "type": "str"},
            {"name": "viz_config", "type": "Dict[str, Any]"}
        ],
        optimizer="BootstrapFewShot",
        latency_budget_ms=80,
        min_confidence=0.7
    ),

    # Phase 4: Reflector
    "MemoryWorthinessSignature": DSPySignatureContract(
        signature_name="MemoryWorthinessSignature",
        phase="reflector",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "response", "type": "str"},
            {"name": "evidence_used", "type": "int"}
        ],
        output_fields=[
            {"name": "worth_remembering", "type": "bool"},
            {"name": "memory_type", "type": "MemoryType"},
            {"name": "extracted_facts", "type": "List[Dict]"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=100,
        min_confidence=0.8
    ),
    "ProcedureLearningSignature": DSPySignatureContract(
        signature_name="ProcedureLearningSignature",
        phase="reflector",
        input_fields=[
            {"name": "query", "type": "str"},
            {"name": "successful_hops", "type": "List[HopType]"},
            {"name": "final_confidence", "type": "float"}
        ],
        output_fields=[
            {"name": "learned_procedure", "type": "Dict"},
            {"name": "should_save", "type": "bool"}
        ],
        optimizer="MIPROv2",
        latency_budget_ms=120,
        min_confidence=0.75
    ),
}


# ============================================================================
# DSPy SIGNAL FLOW CONTRACTS
# ============================================================================

class TrainingSignal(BaseModel):
    """
    Training signal for DSPy optimization.

    Generated by Sender agents, collected by Hub (orchestrator),
    distributed to Recipients after optimization.

    From: E2I DSPy Feedback Learner Architecture V2
    """
    signal_id: str = Field(..., regex=r"^sig_[a-z0-9]{16}$")
    timestamp: datetime
    source_agent: str = Field(..., description="Agent that generated signal")
    source_type: Literal["hub", "hybrid", "sender", "recipient"] = Field(
        ..., description="DSPy role of source agent"
    )
    signature_name: str = Field(..., description="Which DSPy signature this is for")

    # Signal content
    input_data: Dict[str, Any] = Field(..., description="Input to the signature")
    output_data: Dict[str, Any] = Field(..., description="Output from signature")
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="If available")

    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Signal quality")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    latency_ms: int = Field(..., description="Execution latency")
    user_feedback: Optional[int] = Field(None, ge=-1, le=1, description="-1/0/1 feedback")

    # Metadata
    session_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalFlowContract(BaseModel):
    """
    Contract defining DSPy signal flow between agents.

    Defines the Hub-Sender-Recipient pattern for DSPy optimization:
    1. Senders generate TrainingSignals during execution
    2. Hub (orchestrator) collects and batches signals
    3. Hub triggers optimization when thresholds met
    4. Hub distributes optimized prompts to Recipients

    From: E2I DSPy Feedback Learner Architecture V2
    """

    # Agent roles
    hub_agent: str = Field(
        default="orchestrator",
        description="Agent that coordinates DSPy optimization"
    )
    sender_agents: List[str] = Field(
        default=[
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "prediction_synthesizer",
        ],
        description="Agents that generate training signals"
    )
    recipient_agents: List[str] = Field(
        default=[
            "health_score",
            "resource_optimizer",
            "explainer",
        ],
        description="Agents that receive optimized prompts"
    )
    hybrid_agents: List[str] = Field(
        default=[
            "tool_composer",
            "feedback_learner",
        ],
        description="Agents that both send and receive"
    )

    # Optimization thresholds
    min_signals_for_optimization: int = Field(
        default=100,
        ge=10,
        description="Minimum signals before triggering optimization"
    )
    optimization_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum hours between optimization cycles"
    )

    # Quality gates
    min_signal_quality: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="Minimum quality score for signal inclusion"
    )
    min_prompt_improvement: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="Minimum improvement to distribute new prompts"
    )

    # Signature-to-agent mapping
    signature_assignments: Dict[str, List[str]] = Field(
        default={
            "QueryRewriteSignature": ["orchestrator", "explainer"],
            "EntityExtractionSignature": ["orchestrator"],
            "IntentClassificationSignature": ["orchestrator"],
            "InvestigationPlanSignature": ["experiment_designer"],
            "HopDecisionSignature": ["drift_monitor"],
            "EvidenceRelevanceSignature": ["gap_analyzer"],
            "EvidenceSynthesisSignature": ["causal_impact", "heterogeneous_optimizer", "prediction_synthesizer"],
            "AgentRoutingSignature": ["orchestrator"],
            "VisualizationConfigSignature": ["tool_composer"],
            "MemoryWorthinessSignature": ["feedback_learner"],
            "ProcedureLearningSignature": ["feedback_learner"],
        },
        description="Which agents use which signatures"
    )


# Default signal flow configuration
DEFAULT_SIGNAL_FLOW = SignalFlowContract()
```

### Cognitive Workflow Integration

```python
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, Field

class CognitiveWorkflowRequest(BaseModel):
    """
    Request to execute cognitive RAG workflow.

    Used by: Orchestrator → CognitiveRAG
    """

    # === QUERY ===
    query: str = Field(..., description="User query")
    conversation_id: str = Field(..., description="Conversation session ID")

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation turns"
    )

    user_expertise: Literal["executive", "analyst", "data_scientist", "developer"] = Field(
        default="analyst",
        description="User expertise level"
    )

    # === CONFIG ===
    max_hops: int = Field(default=5, ge=1, le=10, description="Maximum investigation hops")

    enable_learning: bool = Field(
        default=True,
        description="Enable reflector phase for learning"
    )

    collect_dspy_signals: bool = Field(
        default=True,
        description="Collect DSPy signals for optimization"
    )

    # === MEMORY CONFIG ===
    memory_config: Optional[MemoryBackendConfig] = Field(
        None,
        description="Override memory backend configuration"
    )


class CognitiveWorkflowResponse(BaseModel):
    """
    Response from cognitive RAG workflow.

    Returned by: CognitiveRAG → Orchestrator
    """

    # === RESPONSE ===
    response: str = Field(..., description="Final synthesized response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")

    # === EVIDENCE ===
    evidence_count: int = Field(..., description="Number of evidence pieces used")
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Sources of evidence"
    )

    # === ROUTING ===
    routed_agents: List[str] = Field(
        default_factory=list,
        description="Agents routed to during Agent phase"
    )

    # === VISUALIZATION ===
    visualization_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Vega-Lite visualization spec (if applicable)"
    )

    # === LEARNING ===
    learned_facts: int = Field(default=0, description="New facts learned")
    learned_procedures: int = Field(default=0, description="New procedures learned")

    # === DSPy SIGNALS ===
    dspy_signals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Signals for DSPy optimization"
    )

    # === METADATA ===
    phase_latencies: Dict[str, int] = Field(
        default_factory=dict,
        description="Latency per cognitive phase (ms)"
    )
    total_latency_ms: int = Field(..., description="Total workflow latency")
    hop_count: int = Field(..., description="Investigation hops performed")

    # === ERRORS ===
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Any errors encountered"
    )
```

### Cognitive RAG ↔ Agent Integration

```yaml
# Cognitive RAG provides context to all non-orchestrator agents
cognitive_rag_agent_integration:

  # Tier 2: Causal Agents
  tier_2_context:
    causal_impact:
      - causal_paths_from_graph
      - historical_effects
      - segment_context
    gap_analyzer:
      - opportunity_patterns
      - historical_gaps
      - roi_benchmarks
    heterogeneous_optimizer:
      - segment_definitions
      - cate_history
      - optimization_patterns

  # Tier 3: Monitoring Agents
  tier_3_context:
    drift_monitor:
      - drift_history
      - baseline_metrics
      - alert_patterns
    experiment_designer:
      - experiment_outcomes
      - power_calculations
      - design_patterns
    health_score:
      - health_baselines
      - threshold_history
      - anomaly_patterns

  # Tier 4: ML Agents
  tier_4_context:
    prediction_synthesizer:
      - model_performance_history
      - ensemble_patterns
      - calibration_data
    resource_optimizer:
      - allocation_history
      - constraint_patterns
      - optimization_outcomes

  # Tier 5: Self-Improvement Agents
  tier_5_context:
    explainer:
      - explanation_templates
      - user_preference_history
      - feedback_patterns
    feedback_learner:
      - feedback_patterns
      - optimization_outcomes
      - dspy_signals_history
```

---

## System Health Monitoring

### Health Check Contract

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class SystemHealthRequest(BaseModel):
    """
    Request for system health check.

    Endpoint: GET /api/v1/health
    """

    # === SCOPE ===
    components: Optional[List[str]] = Field(
        None,
        description="Specific components to check (None = all)"
    )

    depth: Literal["shallow", "deep"] = Field(
        default="shallow",
        description="Health check depth"
    )

    # === OPTIONS ===
    include_metrics: bool = Field(
        default=True,
        description="Include detailed metrics"
    )

    include_diagnostics: bool = Field(
        default=False,
        description="Include diagnostic information"
    )

class ComponentHealth(BaseModel):
    """
    Health status for a single component.
    """

    # === IDENTIFICATION ===
    component_name: str = Field(..., description="Component name")

    component_type: Literal[
        "frontend",
        "api",
        "database",
        "agent",
        "rag",
        "memory",
        "mlops"
    ] = Field(..., description="Component type")

    # === STATUS ===
    status: Literal["healthy", "degraded", "unhealthy", "unknown"] = Field(
        ...,
        description="Health status"
    )

    health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Health score (1.0 = perfect)"
    )

    # === METRICS ===
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Component metrics"
    )
    # Example: {"latency_p99_ms": 150, "error_rate": 0.01, "cpu_usage": 0.45}

    # === ISSUES ===
    active_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active issues"
    )

    # === METADATA ===
    last_check: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last health check timestamp"
    )

    uptime_seconds: int = Field(..., description="Component uptime")

class SystemHealthResponse(BaseModel):
    """
    Response for system health check.
    """

    # === OVERALL STATUS ===
    overall_status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall system status"
    )

    overall_health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall health score"
    )

    # === COMPONENTS ===
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Component health statuses"
    )

    # === SUMMARY ===
    healthy_count: int = Field(..., description="Number of healthy components")

    degraded_count: int = Field(..., description="Number of degraded components")

    unhealthy_count: int = Field(..., description="Number of unhealthy components")

    # === ALERTS ===
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active system alerts"
    )

    # === METADATA ===
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )

    check_duration_ms: int = Field(..., description="Health check duration")
```

---

## End-to-End Workflow Examples

### Example 1: Simple Causal Query

```yaml
# Workflow: "What caused the drop in NRx for Kisqali?"
workflow_simple_causal:

  # Step 1: Frontend → API
  frontend_request:
    query: "What caused the drop in NRx for Kisqali?"
    session_id: sess_abc123
    user_expertise: analyst

  # Step 2: API → NLP Layer
  nlp_parsing:
    original_query: "What caused the drop in NRx for Kisqali?"
    primary_intent: causal
    entities:
      brands: ["Kisqali"]
      kpis: ["NRx"]
      direction: ["drop"]
    parse_time_ms: 340

  # Step 3: NLP → Orchestrator
  orchestrator_planning:
    selected_agents: ["causal_impact", "explainer"]
    execution_mode: sequential
    estimated_time_ms: 25000

  # Step 4: Orchestrator → causal_impact
  causal_impact_dispatch:
    target_agent: causal_impact
    input:
      query: "What caused the drop in NRx for Kisqali?"
      parsed_query: {...}
      rag_context: {...}

  # Step 5: causal_impact execution
  causal_impact_execution:
    # Retrieve RAG context
    rag_retrieval:
      indexes: ["causal_paths", "business_metrics"]
      top_k: 5
      results: [...]

    # Query causal engine
    causal_analysis:
      treatment_var: "targeting_strategy"
      outcome_var: "NRx"
      ate: -0.12
      confidence_interval: [-0.18, -0.06]
      refutation_passed: true

    # Generate interpretation
    llm_interpretation:
      model: claude-sonnet-4-20250514
      tokens: 1200
      narrative: "Analysis shows..."

    # Return result
    result:
      confidence: 0.85
      execution_time_ms: 15000

  # Step 6: Orchestrator → explainer
  explainer_dispatch:
    target_agent: explainer
    input:
      query: "What caused the drop in NRx for Kisqali?"
      analysis_results: [causal_impact_result]
      user_expertise: analyst

  # Step 7: explainer execution
  explainer_execution:
    # Generate narrative
    llm_generation:
      model: claude-opus-4-20250514
      tokens: 2500
      narrative: "Based on causal analysis..."

    # Extract insights
    insights: [...]

    # Return result
    result:
      confidence: 0.90
      execution_time_ms: 8000

  # Step 8: Orchestrator → API
  orchestrator_response:
    status: completed
    agents_used: ["causal_impact", "explainer"]
    total_execution_time_ms: 23000

  # Step 9: API → Frontend
  frontend_response:
    query_id: qry_xyz789
    status: completed
    response: "Based on causal analysis..."
    insights: [...]
    confidence: 0.85
    execution_time_ms: 24500
```

### Example 2: ML Training Pipeline (Tier 0)

```yaml
# Workflow: "Train a conversion prediction model"
workflow_ml_training:

  # Step 1: Orchestrator → scope_definer
  scope_definer_execution:
    input:
      business_question: "Predict patient conversion"
      success_criteria: {min_f1: 0.85}
    output:
      problem_type: "binary_classification"
      target_variable: "patient_conversion"
      experiment_id: "exp_abc123"
    execution_time_ms: 3000

  # Step 2: scope_definer → data_preparer (sequential)
  data_preparer_execution:
    input:
      experiment_id: "exp_abc123"
      scope_spec: {...}

    # QC Gate (CRITICAL)
    qc_check:
      status: "passed"
      qc_score: 0.92
      blocking_issues: []

    # Baseline metrics
    baseline_computation:
      training_samples: 42000
      target_rate: 0.34

    output:
      qc_passed: true
      data_ready: true
      baseline_metrics: {...}
    execution_time_ms: 45000

  # Step 3: data_preparer → model_selector (sequential)
  model_selector_execution:
    input:
      experiment_id: "exp_abc123"
      problem_type: "binary_classification"
    output:
      selected_algorithms: ["XGBoost", "LightGBM", "RandomForest"]
      search_space: {...}
    execution_time_ms: 80000

  # Step 4: model_selector → model_trainer (sequential)
  model_trainer_execution:
    input:
      experiment_id: "exp_abc123"
      algorithms: ["XGBoost", "LightGBM", "RandomForest"]

    # MLflow experiment tracking
    mlflow_integration:
      experiment_name: "e2i-kisqali-conversion"
      run_id: "mlflow_run_xyz789"

    # Model training (long-running)
    training:
      duration_ms: 1800000  # 30 minutes
      trials: 100
      best_model: "XGBoost"

    # Model validation
    validation:
      f1_score: 0.87
      precision: 0.85
      recall: 0.89

    output:
      model_id: "model_exp_abc123_v1"
      mlflow_run_id: "mlflow_run_xyz789"
      validation_passed: true
    execution_time_ms: 1800000

  # Step 5: model_trainer → feature_analyzer (sequential)
  feature_analyzer_execution:
    input:
      model_id: "model_exp_abc123_v1"
      experiment_id: "exp_abc123"

    # SHAP analysis
    shap_computation:
      samples: 1000
      features: 15

    # LLM interpretation
    interpretation:
      model: claude-sonnet-4-20250514
      tokens: 800

    output:
      feature_importance: [...]
      interactions: [...]
      interpretation: "..."
    execution_time_ms: 95000

  # Step 6: feature_analyzer → model_deployer (sequential)
  model_deployer_execution:
    input:
      model_id: "model_exp_abc123_v1"

    # BentoML deployment
    deployment:
      bento_tag: "e2i-conversion-v1"
      endpoint: "https://api.example.com/predict/conversion"

    output:
      deployed: true
      endpoint_url: "https://api.example.com/predict/conversion"
      health_check_passed: true
    execution_time_ms: 25000

  # Step 7: Cross-cutting observability
  observability_tracking:
    # Opik traces
    traces:
      - agent: scope_definer
        duration_ms: 3000
        status: success
      - agent: data_preparer
        duration_ms: 45000
        status: success
      # ... all agents

    # MLflow experiment
    mlflow_experiment:
      experiment_id: "mlflow_exp_123"
      run_id: "mlflow_run_xyz789"
      metrics_logged: 25
      artifacts_logged: 8

    # Total pipeline time
    total_duration_ms: 2048000  # ~34 minutes
```

---

## Platform-Level Validation Rules

### Pre-Deployment Validation Checklist

```yaml
validation_checklist:

  # 1. Contract Compliance
  contract_validation:
    - rule: "All agent inputs/outputs use Pydantic models"
      severity: critical

    - rule: "All agent states use TypedDict (LangGraph)"
      severity: critical

    - rule: "All handoffs follow agent-handoff.yaml format"
      severity: high

    - rule: "All API endpoints follow QueryRequest/QueryResponse format"
      severity: critical

  # 2. Error Handling
  error_handling_validation:
    - rule: "All agents implement BaseAgent.execute()"
      severity: critical

    - rule: "All agents handle timeouts gracefully"
      severity: high

    - rule: "All agents support retry with exponential backoff"
      severity: medium

    - rule: "All agents log errors to observability system"
      severity: high

  # 3. Observability
  observability_validation:
    - rule: "All agents create Opik spans"
      severity: high

    - rule: "All ML training logs to MLflow"
      severity: critical

    - rule: "All database queries tracked"
      severity: medium

    - rule: "All LLM calls tracked with token counts"
      severity: high

  # 4. Performance
  performance_validation:
    - rule: "Orchestrator responds within 2s SLA"
      severity: critical

    - rule: "Health Score agent responds within 5s SLA"
      severity: high

    - rule: "Drift Monitor agent responds within 10s SLA"
      severity: high

    - rule: "Total query response within max_response_time_seconds"
      severity: high

  # 5. Data Quality
  data_quality_validation:
    - rule: "Data Preparer QC gate blocks if score < 0.75"
      severity: critical

    - rule: "All data leakage tests pass"
      severity: critical

    - rule: "Baseline metrics computed before training"
      severity: critical

    - rule: "Feature distributions validated"
      severity: high

  # 6. Security
  security_validation:
    - rule: "API requests validated with Pydantic"
      severity: critical

    - rule: "Database queries use parameterized statements"
      severity: critical

    - rule: "No secrets in logs or error messages"
      severity: critical

    - rule: "User input sanitized before LLM calls"
      severity: high

  # 7. Integration
  integration_validation:
    - rule: "NLP → Orchestrator contract validated"
      severity: critical

    - rule: "Orchestrator → Agent contracts validated"
      severity: critical

    - rule: "Agent → RAG contracts validated"
      severity: high

    - rule: "Agent → Memory contracts validated"
      severity: high

    - rule: "Frontend → API contracts validated"
      severity: critical
```

---

## Testing Requirements

All integration points must have:

1. **Contract Tests**:
   - Validate request/response schemas
   - Validate field types and constraints
   - Validate required fields present

2. **Integration Tests**:
   - Test end-to-end workflows
   - Test error propagation
   - Test fallback mechanisms
   - Test timeout handling

3. **Load Tests**:
   - Test under expected load
   - Test graceful degradation
   - Test resource limits

4. **Chaos Tests**:
   - Test with random failures
   - Test circuit breaker
   - Test retry logic

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-18 | E2I Team | Initial integration contracts |
| 1.1 | 2025-12-30 | E2I Team | V4.4: Added discovery params to CausalAnalysisRequest/Response |
| 1.2 | 2025-12-31 | E2I Team | V4.4: Added Causal Engine Discovery Contract section (DiscoveryRunner, DriverRanker, DiscoveryGate, StructuralDrift interfaces) |

---

## Related Documents

- [base-contract.md](base-contract.md) - Base agent contracts
- [orchestrator-contracts.md](orchestrator-contracts.md) - Orchestrator contracts
- [agent-handoff.yaml](agent-handoff.yaml) - Agent handoff formats
- [tier0-contracts.md](tier0-contracts.md) - ML Foundation contracts
- [tier2-contracts.md](tier2-contracts.md) - Causal inference contracts
- [tier3-contracts.md](tier3-contracts.md) - Design & monitoring contracts
- [tier4-contracts.md](tier4-contracts.md) - ML prediction contracts
- [tier5-contracts.md](tier5-contracts.md) - Self-improvement contracts

---

**End of Integration Contracts**
