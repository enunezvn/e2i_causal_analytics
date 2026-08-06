"""
E2I Cognitive RAG + DSPy Integration
=====================================
Version: 4.3

This module extends the 4-phase cognitive cycle with DSPy optimization
for each node in the retrieval and reasoning pipeline.

Architecture:
- Phase 1: Summarizer Node → DSPy Query Rewriting
- Phase 2: Investigator Node → DSPy Hop Decision Making
- Phase 3: Agent Node → DSPy Response Synthesis
- Phase 4: Reflector Node → DSPy Training Signal Collection (existing)

The key insight: Each phase has LLM-driven decisions that can be
optimized through DSPy signatures and modules.

GEPA Migration (v4.3):
- Added GEPA as primary optimizer (10%+ improvement over MIPROv2)
- CognitiveRAGOptimizer now supports optimizer_type="gepa" or "miprov2"
- Integrated with RAGAS feedback for RAG-specific quality evaluation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import dspy
from dspy.teleprompt import MIPROv2

logger = logging.getLogger(__name__)

# Conditional GEPA import
try:
    from src.optimization.gepa import (
        create_gepa_optimizer,
        save_optimized_module,
    )
    from src.optimization.gepa.integration.ragas_feedback import (
        RAGASFeedbackProvider,
        create_ragas_metric,
    )

    GEPA_AVAILABLE = True
    logger.info("GEPA optimizer loaded for Cognitive RAG")
except ImportError:
    GEPA_AVAILABLE = False
    logger.info("GEPA not available - using MIPROv2 optimizer")
    create_gepa_optimizer = None  # type: ignore[assignment]
    save_optimized_module = None  # type: ignore[assignment]
    RAGASFeedbackProvider = None  # type: ignore[assignment, misc]
    create_ragas_metric = None  # type: ignore[assignment]

# =============================================================================
# 1. MEMORY TYPES & HOP DEFINITIONS
# =============================================================================


class MemoryType(Enum):
    """The 4 memory types in the Agentic Memory architecture."""

    WORKING = "working"  # Redis + LangGraph MemorySaver
    EPISODIC = "episodic"  # Supabase + pgvector (experiences)
    SEMANTIC = "semantic"  # FalkorDB + Graphity (relationships)
    PROCEDURAL = "procedural"  # Supabase + pgvector (skills)


def _coerce_memory_type(raw: str) -> Optional[MemoryType]:
    """Validate-or-default an LLM-produced next_memory string to a MemoryType.

    The DSPy hop-decider (HopDecisionSignature.next_memory) is free text. Casting
    it directly via MemoryType(raw) raises ValueError on any off-vocabulary value
    (capitalization variants, hallucinated tokens, trailing whitespace). Today
    that crash is latent (an off-vocab value is filtered out by
    _retrieve_from_memory before reaching the cast), but the case-insensitive
    routing added in this change would expose it -- and if it ever fired it would
    unwind out of InvestigatorModule.forward and be swallowed into a degraded
    HTTP-200 dict by CausalRAG.cognitive_search, silently truncating the whole
    investigation (C2). Coerce tolerantly; return None for the uncoercible so the
    caller can skip that one item instead of discarding the entire board.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    try:
        return MemoryType(candidate)
    except ValueError:
        return None


class HopType(Enum):
    """Multi-hop investigation sequence."""

    HOP_1_EPISODIC = "episodic"  # "What happened?"
    HOP_2_SEMANTIC = "semantic"  # "Who/What related?"
    HOP_3_PROCEDURAL = "procedural"  # "How to solve?"
    HOP_4_REFINEMENT = "refinement"  # Additional context


@dataclass
class Evidence:
    """A piece of evidence retrieved during investigation."""

    source: MemoryType
    hop_number: int
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Per-chunk cap for learning_signals.retrieved_chunks (#1489). Retrieval
# chunks are normally hundreds to a couple of thousand characters, so this cap
# is not reached on real traffic; it exists so one pathological evidence item
# cannot write an unbounded JSONB blob on a live turn. A cut chunk is MARKED
# (``truncated``) rather than silently shortened — an unmarked cut would
# understate what the answer was grounded in, and faithfulness is judged
# against exactly this text.
MAX_CHUNK_CONTENT_CHARS = 4000


def _chunks_from_evidence(evidence_board: List["Evidence"]) -> List[Dict[str, Any]]:
    """Evidence the turn really retrieved, as the retrieved_chunks payload.

    Every evidence item produces exactly one chunk: dropping any would
    misreport what retrieval returned, and the parallel ``retrieval_scores``
    array is index-aligned with this list.
    """
    chunks: List[Dict[str, Any]] = []
    for item in evidence_board:
        content = str(item.content)
        chunk: Dict[str, Any] = {
            "content": content[:MAX_CHUNK_CONTENT_CHARS],
            "source": item.source.value,
            "hop": item.hop_number,
        }
        if len(content) > MAX_CHUNK_CONTENT_CHARS:
            chunk["truncated"] = True
        chunks.append(chunk)
    return chunks


@dataclass
class CognitiveState:
    """State passed through the 4-phase cognitive cycle."""

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

    # Phase 3: Agent outputs
    response: str = ""
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    routed_agents: List[str] = field(default_factory=list)

    # Phase 4: Reflector outputs
    worth_remembering: bool = False
    extracted_facts: List[Dict] = field(default_factory=list)
    learned_procedures: List[Dict] = field(default_factory=list)
    dspy_signals: List[Dict] = field(default_factory=list)


# =============================================================================
# 2. PHASE 1: SUMMARIZER NODE - DSPy Signatures
# =============================================================================


class QueryRewriteSignature(dspy.Signature):
    """
    Rewrite user query for optimal retrieval across memory stores.
    Pharmaceutical domain-aware query expansion.
    """

    original_query: str = dspy.InputField(desc="The user's original natural language question")
    conversation_context: str = dspy.InputField(desc="Recent conversation history for context")
    domain_vocabulary: str = dspy.InputField(
        desc="Available domain terms: brands, regions, stages, HCP types"
    )

    rewritten_query: str = dspy.OutputField(
        desc="Optimized query for hybrid retrieval (dense + sparse + graph)"
    )
    search_keywords: list = dspy.OutputField(desc="Key terms for full-text search")
    graph_entities: list = dspy.OutputField(desc="Entities to anchor graph traversal")


class EntityExtractionSignature(dspy.Signature):
    """
    Extract pharmaceutical domain entities from user query.
    Maps to E2I domain vocabulary.
    """

    query: str = dspy.InputField(desc="User query or message")
    domain_vocabulary: str = dspy.InputField(desc="Domain vocabulary YAML")

    brands: list = dspy.OutputField(desc="Brand names mentioned (Remibrutinib, Fabhalta, Kisqali)")
    regions: list = dspy.OutputField(desc="Geographic regions (Northeast, Midwest, etc.)")
    hcp_types: list = dspy.OutputField(desc="HCP specialties (Oncologist, Rheumatologist, etc.)")
    patient_stages: list = dspy.OutputField(
        desc="Patient journey stages (Diagnosis, Treatment, etc.)"
    )
    time_references: list = dspy.OutputField(desc="Temporal references (last quarter, YTD, etc.)")


class IntentClassificationSignature(dspy.Signature):
    """
    Classify user intent for agent routing.
    Determines which E2I agents should handle the query.
    """

    query: str = dspy.InputField(desc="User query")
    extracted_entities: str = dspy.InputField(desc="Extracted entities JSON")

    primary_intent: str = dspy.OutputField(
        desc="Primary intent: CAUSAL_ANALYSIS | GAP_ANALYSIS | PREDICTION | EXPERIMENT_DESIGN | EXPLANATION | GENERAL"
    )
    secondary_intents: list = dspy.OutputField(desc="Additional relevant intents")
    requires_visualization: bool = dspy.OutputField(
        desc="Whether query requires chart/graph output"
    )
    complexity: str = dspy.OutputField(desc="Query complexity: SIMPLE | MODERATE | COMPLEX")


class SummarizerModule(dspy.Module):
    """
    DSPy module for Phase 1: Summarizer Node.
    Prepares the query for multi-hop investigation.
    """

    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought(QueryRewriteSignature)
        self.extract = dspy.Predict(EntityExtractionSignature)
        self.classify = dspy.ChainOfThought(IntentClassificationSignature)

    def forward(
        self, original_query: str, conversation_context: str, domain_vocabulary: str
    ) -> CognitiveState:
        # Step 1: Extract entities
        entities = self.extract(query=original_query, domain_vocabulary=domain_vocabulary)

        # Step 2: Rewrite query for retrieval
        rewritten = self.rewrite(
            original_query=original_query,
            conversation_context=conversation_context,
            domain_vocabulary=domain_vocabulary,
        )

        # Step 3: Classify intent. The classify signature takes a STRING
        # (extracted_entities: str dspy.InputField), so keep the JSON-ish
        # rendering for the LM input only.
        entities_dict = {
            "brands": entities.brands,
            "regions": entities.regions,
            "hcp_types": entities.hcp_types,
            "patient_stages": entities.patient_stages,
            "time_references": entities.time_references,
        }
        entities_json = str(entities_dict)

        intent = self.classify(query=original_query, extracted_entities=entities_json)

        # CognitiveState.extracted_entities is typed List[str] and is surfaced
        # verbatim as CognitiveRAGResponse.entities (List[str]). Returning the
        # str(dict) rendering here put a STRING into that List[str] channel,
        # which propagated to the response and made pydantic reject it with
        # ``list_type`` -> the live HTTP 400 (#953). Flatten the per-category
        # LM outputs into a single de-duplicated List[str] of entity values so
        # the state field and the response field are correctly typed at the
        # source. Each category OutputField is a list; tolerate a non-list LM
        # return defensively (an LM can emit null/str for a list field).
        extracted_entities: List[str] = []
        for value in entities_dict.values():
            if isinstance(value, (list, tuple, set)):
                extracted_entities.extend(str(v) for v in value if v is not None)
            elif value not in (None, ""):
                extracted_entities.append(str(value))
        # De-duplicate while preserving first-seen order (dict keys are ordered
        # and unique; clearer than a side-effecting set comprehension).
        deduped_entities: List[str] = list(dict.fromkeys(extracted_entities))

        return {  # type: ignore[return-value]
            "rewritten_query": rewritten.rewritten_query,
            "search_keywords": rewritten.search_keywords,
            "graph_entities": rewritten.graph_entities,
            "extracted_entities": deduped_entities,
            "primary_intent": intent.primary_intent,
            "secondary_intents": intent.secondary_intents,
            "requires_visualization": intent.requires_visualization,
            "complexity": intent.complexity,
        }


# =============================================================================
# 3. PHASE 2: INVESTIGATOR NODE - DSPy Signatures
# =============================================================================


class InvestigationPlanSignature(dspy.Signature):
    """
    Plan the multi-hop investigation strategy.
    Determines which memory stores to query and in what order.
    """

    query: str = dspy.InputField(desc="Rewritten query for retrieval")
    intent: str = dspy.InputField(desc="Classified intent")
    entities: str = dspy.InputField(desc="Extracted entities")

    investigation_goal: str = dspy.OutputField(
        desc="Clear statement of what we're trying to discover"
    )
    hop_strategy: list = dspy.OutputField(
        desc="Ordered list of memory types to query: [episodic, semantic, procedural, ...]"
    )
    max_hops: int = dspy.OutputField(desc="Maximum number of hops needed (1-4)")
    early_stop_criteria: str = dspy.OutputField(
        desc="Conditions under which to stop investigation early"
    )


class HopDecisionSignature(dspy.Signature):
    """
    Decide the next retrieval hop based on accumulated evidence.
    This is the core iterative retrieval decision point.
    """

    investigation_goal: str = dspy.InputField(desc="What we're trying to discover")
    current_evidence: str = dspy.InputField(desc="Evidence collected so far")
    hop_number: int = dspy.InputField(desc="Current hop number (1-4)")
    available_memories: list = dspy.InputField(desc="Memory types not yet queried")

    next_memory: str = dspy.OutputField(
        desc="Next memory type to query: episodic | semantic | procedural | STOP"
    )
    retrieval_query: str = dspy.OutputField(desc="Specific query for the next memory store")
    reasoning: str = dspy.OutputField(desc="Why this hop is needed or why to stop")
    confidence: float = dspy.OutputField(desc="Confidence that more evidence is needed (0.0-1.0)")


class EvidenceRelevanceSignature(dspy.Signature):
    """
    Score retrieved evidence for relevance to investigation goal.
    Filters noise and ranks evidence quality.
    """

    investigation_goal: str = dspy.InputField(desc="What we're trying to discover")
    evidence_item: str = dspy.InputField(desc="A single piece of retrieved evidence")
    source_memory: str = dspy.InputField(desc="Which memory store this came from")

    relevance_score: float = dspy.OutputField(desc="Relevance score 0.0-1.0")
    key_insight: str = dspy.OutputField(desc="The key insight this evidence provides")
    follow_up_needed: bool = dspy.OutputField(
        desc="Whether this evidence suggests follow-up queries"
    )


class InvestigatorModule(dspy.Module):
    """
    DSPy module for Phase 2: Investigator Node.
    Implements iterative multi-hop retrieval with learned hop decisions.
    """

    def __init__(self, memory_backends: Dict[str, Any]):
        super().__init__()
        self.plan = dspy.ChainOfThought(InvestigationPlanSignature)
        self.decide_hop = dspy.ChainOfThought(HopDecisionSignature)
        self.score_evidence = dspy.Predict(EvidenceRelevanceSignature)
        self.memory_backends = memory_backends
        self.max_hops = 4

    async def forward(self, rewritten_query: str, intent: str, entities: str) -> Dict:
        # Step 1: Plan investigation. self.plan is a SYNC DSPy ChainOfThought
        # (blocking LLM call); run it off the event loop so the gunicorn worker
        # keeps sending heartbeats during the call (#953 RC2). Output is
        # assigned exactly as before -- only WHERE the work runs changes.
        plan = await asyncio.to_thread(
            self.plan, query=rewritten_query, intent=intent, entities=entities
        )

        evidence_board: List[Evidence] = []
        queried_memories = set()

        # Step 2: Iterative hop execution
        for hop_num in range(1, self.max_hops + 1):
            available = [
                m for m in ["episodic", "semantic", "procedural"] if m not in queried_memories
            ]

            if not available:
                break

            # Decide next hop. self.decide_hop is a SYNC DSPy call -> offload.
            decision = await asyncio.to_thread(
                self.decide_hop,
                investigation_goal=plan.investigation_goal,
                current_evidence=str([e.__dict__ for e in evidence_board]),
                hop_number=hop_num,
                available_memories=available,
            )

            if decision.next_memory == "STOP":
                break

            # Execute retrieval
            raw_evidence = await self._retrieve_from_memory(
                decision.next_memory, decision.retrieval_query
            )

            # Score and filter evidence. Coerce the decider's free-text
            # next_memory ONCE per hop; skip the whole scoring block if it is
            # off-vocabulary (C2) instead of crashing the investigation.
            hop_memory = _coerce_memory_type(decision.next_memory)
            if hop_memory is None:
                logger.warning(
                    "Investigator hop %d returned off-vocabulary next_memory %r; "
                    "skipping evidence from this hop (no valid MemoryType)",
                    hop_num,
                    decision.next_memory,
                )
            else:
                for item in raw_evidence:
                    # self.score_evidence is a SYNC DSPy Predict -> offload.
                    scored = await asyncio.to_thread(
                        self.score_evidence,
                        investigation_goal=plan.investigation_goal,
                        evidence_item=item["content"],
                        source_memory=hop_memory.value,
                    )

                    if scored.relevance_score >= 0.5:  # Threshold
                        evidence_board.append(
                            Evidence(
                                source=hop_memory,
                                hop_number=hop_num,
                                content=item["content"],
                                relevance_score=scored.relevance_score,
                                metadata={"key_insight": scored.key_insight},
                            )
                        )

            queried_memories.add(decision.next_memory)

            # Check early stop
            if decision.confidence < 0.3:  # Low confidence = sufficient evidence
                break

        return {
            "investigation_goal": plan.investigation_goal,
            "evidence_board": evidence_board,
            "hop_count": len(queried_memories),
            "sufficient_evidence": len(evidence_board) >= 2,
        }

    async def _retrieve_from_memory(self, memory_type: str, query: str) -> List[Dict[Any, Any]]:
        """Execute retrieval against the appropriate memory backend."""
        coerced = _coerce_memory_type(memory_type)
        if coerced is None:
            logger.warning(
                "Investigator hop requested off-vocabulary memory type %r; "
                "no backend, skipping hop",
                memory_type,
            )
            return []
        memory_type = coerced.value
        backend = self.memory_backends.get(memory_type)
        if not backend:
            # Coercible but unregistered (e.g. 'working' is a valid MemoryType
            # never offered as a hop target and has no backend). Log so a missing
            # backend is distinguishable from an off-vocabulary input.
            logger.warning("No backend registered for memory type %r; skipping hop", memory_type)
            return []

        result: List[Dict[Any, Any]]
        if memory_type == "episodic":
            # pgvector semantic search
            result = await backend.vector_search(query, limit=5)
        elif memory_type == "semantic":
            # FalkorDB graph traversal
            result = await backend.graph_query(query, max_depth=2)
        elif memory_type == "procedural":
            # pgvector similarity on tool sequences
            result = await backend.procedure_search(query, limit=3)
        else:
            return []

        return result


# =============================================================================
# 4. PHASE 3: AGENT NODE - DSPy Signatures
# =============================================================================


class EvidenceSynthesisSignature(dspy.Signature):
    """
    Synthesize collected evidence into a coherent response.
    Integrates insights from multiple memory hops.
    """

    user_query: str = dspy.InputField(desc="Original user question")
    investigation_goal: str = dspy.InputField(desc="What we investigated")
    evidence_board: str = dspy.InputField(desc="Collected evidence JSON")
    intent: str = dspy.InputField(desc="User intent classification")

    synthesis: str = dspy.OutputField(desc="Synthesized answer integrating all evidence")
    confidence_statement: str = dspy.OutputField(
        desc="Statement about confidence level and evidence quality"
    )
    evidence_citations: list = dspy.OutputField(
        desc="Which pieces of evidence support the synthesis"
    )


class AgentRoutingSignature(dspy.Signature):
    """
    Determine which E2I agents should process this query.
    Routes to appropriate tier based on intent and evidence.
    """

    intent: str = dspy.InputField(desc="Primary intent")
    complexity: str = dspy.InputField(desc="Query complexity")
    evidence_summary: str = dspy.InputField(desc="Summary of collected evidence")

    primary_agent: str = dspy.OutputField(
        desc="Primary agent: orchestrator | causal_impact | gap_analyzer | experiment_designer | explainer | prediction_synthesizer"
    )
    supporting_agents: list = dspy.OutputField(desc="Additional agents to involve")
    requires_deep_reasoning: bool = dspy.OutputField(
        desc="Whether to use Deep agent (extended thinking)"
    )


class VisualizationConfigSignature(dspy.Signature):
    """
    Generate visualization configuration for the response.
    Maps insights to appropriate chart types.
    """

    synthesis: str = dspy.InputField(desc="Synthesized answer")
    data_types: list = dspy.InputField(desc="Types of data in evidence")
    user_preference: str = dspy.InputField(desc="User's visualization preferences if any")

    chart_type: str = dspy.OutputField(
        desc="Chart type: bar | line | scatter | heatmap | sankey | network | none"
    )
    chart_config: str = dspy.OutputField(desc="JSON configuration for the chart")
    highlights: list = dspy.OutputField(desc="Key data points to highlight")


# Agent name under which the optimized synthesis prompt is BOTH saved by the
# nightly RAG leg (src/tasks/dspy_optimization_tasks.py) and loaded here. It is
# one constant imported by both sides on purpose: a save-name literal and a
# load-name literal that can drift is how the 2026-06-08 artifact sat unshipped
# for six weeks while everyone believed the tuned module was live
# (docs/reports/dspy_lane_ab_20260718.md section 7).
#
# Resolves under ./optimized_modules/<name>/ — CWD-relative, matching the
# /app/optimized_modules named volume that
# tests/integration/test_optimized_artifacts_compose_wiring.py pins as the
# handshake between the worker that writes and the api that reads.
OPTIMIZED_SYNTHESIS_AGENT_NAME = "cognitive_rag_synthesis"

# Must match load_optimized_module's default, since the signature probe below
# resolves the same directory the loader will read.
OPTIMIZED_MODULES_DIR = "./optimized_modules"

# {"attempted": bool, "signature": Optional[tuple], "module": Optional[dspy.Module]}.
# Module-level rather than per-instance because AgentModule is rebuilt per
# workflow construction and the load is a parse, not just a stat.
_OPTIMIZED_SYNTHESIS_CACHE: Dict[str, Any] = {
    "attempted": False,
    "signature": None,
    "module": None,
}


def _artifact_signature() -> Optional[tuple]:
    """(path, mtime_ns) of the newest saved artifact, or None if there is none.

    Cheap on purpose — one directory glob plus a stat, microseconds against the
    LLM call that follows. It is what lets a long-lived worker notice an
    artifact that appeared (or was replaced) after it last looked: nothing else
    invalidates the cache, because docker-compose mounts optimized_modules
    read-only into api and writable into worker_medium with no signal between
    them.

    "Newest" MUST come from versioning.newest_saved_artifact — the same
    resolver load_optimized_module uses — because this signature decides
    whether the cached module is stale. When the two sites ranked names
    independently they both inverted lexicographically at gepa_v10 (#1496),
    and fixing either one alone desynchronizes the cache key from what the
    loader actually parses. Import deferred like _load_optimized_module's: the
    gepa package costs ~1s the first time it is pulled into a process, and a
    module-level import would put that on every importer of this module.

    The import lives INSIDE the try: this probe is a fail-soft seam
    (AgentModule construction calls it on every workflow build with no catch
    above it), so a gepa import failure must degrade to the base prompt — with
    a WARNING, not the cached-miss INFO line — never break RAG construction.
    _load_optimized_module's twin import needs no such guard because it runs
    inside the caller's transient-failure ``except Exception`` block.
    """
    directory = Path(OPTIMIZED_MODULES_DIR) / OPTIMIZED_SYNTHESIS_AGENT_NAME
    try:
        from src.optimization.gepa.versioning import newest_saved_artifact

        newest = newest_saved_artifact(directory)
        if newest is None:
            return None
        return (str(newest), newest.stat().st_mtime_ns)
    except ImportError as e:
        logger.warning(
            "Cannot resolve optimized synthesis artifacts (gepa import failed); "
            "using base prompt: %s",
            e,
        )
        return None
    except OSError:
        return None


def _load_optimized_module() -> Any:
    """Load the saved synthesis module. Separate seam so tests can drive failures."""
    from src.optimization.gepa.versioning import load_optimized_module

    module, _meta = load_optimized_module(
        lambda: dspy.ChainOfThought(EvidenceSynthesisSignature),
        agent_name=OPTIMIZED_SYNTHESIS_AGENT_NAME,
    )
    return module


def load_optimized_synthesis_module(reset: bool = False) -> Optional[Any]:
    """Return the GEPA-optimized synthesis module, or None to use the base prompt.

    Mirrors ``pattern_analyzer._load_optimized_pattern_module`` deliberately:

    - An intentional miss (no artifact saved yet -> FileNotFoundError) is CACHED,
      so a cold install does not stat the filesystem on every workflow build.
    - A transient failure (corrupt read, import race) is NOT cached, so a later
      cycle retries once the condition clears. Caching it would strand the
      runtime on the base prompt until the process restarted.
    - The miss logs at INFO, not DEBUG: prod ran on the silent fallback for six
      weeks believing the tuned module was live. This is quiet in volume because
      the miss is cached.

    Args:
        reset: Re-probe even if a previous outcome was cached (tests, and any
            caller that has just written a new artifact).
    """
    if reset:
        _OPTIMIZED_SYNTHESIS_CACHE.update({"attempted": False, "signature": None, "module": None})

    signature = _artifact_signature()
    # Keyed on the artifact rather than on "have we looked before". Caching the
    # bare outcome meant a worker that probed before the first nightly success
    # served the base prompt until restart, and a worker holding version N never
    # saw N+1 — with no invalidation path in either direction.
    if (
        _OPTIMIZED_SYNTHESIS_CACHE["attempted"]
        and _OPTIMIZED_SYNTHESIS_CACHE["signature"] == signature
    ):
        return _OPTIMIZED_SYNTHESIS_CACHE["module"]

    if signature is None:
        logger.info(
            "No optimized cognitive-RAG synthesis module saved yet (%s); using base prompt",
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
        )
        _OPTIMIZED_SYNTHESIS_CACHE.update({"attempted": True, "signature": None, "module": None})
        return None

    try:
        module = _load_optimized_module()
    except FileNotFoundError:
        logger.info(
            "No optimized cognitive-RAG synthesis module saved yet (%s); using base prompt",
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
        )
        _OPTIMIZED_SYNTHESIS_CACHE.update(
            {"attempted": True, "signature": signature, "module": None}
        )
        return None
    except Exception as e:  # noqa: BLE001 - transient: do NOT cache, allow retry
        logger.warning("Failed to load optimized synthesis module (will retry next build): %s", e)
        # Leave `attempted`/`signature` untouched so the next call retries this
        # same artifact rather than being stranded on the base prompt.
        _OPTIMIZED_SYNTHESIS_CACHE["module"] = None
        return None

    logger.info(
        "Loaded optimized cognitive-RAG synthesis module (%s)", OPTIMIZED_SYNTHESIS_AGENT_NAME
    )
    _OPTIMIZED_SYNTHESIS_CACHE.update({"attempted": True, "signature": signature, "module": module})
    return module


class AgentModule(dspy.Module):
    """
    DSPy module for Phase 3: Agent Node.
    Synthesizes evidence and routes to appropriate E2I agents.
    """

    def __init__(self, agent_registry: Dict[str, Any]):
        super().__init__()
        # Consume the nightly leg's artifact when one exists (#1486). Falling
        # back to the base signature keeps a cold install and a corrupt artifact
        # both serving traffic on the shipped prompt.
        self.synthesize = load_optimized_synthesis_module() or dspy.ChainOfThought(
            EvidenceSynthesisSignature
        )
        self.route = dspy.Predict(AgentRoutingSignature)
        self.visualize = dspy.Predict(VisualizationConfigSignature)
        self.agent_registry = agent_registry

    async def forward(self, state: CognitiveState) -> CognitiveState:
        # Step 1: Synthesize evidence. self.synthesize is a SYNC DSPy
        # ChainOfThought (blocking LLM call); offload so the worker heartbeat
        # keeps firing (#953 RC2). Output assigned exactly as before.
        synthesis = await asyncio.to_thread(
            self.synthesize,
            user_query=state.user_query,
            investigation_goal=state.investigation_goal,
            evidence_board=str([e.__dict__ for e in state.evidence_board]),
            intent=state.detected_intent,
        )

        # Step 2: Determine agent routing. self.route is a SYNC DSPy Predict.
        routing = await asyncio.to_thread(
            self.route,
            intent=state.detected_intent,
            complexity="COMPLEX" if state.hop_count > 2 else "MODERATE",
            evidence_summary=synthesis.synthesis[:500],
        )

        # Step 3: Execute primary agent
        primary_agent = self.agent_registry.get(routing.primary_agent)
        if primary_agent:
            agent_response = await primary_agent.process(
                query=state.user_query, evidence=state.evidence_board, synthesis=synthesis.synthesis
            )
            state.response = agent_response
        else:
            state.response = synthesis.synthesis

        # Step 4: Generate visualization if needed
        if state.detected_intent in ["CAUSAL_ANALYSIS", "GAP_ANALYSIS", "PREDICTION"]:
            # self.visualize is a SYNC DSPy Predict -> offload.
            viz = await asyncio.to_thread(
                self.visualize,
                synthesis=synthesis.synthesis,
                data_types=["temporal", "categorical", "causal"],
                user_preference="",
            )
            state.visualization_config = {
                "chart_type": viz.chart_type,
                "config": viz.chart_config,
                "highlights": viz.highlights,
            }

        state.routed_agents = [routing.primary_agent] + routing.supporting_agents

        return state


# =============================================================================
# 5. PHASE 4: REFLECTOR NODE - DSPy Training Signal Collection
# =============================================================================


class MemoryWorthinessSignature(dspy.Signature):
    """
    Evaluate if this interaction is worth remembering.
    Determines what to store in long-term memory.
    """

    user_query: str = dspy.InputField(desc="Original query")
    response: str = dspy.InputField(desc="Generated response")
    evidence_count: int = dspy.InputField(desc="Number of evidence pieces used")
    user_feedback: str = dspy.InputField(desc="User feedback if available")

    worth_remembering: bool = dspy.OutputField(desc="Whether to store in episodic memory")
    memory_type: str = dspy.OutputField(
        desc="Which memory: episodic | semantic | procedural | none"
    )
    importance_score: float = dspy.OutputField(desc="Importance for future retrieval (0.0-1.0)")
    key_facts: list = dspy.OutputField(desc="Facts to extract for semantic memory")


class ProcedureLearningSignature(dspy.Signature):
    """
    Extract successful tool/agent sequences for procedural memory.
    Learns patterns that worked well.
    """

    query_type: str = dspy.InputField(desc="Type of query handled")
    agents_used: list = dspy.InputField(desc="Agents that processed this query")
    hop_sequence: list = dspy.InputField(desc="Memory hops executed")
    success_indicators: str = dspy.InputField(desc="Signals of successful response")

    procedure_pattern: str = dspy.OutputField(desc="Generalized procedure pattern")
    trigger_conditions: list = dspy.OutputField(desc="When to apply this procedure")
    expected_outcome: str = dspy.OutputField(desc="What outcome this procedure achieves")


class ReflectorModule(dspy.Module):
    """
    DSPy module for Phase 4: Reflector Node.
    Handles asynchronous learning and DSPy signal collection.
    """

    def __init__(self, memory_writers: Dict[str, Any], signal_collector: Any):
        super().__init__()
        self.evaluate = dspy.Predict(MemoryWorthinessSignature)
        self.learn_procedure = dspy.Predict(ProcedureLearningSignature)
        self.memory_writers = memory_writers
        self.signal_collector = signal_collector

    async def forward(
        self, state: CognitiveState, user_feedback: Optional[str] = None
    ) -> CognitiveState:
        # Step 1: Evaluate memory worthiness. The DSPy evaluator is a SYNC
        # predictor (dspy.Predict); run it off the event loop so the gunicorn
        # heartbeat keeps firing during the ~LLM-latency call (#953 RC2).
        evaluation = await asyncio.to_thread(
            self.evaluate,
            user_query=state.user_query,
            response=state.response,
            evidence_count=len(state.evidence_board),
            user_feedback=user_feedback or "",
        )

        state.worth_remembering = evaluation.worth_remembering

        # Step 2: Store in appropriate memory.
        #
        # These long-term-memory writes are BEST-EFFORT: a write failure must
        # never abort the request or surface in-band as the HTTP-200 ``error``
        # field (which is exactly what happened pre-#953 when the call hit the
        # nonexistent ``.store`` method on EpisodicMemoryBackend -> the
        # AttributeError unwound into CausalRAG.cognitive_search's except
        # clause). We call the REAL backend signatures and guard each write so
        # it logs-and-continues instead of raising.
        if evaluation.worth_remembering:
            if evaluation.memory_type == "episodic":
                # EpisodicMemoryBackend.store_episode(content, episode_type,
                # metadata) -> episodic_memories(event_type, agent_name, ...).
                # The synthesized response is what is worth remembering; the
                # original query + importance live in metadata.
                #
                # BOTH the event_type and agent_name we write must be valid
                # against constrained DB enums or the insert dies with postgrest
                # 22P02 and the episode is lost:
                #   - event_type maps to ``memory_event_type`` (NOT NULL). Its
                #     vocabulary does NOT include "conversation"; "agent_action"
                #     is the faithful label for the cognitive-RAG cycle (an agent
                #     produced this synthesized response). Verified against the
                #     live DB enum.
                #   - agent_name maps to ``e2i_agent_name`` (nullable), whose
                #     vocabulary is the 23 registered E2I agents. The cognitive
                #     RAG cycle is NOT one of them, so we OMIT agent_name (the
                #     writer drops None keys -> column left NULL) rather than
                #     invent a fake-but-valid agent.
                await self._best_effort(
                    "episodic.store_episode",
                    self.memory_writers["episodic"].store_episode(
                        content=state.response,
                        episode_type="agent_action",
                        metadata={
                            "query": state.user_query,
                            "importance_score": evaluation.importance_score,
                            "session_id": state.conversation_id,
                        },
                    ),
                )

            if evaluation.key_facts:
                state.extracted_facts = evaluation.key_facts
                # SemanticMemoryBackend exposes store_relationship(source,
                # target, rel_type, properties) -- there is no ``add_fact``. A
                # "fact" only maps cleanly onto a graph edge when it carries a
                # source AND target; ill-formed facts are skipped (best-effort)
                # rather than crashing the write-back.
                for fact in evaluation.key_facts:
                    edge = self._fact_to_relationship(fact)
                    if edge is None:
                        continue
                    source, target, rel_type, properties = edge
                    await self._best_effort(
                        "semantic.store_relationship",
                        self.memory_writers["semantic"].store_relationship(
                            source_entity=source,
                            target_entity=target,
                            relationship_type=rel_type,
                            properties=properties,
                        ),
                    )

        # Step 3: Learn procedures from successful interactions. learn_procedure
        # is a SYNC DSPy predictor -> offload (#953 RC2).
        if user_feedback and "positive" in user_feedback.lower():
            procedure = await asyncio.to_thread(
                self.learn_procedure,
                query_type=state.detected_intent,
                agents_used=state.routed_agents,
                hop_sequence=[e.source.value for e in state.evidence_board],
                success_indicators=user_feedback,
            )

            state.learned_procedures.append(
                {
                    "pattern": procedure.procedure_pattern,
                    "triggers": procedure.trigger_conditions,
                    "outcome": procedure.expected_outcome,
                }
            )

            # ProceduralMemoryBackend.store_procedure(procedure_name,
            # tool_sequence, trigger_pattern, intent, ...) -- positional fields,
            # not a single dict. Map the learned pattern onto that contract.
            learned = state.learned_procedures[-1]
            triggers = learned.get("triggers")
            if isinstance(triggers, list) and triggers:
                trigger_pattern: Optional[str] = str(triggers[0])
            elif isinstance(triggers, str):
                trigger_pattern = triggers
            else:
                trigger_pattern = None
            await self._best_effort(
                "procedural.store_procedure",
                self.memory_writers["procedural"].store_procedure(
                    procedure_name=str(
                        learned.get("pattern") or state.detected_intent or "procedure"
                    ),
                    tool_sequence=[{"agent": agent} for agent in state.routed_agents],
                    trigger_pattern=trigger_pattern,
                    intent=state.detected_intent or None,
                ),
            )

        # Step 4: Collect DSPy training signals for Feedback Learner
        state.dspy_signals = self._collect_training_signals(state, user_feedback)
        await self.signal_collector.collect(state.dspy_signals)

        return state

    @staticmethod
    async def _best_effort(label: str, awaitable: Any) -> None:
        """Await a best-effort memory write, logging-and-swallowing any error.

        Long-term-memory consolidation is non-critical to producing the user's
        response. A failure here (a backend signature drift, a transient
        Supabase/FalkorDB outage) must NEVER unwind into the response path and
        be surfaced as error-as-data (#953). The underlying backend methods
        already swallow their own exceptions and return None/False, but this
        guard is the contract enforcement point for any future writer.
        """
        try:
            await awaitable
        except Exception as exc:  # noqa: BLE001 -- best-effort by design
            logger.warning("Best-effort memory write %s failed: %s", label, exc)

    @staticmethod
    def _fact_to_relationship(
        fact: Any,
    ) -> Optional[tuple[str, str, str, Dict[str, Any]]]:
        """Map a Reflector ``key_facts`` entry onto a semantic-graph edge.

        ``MemoryWorthinessSignature.key_facts`` is a free-text list, so an entry
        may be a dict (preferred) or a bare string. We can only persist a fact
        as a relationship when it carries BOTH a source and a target; anything
        else is skipped (returns None) rather than guessed-at, so we never
        fabricate edges. ``store_relationship`` is the only write method the
        semantic backend exposes.
        """
        if not isinstance(fact, dict):
            return None
        source = fact.get("source") or fact.get("source_entity")
        target = fact.get("target") or fact.get("target_entity")
        if not source or not target:
            return None
        rel_type = str(
            fact.get("relationship")
            or fact.get("relationship_type")
            or fact.get("rel_type")
            or "RELATED_TO"
        )
        raw_props = fact.get("properties")
        properties: Dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
        # Stamp runtime provenance so this LLM-derived reflection edge is
        # recognisable as NON-curated. The Knowledge Graph's curated view filters
        # on ``r.agent IS NULL`` (only seed/sync gold-standard carries no agent
        # tag); without this, a reflection-written CAUSES/IMPACTS edge would leak
        # into the gold-standard view. RAG retrieval ignores this key, so it has
        # no effect on read paths. See semantic_memory.list_relationships(curated_only).
        properties = {"agent": "rag_reflection", **properties}
        return str(source), str(target), rel_type, properties

    def _collect_training_signals(
        self, state: CognitiveState, user_feedback: Optional[str]
    ) -> List[Dict]:
        """
        Collect training signals for DSPy optimization.
        These flow to the Feedback Learner agent via SignalCollectorAdapter.

        Signal format must match SignalCollectorAdapter.collect() expectations:
        - type: Signal type (e.g., "summarizer", "investigator", "agent")
        - query: The input query/prompt
        - response: The output/response
        - reward: Quality score (0.0 to 1.0)
        - feedback: Optional user feedback dict
        - metadata: Additional context
        """
        signals = []

        # Calculate rewards based on workflow outcomes
        summarizer_reward = (
            min(1.0, len(state.evidence_board) / 4.0) if state.sufficient_evidence else 0.3
        )
        investigator_reward = (
            min(1.0, sum(e.relevance_score for e in state.evidence_board) / 3.0)
            if state.evidence_board
            else 0.0
        )
        # Synthesis reward graded by observable outcome quality, mirroring how
        # the two rewards above are derived. The previous constant
        # ``0.8 if state.response`` had zero variance, so every downstream
        # consumer (GEPA gating, feedback-learner pattern analysis) learned
        # nothing from it. Base 0.5 for producing any response keeps completed
        # turns inside the GEPA fuel band (reward >= 0.5), as before.
        if state.response:
            agent_reward = 0.5
            # Grounded in retrieved evidence (same 4-item scale as summarizer)
            agent_reward += 0.2 * min(1.0, len(state.evidence_board) / 4.0)
            # Substantive answer, not a one-liner
            agent_reward += 0.1 if len(state.response) >= 200 else 0.0
            # Investigation deemed its evidence sufficient
            agent_reward += 0.1 if state.sufficient_evidence else 0.0
            # Produced an actionable artifact alongside the text
            agent_reward += 0.1 if state.visualization_config else 0.0
            agent_reward = min(1.0, agent_reward)
        else:
            agent_reward = 0.0

        # Adjust rewards based on user feedback if provided
        if user_feedback:
            feedback_boost = 0.2 if "positive" in user_feedback.lower() else -0.1
            summarizer_reward = min(1.0, max(0.0, summarizer_reward + feedback_boost))
            investigator_reward = min(1.0, max(0.0, investigator_reward + feedback_boost))
            agent_reward = min(1.0, max(0.0, agent_reward + feedback_boost))

        # Signal for Summarizer optimization (query rewrite, entity extraction, intent)
        signals.append(
            {
                "type": "summarizer",
                "query": state.user_query,
                "response": state.rewritten_query or state.user_query,
                "reward": summarizer_reward,
                "feedback": {"user_feedback": user_feedback} if user_feedback else None,
                "metadata": {
                    "entities": state.extracted_entities,
                    "intent": state.detected_intent,
                    "conversation_id": state.conversation_id,
                },
            }
        )

        # Signal for Investigator optimization (multi-hop retrieval)
        signals.append(
            {
                "type": "investigator",
                "query": state.investigation_goal or state.rewritten_query or state.user_query,
                "response": f"Found {len(state.evidence_board)} evidence items in {state.hop_count} hops",
                "reward": investigator_reward,
                "feedback": (
                    {"sufficient_evidence": state.sufficient_evidence}
                    if state.sufficient_evidence
                    else None
                ),
                "metadata": {
                    "hop_count": state.hop_count,
                    "evidence_count": len(state.evidence_board),
                    "evidence_sources": [e.source.value for e in state.evidence_board],
                    "conversation_id": state.conversation_id,
                },
            }
        )

        # Signal for Agent/Synthesis optimization.
        #
        # #1489 deferral 1: this is the ONLY signal that carries the turn's
        # retrieved evidence. database/ml/022 added learning_signals
        # .retrieved_chunks / .retrieval_scores "for RAGAS evaluation" and
        # never got a producer (3,959 rows, 0 populated, measured 2026-08-06);
        # the evidence has always been right here in state.evidence_board.
        # Attaching it to the summarizer/investigator signals too would store
        # one turn's retrieval three times and make any per-row count of
        # retrieved chunks read 3x the truth — and it is THIS row whose
        # ``response`` those chunks grounded, so it is the row a RAGAS judge
        # scores. No LLM call is added: the evidence is already in the state.
        signals.append(
            {
                "type": "agent",
                "query": f"Intent: {state.detected_intent}, Evidence: {len(state.evidence_board)} items",
                "response": state.response[:500] if state.response else "",
                "reward": agent_reward,
                "feedback": {"user_feedback": user_feedback} if user_feedback else None,
                "retrieved_chunks": _chunks_from_evidence(state.evidence_board),
                "retrieval_scores": [float(e.relevance_score) for e in state.evidence_board],
                "metadata": {
                    "routed_agents": state.routed_agents,
                    "has_visualization": bool(state.visualization_config),
                    "response_length": len(state.response) if state.response else 0,
                    "conversation_id": state.conversation_id,
                },
            }
        )

        return signals


# =============================================================================
# 6. COMPLETE COGNITIVE WORKFLOW WITH DSPy
# =============================================================================

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


def create_dspy_cognitive_workflow(
    memory_backends: Dict[str, Any],
    memory_writers: Dict[str, Any],
    agent_registry: Dict[str, Any],
    signal_collector: Any,
    domain_vocabulary: str,
) -> StateGraph:
    """
    Create the complete 4-phase cognitive workflow with DSPy optimization.

    Each phase uses DSPy modules that can be independently optimized
    by the Feedback Learner agent.
    """

    # Initialize DSPy modules
    summarizer = SummarizerModule()
    investigator = InvestigatorModule(memory_backends)
    agent = AgentModule(agent_registry)
    reflector = ReflectorModule(memory_writers, signal_collector)

    graph = StateGraph(CognitiveState)

    # Phase 1: Summarizer Node
    async def summarizer_node(state: CognitiveState) -> CognitiveState:
        # summarizer.forward is SYNC and fires three blocking DSPy predictors
        # (extract / rewrite / classify). Run the whole call off the event loop
        # so the gunicorn worker keeps sending heartbeats during the ~LLM
        # latency -- otherwise the loop blocks, the worker misses its
        # heartbeat, gunicorn kills it (WORKER TIMEOUT) and nginx returns 502
        # (#953 RC2). The returned dict is consumed exactly as before.
        result = await asyncio.to_thread(
            summarizer.forward,
            original_query=state.user_query,
            conversation_context=state.compressed_history,
            domain_vocabulary=domain_vocabulary,
        )

        # Defensive coercion at the channel-write boundary: an LM can return
        # null/non-list for a list field, and CognitiveState is a plain
        # @dataclass that does not coerce. extracted_entities is surfaced
        # verbatim as CognitiveRAGResponse.entities (List[str]); a None or
        # non-list here would make the response fail pydantic validation (the
        # #953 list_type 400). SummarizerModule.forward already returns a flat
        # List[str], but coerce again so this channel is never None/non-list.
        state.rewritten_query = result["rewritten_query"] or ""  # type: ignore[index]
        entities_out = result["extracted_entities"]  # type: ignore[index]
        state.extracted_entities = (
            list(entities_out) if isinstance(entities_out, (list, tuple)) else []
        )
        state.detected_intent = result["primary_intent"] or ""  # type: ignore[index]

        return state

    # Phase 2: Investigator Node
    async def investigator_node(state: CognitiveState) -> CognitiveState:
        result = await investigator.forward(
            rewritten_query=state.rewritten_query,
            intent=state.detected_intent,
            entities=str(state.extracted_entities),
        )

        state.investigation_goal = result["investigation_goal"]
        state.evidence_board = result["evidence_board"]
        state.hop_count = result["hop_count"]
        state.sufficient_evidence = result["sufficient_evidence"]

        return state

    # Phase 3: Agent Node
    async def agent_node(state: CognitiveState) -> CognitiveState:
        return await agent.forward(state)

    # Phase 4: Reflector Node (async, runs after response)
    async def reflector_node(state: CognitiveState) -> CognitiveState:
        return await reflector.forward(state)

    # Build graph
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("investigator", investigator_node)
    graph.add_node("agent", agent_node)
    graph.add_node("reflector", reflector_node)

    graph.set_entry_point("summarizer")
    graph.add_edge("summarizer", "investigator")
    graph.add_edge("investigator", "agent")
    graph.add_edge("agent", "reflector")
    graph.add_edge("reflector", END)

    return graph.compile(checkpointer=MemorySaver())  # type: ignore[return-value]


# =============================================================================
# 7. DSPy OPTIMIZATION TARGETS FOR RAG
# =============================================================================


# Type alias for optimizer selection
OptimizerType = Literal["miprov2", "gepa"]


class CognitiveRAGOptimizer:
    """
    Optimizer specifically for the 4-phase cognitive RAG system.
    Defines metrics and optimization strategies for each phase.

    GEPA Migration (v4.3):
    - Added GEPA as primary optimizer (10%+ improvement over MIPROv2)
    - Supports optimizer_type="gepa" (default) or "miprov2" (legacy)
    - Integrates RAGAS feedback for RAG-specific quality evaluation
    - Falls back to MIPROv2 if GEPA is not available
    """

    def __init__(
        self,
        feedback_learner: Any,
        optimizer_type: OptimizerType = "gepa",
    ):
        self.feedback_learner = feedback_learner

        # Select optimizer based on availability and preference
        if optimizer_type == "gepa" and GEPA_AVAILABLE:
            self.optimizer_type = "gepa"
            logger.info("CognitiveRAGOptimizer using GEPA optimizer")
        else:
            self.optimizer_type = "miprov2"
            if optimizer_type == "gepa":
                logger.warning("GEPA not available, falling back to MIPROv2")
            logger.info("CognitiveRAGOptimizer using MIPROv2 optimizer")

    def summarizer_metric(self, example, prediction, trace=None) -> float:
        """
        Metric for Summarizer optimization.
        Good summarization leads to better retrieval.
        """
        score = 0.0

        # Helper to get value from prediction (handles both dict and object)
        def get_val(key, default=""):
            if isinstance(prediction, dict):
                return prediction.get(key, default)
            return getattr(prediction, key, default)

        # Query rewrite should be more specific than original
        rewritten = get_val("rewritten_query", "")
        original = getattr(example, "original_query", "")
        if len(str(rewritten)) > len(str(original)):
            score += 0.2

        # Should extract at least one entity
        if get_val("graph_entities"):
            score += 0.3

        # Intent should be confident (not GENERAL)
        if get_val("primary_intent") != "GENERAL":
            score += 0.3

        # Search keywords should be pharmaceutical domain-specific
        pharma_terms = ["hcp", "patient", "brand", "conversion", "adoption"]
        if any(term in str(get_val("search_keywords")).lower() for term in pharma_terms):
            score += 0.2

        return score

    def investigator_metric(self, example, prediction, trace=None) -> float:
        """
        Metric for Investigator optimization.
        Good investigation finds relevant evidence efficiently.
        """
        score = 0.0

        # Should find evidence
        if hasattr(prediction, "evidence_board") and prediction.evidence_board:
            evidence_count = len(prediction.evidence_board)
            score += min(0.4, evidence_count * 0.1)  # Up to 0.4 for 4 pieces

        # Evidence should be relevant (high scores)
        if hasattr(prediction, "evidence_board"):
            avg_relevance = sum(e.relevance_score for e in prediction.evidence_board) / max(
                1, len(prediction.evidence_board)
            )
            score += avg_relevance * 0.3

        # Efficiency: fewer hops is better if evidence is sufficient
        if prediction.sufficient_evidence and prediction.hop_count <= 2:
            score += 0.3
        elif prediction.sufficient_evidence:
            score += 0.15

        return score

    def agent_metric(self, example, prediction, trace=None) -> float:
        """
        Metric for Agent/Synthesis optimization.
        Good synthesis integrates evidence coherently.
        """
        score = 0.0

        # Response should be substantive
        if len(prediction.response) > 200:
            score += 0.2

        # Should cite evidence
        if prediction.evidence_citations:
            score += min(0.3, len(prediction.evidence_citations) * 0.1)

        # Confidence statement should be present
        if prediction.confidence_statement and len(prediction.confidence_statement) > 20:
            score += 0.2

        # Visualization config should match intent
        if example.requires_visualization and prediction.chart_type != "none":
            score += 0.3

        return score

    async def optimize_phase(
        self,
        phase: Literal["summarizer", "investigator", "agent"],
        training_signals: List[Dict],
        budget: Union[int, str] = "medium",
    ) -> dspy.Module:
        """
        Run optimization for a specific phase.

        Args:
            phase: Which RAG phase to optimize (summarizer, investigator, agent)
            training_signals: Training signals collected from the Reflector
            budget: GEPA budget preset ("light", "medium", "heavy") or MIPROv2 trials count

        Returns:
            Optimized DSPy module for the phase
        """
        if self.optimizer_type == "gepa":
            return await self._optimize_with_gepa(phase, training_signals, budget)
        else:
            return await self._optimize_with_miprov2(phase, training_signals, budget)

    async def _optimize_with_gepa(
        self,
        phase: Literal["summarizer", "investigator", "agent"],
        training_signals: List[Dict],
        budget: Union[int, str] = "medium",
    ) -> dspy.Module:
        """
        Run GEPA optimization for a specific RAG phase.

        GEPA provides 10%+ improvement over MIPROv2 with:
        - Reflective evolution with rich textual feedback
        - RAGAS-based quality evaluation for RAG components
        - Better generalization through diverse candidate sampling
        """
        modules = {
            "summarizer": SummarizerModule,
            "investigator": InvestigatorModule,
            "agent": AgentModule,
        }

        # Convert signals to DSPy examples
        trainset = self._signals_to_examples(training_signals, phase)

        if not trainset:
            logger.warning(f"No training examples for {phase}, skipping optimization")
            return modules[phase]()

        # Split into train/val (80/20)
        split_idx = int(len(trainset) * 0.8)
        train_examples = trainset[:split_idx] if split_idx > 0 else trainset
        val_examples = trainset[split_idx:] if split_idx < len(trainset) else trainset[-2:]

        # Create RAGAS-based metric for RAG evaluation
        # Uses correct signature: create_ragas_metric(provider, agent_name, weights)
        phase_weights = self._get_phase_weights(phase)
        ragas_metric = create_ragas_metric(
            agent_name=f"cognitive_rag_{phase}",
            weights=phase_weights,
        )

        # Convert budget string to GEPA format if needed
        budget_preset = budget if isinstance(budget, str) else "medium"

        # Create GEPA optimizer
        optimizer = create_gepa_optimizer(
            metric=ragas_metric,  # type: ignore[arg-type]
            trainset=train_examples,
            valset=val_examples,
            budget=budget_preset,
            enable_tool_optimization=False,  # RAG doesn't use external tools
            seed=42,
        )

        # Get module to optimize
        module_class = modules[phase]
        module = module_class() if phase == "summarizer" else module_class({})

        # Run optimization
        logger.info(f"Starting GEPA optimization for {phase} with {len(train_examples)} examples")
        optimized = optimizer.compile(module, trainset=train_examples)

        # Save optimized module if successful
        if optimized and hasattr(optimizer, "best_score"):
            try:
                version_id = await save_optimized_module(  # type: ignore[call-arg, misc]
                    agent_name=f"cognitive_rag_{phase}",
                    optimized_module=optimized,
                    budget=budget_preset,
                    score=optimizer.best_score,
                )
                logger.info(f"Saved optimized {phase} module: {version_id}")
            except Exception as e:
                logger.warning(f"Could not save optimized module: {e}")

        return optimized

    async def _optimize_with_miprov2(
        self,
        phase: Literal["summarizer", "investigator", "agent"],
        training_signals: List[Dict],
        budget: Union[int, str] = 50,
    ) -> dspy.Module:
        """Run legacy MIPROv2 optimization for a specific phase."""
        modules = {
            "summarizer": SummarizerModule,
            "investigator": InvestigatorModule,
            "agent": AgentModule,
        }

        metrics = {
            "summarizer": self.summarizer_metric,
            "investigator": self.investigator_metric,
            "agent": self.agent_metric,
        }

        # Convert signals to DSPy examples
        trainset = self._signals_to_examples(training_signals, phase)

        if not trainset:
            logger.warning(f"No training examples for {phase}, skipping optimization")
            return modules[phase]()

        # Convert budget string to int if needed
        num_trials = budget if isinstance(budget, int) else 50

        optimizer = MIPROv2(
            metric=metrics[phase],
            auto=None,  # Disable auto mode to allow manual configuration
            num_candidates=10,
            max_bootstrapped_demos=4,
            num_threads=4,
        )

        module_class = modules[phase]
        module = module_class() if phase == "summarizer" else module_class({})
        optimized = optimizer.compile(module, trainset=trainset, num_trials=num_trials)

        return optimized

    def _get_phase_metric(self, phase: str):
        """Get the metric function for a phase."""
        metrics = {
            "summarizer": self.summarizer_metric,
            "investigator": self.investigator_metric,
            "agent": self.agent_metric,
        }
        return metrics.get(phase, self.summarizer_metric)

    def _get_phase_weights(self, phase: str) -> Optional[dict[str, float]]:
        """Get RAGAS metric weights optimized for a specific RAG phase.

        Different phases have different priorities:
        - Summarizer: Focus on relevancy (query understanding)
        - Investigator: Focus on precision and faithfulness (retrieval quality)
        - Agent: Balanced focus on all metrics (synthesis quality)

        Args:
            phase: The RAG phase name

        Returns:
            Dict of weights or None for default equal weights
        """
        weights = {
            "summarizer": {
                "faithfulness": 0.15,
                "answer_relevancy": 0.45,
                "context_precision": 0.25,
                "context_recall": 0.15,
            },
            "investigator": {
                "faithfulness": 0.30,
                "answer_relevancy": 0.15,
                "context_precision": 0.35,
                "context_recall": 0.20,
            },
            "agent": {
                "faithfulness": 0.30,
                "answer_relevancy": 0.30,
                "context_precision": 0.20,
                "context_recall": 0.20,
            },
        }
        return weights.get(phase)  # None for unknown phases uses default

    def _signals_to_examples(self, signals: List[Dict], phase: str) -> List[dspy.Example]:
        """Convert training signals to DSPy Examples."""
        examples = []
        for signal in signals:
            if signal["phase"] == phase and signal.get("success"):
                example = dspy.Example(**signal["input"], **signal["output"])
                examples.append(example.with_inputs(*signal["input"].keys()))
        return examples


# =============================================================================
# 8. PRODUCTION FACTORY & USAGE
# =============================================================================


def create_production_cognitive_workflow(
    supabase_client: Optional[Any] = None,
    falkordb_memory: Optional[Any] = None,
    memory_connector: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    domain_vocabulary: Optional[str] = None,
    lm_model: Optional[str] = None,
    configure_dspy: bool = True,
) -> Any:
    """
    Create a production cognitive workflow with real memory backends.

    This factory function wires up the CognitiveRAGWorkflow with real
    memory implementations (Supabase, FalkorDB) instead of mocks.

    Args:
        supabase_client: Supabase client for database access
        falkordb_memory: FalkorDBSemanticMemory instance
        memory_connector: MemoryConnector instance for hybrid retrieval
        embedding_model: Embedding model for vector operations
        agent_registry: Dict of available specialized agents
        domain_vocabulary: Domain vocabulary for query understanding
        lm_model: DSPy language model to use
        configure_dspy: Whether to configure DSPy LM (set False if already configured)

    Returns:
        LangGraph workflow configured with production backends

    Example:
        from supabase import create_client
        from src.rag.memory_connector import MemoryConnector
        from src.memory.semantic_memory import FalkorDBSemanticMemory

        client = create_client(url, key)
        connector = MemoryConnector(client)
        falkordb = FalkorDBSemanticMemory(...)

        workflow = create_production_cognitive_workflow(
            supabase_client=client,
            falkordb_memory=falkordb,
            memory_connector=connector,
        )

        # The workflow is compiled with a MemorySaver checkpointer, so a
        # thread_id config is REQUIRED (omitting it raises ValueError).
        state = CognitiveState(
            user_query="Why did Kisqali adoption increase?",
            conversation_id="session-123",
        )
        result = await workflow.ainvoke(
            state,
            config={"configurable": {"thread_id": state.conversation_id}},
        )
    """
    # Import adapters here to avoid circular imports
    from src.rag.memory_adapters import (
        EpisodicMemoryAdapter,
        ProceduralMemoryAdapter,
        SemanticMemoryAdapter,
        SignalCollectorAdapter,
    )

    # Configure DSPy if requested. Resolve the model from env (provider-aware)
    # when not explicitly given, so we never reconfigure onto a retired model
    # the deployed key cannot serve.
    #
    # #1475: guard on an already-configured LM, mirroring every
    # request-reachable configure site (canonical pattern:
    # src/api/routes/chatbot_dspy.py:62). dspy 3.1.0's FIRST configure
    # permanently binds an owner thread/task and a later configure from a
    # different thread raises RuntimeError — without this guard any future
    # non-owner-thread caller crashes even though an LM is already usable.
    # When the guard skips, the EXISTING global LM wins over ``lm_model``;
    # callers needing a different model must configure it on the owner thread.
    if configure_dspy:
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            logger.debug(
                "DSPy LM already configured — skipping configure "
                "(dspy 3.1.0 cross-thread configure would raise)"
            )
        else:
            from src.optimization.dspy_lm import get_default_dspy_model

            lm = dspy.LM(lm_model or get_default_dspy_model())
            dspy.configure(lm=lm)

    # Create adapters that wrap real backends
    episodic_adapter = EpisodicMemoryAdapter(
        memory_connector=memory_connector,
        embedding_model=embedding_model,
    )
    semantic_adapter = SemanticMemoryAdapter(
        falkordb_memory=falkordb_memory,
        memory_connector=memory_connector,
    )
    procedural_adapter = ProceduralMemoryAdapter(
        supabase_client=supabase_client,
        embedding_model=embedding_model,
    )
    signal_collector = SignalCollectorAdapter(
        supabase_client=supabase_client,
    )

    # Configure memory backends for the workflow
    memory_backends = {
        "episodic": episodic_adapter,
        "semantic": semantic_adapter,
        "procedural": procedural_adapter,
    }

    # Create the workflow
    return create_dspy_cognitive_workflow(
        memory_backends=memory_backends,
        memory_writers=memory_backends,  # Adapters handle writes too
        agent_registry=agent_registry or {},
        signal_collector=signal_collector,
        domain_vocabulary=domain_vocabulary or _default_domain_vocabulary(),
    )


def _default_domain_vocabulary() -> str:
    """Return default E2I domain vocabulary."""
    return """
    brands: [Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2- breast cancer)]
    kpis: [TRx, NRx, conversion_rate, market_share, adoption_rate]
    entities: [HCP, patient, territory, region, therapeutic_area]
    metrics: [prescriptions, visits, detailing, samples]
    """


async def main():
    """Example usage of DSPy-enhanced cognitive RAG."""

    # Configure DSPy (provider-aware default; honors LLM_PROVIDER / DSPY_LM_MODEL)
    from src.optimization.dspy_lm import get_default_dspy_model

    lm = dspy.LM(get_default_dspy_model())
    dspy.configure(lm=lm)

    # Mock backends (for demo - use create_production_cognitive_workflow for production)
    memory_backends = {
        "episodic": MockEpisodicMemory(),
        "semantic": MockSemanticMemory(),
        "procedural": MockProceduralMemory(),
    }

    # Create workflow
    workflow = create_dspy_cognitive_workflow(
        memory_backends=memory_backends,
        memory_writers=memory_backends,  # Same for demo
        agent_registry={},
        signal_collector=MockSignalCollector(),
        domain_vocabulary="brands: [Remibrutinib, Fabhalta, Kisqali]...",
    )

    # Run cognitive cycle. The workflow is compiled with a MemorySaver
    # checkpointer, so ainvoke REQUIRES a thread_id config; we seed it from the
    # same conversation_id the state carries.
    initial_state = CognitiveState(
        user_query="Why did Kisqali adoption increase in the Northeast last quarter?",
        conversation_id="demo-123",
    )

    result = await workflow.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": initial_state.conversation_id}},
    )

    print(f"Response: {result.response}")
    print(f"Hops: {result.hop_count}")
    print(f"Evidence: {len(result.evidence_board)} pieces")
    print(f"DSPy signals collected: {len(result.dspy_signals)}")


# =============================================================================
# MOCK BACKENDS (for testing and demos only)
# =============================================================================


class MockEpisodicMemory:
    async def vector_search(self, query, limit):
        return [{"content": "Kisqali adoption increased 15% in Q3..."}]


class MockSemanticMemory:
    async def graph_query(self, query, max_depth):
        return [{"content": "Northeast region CONNECTED_TO high oncologist density..."}]


class MockProceduralMemory:
    async def procedure_search(self, query, limit):
        return [{"content": "For adoption analysis: query episodic → check regional factors..."}]


class MockSignalCollector:
    async def collect(self, signals):
        print(f"Collected {len(signals)} training signals")


if __name__ == "__main__":
    asyncio.run(main())
