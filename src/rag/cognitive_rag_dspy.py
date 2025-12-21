"""
E2I Cognitive RAG + DSPy Integration
=====================================

This module extends the 4-phase cognitive cycle with DSPy optimization
for each node in the retrieval and reasoning pipeline.

Architecture:
- Phase 1: Summarizer Node → DSPy Query Rewriting
- Phase 2: Investigator Node → DSPy Hop Decision Making
- Phase 3: Agent Node → DSPy Response Synthesis
- Phase 4: Reflector Node → DSPy Training Signal Collection (existing)

The key insight: Each phase has LLM-driven decisions that can be
optimized through DSPy signatures and modules.
"""

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot, COPRO
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import asyncio


# =============================================================================
# 1. MEMORY TYPES & HOP DEFINITIONS
# =============================================================================

class MemoryType(Enum):
    """The 4 memory types in the Agentic Memory architecture."""
    WORKING = "working"       # Redis + LangGraph MemorySaver
    EPISODIC = "episodic"     # Supabase + pgvector (experiences)
    SEMANTIC = "semantic"     # FalkorDB + Graphity (relationships)
    PROCEDURAL = "procedural" # Supabase + pgvector (skills)


class HopType(Enum):
    """Multi-hop investigation sequence."""
    HOP_1_EPISODIC = "episodic"      # "What happened?"
    HOP_2_SEMANTIC = "semantic"      # "Who/What related?"
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
    original_query: str = dspy.InputField(
        desc="The user's original natural language question"
    )
    conversation_context: str = dspy.InputField(
        desc="Recent conversation history for context"
    )
    domain_vocabulary: str = dspy.InputField(
        desc="Available domain terms: brands, regions, stages, HCP types"
    )
    
    rewritten_query: str = dspy.OutputField(
        desc="Optimized query for hybrid retrieval (dense + sparse + graph)"
    )
    search_keywords: list = dspy.OutputField(
        desc="Key terms for full-text search"
    )
    graph_entities: list = dspy.OutputField(
        desc="Entities to anchor graph traversal"
    )


class EntityExtractionSignature(dspy.Signature):
    """
    Extract pharmaceutical domain entities from user query.
    Maps to E2I domain vocabulary.
    """
    query: str = dspy.InputField(desc="User query or message")
    domain_vocabulary: str = dspy.InputField(desc="Domain vocabulary YAML")
    
    brands: list = dspy.OutputField(
        desc="Brand names mentioned (Remibrutinib, Fabhalta, Kisqali)"
    )
    regions: list = dspy.OutputField(
        desc="Geographic regions (Northeast, Midwest, etc.)"
    )
    hcp_types: list = dspy.OutputField(
        desc="HCP specialties (Oncologist, Rheumatologist, etc.)"
    )
    patient_stages: list = dspy.OutputField(
        desc="Patient journey stages (Diagnosis, Treatment, etc.)"
    )
    time_references: list = dspy.OutputField(
        desc="Temporal references (last quarter, YTD, etc.)"
    )


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
    secondary_intents: list = dspy.OutputField(
        desc="Additional relevant intents"
    )
    requires_visualization: bool = dspy.OutputField(
        desc="Whether query requires chart/graph output"
    )
    complexity: str = dspy.OutputField(
        desc="Query complexity: SIMPLE | MODERATE | COMPLEX"
    )


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
        self,
        original_query: str,
        conversation_context: str,
        domain_vocabulary: str
    ) -> CognitiveState:
        # Step 1: Extract entities
        entities = self.extract(
            query=original_query,
            domain_vocabulary=domain_vocabulary
        )
        
        # Step 2: Rewrite query for retrieval
        rewritten = self.rewrite(
            original_query=original_query,
            conversation_context=conversation_context,
            domain_vocabulary=domain_vocabulary
        )
        
        # Step 3: Classify intent
        entities_json = str({
            "brands": entities.brands,
            "regions": entities.regions,
            "hcp_types": entities.hcp_types,
            "patient_stages": entities.patient_stages,
            "time_references": entities.time_references
        })
        
        intent = self.classify(
            query=original_query,
            extracted_entities=entities_json
        )
        
        return {
            "rewritten_query": rewritten.rewritten_query,
            "search_keywords": rewritten.search_keywords,
            "graph_entities": rewritten.graph_entities,
            "extracted_entities": entities_json,
            "primary_intent": intent.primary_intent,
            "secondary_intents": intent.secondary_intents,
            "requires_visualization": intent.requires_visualization,
            "complexity": intent.complexity
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
    max_hops: int = dspy.OutputField(
        desc="Maximum number of hops needed (1-4)"
    )
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
    retrieval_query: str = dspy.OutputField(
        desc="Specific query for the next memory store"
    )
    reasoning: str = dspy.OutputField(
        desc="Why this hop is needed or why to stop"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence that more evidence is needed (0.0-1.0)"
    )


class EvidenceRelevanceSignature(dspy.Signature):
    """
    Score retrieved evidence for relevance to investigation goal.
    Filters noise and ranks evidence quality.
    """
    investigation_goal: str = dspy.InputField(desc="What we're trying to discover")
    evidence_item: str = dspy.InputField(desc="A single piece of retrieved evidence")
    source_memory: str = dspy.InputField(desc="Which memory store this came from")
    
    relevance_score: float = dspy.OutputField(
        desc="Relevance score 0.0-1.0"
    )
    key_insight: str = dspy.OutputField(
        desc="The key insight this evidence provides"
    )
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
    
    async def forward(
        self,
        rewritten_query: str,
        intent: str,
        entities: str
    ) -> Dict:
        # Step 1: Plan investigation
        plan = self.plan(
            query=rewritten_query,
            intent=intent,
            entities=entities
        )
        
        evidence_board = []
        queried_memories = set()
        
        # Step 2: Iterative hop execution
        for hop_num in range(1, self.max_hops + 1):
            available = [m for m in ["episodic", "semantic", "procedural"] 
                        if m not in queried_memories]
            
            if not available:
                break
            
            # Decide next hop
            decision = self.decide_hop(
                investigation_goal=plan.investigation_goal,
                current_evidence=str([e.__dict__ for e in evidence_board]),
                hop_number=hop_num,
                available_memories=available
            )
            
            if decision.next_memory == "STOP":
                break
            
            # Execute retrieval
            raw_evidence = await self._retrieve_from_memory(
                decision.next_memory,
                decision.retrieval_query
            )
            
            # Score and filter evidence
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
            
            queried_memories.add(decision.next_memory)
            
            # Check early stop
            if decision.confidence < 0.3:  # Low confidence = sufficient evidence
                break
        
        return {
            "investigation_goal": plan.investigation_goal,
            "evidence_board": evidence_board,
            "hop_count": len(queried_memories),
            "sufficient_evidence": len(evidence_board) >= 2
        }
    
    async def _retrieve_from_memory(
        self,
        memory_type: str,
        query: str
    ) -> List[Dict]:
        """Execute retrieval against the appropriate memory backend."""
        backend = self.memory_backends.get(memory_type)
        if not backend:
            return []
        
        if memory_type == "episodic":
            # pgvector semantic search
            return await backend.vector_search(query, limit=5)
        elif memory_type == "semantic":
            # FalkorDB graph traversal
            return await backend.graph_query(query, max_depth=2)
        elif memory_type == "procedural":
            # pgvector similarity on tool sequences
            return await backend.procedure_search(query, limit=3)
        
        return []


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
    
    synthesis: str = dspy.OutputField(
        desc="Synthesized answer integrating all evidence"
    )
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
    supporting_agents: list = dspy.OutputField(
        desc="Additional agents to involve"
    )
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
    chart_config: str = dspy.OutputField(
        desc="JSON configuration for the chart"
    )
    highlights: list = dspy.OutputField(
        desc="Key data points to highlight"
    )


class AgentModule(dspy.Module):
    """
    DSPy module for Phase 3: Agent Node.
    Synthesizes evidence and routes to appropriate E2I agents.
    """
    
    def __init__(self, agent_registry: Dict[str, Any]):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(EvidenceSynthesisSignature)
        self.route = dspy.Predict(AgentRoutingSignature)
        self.visualize = dspy.Predict(VisualizationConfigSignature)
        self.agent_registry = agent_registry
    
    async def forward(
        self,
        state: CognitiveState
    ) -> CognitiveState:
        # Step 1: Synthesize evidence
        synthesis = self.synthesize(
            user_query=state.user_query,
            investigation_goal=state.investigation_goal,
            evidence_board=str([e.__dict__ for e in state.evidence_board]),
            intent=state.detected_intent
        )
        
        # Step 2: Determine agent routing
        routing = self.route(
            intent=state.detected_intent,
            complexity="COMPLEX" if state.hop_count > 2 else "MODERATE",
            evidence_summary=synthesis.synthesis[:500]
        )
        
        # Step 3: Execute primary agent
        primary_agent = self.agent_registry.get(routing.primary_agent)
        if primary_agent:
            agent_response = await primary_agent.process(
                query=state.user_query,
                evidence=state.evidence_board,
                synthesis=synthesis.synthesis
            )
            state.response = agent_response
        else:
            state.response = synthesis.synthesis
        
        # Step 4: Generate visualization if needed
        if state.detected_intent in ["CAUSAL_ANALYSIS", "GAP_ANALYSIS", "PREDICTION"]:
            viz = self.visualize(
                synthesis=synthesis.synthesis,
                data_types=["temporal", "categorical", "causal"],
                user_preference=""
            )
            state.visualization_config = {
                "chart_type": viz.chart_type,
                "config": viz.chart_config,
                "highlights": viz.highlights
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
    
    worth_remembering: bool = dspy.OutputField(
        desc="Whether to store in episodic memory"
    )
    memory_type: str = dspy.OutputField(
        desc="Which memory: episodic | semantic | procedural | none"
    )
    importance_score: float = dspy.OutputField(
        desc="Importance for future retrieval (0.0-1.0)"
    )
    key_facts: list = dspy.OutputField(
        desc="Facts to extract for semantic memory"
    )


class ProcedureLearningSignature(dspy.Signature):
    """
    Extract successful tool/agent sequences for procedural memory.
    Learns patterns that worked well.
    """
    query_type: str = dspy.InputField(desc="Type of query handled")
    agents_used: list = dspy.InputField(desc="Agents that processed this query")
    hop_sequence: list = dspy.InputField(desc="Memory hops executed")
    success_indicators: str = dspy.InputField(desc="Signals of successful response")
    
    procedure_pattern: str = dspy.OutputField(
        desc="Generalized procedure pattern"
    )
    trigger_conditions: list = dspy.OutputField(
        desc="When to apply this procedure"
    )
    expected_outcome: str = dspy.OutputField(
        desc="What outcome this procedure achieves"
    )


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
        self,
        state: CognitiveState,
        user_feedback: Optional[str] = None
    ) -> CognitiveState:
        # Step 1: Evaluate memory worthiness
        evaluation = self.evaluate(
            user_query=state.user_query,
            response=state.response,
            evidence_count=len(state.evidence_board),
            user_feedback=user_feedback or ""
        )
        
        state.worth_remembering = evaluation.worth_remembering
        
        # Step 2: Store in appropriate memory
        if evaluation.worth_remembering:
            if evaluation.memory_type == "episodic":
                await self.memory_writers["episodic"].store({
                    "query": state.user_query,
                    "response": state.response,
                    "importance": evaluation.importance_score
                })
            
            if evaluation.key_facts:
                state.extracted_facts = evaluation.key_facts
                for fact in evaluation.key_facts:
                    await self.memory_writers["semantic"].add_fact(fact)
        
        # Step 3: Learn procedures from successful interactions
        if user_feedback and "positive" in user_feedback.lower():
            procedure = self.learn_procedure(
                query_type=state.detected_intent,
                agents_used=state.routed_agents,
                hop_sequence=[e.source.value for e in state.evidence_board],
                success_indicators=user_feedback
            )
            
            state.learned_procedures.append({
                "pattern": procedure.procedure_pattern,
                "triggers": procedure.trigger_conditions,
                "outcome": procedure.expected_outcome
            })
            
            await self.memory_writers["procedural"].store_procedure(
                state.learned_procedures[-1]
            )
        
        # Step 4: Collect DSPy training signals for Feedback Learner
        state.dspy_signals = self._collect_training_signals(state, user_feedback)
        await self.signal_collector.collect(state.dspy_signals)
        
        return state
    
    def _collect_training_signals(
        self,
        state: CognitiveState,
        user_feedback: Optional[str]
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
        summarizer_reward = min(1.0, len(state.evidence_board) / 4.0) if state.sufficient_evidence else 0.3
        investigator_reward = min(1.0, sum(e.relevance_score for e in state.evidence_board) / 3.0) if state.evidence_board else 0.0
        agent_reward = 0.8 if state.response else 0.0

        # Adjust rewards based on user feedback if provided
        if user_feedback:
            feedback_boost = 0.2 if "positive" in user_feedback.lower() else -0.1
            summarizer_reward = min(1.0, max(0.0, summarizer_reward + feedback_boost))
            investigator_reward = min(1.0, max(0.0, investigator_reward + feedback_boost))
            agent_reward = min(1.0, max(0.0, agent_reward + feedback_boost))

        # Signal for Summarizer optimization (query rewrite, entity extraction, intent)
        signals.append({
            "type": "summarizer",
            "query": state.user_query,
            "response": state.rewritten_query or state.user_query,
            "reward": summarizer_reward,
            "feedback": {"user_feedback": user_feedback} if user_feedback else None,
            "metadata": {
                "entities": state.extracted_entities,
                "intent": state.detected_intent,
                "conversation_id": state.conversation_id,
            }
        })

        # Signal for Investigator optimization (multi-hop retrieval)
        signals.append({
            "type": "investigator",
            "query": state.investigation_goal or state.rewritten_query or state.user_query,
            "response": f"Found {len(state.evidence_board)} evidence items in {state.hop_count} hops",
            "reward": investigator_reward,
            "feedback": {"sufficient_evidence": state.sufficient_evidence} if state.sufficient_evidence else None,
            "metadata": {
                "hop_count": state.hop_count,
                "evidence_count": len(state.evidence_board),
                "evidence_sources": [e.source.value for e in state.evidence_board],
                "conversation_id": state.conversation_id,
            }
        })

        # Signal for Agent/Synthesis optimization
        signals.append({
            "type": "agent",
            "query": f"Intent: {state.detected_intent}, Evidence: {len(state.evidence_board)} items",
            "response": state.response[:500] if state.response else "",
            "reward": agent_reward,
            "feedback": {"user_feedback": user_feedback} if user_feedback else None,
            "metadata": {
                "routed_agents": state.routed_agents,
                "has_visualization": bool(state.visualization_config),
                "response_length": len(state.response) if state.response else 0,
                "conversation_id": state.conversation_id,
            }
        })

        return signals


# =============================================================================
# 6. COMPLETE COGNITIVE WORKFLOW WITH DSPy
# =============================================================================

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def create_dspy_cognitive_workflow(
    memory_backends: Dict[str, Any],
    memory_writers: Dict[str, Any],
    agent_registry: Dict[str, Any],
    signal_collector: Any,
    domain_vocabulary: str
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
        result = summarizer.forward(
            original_query=state.user_query,
            conversation_context=state.compressed_history,
            domain_vocabulary=domain_vocabulary
        )
        
        state.rewritten_query = result["rewritten_query"]
        state.extracted_entities = result["extracted_entities"]
        state.detected_intent = result["primary_intent"]
        
        return state
    
    # Phase 2: Investigator Node
    async def investigator_node(state: CognitiveState) -> CognitiveState:
        result = await investigator.forward(
            rewritten_query=state.rewritten_query,
            intent=state.detected_intent,
            entities=state.extracted_entities
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
    
    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# 7. DSPy OPTIMIZATION TARGETS FOR RAG
# =============================================================================

class CognitiveRAGOptimizer:
    """
    Optimizer specifically for the 4-phase cognitive RAG system.
    Defines metrics and optimization strategies for each phase.
    """
    
    def __init__(self, feedback_learner: Any):
        self.feedback_learner = feedback_learner
    
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
        if hasattr(prediction, 'evidence_board') and prediction.evidence_board:
            evidence_count = len(prediction.evidence_board)
            score += min(0.4, evidence_count * 0.1)  # Up to 0.4 for 4 pieces
        
        # Evidence should be relevant (high scores)
        if hasattr(prediction, 'evidence_board'):
            avg_relevance = sum(e.relevance_score for e in prediction.evidence_board) / max(1, len(prediction.evidence_board))
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
        budget: int = 50
    ) -> dspy.Module:
        """Run MIPROv2 optimization for a specific phase."""
        modules = {
            "summarizer": SummarizerModule,
            "investigator": InvestigatorModule,
            "agent": AgentModule
        }
        
        metrics = {
            "summarizer": self.summarizer_metric,
            "investigator": self.investigator_metric,
            "agent": self.agent_metric
        }
        
        # Convert signals to DSPy examples
        trainset = self._signals_to_examples(training_signals, phase)
        
        optimizer = MIPROv2(
            metric=metrics[phase],
            auto=None,  # Disable auto mode to allow manual configuration
            num_candidates=10,
            max_bootstrapped_demos=4,
            num_threads=4
        )
        
        module_class = modules[phase]
        optimized = optimizer.compile(
            module_class(),
            trainset=trainset,
            num_trials=budget
        )
        
        return optimized
    
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
    lm_model: str = "anthropic/claude-sonnet-4-20250514",
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

        result = await workflow.ainvoke(CognitiveState(
            user_query="Why did Kisqali adoption increase?"
        ))
    """
    # Import adapters here to avoid circular imports
    from src.rag.memory_adapters import (
        EpisodicMemoryAdapter,
        SemanticMemoryAdapter,
        ProceduralMemoryAdapter,
        SignalCollectorAdapter,
    )

    # Configure DSPy if requested
    if configure_dspy:
        lm = dspy.LM(lm_model)
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

    # Configure DSPy
    lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
    dspy.configure(lm=lm)

    # Mock backends (for demo - use create_production_cognitive_workflow for production)
    memory_backends = {
        "episodic": MockEpisodicMemory(),
        "semantic": MockSemanticMemory(),
        "procedural": MockProceduralMemory()
    }

    # Create workflow
    workflow = create_dspy_cognitive_workflow(
        memory_backends=memory_backends,
        memory_writers=memory_backends,  # Same for demo
        agent_registry={},
        signal_collector=MockSignalCollector(),
        domain_vocabulary="brands: [Remibrutinib, Fabhalta, Kisqali]..."
    )

    # Run cognitive cycle
    initial_state = CognitiveState(
        user_query="Why did Kisqali adoption increase in the Northeast last quarter?",
        conversation_id="demo-123"
    )

    result = await workflow.ainvoke(initial_state)

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
