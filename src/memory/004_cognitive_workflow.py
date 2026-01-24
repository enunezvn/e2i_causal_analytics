"""
E2I Agentic Memory - Cognitive Workflow
LangGraph State Machine for Deep Agentic Memory

This module implements the 4-phase cognitive cycle:
1. Summarizer Node - Context pruning and compression
2. Investigator Node - Multi-hop retrieval across memory types
3. Agent Node - Synthesis and response generation
4. Reflector Node - Asynchronous learning and memory updates

Version: 1.1
"""

import hashlib
import logging
import operator
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# EVIDENCE EVALUATION CACHE
# ============================================================================
# Moved to src/memory/evidence_cache.py for proper module imports
from src.memory.evidence_cache import (
    EvidenceEvaluationCache,
    get_evidence_cache,
    is_evidence_cache_enabled,
)


# ============================================================================
# STATE DEFINITIONS
# ============================================================================


class EvidenceItem(BaseModel):
    """Single piece of evidence from investigation."""

    hop_number: int
    source: Literal["episodic", "procedural", "semantic", "agent_output"]
    query_type: str
    content: str
    raw_data: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    selected: bool = False


class Message(BaseModel):
    """Conversation message."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CognitiveState(TypedDict):
    """
    Complete state for the cognitive workflow.

    This state flows through all 4 phases and accumulates
    context, evidence, and learning signals.
    """

    # === Session Identity ===
    session_id: str
    cycle_id: str
    user_id: Optional[str]

    # === Current Input ===
    user_query: str
    query_embedding: Optional[List[float]]
    detected_intent: Optional[str]
    detected_entities: Dict[str, List[str]]  # {"brands": [], "regions": [], ...}

    # === Conversation Context ===
    messages: Annotated[List[Message], operator.add]  # Append-only
    conversation_summary: Optional[str]  # Compressed history
    message_count: int

    # === Phase 1: Summarizer Output ===
    context_compressed: bool
    compression_ratio: Optional[float]

    # === Phase 2: Investigator Output ===
    investigation_goal: Optional[str]
    current_hop: int
    max_hops: int
    evidence_trail: List[EvidenceItem]
    investigation_complete: bool
    investigation_decision: Optional[str]  # "sufficient", "max_hops", "no_data"

    # === Phase 3: Agent Output ===
    agents_to_invoke: List[str]
    agent_outputs: Dict[str, Any]
    synthesized_response: Optional[str]
    visualization_config: Optional[Dict[str, Any]]
    confidence_score: Optional[float]

    # === Phase 4: Reflector Output ===
    worth_remembering: bool
    new_facts: List[Dict[str, Any]]  # Triplets for semantic memory
    new_procedures: List[Dict[str, Any]]  # Tool sequences for procedural
    feedback_signals: List[Dict[str, Any]]

    # === Timing ===
    phase_timings: Dict[str, Dict[str, datetime]]

    # === Error Handling ===
    error: Optional[str]
    phase_completed: Literal["init", "summarizer", "investigator", "agent", "reflector"]


# ============================================================================
# PHASE 1: SUMMARIZER NODE
# Context pruning and compression
# ============================================================================


async def summarizer_node(state: CognitiveState) -> CognitiveState:
    """
    Phase 1: Input & Context Pruning

    Responsibilities:
    - Ingest user message into state
    - Compress old messages if history > threshold
    - Extract entities and intent from query
    - Prepare clean context for investigation
    """
    from .memory_backends import get_embedding_service, get_llm_service

    # Record phase start
    state["phase_timings"]["summarizer"] = {"start": datetime.now(timezone.utc)}

    # Add user message to conversation
    user_message = Message(role="user", content=state["user_query"])
    state["messages"] = [user_message]  # Will be appended via operator.add
    state["message_count"] += 1

    # Generate query embedding for similarity search
    embedding_service = get_embedding_service()
    state["query_embedding"] = await embedding_service.embed(state["user_query"])

    # Context compression if needed (threshold: 10 messages)
    COMPRESSION_THRESHOLD = 10
    if state["message_count"] > COMPRESSION_THRESHOLD:
        llm = get_llm_service()

        # Get oldest messages to compress
        old_messages = state["messages"][:-5]  # Keep last 5
        messages_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])

        # Generate summary
        summary_prompt = f"""Summarize this conversation history concisely,
        preserving key context, entities mentioned, and any decisions made:

        {messages_text}

        Summary:"""

        state["conversation_summary"] = await llm.complete(summary_prompt)
        state["context_compressed"] = True
        state["compression_ratio"] = len(state["conversation_summary"]) / len(messages_text)
    else:
        state["context_compressed"] = False

    # Extract entities from query using vocabulary
    state["detected_entities"] = await extract_entities(state["user_query"])

    # Detect intent for routing
    state["detected_intent"] = await detect_intent(state["user_query"])

    # Record phase end
    state["phase_timings"]["summarizer"]["end"] = datetime.now(timezone.utc)
    state["phase_completed"] = "summarizer"

    return state


async def extract_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract E2I domain entities from query.
    Uses vocabulary from memory_vocabulary.yaml
    """
    from pathlib import Path

    import yaml

    # Load vocabulary
    vocab_path = Path(__file__).parent.parent.parent / "config" / "domain_vocabulary.yaml"
    with open(vocab_path) as f:
        vocab = yaml.safe_load(f)

    entities = {"brands": [], "regions": [], "kpis": [], "agents": [], "time_periods": []}

    query_lower = query.lower()

    # Match brands
    for brand_id, brand_info in vocab["entity_types"]["brand"]["vocabulary"].items():
        if brand_info["brand_name"].lower() in query_lower:
            entities["brands"].append(brand_id)
        for alias in brand_info.get("aliases", []):
            if alias.lower() in query_lower:
                entities["brands"].append(brand_id)
                break

    # Match regions
    for region_id in vocab["entity_types"]["region"]["vocabulary"].keys():
        if region_id in query_lower:
            entities["regions"].append(region_id)

    # Match KPI categories
    for _category, kpis in vocab["entity_types"]["kpi"]["categories"].items():
        for kpi in kpis:
            if kpi.replace("_", " ") in query_lower:
                entities["kpis"].append(kpi)

    return entities


async def detect_intent(query: str) -> str:
    """
    Detect query intent for agent routing.
    Uses intent_vocabulary from memory_vocabulary.yaml
    """
    from pathlib import Path

    import yaml

    vocab_path = Path(__file__).parent.parent.parent / "config" / "domain_vocabulary.yaml"
    with open(vocab_path) as f:
        vocab = yaml.safe_load(f)

    query_lower = query.lower()
    intent_scores = {}

    for intent_name, intent_info in vocab["intent_vocabulary"].items():
        score = sum(1 for kw in intent_info["keywords"] if kw in query_lower)
        if score > 0:
            intent_scores[intent_name] = score

    if intent_scores:
        return max(intent_scores, key=intent_scores.get)
    return "exploration_intents"  # Default


# ============================================================================
# PHASE 2: INVESTIGATOR NODE
# Multi-hop retrieval across memory stores
# ============================================================================


async def investigator_node(state: CognitiveState) -> CognitiveState:
    """
    Phase 2: Multi-Hop Investigation

    Responsibilities:
    - Set investigation goal based on query
    - Execute iterative retrieval (up to max_hops)
    - Build evidence trail from multiple memory sources
    - Decide when sufficient evidence is gathered
    """
    from .memory_backends import (
        find_relevant_procedures,
        get_llm_service,
        query_semantic_graph,
        search_episodic_memory,
    )

    state["phase_timings"]["investigator"] = {"start": datetime.now(timezone.utc)}

    # Set investigation goal
    llm = get_llm_service()
    goal_prompt = f"""Given this user query: "{state['user_query']}"
    And detected intent: {state['detected_intent']}
    And entities: {state['detected_entities']}

    What specific information do we need to find to answer this query?
    Be specific about what facts, relationships, or historical events are needed.

    Investigation goal:"""

    state["investigation_goal"] = await llm.complete(goal_prompt)

    # Initialize investigation
    state["current_hop"] = 0
    state["evidence_trail"] = []
    state["investigation_complete"] = False

    # Investigation loop
    while state["current_hop"] < state["max_hops"] and not state["investigation_complete"]:
        state["current_hop"] += 1
        hop_evidence = []

        # Hop 1: Always start with episodic memory (what happened before?)
        if state["current_hop"] == 1:
            episodic_results = await search_episodic_memory(
                embedding=state["query_embedding"],
                filters={"event_type": None, "entities": state["detected_entities"]},  # All types
                limit=5,
            )

            for result in episodic_results:
                hop_evidence.append(
                    EvidenceItem(
                        hop_number=1,
                        source="episodic",
                        query_type="vector_search",
                        content=result["description"],
                        raw_data=result,
                        relevance_score=result.get("similarity", 0.0),
                    )
                )

        # Hop 2: Check procedural memory (how did we handle similar queries?)
        elif state["current_hop"] == 2:
            procedures = await find_relevant_procedures(embedding=state["query_embedding"], limit=3)

            for proc in procedures:
                hop_evidence.append(
                    EvidenceItem(
                        hop_number=2,
                        source="procedural",
                        query_type="procedure_match",
                        content=f"Procedure: {proc['procedure_name']} (success rate: {proc['success_rate']:.0%})",
                        raw_data=proc,
                        relevance_score=proc.get("similarity", 0.0),
                    )
                )

        # Hop 3+: Query semantic graph based on evidence so far
        else:
            # Build graph query based on accumulated evidence
            graph_query = await build_graph_query(state)

            graph_results = await query_semantic_graph(graph_query)

            for result in graph_results:
                hop_evidence.append(
                    EvidenceItem(
                        hop_number=state["current_hop"],
                        source="semantic",
                        query_type="graph_traversal",
                        content=f"{result['subject']} -{result['predicate']}-> {result['object']}",
                        raw_data=result,
                        relevance_score=result.get("confidence", 0.5),
                    )
                )

        # Add hop evidence to trail
        state["evidence_trail"].extend(hop_evidence)

        # Decide: Do we have enough evidence?
        decision = await evaluate_evidence(state, hop_evidence)

        if decision == "sufficient":
            state["investigation_complete"] = True
            state["investigation_decision"] = "sufficient"
        elif decision == "no_more_relevant":
            state["investigation_complete"] = True
            state["investigation_decision"] = "no_data"

    # If we hit max hops
    if not state["investigation_complete"]:
        state["investigation_complete"] = True
        state["investigation_decision"] = "max_hops"

    # Select most relevant evidence for agent phase
    state["evidence_trail"] = select_top_evidence(state["evidence_trail"], top_k=10)

    state["phase_timings"]["investigator"]["end"] = datetime.now(timezone.utc)
    state["phase_completed"] = "investigator"

    return state


async def build_graph_query(state: CognitiveState) -> Dict[str, Any]:
    """Build semantic graph query based on accumulated evidence."""
    # Extract entity IDs from evidence for graph traversal
    entity_ids = []
    for evidence in state["evidence_trail"]:
        if evidence.raw_data:
            # Extract entity references from evidence
            entities = evidence.raw_data.get("entities", {})
            for _entity_type, ids in entities.items():
                entity_ids.extend(ids)

    return {
        "start_nodes": entity_ids,
        "relationship_types": ["CAUSES", "IMPACTS", "PRESCRIBES", "TREATED_BY"],
        "max_depth": 2,
        "filters": state["detected_entities"],
    }


async def evaluate_evidence(state: CognitiveState, new_evidence: List[EvidenceItem]) -> str:
    """
    Evaluate if current evidence is sufficient to answer the query.

    Uses a multi-tier evaluation strategy:
    1. Empty evidence check - fast path for no new evidence
    2. Heuristic check - if 3+ high-relevance items, consider sufficient
    3. Cache check - avoid redundant LLM calls for similar evaluations
    4. LLM judgment - final arbiter when heuristics are inconclusive
    """
    from .memory_backends import get_llm_service

    if not new_evidence:
        return "no_more_relevant"

    # Quick heuristic: if we have high-relevance evidence, we might be done
    high_relevance = [e for e in new_evidence if e.relevance_score > 0.8]
    if len(high_relevance) >= 3:
        return "sufficient"

    # Build evidence summary for cache lookup and LLM prompt
    evidence_summary = "\n".join(
        [
            f"- [{e.source}] {e.content[:100]} (relevance: {e.relevance_score:.2f})"
            for e in state["evidence_trail"][-10:]  # Last 10 pieces
        ]
    )

    investigation_goal = state.get("investigation_goal", state["user_query"])

    # Check cache before making LLM call
    if is_evidence_cache_enabled():
        cache = get_evidence_cache()
        cached_result = cache.get(investigation_goal, evidence_summary)
        if cached_result:
            return cached_result

    # Otherwise, use LLM to judge
    llm = get_llm_service()

    judge_prompt = f"""Goal: {investigation_goal}

    Evidence collected so far:
    {evidence_summary}

    Is this evidence sufficient to answer the original query: "{state['user_query']}"?

    Answer only: SUFFICIENT, NEED_MORE, or NO_RELEVANT_DATA"""

    judgment = await llm.complete(judge_prompt)

    # Parse result
    if "SUFFICIENT" in judgment.upper():
        result = "sufficient"
    elif "NO_RELEVANT" in judgment.upper():
        result = "no_more_relevant"
    else:
        result = "need_more"

    # Cache the result for future similar evaluations
    if is_evidence_cache_enabled():
        cache = get_evidence_cache()
        cache.set(investigation_goal, evidence_summary, result)

    return result


def select_top_evidence(evidence_trail: List[EvidenceItem], top_k: int = 10) -> List[EvidenceItem]:
    """Select top-k most relevant evidence items."""
    sorted_evidence = sorted(evidence_trail, key=lambda e: e.relevance_score, reverse=True)
    for _i, e in enumerate(sorted_evidence[:top_k]):
        e.selected = True
    return sorted_evidence


# ============================================================================
# PHASE 3: AGENT NODE
# Synthesis and response generation
# ============================================================================


async def agent_node(state: CognitiveState) -> CognitiveState:
    """
    Phase 3: Synthesis & Action

    Responsibilities:
    - Route to appropriate E2I agents based on intent
    - Invoke agents with context and evidence
    - Synthesize agent outputs into coherent response
    - Generate visualization configuration if needed
    """
    from .agent_registry import invoke_agent
    from .memory_backends import get_llm_service

    state["phase_timings"]["agent"] = {"start": datetime.now(timezone.utc)}

    # Determine which agents to invoke based on intent
    intent_to_agents = {
        "causal_intents": ["causal_impact", "explainer"],
        "trend_intents": ["drift_monitor", "explainer"],
        "comparison_intents": ["gap_analyzer", "explainer"],
        "optimization_intents": ["heterogeneous_optimizer", "resource_optimizer"],
        "experiment_intents": ["experiment_designer"],
    }

    state["agents_to_invoke"] = intent_to_agents.get(state["detected_intent"], ["orchestrator"])

    # Build context for agents
    selected_evidence = [e for e in state["evidence_trail"] if e.selected]
    evidence_context = "\n".join([f"[{e.source}] {e.content}" for e in selected_evidence])

    agent_context = {
        "user_query": state["user_query"],
        "conversation_summary": state["conversation_summary"],
        "detected_entities": state["detected_entities"],
        "evidence": evidence_context,
        "session_id": state["session_id"],
    }

    # Invoke each agent
    state["agent_outputs"] = {}
    for agent_name in state["agents_to_invoke"]:
        try:
            output = await invoke_agent(agent_name, agent_context)
            state["agent_outputs"][agent_name] = output
        except Exception as e:
            state["agent_outputs"][agent_name] = {"error": str(e)}

    # Synthesize response
    llm = get_llm_service()

    agent_outputs_text = "\n\n".join(
        [
            f"=== {agent} ===\n{output}"
            for agent, output in state["agent_outputs"].items()
            if "error" not in output
        ]
    )

    synthesis_prompt = f"""You are synthesizing a response for the E2I Causal Analytics Dashboard.

User Query: {state['user_query']}

Context from investigation:
{evidence_context}

Agent Analysis Results:
{agent_outputs_text}

Generate a clear, actionable response that:
1. Directly answers the user's question
2. Cites specific evidence and causal relationships
3. Provides confidence levels for claims
4. Suggests next steps or follow-up questions if appropriate

Response:"""

    state["synthesized_response"] = await llm.complete(synthesis_prompt)

    # Generate visualization config if relevant
    if state["detected_intent"] in ["causal_intents", "trend_intents", "comparison_intents"]:
        state["visualization_config"] = await generate_viz_config(
            state["detected_intent"], state["agent_outputs"]
        )

    # Calculate confidence
    if selected_evidence:
        state["confidence_score"] = sum(e.relevance_score for e in selected_evidence) / len(
            selected_evidence
        )
    else:
        state["confidence_score"] = 0.5

    # Add assistant message
    assistant_message = Message(role="assistant", content=state["synthesized_response"])
    state["messages"] = [assistant_message]

    state["phase_timings"]["agent"]["end"] = datetime.now(timezone.utc)
    state["phase_completed"] = "agent"

    return state


async def generate_viz_config(intent: str, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visualization configuration based on intent and outputs."""
    viz_types = {"causal_intents": "sankey", "trend_intents": "line", "comparison_intents": "bar"}

    return {
        "chart_type": viz_types.get(intent, "table"),
        "data": agent_outputs,
        "config": {"title": "Analysis Results", "show_confidence": True},
    }


# ============================================================================
# PHASE 4: REFLECTOR NODE
# Asynchronous learning and self-updating
# ============================================================================


async def reflector_node(state: CognitiveState) -> CognitiveState:
    """
    Phase 4: Learning & Self-Updating

    Responsibilities:
    - Evaluate if interaction is worth remembering
    - Extract new facts as triplets for semantic memory
    - Learn successful tool sequences for procedural memory
    - Create episodic memory entry
    - Collect learning signals for DSPy optimization

    This node runs AFTER the response is sent (asynchronous).
    """
    from .memory_backends import (
        get_llm_service,
        insert_episodic_memory,
        insert_procedural_memory,
        sync_to_semantic_graph,
    )

    state["phase_timings"]["reflector"] = {"start": datetime.now(timezone.utc)}

    llm = get_llm_service()

    # === Selective Attention: Is this worth remembering? ===
    evaluation_prompt = f"""Evaluate this interaction for learning potential.

User Query: {state['user_query']}
Response Confidence: {state['confidence_score']:.2f}
Evidence Items: {len(state['evidence_trail'])}
Agents Used: {state['agents_to_invoke']}

Is this interaction worth remembering? Consider:
- Did we discover new causal relationships?
- Was this a novel query pattern?
- Did the user receive actionable insights?
- Could this help similar future queries?

Answer: REMEMBER or SKIP, followed by a brief reason."""

    evaluation = await llm.complete(evaluation_prompt)
    state["worth_remembering"] = "REMEMBER" in evaluation.upper()

    if not state["worth_remembering"]:
        state["phase_timings"]["reflector"]["end"] = datetime.now(timezone.utc)
        state["phase_completed"] = "reflector"
        return state

    # === Extract New Facts (Triplets) ===
    fact_extraction_prompt = f"""Extract factual relationships discovered in this interaction.

Query: {state['user_query']}
Evidence: {[e.content for e in state['evidence_trail'] if e.selected]}
Agent Outputs: {state['agent_outputs']}

Format as triplets: (Subject, Predicate, Object)
Only include NEW facts that we learned, not general knowledge.

Triplets (one per line, or NONE if no new facts):"""

    facts_text = await llm.complete(fact_extraction_prompt)
    state["new_facts"] = parse_triplets(facts_text)

    # Sync new facts to semantic graph
    for fact in state["new_facts"]:
        await sync_to_semantic_graph(fact)

    # === Learn Successful Tool Sequences ===
    if state["confidence_score"] and state["confidence_score"] > 0.7:
        # This was a successful interaction - learn the procedure
        tool_sequence = []
        for agent in state["agents_to_invoke"]:
            output = state["agent_outputs"].get(agent, {})
            if "error" not in output:
                tool_sequence.append(
                    {
                        "agent": agent,
                        "input": state["detected_entities"],
                        "output_summary": str(output)[:200],
                    }
                )

        if tool_sequence:
            procedure = {
                "procedure_name": f"{state['detected_intent']}_procedure",
                "procedure_type": "tool_sequence",
                "tool_sequence": tool_sequence,
                "trigger_pattern": state["user_query"],
                "intent_keywords": list(state["detected_entities"].get("kpis", [])),
                "success_count": 1,
                "usage_count": 1,
            }
            state["new_procedures"] = [procedure]
            await insert_procedural_memory(procedure, state["query_embedding"])

    # === Create Episodic Memory ===
    episodic_entry = {
        "event_type": "user_query",
        "event_subtype": state["detected_intent"],
        "description": f"User asked about {', '.join(state['detected_entities'].get('brands', ['unknown']))} "
        f"with intent '{state['detected_intent']}'. "
        f"Response confidence: {state['confidence_score']:.0%}",
        "entities": state["detected_entities"],
        "outcome_type": "success" if state["confidence_score"] > 0.6 else "partial",
        "agent_name": ",".join(state["agents_to_invoke"]),
    }
    await insert_episodic_memory(episodic_entry, state["query_embedding"])

    # === Collect Learning Signals for DSPy ===
    state["feedback_signals"] = [
        {
            "signal_type": (
                "outcome_success" if state["confidence_score"] > 0.7 else "outcome_partial"
            ),
            "signal_value": state["confidence_score"],
            "applies_to_type": "procedure",
            "is_training_example": state["confidence_score"] > 0.8,  # High-quality examples
            "dspy_metric_name": "response_quality",
            "dspy_metric_value": state["confidence_score"],
        }
    ]

    state["phase_timings"]["reflector"]["end"] = datetime.now(timezone.utc)
    state["phase_completed"] = "reflector"

    return state


def parse_triplets(text: str) -> List[Dict[str, Any]]:
    """Parse triplet text into structured format."""
    triplets = []

    if "NONE" in text.upper():
        return triplets

    import re

    # Match (Subject, Predicate, Object) pattern
    pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"
    matches = re.findall(pattern, text)

    for match in matches:
        triplets.append(
            {
                "subject": match[0].strip(),
                "predicate": match[1].strip(),
                "object": match[2].strip(),
                "confidence": 0.8,  # Default confidence for extracted facts
            }
        )

    return triplets


# ============================================================================
# ROUTING LOGIC
# ============================================================================


def should_continue_to_agent(state: CognitiveState) -> Literal["agent", "error"]:
    """Route from investigator to agent or handle errors."""
    if state.get("error"):
        return "error"
    return "agent"


def should_continue_to_reflector(state: CognitiveState) -> Literal["reflector", "end"]:
    """Route from agent to reflector or end."""
    # Always go to reflector for learning (runs async)
    return "reflector"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_cognitive_workflow() -> StateGraph:
    """
    Create the LangGraph state machine for the cognitive workflow.

    Flow:
    START -> summarizer -> investigator -> agent -> reflector -> END

    The reflector runs asynchronously after the response is sent.
    """

    # Initialize graph with state schema
    workflow = StateGraph(CognitiveState)

    # Add nodes (phases)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reflector", reflector_node)

    # Add edges (transitions)
    workflow.set_entry_point("summarizer")
    workflow.add_edge("summarizer", "investigator")
    workflow.add_conditional_edges(
        "investigator", should_continue_to_agent, {"agent": "agent", "error": END}
    )
    workflow.add_conditional_edges(
        "agent", should_continue_to_reflector, {"reflector": "reflector", "end": END}
    )
    workflow.add_edge("reflector", END)

    return workflow


def get_initial_state(
    user_query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_hops: int = 4,
) -> CognitiveState:
    """Create initial state for a new cognitive cycle."""
    return CognitiveState(
        session_id=session_id or str(uuid.uuid4()),
        cycle_id=str(uuid.uuid4()),
        user_id=user_id,
        user_query=user_query,
        query_embedding=None,
        detected_intent=None,
        detected_entities={},
        messages=[],
        conversation_summary=None,
        message_count=0,
        context_compressed=False,
        compression_ratio=None,
        investigation_goal=None,
        current_hop=0,
        max_hops=max_hops,
        evidence_trail=[],
        investigation_complete=False,
        investigation_decision=None,
        agents_to_invoke=[],
        agent_outputs={},
        synthesized_response=None,
        visualization_config=None,
        confidence_score=None,
        worth_remembering=False,
        new_facts=[],
        new_procedures=[],
        feedback_signals=[],
        phase_timings={},
        error=None,
        phase_completed="init",
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def run_cognitive_cycle(
    user_query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    checkpointer: Optional[MemorySaver] = None,
) -> CognitiveState:
    """
    Execute a complete cognitive cycle for a user query.

    Args:
        user_query: The user's natural language question
        session_id: Optional session ID for continuity
        user_id: Optional user ID for personalization
        checkpointer: Optional LangGraph checkpointer for state persistence

    Returns:
        Final state with response and learning outputs
    """
    # Create workflow
    workflow = create_cognitive_workflow()

    # Compile with optional checkpointer
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()

    # Create initial state
    initial_state = get_initial_state(user_query, session_id, user_id)

    # Run the workflow
    config = {"configurable": {"thread_id": initial_state["session_id"]}}
    final_state = await app.ainvoke(initial_state, config)

    return final_state


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Example query
        result = await run_cognitive_cycle(
            user_query="Why did Kisqali adoption increase in the Northeast last quarter?",
            user_id="analyst_001",
        )

        print("=" * 60)
        print("COGNITIVE CYCLE COMPLETE")
        print("=" * 60)
        print(f"\nDetected Intent: {result['detected_intent']}")
        print(f"Entities: {result['detected_entities']}")
        print(f"Investigation Hops: {result['current_hop']}")
        print(f"Evidence Items: {len(result['evidence_trail'])}")
        print(f"Agents Used: {result['agents_to_invoke']}")
        print(f"Confidence: {result['confidence_score']:.0%}")
        print(f"\nResponse:\n{result['synthesized_response']}")
        print(f"\nWorth Remembering: {result['worth_remembering']}")
        print(f"New Facts Learned: {len(result['new_facts'])}")
        print(f"New Procedures Learned: {len(result['new_procedures'])}")

    asyncio.run(main())
