"""
Simplified Cognitive Cycle Test
This test uses inline implementations to avoid relative import issues.
"""

import asyncio
import os
import json
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from pathlib import Path
import operator

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ============================================================================
# SERVICE CLIENTS
# ============================================================================

def get_supabase_client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    return create_client(url, key)


def get_falkordb_client():
    from falkordb import FalkorDB
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6381"))  # e2i port
    return FalkorDB(host=host, port=port)


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class OpenAIEmbeddings:
    def __init__(self):
        import openai
        self.client = openai.OpenAI()
        self.model = "text-embedding-3-small"

    async def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class EvidenceItem(BaseModel):
    hop_number: int
    source: str
    query_type: str
    content: str
    raw_data: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    selected: bool = False


class CognitiveState(TypedDict):
    session_id: str
    cycle_id: str
    user_id: Optional[str]
    user_query: str
    query_embedding: Optional[List[float]]
    detected_intent: Optional[str]
    detected_entities: Dict[str, List[str]]
    messages: List[Dict[str, Any]]
    conversation_summary: Optional[str]
    message_count: int
    context_compressed: bool
    compression_ratio: Optional[float]
    investigation_goal: Optional[str]
    current_hop: int
    max_hops: int
    evidence_trail: List[Dict[str, Any]]
    investigation_complete: bool
    investigation_decision: Optional[str]
    agents_to_invoke: List[str]
    agent_outputs: Dict[str, Any]
    synthesized_response: Optional[str]
    visualization_config: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    worth_remembering: bool
    new_facts: List[Dict[str, Any]]
    new_procedures: List[Dict[str, Any]]
    feedback_signals: List[Dict[str, Any]]
    phase_timings: Dict[str, Dict[str, str]]
    error: Optional[str]
    phase_completed: str


# ============================================================================
# COGNITIVE WORKFLOW PHASES
# ============================================================================

async def summarizer_node(state: CognitiveState) -> CognitiveState:
    """Phase 1: Input & Context Pruning"""
    print("  [Phase 1] Summarizer: Processing query...")
    state["phase_timings"]["summarizer"] = {"start": datetime.now(timezone.utc).isoformat()}

    # Generate embedding
    embeddings = OpenAIEmbeddings()
    state["query_embedding"] = await embeddings.embed(state["user_query"])

    # Extract entities (simple keyword matching)
    query_lower = state["user_query"].lower()
    entities = {"brands": [], "regions": [], "kpis": []}

    for brand in ["kisqali", "fabhalta", "remibrutinib"]:
        if brand in query_lower:
            entities["brands"].append(brand)

    for region in ["northeast", "south", "midwest", "west"]:
        if region in query_lower:
            entities["regions"].append(region)

    state["detected_entities"] = entities

    # Detect intent
    if any(w in query_lower for w in ["why", "cause", "because", "reason"]):
        state["detected_intent"] = "causal_intents"
    elif any(w in query_lower for w in ["trend", "change", "increase", "decrease"]):
        state["detected_intent"] = "trend_intents"
    elif any(w in query_lower for w in ["compare", "versus", "vs", "difference"]):
        state["detected_intent"] = "comparison_intents"
    else:
        state["detected_intent"] = "exploration_intents"

    state["phase_timings"]["summarizer"]["end"] = datetime.now(timezone.utc).isoformat()
    state["phase_completed"] = "summarizer"
    state["context_compressed"] = False

    print(f"    Intent: {state['detected_intent']}")
    print(f"    Entities: {entities}")

    return state


async def investigator_node(state: CognitiveState) -> CognitiveState:
    """Phase 2: Multi-Hop Investigation"""
    print("  [Phase 2] Investigator: Gathering evidence...")
    state["phase_timings"]["investigator"] = {"start": datetime.now(timezone.utc).isoformat()}

    llm = AnthropicLLM()

    # Set investigation goal
    goal_prompt = f"""Given this user query: "{state['user_query']}"
    And detected intent: {state['detected_intent']}
    And entities: {state['detected_entities']}

    What specific information do we need to answer this query? Be specific. Keep it to 2-3 sentences."""

    state["investigation_goal"] = await llm.complete(goal_prompt, max_tokens=200)
    print(f"    Goal: {state['investigation_goal'][:100]}...")

    state["current_hop"] = 0
    state["evidence_trail"] = []

    # Hop 1: Query semantic graph (FalkorDB)
    state["current_hop"] = 1
    print(f"    Hop {state['current_hop']}: Querying semantic graph...")

    try:
        client = get_falkordb_client()
        graph = client.select_graph("e2i_semantic")

        # Query for Kisqali relationships
        result = graph.query("""
            MATCH (h:HCP)-[r:PRESCRIBES]->(b:Brand {name: 'Kisqali'})
            RETURN h.name as hcp, h.specialty as specialty, r.volume_monthly as volume, r.market_share as share
        """)

        for record in result.result_set:
            state["evidence_trail"].append({
                "hop_number": 1,
                "source": "semantic",
                "query_type": "graph_traversal",
                "content": f"HCP {record[0]} ({record[1]}) prescribes Kisqali: volume={record[2]}, share={record[3]}",
                "relevance_score": 0.85,
                "selected": True
            })

        # Query HCP influence network
        result = graph.query("""
            MATCH (h1:HCP)-[r:INFLUENCES]->(h2:HCP)
            RETURN h1.name, h2.name, r.influence_strength
        """)

        for record in result.result_set:
            state["evidence_trail"].append({
                "hop_number": 1,
                "source": "semantic",
                "query_type": "influence_network",
                "content": f"{record[0]} influences {record[1]} (strength: {record[2]})",
                "relevance_score": 0.75,
                "selected": True
            })

    except Exception as e:
        print(f"    Warning: FalkorDB query failed: {e}")

    # Hop 2: Query episodic memory (Supabase)
    state["current_hop"] = 2
    print(f"    Hop {state['current_hop']}: Querying episodic memory...")

    try:
        supabase = get_supabase_client()
        result = supabase.table("episodic_memories").select("*").limit(5).execute()

        for mem in result.data:
            state["evidence_trail"].append({
                "hop_number": 2,
                "source": "episodic",
                "query_type": "vector_search",
                "content": mem.get("description", "No description"),
                "relevance_score": 0.7,
                "selected": True
            })
    except Exception as e:
        print(f"    Warning: Supabase query failed: {e}")

    state["investigation_complete"] = True
    state["investigation_decision"] = "sufficient" if state["evidence_trail"] else "no_data"

    print(f"    Evidence collected: {len(state['evidence_trail'])} items")

    state["phase_timings"]["investigator"]["end"] = datetime.now(timezone.utc).isoformat()
    state["phase_completed"] = "investigator"

    return state


async def agent_node(state: CognitiveState) -> CognitiveState:
    """Phase 3: Synthesis & Response Generation"""
    print("  [Phase 3] Agent: Synthesizing response...")
    state["phase_timings"]["agent"] = {"start": datetime.now(timezone.utc).isoformat()}

    llm = AnthropicLLM()

    # Determine agents to invoke based on intent
    intent_to_agents = {
        "causal_intents": ["causal_impact", "explainer"],
        "trend_intents": ["drift_monitor", "explainer"],
        "comparison_intents": ["gap_analyzer", "explainer"],
        "exploration_intents": ["orchestrator"]
    }
    state["agents_to_invoke"] = intent_to_agents.get(state["detected_intent"], ["orchestrator"])

    # Build evidence context
    evidence_context = "\n".join([
        f"- [{e['source']}] {e['content']}"
        for e in state["evidence_trail"] if e.get("selected")
    ])

    if not evidence_context:
        evidence_context = "No specific evidence collected from memory stores."

    # Generate response
    synthesis_prompt = f"""You are the E2I Causal Analytics assistant. Answer this query based on the evidence.

User Query: {state['user_query']}

Investigation Goal: {state['investigation_goal']}

Evidence from Memory Systems:
{evidence_context}

Detected Intent: {state['detected_intent']}
Detected Entities: {state['detected_entities']}

Provide a clear, analytical response that:
1. Directly addresses the user's question
2. References the evidence where applicable
3. Suggests potential causal factors
4. Recommends follow-up analysis if needed

Response:"""

    state["synthesized_response"] = await llm.complete(synthesis_prompt, max_tokens=800)

    # Mock agent outputs
    state["agent_outputs"] = {
        agent: {"status": "completed", "confidence": 0.8}
        for agent in state["agents_to_invoke"]
    }

    # Calculate confidence
    if state["evidence_trail"]:
        scores = [e.get("relevance_score", 0.5) for e in state["evidence_trail"]]
        state["confidence_score"] = sum(scores) / len(scores)
    else:
        state["confidence_score"] = 0.5

    state["phase_timings"]["agent"]["end"] = datetime.now(timezone.utc).isoformat()
    state["phase_completed"] = "agent"

    print(f"    Agents invoked: {state['agents_to_invoke']}")
    print(f"    Confidence: {state['confidence_score']:.0%}")

    return state


async def reflector_node(state: CognitiveState) -> CognitiveState:
    """Phase 4: Learning & Reflection"""
    print("  [Phase 4] Reflector: Evaluating for learning...")
    state["phase_timings"]["reflector"] = {"start": datetime.now(timezone.utc).isoformat()}

    llm = AnthropicLLM()

    # Evaluate if worth remembering
    eval_prompt = f"""Evaluate this interaction for learning potential:

Query: {state['user_query']}
Confidence: {state['confidence_score']:.2f}
Evidence Items: {len(state['evidence_trail'])}

Should this be remembered? Answer REMEMBER or SKIP with brief reason."""

    evaluation = await llm.complete(eval_prompt, max_tokens=100)
    state["worth_remembering"] = "REMEMBER" in evaluation.upper()

    print(f"    Worth remembering: {state['worth_remembering']}")

    # Extract facts if worth remembering
    state["new_facts"] = []
    state["new_procedures"] = []

    if state["worth_remembering"] and state["confidence_score"] > 0.6:
        # Record a procedural memory
        state["new_procedures"].append({
            "procedure_name": f"{state['detected_intent']}_procedure",
            "tool_sequence": state["agents_to_invoke"],
            "success_rate": state["confidence_score"]
        })

        # Try to store episodic memory
        try:
            supabase = get_supabase_client()
            episodic_entry = {
                "memory_id": str(uuid.uuid4()),
                "event_type": "user_query",
                "event_subtype": state["detected_intent"],
                "description": f"User asked: {state['user_query'][:200]}",
                "entities": json.dumps(state["detected_entities"]),
                "outcome_type": "success" if state["confidence_score"] > 0.6 else "partial_success",
                "importance_score": state["confidence_score"],
                "brand": state["detected_entities"].get("brands", [""])[0] if state["detected_entities"].get("brands") else None,
                "region": state["detected_entities"].get("regions", [""])[0] if state["detected_entities"].get("regions") else None
            }
            supabase.table("episodic_memories").insert(episodic_entry).execute()
            print("    ✓ Episodic memory stored")
        except Exception as e:
            print(f"    Warning: Could not store episodic memory: {e}")

    state["phase_timings"]["reflector"]["end"] = datetime.now(timezone.utc).isoformat()
    state["phase_completed"] = "reflector"

    return state


# ============================================================================
# WORKFLOW CONSTRUCTION
# ============================================================================

def create_cognitive_workflow():
    workflow = StateGraph(CognitiveState)

    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reflector", reflector_node)

    workflow.set_entry_point("summarizer")
    workflow.add_edge("summarizer", "investigator")
    workflow.add_edge("investigator", "agent")
    workflow.add_edge("agent", "reflector")
    workflow.add_edge("reflector", END)

    return workflow


def get_initial_state(user_query: str, user_id: Optional[str] = None) -> CognitiveState:
    return CognitiveState(
        session_id=str(uuid.uuid4()),
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
        max_hops=4,
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
        phase_completed="init"
    )


async def run_cognitive_cycle(user_query: str, user_id: Optional[str] = None):
    workflow = create_cognitive_workflow()
    app = workflow.compile(checkpointer=MemorySaver())

    initial_state = get_initial_state(user_query, user_id)
    config = {"configurable": {"thread_id": initial_state["session_id"]}}

    return await app.ainvoke(initial_state, config)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(result: CognitiveState, output_path: Optional[str] = None) -> str:
    """Generate a comprehensive markdown report from the cognitive cycle results."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Calculate phase durations
    phase_durations = {}
    for phase, timings in result.get("phase_timings", {}).items():
        if "start" in timings and "end" in timings:
            start = datetime.fromisoformat(timings["start"].replace("+00:00", ""))
            end = datetime.fromisoformat(timings["end"].replace("+00:00", ""))
            duration_ms = (end - start).total_seconds() * 1000
            phase_durations[phase] = f"{duration_ms:.0f}ms"

    # Build evidence table
    evidence_rows = []
    for e in result.get("evidence_trail", []):
        evidence_rows.append(
            f"| {e.get('hop_number', '-')} | {e.get('source', '-')} | {e.get('query_type', '-')} | {e.get('content', '-')[:60]}... | {e.get('relevance_score', 0):.0%} |"
        )
    evidence_table = "\n".join(evidence_rows) if evidence_rows else "| - | - | - | No evidence collected | - |"

    # Build agent outputs table
    agent_rows = []
    for agent, output in result.get("agent_outputs", {}).items():
        status = output.get("status", "unknown")
        confidence = output.get("confidence", 0)
        agent_rows.append(f"| {agent} | {status} | {confidence:.0%} |")
    agent_table = "\n".join(agent_rows) if agent_rows else "| - | - | - |"

    # Entities formatting
    entities = result.get("detected_entities", {})
    entities_list = []
    for entity_type, values in entities.items():
        if values:
            entities_list.append(f"- **{entity_type.title()}**: {', '.join(values)}")
    entities_section = "\n".join(entities_list) if entities_list else "- No entities detected"

    # New learnings section
    new_facts = result.get("new_facts", [])
    new_procedures = result.get("new_procedures", [])
    learnings_section = ""
    if new_facts:
        learnings_section += "\n### New Facts\n"
        for fact in new_facts:
            learnings_section += f"- {fact}\n"
    if new_procedures:
        learnings_section += "\n### New Procedures\n"
        for proc in new_procedures:
            learnings_section += f"- **{proc.get('procedure_name', 'Unknown')}**: {proc.get('tool_sequence', [])} (success rate: {proc.get('success_rate', 0):.0%})\n"
    if not learnings_section:
        learnings_section = "\nNo new learnings recorded for this cycle."

    # Build the full report
    report = f"""# E2I Cognitive Cycle Report

**Generated**: {timestamp}
**Session ID**: `{result.get('session_id', 'N/A')}`
**Cycle ID**: `{result.get('cycle_id', 'N/A')}`
**User ID**: `{result.get('user_id', 'Anonymous')}`

---

## Query Analysis

### User Query
> {result.get('user_query', 'N/A')}

### Detected Intent
`{result.get('detected_intent', 'N/A')}`

### Detected Entities
{entities_section}

### Investigation Goal
{result.get('investigation_goal', 'N/A')}

---

## Phase Execution Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Summarizer | {phase_durations.get('summarizer', 'N/A')} | ✅ Complete |
| Investigator | {phase_durations.get('investigator', 'N/A')} | ✅ Complete |
| Agent | {phase_durations.get('agent', 'N/A')} | ✅ Complete |
| Reflector | {phase_durations.get('reflector', 'N/A')} | ✅ Complete |

---

## Evidence Trail

**Total Evidence Items**: {len(result.get('evidence_trail', []))}
**Investigation Decision**: `{result.get('investigation_decision', 'N/A')}`

| Hop | Source | Query Type | Content | Relevance |
|-----|--------|------------|---------|-----------|
{evidence_table}

---

## Agent Execution

**Agents Invoked**: {', '.join(result.get('agents_to_invoke', ['None']))}

| Agent | Status | Confidence |
|-------|--------|------------|
{agent_table}

---

## Synthesized Response

**Overall Confidence**: {f"{result.get('confidence_score', 0):.0%}" if result.get('confidence_score') else "N/A"}

{result.get('synthesized_response', 'No response generated.')}

---

## Learning & Reflection

**Worth Remembering**: {'✅ Yes' if result.get('worth_remembering') else '❌ No'}
{learnings_section}

---

## Metadata

| Property | Value |
|----------|-------|
| Context Compressed | {'Yes' if result.get('context_compressed') else 'No'} |
| Compression Ratio | {result.get('compression_ratio', 'N/A')} |
| Max Hops Configured | {result.get('max_hops', 4)} |
| Hops Executed | {result.get('current_hop', 0)} |
| Error | {result.get('error', 'None')} |

---

*Report generated by E2I Agentic Memory System*
"""

    # Save to file if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.write_text(report, encoding="utf-8")
        print(f"\n📄 Report saved to: {output_file.absolute()}")

    return report


# ============================================================================
# MAIN TEST
# ============================================================================

async def main():
    print("="*70)
    print("E2I COGNITIVE CYCLE TEST")
    print("="*70)

    test_query = "Why did Kisqali adoption increase in the Northeast last quarter?"
    print(f"\nQuery: {test_query}\n")
    print("-"*70)

    result = await run_cognitive_cycle(test_query, "test_analyst")

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nIntent: {result['detected_intent']}")
    print(f"Entities: {result['detected_entities']}")
    print(f"Evidence Items: {len(result['evidence_trail'])}")
    print(f"Agents Used: {result['agents_to_invoke']}")
    print(f"Confidence: {result['confidence_score']:.0%}" if result['confidence_score'] else "N/A")
    print(f"Worth Remembering: {result['worth_remembering']}")
    print(f"\n{'='*70}")
    print("RESPONSE")
    print("="*70)
    print(result['synthesized_response'])

    # Generate markdown report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)

    report_filename = f"cognitive_cycle_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    report = generate_markdown_report(result, report_filename)

    print(f"\n✅ Cognitive cycle complete!")


if __name__ == "__main__":
    asyncio.run(main())
