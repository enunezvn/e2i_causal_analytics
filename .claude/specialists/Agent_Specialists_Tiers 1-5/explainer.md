# Tier 5: Explainer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 5 (Self-Improvement) |
| **Agent Type** | Deep (Extended Reasoning) |
| **Model Tier** | Opus/Sonnet |
| **Latency Tolerance** | High (up to 45s) |
| **Critical Path** | No - can run async |

## Domain Scope

You are the specialist for the Tier 5 Explainer Agent:
- `src/agents/explainer/` - Natural language explanations

This is a **Deep Reasoning Agent** for:
- Synthesizing complex analyses into clear narratives
- Adapting explanations to user expertise
- Generating actionable insights
- Creating educational content

## Design Principles

### Deep Reasoning for Synthesis
The Explainer requires extended thinking to:
- Understand complex analytical outputs
- Identify key insights
- Structure compelling narratives
- Adapt to audience expertise level

### Responsibilities
1. **Analysis Synthesis** - Combine multi-agent outputs
2. **Narrative Generation** - Create clear explanations
3. **Insight Extraction** - Identify actionable takeaways
4. **Audience Adaptation** - Tailor to expertise level

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXPLAINER AGENT                             │
│                      (Deep Pattern)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                CONTEXT ASSEMBLER                         │    │
│  │   Gather all analysis results, user context, history    │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            DEEP REASONING ENGINE                         │    │
│  │   • Extended thinking for complex synthesis              │    │
│  │   • Narrative structure planning                         │    │
│  │   • Insight prioritization                               │    │
│  │   • Audience adaptation                                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              NARRATIVE GENERATOR                         │    │
│  │   • Executive summary  • Detailed explanation            │    │
│  │   • Visual suggestions  • Follow-up questions            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
explainer/
├── agent.py              # Main ExplainerAgent class (v4.3: smart LLM selection)
├── config.py             # Configuration and complexity scoring (v4.3)
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── dspy_integration.py   # DSPy signal collection
├── mlflow_tracker.py     # MLflow experiment tracking
├── memory_hooks.py       # Tri-memory integration
├── nodes/
│   ├── context_assembler.py  # Gather analysis context
│   ├── deep_reasoner.py      # Extended reasoning node (LLM optional)
│   └── narrative_generator.py # Generate explanations (4 formats)
└── prompts.py            # LLM prompts for explanation
```

## Smart LLM Mode Selection (v4.3)

The Explainer agent now supports automatic LLM mode selection based on input complexity:

```python
from src.agents.explainer import ExplainerAgent, ExplainerConfig

# Auto mode (default) - uses complexity scoring
agent = ExplainerAgent(use_llm=None)  # Automatically decides

# Explicit modes
agent = ExplainerAgent(use_llm=True)   # Always use LLM
agent = ExplainerAgent(use_llm=False)  # Always deterministic

# Custom threshold
config = ExplainerConfig(llm_threshold=0.7)  # Higher threshold
agent = ExplainerAgent(use_llm=None, config=config)
```

### Complexity Scoring

The agent evaluates input complexity based on:
- **Result count** (25%): More results = higher complexity
- **Query complexity** (30%): Keywords like "why", "explain", "compare"
- **Causal discovery** (25%): Presence of DAG data triggers LLM
- **Expertise level** (20%): Executive needs more synthesis

Default threshold: 0.5 (scores above trigger LLM mode)

## LangGraph State Definition

```python
# src/agents/explainer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class AnalysisContext(TypedDict):
    """Context from prior analysis"""
    source_agent: str
    analysis_type: str
    key_findings: List[str]
    data_summary: Dict[str, Any]
    confidence: float
    warnings: List[str]

class Insight(TypedDict):
    """Extracted insight"""
    insight_id: str
    category: Literal["finding", "recommendation", "warning", "opportunity"]
    statement: str
    supporting_evidence: List[str]
    confidence: float
    priority: int
    actionability: Literal["immediate", "short_term", "long_term", "informational"]

class NarrativeSection(TypedDict):
    """Section of generated narrative"""
    section_type: str
    title: str
    content: str
    supporting_data: Optional[Dict[str, Any]]

class ExplainerState(TypedDict):
    """Complete state for Explainer agent"""
    
    # === INPUT ===
    query: str
    analysis_results: List[Dict[str, Any]]  # From upstream agents
    user_expertise: Literal["executive", "analyst", "data_scientist"]
    output_format: Literal["narrative", "structured", "presentation", "brief"]
    focus_areas: Optional[List[str]]
    
    # === CONTEXT ===
    analysis_context: Optional[List[AnalysisContext]]
    user_context: Optional[Dict[str, Any]]
    conversation_history: Optional[List[Dict]]
    
    # === REASONING OUTPUTS ===
    extracted_insights: Optional[List[Insight]]
    narrative_structure: Optional[List[str]]
    key_themes: Optional[List[str]]
    
    # === NARRATIVE OUTPUTS ===
    executive_summary: Optional[str]
    detailed_explanation: Optional[str]
    narrative_sections: Optional[List[NarrativeSection]]
    
    # === SUPPLEMENTARY OUTPUTS ===
    visual_suggestions: Optional[List[Dict[str, Any]]]
    follow_up_questions: Optional[List[str]]
    related_analyses: Optional[List[str]]
    
    # === EXECUTION METADATA ===
    assembly_latency_ms: int
    reasoning_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model_used: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "assembling", "reasoning", "generating", "completed", "failed"]
```

## Node Implementations

### Context Assembler Node

```python
# src/agents/explainer/nodes/context_assembler.py

import time
from typing import List, Dict, Any

from ..state import ExplainerState, AnalysisContext

class ContextAssemblerNode:
    """
    Assemble context from multiple analysis results
    Prepares input for deep reasoning
    """
    
    def __init__(self, conversation_store=None):
        self.conversation_store = conversation_store
    
    async def execute(self, state: ExplainerState) -> ExplainerState:
        start_time = time.time()
        
        try:
            analysis_results = state["analysis_results"]
            
            # Extract context from each analysis
            contexts = []
            for result in analysis_results:
                context = self._extract_context(result)
                if context:
                    contexts.append(context)
            
            # Get user context if available
            user_context = await self._get_user_context(state)
            
            # Get conversation history if available
            history = await self._get_conversation_history(state)
            
            assembly_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "analysis_context": contexts,
                "user_context": user_context,
                "conversation_history": history,
                "assembly_latency_ms": assembly_time,
                "status": "reasoning"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "context_assembler", "error": str(e)}],
                "status": "failed"
            }
    
    def _extract_context(self, result: Dict[str, Any]) -> AnalysisContext:
        """Extract standardized context from analysis result"""
        
        return AnalysisContext(
            source_agent=result.get("agent", "unknown"),
            analysis_type=result.get("analysis_type", "unknown"),
            key_findings=result.get("key_findings", []),
            data_summary={
                k: v for k, v in result.items() 
                if k not in ["agent", "analysis_type", "key_findings", "narrative"]
            },
            confidence=result.get("confidence", 0.5),
            warnings=result.get("warnings", [])
        )
    
    async def _get_user_context(self, state: ExplainerState) -> Dict[str, Any]:
        """Get user context for personalization"""
        return {
            "expertise": state["user_expertise"],
            "focus_areas": state.get("focus_areas", [])
        }
    
    async def _get_conversation_history(self, state: ExplainerState) -> List[Dict]:
        """Get relevant conversation history"""
        if self.conversation_store and state.get("session_id"):
            return await self.conversation_store.get_recent(
                session_id=state["session_id"],
                limit=5
            )
        return []
```

### Deep Reasoner Node

```python
# src/agents/explainer/nodes/deep_reasoner.py

import asyncio
import time
import json
from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import ExplainerState, Insight

class DeepReasonerNode:
    """
    Deep reasoning for insight extraction and narrative planning
    This is where extended thinking adds value
    """
    
    def __init__(self):
        # Use Opus for complex synthesis
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Or Opus for very complex cases
            max_tokens=8192,
            timeout=120
        )
        self.fallback_llm = ChatAnthropic(
            model="claude-haiku-4-20250414",
            max_tokens=4096,
            timeout=60
        )
    
    async def execute(self, state: ExplainerState) -> ExplainerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            prompt = self._build_reasoning_prompt(state)
            
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=120
                )
                model_used = "claude-sonnet-4-20250514"
            except (asyncio.TimeoutError, Exception) as e:
                response = await self.fallback_llm.ainvoke(
                    self._build_simplified_prompt(state)
                )
                model_used = "claude-haiku-4-20250414 (fallback)"
                state = {**state, "warnings": state.get("warnings", []) + [f"Used fallback: {str(e)}"]}
            
            # Parse reasoning output
            parsed = self._parse_reasoning(response.content)
            
            reasoning_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "extracted_insights": parsed.get("insights", []),
                "narrative_structure": parsed.get("structure", []),
                "key_themes": parsed.get("themes", []),
                "reasoning_latency_ms": reasoning_time,
                "model_used": model_used,
                "status": "generating"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "deep_reasoner", "error": str(e)}],
                "status": "failed"
            }
    
    def _build_reasoning_prompt(self, state: ExplainerState) -> str:
        """Build comprehensive reasoning prompt"""
        
        expertise_guidance = {
            "executive": """
- Focus on business impact and ROI
- Lead with bottom-line implications
- Use plain language, avoid jargon
- Emphasize actionable recommendations
- Keep technical details minimal""",
            "analyst": """
- Balance business context with methodology
- Include key statistics with interpretation
- Explain confidence levels and caveats
- Provide actionable next steps
- Reference underlying data appropriately""",
            "data_scientist": """
- Include full technical detail
- Discuss methodology and assumptions
- Address statistical validity and limitations
- Provide reproducibility context
- Suggest advanced follow-up analyses"""
        }
        
        contexts_str = "\n\n".join([
            f"### Analysis from {ctx['source_agent']}\n"
            f"Type: {ctx['analysis_type']}\n"
            f"Confidence: {ctx['confidence']}\n"
            f"Key Findings:\n" + "\n".join(f"- {f}" for f in ctx['key_findings']) + "\n"
            f"Data Summary: {json.dumps(ctx['data_summary'], indent=2)}"
            for ctx in state.get("analysis_context", [])
        ])
        
        return f"""You are an expert pharmaceutical analytics communicator. Your task is to synthesize complex analyses into clear, actionable explanations.

## Analysis Results to Explain

{contexts_str}

## Target Audience

Expertise Level: {state['user_expertise']}
{expertise_guidance.get(state['user_expertise'], '')}

Focus Areas: {state.get('focus_areas', ['General overview'])}

## Original Query

"{state['query']}"

---

## Your Task

Perform deep reasoning to:

### 1. Extract Key Insights
Identify the 3-5 most important insights from the analyses. For each:
- State the insight clearly
- Categorize (finding, recommendation, warning, opportunity)
- Assess confidence and actionability
- Identify supporting evidence

### 2. Plan Narrative Structure
Determine the optimal structure for explaining these results:
- What should come first to capture attention?
- How should findings build on each other?
- Where should caveats and limitations appear?
- How to end with clear next steps?

### 3. Identify Key Themes
What are the 2-3 overarching themes that tie the analyses together?

### Output Format (JSON)

```json
{{
  "insights": [
    {{
      "insight_id": "1",
      "category": "finding|recommendation|warning|opportunity",
      "statement": "Clear statement of the insight",
      "supporting_evidence": ["Evidence 1", "Evidence 2"],
      "confidence": 0.0-1.0,
      "priority": 1-5,
      "actionability": "immediate|short_term|long_term|informational"
    }}
  ],
  "structure": [
    "Section 1: Executive Summary",
    "Section 2: Key Finding",
    "Section 3: Methodology Context",
    "Section 4: Recommendations",
    "Section 5: Next Steps"
  ],
  "themes": [
    "Theme 1 description",
    "Theme 2 description"
  ]
}}
```"""

    def _build_simplified_prompt(self, state: ExplainerState) -> str:
        """Simplified prompt for fallback"""
        
        findings = []
        for ctx in state.get("analysis_context", []):
            findings.extend(ctx.get("key_findings", []))
        
        return f"""Summarize these analysis findings for a {state['user_expertise']} audience:

{chr(10).join(f'- {f}' for f in findings[:10])}

Provide JSON with:
- insights: List of top 3 insights (statement, category, confidence)
- structure: Suggested explanation structure
- themes: Key themes"""

    def _parse_reasoning(self, content: str) -> Dict[str, Any]:
        """Parse reasoning output"""
        import re
        
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract what we can
        return {
            "insights": [],
            "structure": ["Summary", "Findings", "Recommendations"],
            "themes": []
        }
```

### Narrative Generator Node

```python
# src/agents/explainer/nodes/narrative_generator.py

import asyncio
import time
from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import ExplainerState, NarrativeSection

class NarrativeGeneratorNode:
    """
    Generate final narrative explanations
    Uses structured insights from reasoning phase
    """
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            timeout=90
        )
    
    async def execute(self, state: ExplainerState) -> ExplainerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            output_format = state.get("output_format", "narrative")
            
            if output_format == "brief":
                result = await self._generate_brief(state)
            elif output_format == "structured":
                result = await self._generate_structured(state)
            elif output_format == "presentation":
                result = await self._generate_presentation(state)
            else:
                result = await self._generate_narrative(state)
            
            # Generate supplementary content
            visuals = self._suggest_visuals(state)
            follow_ups = self._generate_follow_ups(state)
            
            generation_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("assembly_latency_ms", 0) +
                state.get("reasoning_latency_ms", 0) +
                generation_time
            )
            
            return {
                **state,
                **result,
                "visual_suggestions": visuals,
                "follow_up_questions": follow_ups,
                "generation_latency_ms": generation_time,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "narrative_generator", "error": str(e)}],
                "status": "failed"
            }
    
    async def _generate_narrative(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate full narrative explanation"""
        
        insights = state.get("extracted_insights", [])
        structure = state.get("narrative_structure", [])
        themes = state.get("key_themes", [])
        expertise = state["user_expertise"]
        
        prompt = f"""Generate a clear, engaging explanation based on these insights:

## Insights
{self._format_insights(insights)}

## Structure to Follow
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(structure))}

## Key Themes
{chr(10).join(f'- {t}' for t in themes)}

## Audience
{expertise.title()} level - adjust language and detail accordingly.

## Requirements
- Write in natural, engaging prose
- Use concrete numbers and examples
- Include clear transitions between sections
- End with actionable next steps
- Keep executive summary under 100 words

Generate the explanation with clear section headers."""

        response = await self.llm.ainvoke(prompt)
        
        # Parse into sections
        sections = self._parse_narrative_sections(response.content)
        
        # Extract executive summary (first section or generate separately)
        exec_summary = sections[0]["content"] if sections else ""
        
        return {
            "executive_summary": exec_summary[:500],
            "detailed_explanation": response.content,
            "narrative_sections": sections
        }
    
    async def _generate_brief(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate brief summary"""
        
        insights = state.get("extracted_insights", [])
        top_insights = sorted(insights, key=lambda x: x.get("priority", 5))[:3]
        
        brief = "**Key Findings:**\n\n"
        for insight in top_insights:
            brief += f"• {insight['statement']}\n"
        
        return {
            "executive_summary": brief,
            "detailed_explanation": brief,
            "narrative_sections": []
        }
    
    async def _generate_structured(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate structured output with clear sections"""
        
        insights = state.get("extracted_insights", [])
        
        sections = []
        
        # Group insights by category
        findings = [i for i in insights if i.get("category") == "finding"]
        recommendations = [i for i in insights if i.get("category") == "recommendation"]
        warnings = [i for i in insights if i.get("category") == "warning"]
        
        if findings:
            sections.append(NarrativeSection(
                section_type="findings",
                title="Key Findings",
                content="\n".join(f"• {f['statement']}" for f in findings),
                supporting_data=None
            ))
        
        if recommendations:
            sections.append(NarrativeSection(
                section_type="recommendations",
                title="Recommendations",
                content="\n".join(f"• {r['statement']}" for r in recommendations),
                supporting_data=None
            ))
        
        if warnings:
            sections.append(NarrativeSection(
                section_type="warnings",
                title="Caveats & Considerations",
                content="\n".join(f"• {w['statement']}" for w in warnings),
                supporting_data=None
            ))
        
        combined = "\n\n".join(f"## {s['title']}\n{s['content']}" for s in sections)
        
        return {
            "executive_summary": f"Analysis complete with {len(findings)} findings and {len(recommendations)} recommendations.",
            "detailed_explanation": combined,
            "narrative_sections": sections
        }
    
    def _format_insights(self, insights: List[Dict]) -> str:
        """Format insights for prompt"""
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"""
{i}. **{insight.get('statement', 'No statement')}**
   Category: {insight.get('category', 'unknown')}
   Confidence: {insight.get('confidence', 0):.0%}
   Actionability: {insight.get('actionability', 'unknown')}
   Evidence: {', '.join(insight.get('supporting_evidence', []))}
""")
        return "\n".join(formatted)
    
    def _parse_narrative_sections(self, content: str) -> List[NarrativeSection]:
        """Parse narrative into sections"""
        import re
        
        sections = []
        # Split by headers (##, ###, etc.)
        parts = re.split(r'\n(#{1,3})\s+(.+?)\n', content)
        
        current_content = []
        current_title = "Introduction"
        
        for part in parts:
            if part.startswith('#'):
                continue
            elif len(part) < 50 and not part.strip().startswith(('-', '•', '*', '1')):
                # Likely a title
                if current_content:
                    sections.append(NarrativeSection(
                        section_type=self._infer_section_type(current_title),
                        title=current_title,
                        content="\n".join(current_content).strip(),
                        supporting_data=None
                    ))
                    current_content = []
                current_title = part.strip()
            else:
                current_content.append(part)
        
        # Add last section
        if current_content:
            sections.append(NarrativeSection(
                section_type=self._infer_section_type(current_title),
                title=current_title,
                content="\n".join(current_content).strip(),
                supporting_data=None
            ))
        
        return sections
    
    def _infer_section_type(self, title: str) -> str:
        """Infer section type from title"""
        title_lower = title.lower()
        if "summary" in title_lower or "overview" in title_lower:
            return "summary"
        elif "finding" in title_lower or "result" in title_lower:
            return "findings"
        elif "recommend" in title_lower or "action" in title_lower:
            return "recommendations"
        elif "caveat" in title_lower or "limit" in title_lower or "warning" in title_lower:
            return "caveats"
        elif "next" in title_lower or "step" in title_lower:
            return "next_steps"
        else:
            return "content"
    
    def _suggest_visuals(self, state: ExplainerState) -> List[Dict[str, Any]]:
        """Suggest visualizations based on analysis"""
        
        suggestions = []
        
        for ctx in state.get("analysis_context", []):
            if ctx["analysis_type"] == "causal_effect_estimation":
                suggestions.append({
                    "type": "effect_plot",
                    "title": "Causal Effect Estimate",
                    "description": "Bar chart showing ATE with confidence interval"
                })
            elif ctx["analysis_type"] == "roi_opportunity_detection":
                suggestions.append({
                    "type": "opportunity_matrix",
                    "title": "ROI Opportunity Matrix",
                    "description": "Scatter plot of ROI vs. implementation difficulty"
                })
            elif ctx["analysis_type"] == "cate_estimation":
                suggestions.append({
                    "type": "segment_effects",
                    "title": "Treatment Effects by Segment",
                    "description": "Grouped bar chart showing CATE by segment"
                })
        
        return suggestions
    
    def _generate_follow_ups(self, state: ExplainerState) -> List[str]:
        """Generate follow-up questions"""
        
        follow_ups = []
        
        insights = state.get("extracted_insights", [])
        
        # Based on insight categories
        has_finding = any(i.get("category") == "finding" for i in insights)
        has_recommendation = any(i.get("category") == "recommendation" for i in insights)
        has_warning = any(i.get("category") == "warning" for i in insights)
        
        if has_finding:
            follow_ups.append("What's driving these findings at the segment level?")
        
        if has_recommendation:
            follow_ups.append("How do we prioritize these recommendations?")
            follow_ups.append("What resources are needed to implement these changes?")
        
        if has_warning:
            follow_ups.append("What additional data would strengthen confidence in these results?")
        
        return follow_ups[:5]
```

## Graph Assembly

```python
# src/agents/explainer/graph.py

from langgraph.graph import StateGraph, END

from .state import ExplainerState
from .nodes.context_assembler import ContextAssemblerNode
from .nodes.deep_reasoner import DeepReasonerNode
from .nodes.narrative_generator import NarrativeGeneratorNode

def build_explainer_graph(conversation_store=None):
    """
    Build the Explainer agent graph
    
    Architecture:
        [assemble] → [reason] → [generate] → END
    """
    
    # Initialize nodes
    assembler = ContextAssemblerNode(conversation_store)
    reasoner = DeepReasonerNode()
    generator = NarrativeGeneratorNode()
    
    # Build graph
    workflow = StateGraph(ExplainerState)
    
    # Add nodes
    workflow.add_node("assemble", assembler.execute)
    workflow.add_node("reason", reasoner.execute)
    workflow.add_node("generate", generator.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Flow
    workflow.set_entry_point("assemble")
    
    workflow.add_conditional_edges(
        "assemble",
        lambda s: "error" if s.get("status") == "failed" else "reason",
        {"reason": "reason", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "reason",
        lambda s: "error" if s.get("status") == "failed" else "generate",
        {"generate": "generate", "error": "error_handler"}
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()

async def error_handler_node(state: ExplainerState) -> ExplainerState:
    return {
        **state,
        "executive_summary": "Unable to generate explanation due to errors.",
        "detailed_explanation": "Please review the errors and try again.",
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract
```python
class ExplainerInput(BaseModel):
    query: str
    analysis_results: List[Dict[str, Any]]
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
    output_format: Literal["narrative", "structured", "presentation", "brief"] = "narrative"
    focus_areas: Optional[List[str]] = None
```

### Output Contract
```python
class ExplainerOutput(BaseModel):
    executive_summary: str
    detailed_explanation: str
    narrative_sections: List[NarrativeSection]
    extracted_insights: List[Insight]
    visual_suggestions: List[Dict[str, Any]]
    follow_up_questions: List[str]
    total_latency_ms: int
```

## Handoff Format

```yaml
explainer_handoff:
  agent: explainer
  analysis_type: explanation
  key_findings:
    - insight_count: <count>
    - recommendation_count: <count>
    - themes: [<theme 1>, <theme 2>]
  outputs:
    executive_summary: <available>
    detailed_explanation: <available>
    sections: <count>
  suggestions:
    visuals: [<visual 1>, <visual 2>]
    follow_ups: [<question 1>, <question 2>]
  requires_further_analysis: <bool>
  suggested_next_agent: <feedback_learner>
```

---

## Cognitive RAG DSPy Integration

### Integration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Explainer Agent                              │
│                     (Tier 5 Deep Reasoning)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
│  │ Orchestrator│───►│         CognitiveRAG DSPy               │    │
│  │   Request   │    │                                         │    │
│  └─────────────┘    │  ┌───────────┐    ┌───────────────┐    │    │
│                     │  │Summarizer │───►│ Investigator  │    │    │
│                     │  │  (Phase 1)│    │   (Phase 2)   │    │    │
│                     │  └───────────┘    └───────┬───────┘    │    │
│                     │                           │            │    │
│                     │         ┌─────────────────▼──────────┐ │    │
│                     │         │    ExplainerCognitiveCtx   │ │    │
│                     │         │  • explanation_patterns    │ │    │
│                     │         │  • audience_preferences    │ │    │
│                     │         │  • narrative_templates     │ │    │
│                     │         │  • feedback_history        │ │    │
│                     │         │  • domain_analogies        │ │    │
│                     │         └─────────────────┬──────────┘ │    │
│                     └───────────────────────────┼────────────┘    │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                  ContextAssemblerNode                         │ │
│  │  • Receives cognitive context with explanation patterns       │ │
│  │  • Selects narrative templates based on audience              │ │
│  │  • Retrieves relevant domain analogies                        │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                   DeepReasonerNode                            │ │
│  │  • Chain-of-thought with cognitive enrichment                 │ │
│  │  • Applies successful explanation patterns                    │ │
│  │  • Integrates feedback_history for preference learning        │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                 NarrativeGeneratorNode                        │ │
│  │  • Generates explanation using learned templates              │ │
│  │  • Adapts language to audience_preferences                    │ │
│  │  • Applies domain_analogies for clarity                       │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│                     ┌───────────────────────────▼────────────────┐│
│                     │            TrainingSignal                  ││
│                     │  • user_rating                             ││
│                     │  • comprehension_score                     ││
│                     │  • follow_up_rate                          ││
│                     │  • action_taken                            ││
│                     └───────────────────────────┬────────────────┘│
│                                                 │                  │
│                     ┌───────────────────────────▼────────────────┐│
│                     │         Memory Contribution                ││
│                     │  Type: SEMANTIC (explanation patterns)     ││
│                     │  Index: successful_explanations            ││
│                     └────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### DSPy Signature Consumption

The Explainer consumes enriched context from the Cognitive RAG's Investigation phase:

```python
class ExplainerCognitiveContext(TypedDict):
    """Cognitive context enriched by CognitiveRAG for explanations."""
    synthesized_summary: str  # Evidence synthesis from Summarizer
    explanation_patterns: List[Dict[str, Any]]  # Successful explanation structures
    audience_preferences: Dict[str, Any]  # User/role preferences for explanations
    narrative_templates: List[Dict[str, Any]]  # Effective narrative structures
    feedback_history: List[Dict[str, Any]]  # Past feedback on explanations
    domain_analogies: List[Dict[str, Any]]  # Domain-appropriate analogies
    causal_narratives: List[Dict[str, Any]]  # Past causal chain explanations
    evidence_confidence: float  # Confidence in retrieved evidence
```

### Node Integration with Cognitive Context

```python
# In src/agents/explainer/nodes/context_assembler.py

async def context_assembler_node(
    state: ExplainerState,
    cognitive_context: Optional[ExplainerCognitiveContext] = None
) -> ExplainerState:
    """
    Assembles explanation context with cognitive enrichment.

    Cognitive enhancements:
    - Selects narrative template based on audience preferences
    - Retrieves relevant domain analogies for technical concepts
    - Incorporates feedback patterns to avoid past issues
    """
    start_time = time.time()

    if cognitive_context:
        audience_preferences = cognitive_context.get("audience_preferences", {})
        narrative_templates = cognitive_context.get("narrative_templates", [])
        domain_analogies = cognitive_context.get("domain_analogies", [])
        feedback_history = cognitive_context.get("feedback_history", [])

        # Select best narrative template for audience
        selected_template = select_template_for_audience(
            state["audience_type"],
            state["explanation_type"],
            narrative_templates,
            audience_preferences
        )

        # Find relevant analogies for technical concepts
        relevant_analogies = find_relevant_analogies(
            state["technical_concepts"],
            domain_analogies,
            state["audience_type"]
        )

        # Extract patterns to avoid from negative feedback
        patterns_to_avoid = extract_negative_patterns(
            feedback_history,
            state["explanation_type"]
        )
    else:
        selected_template = get_default_template(state["explanation_type"])
        relevant_analogies = []
        patterns_to_avoid = []

    return {
        **state,
        "selected_template": selected_template,
        "available_analogies": relevant_analogies,
        "patterns_to_avoid": patterns_to_avoid,
        "cognitive_enrichment_applied": cognitive_context is not None,
        "assembly_latency_ms": int((time.time() - start_time) * 1000)
    }


async def deep_reasoner_node(
    state: ExplainerState,
    cognitive_context: Optional[ExplainerCognitiveContext] = None
) -> ExplainerState:
    """
    Generates chain-of-thought reasoning with cognitive enhancement.

    Cognitive enhancements:
    - Applies successful explanation patterns from history
    - Uses causal narratives for cause-effect explanations
    - Incorporates synthesized summary for grounding
    """
    if cognitive_context:
        explanation_patterns = cognitive_context.get("explanation_patterns", [])
        causal_narratives = cognitive_context.get("causal_narratives", [])
        synthesized_summary = cognitive_context.get("synthesized_summary", "")

        # Find matching explanation pattern
        matched_pattern = match_explanation_pattern(
            state["query"],
            state["explanation_type"],
            explanation_patterns
        )

        # Get relevant causal narrative if explaining causality
        if state["explanation_type"] in ["causal_chain", "impact_analysis"]:
            causal_template = find_causal_narrative(
                state["causal_elements"],
                causal_narratives
            )
        else:
            causal_template = None

        # Ground reasoning in synthesized evidence
        grounding_context = synthesized_summary
    else:
        matched_pattern = None
        causal_template = None
        grounding_context = ""

    # Generate chain-of-thought reasoning
    reasoning_chain = await generate_cot_reasoning(
        query=state["query"],
        explanation_type=state["explanation_type"],
        context=grounding_context,
        pattern=matched_pattern,
        causal_template=causal_template,
        patterns_to_avoid=state.get("patterns_to_avoid", [])
    )

    return {
        **state,
        "reasoning_chain": reasoning_chain,
        "pattern_applied": matched_pattern.get("pattern_id") if matched_pattern else None
    }


async def narrative_generator_node(
    state: ExplainerState,
    cognitive_context: Optional[ExplainerCognitiveContext] = None
) -> ExplainerState:
    """
    Generates final explanation narrative with cognitive adaptation.

    Cognitive enhancements:
    - Adapts language complexity to audience preferences
    - Applies domain analogies at appropriate points
    - Uses template structure for consistency
    """
    template = state.get("selected_template")
    analogies = state.get("available_analogies", [])
    reasoning = state["reasoning_chain"]

    if cognitive_context:
        audience_prefs = cognitive_context.get("audience_preferences", {})

        # Determine language complexity
        complexity_level = audience_prefs.get(
            "complexity_preference",
            infer_complexity(state["audience_type"])
        )

        # Get preferred explanation length
        target_length = audience_prefs.get(
            "length_preference",
            "medium"
        )
    else:
        complexity_level = "medium"
        target_length = "medium"

    # Generate narrative
    explanation = await generate_narrative(
        reasoning_chain=reasoning,
        template=template,
        analogies=analogies,
        complexity=complexity_level,
        target_length=target_length
    )

    return {
        **state,
        "explanation": explanation,
        "complexity_applied": complexity_level,
        "analogies_used": [a["id"] for a in analogies if a.get("used")]
    }
```

### Training Signal for MIPROv2

```python
class ExplainerTrainingSignal:
    """Training signal for MIPROv2 optimization of explanation prompts."""

    def __init__(
        self,
        user_rating: Optional[float],  # 1-5 explicit rating
        comprehension_score: float,  # Inferred from follow-ups
        follow_up_rate: float,  # Rate of clarifying questions
        action_taken: bool,  # User acted on explanation
        explanation_length: int,  # Word count
        target_length: str,  # "short", "medium", "long"
        time_to_feedback: float  # Seconds until user response
    ):
        self.user_rating = user_rating
        self.comprehension_score = comprehension_score
        self.follow_up_rate = follow_up_rate
        self.action_taken = action_taken
        self.explanation_length = explanation_length
        self.target_length = target_length
        self.time_to_feedback = time_to_feedback

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 prompt optimization.

        Weighting:
        - user_rating: 0.35 (direct feedback)
        - comprehension_score: 0.25 (understanding)
        - action_taken: 0.20 (actionable)
        - follow_up_rate: 0.10 (inverse - fewer is better)
        - length_appropriateness: 0.10 (match target)
        """
        # Direct user rating (normalized to 0-1)
        if self.user_rating:
            rating_reward = (self.user_rating - 1) / 4.0 * 0.35
        else:
            rating_reward = 0.15  # Neutral if no rating

        # Comprehension score
        comprehension_reward = self.comprehension_score * 0.25

        # Action taken (binary)
        action_reward = 0.20 if self.action_taken else 0.0

        # Follow-up rate (lower is better, inverted)
        follow_up_penalty = min(self.follow_up_rate, 1.0) * 0.10

        # Length appropriateness
        target_ranges = {
            "short": (50, 150),
            "medium": (150, 400),
            "long": (400, 1000)
        }
        min_len, max_len = target_ranges.get(self.target_length, (150, 400))
        if min_len <= self.explanation_length <= max_len:
            length_reward = 0.10
        else:
            # Penalize being too far from target
            if self.explanation_length < min_len:
                deviation = (min_len - self.explanation_length) / min_len
            else:
                deviation = (self.explanation_length - max_len) / max_len
            length_reward = max(0, 0.10 - deviation * 0.05)

        return max(0.0, (
            rating_reward +
            comprehension_reward +
            action_reward +
            length_reward -
            follow_up_penalty
        ))
```

### Memory Contribution

```python
async def contribute_to_memory(
    state: ExplainerState,
    output: ExplainerOutput,
    feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Contribute successful explanations to semantic memory.

    Memory type: SEMANTIC (explanation patterns are reusable knowledge)
    Index: successful_explanations

    Only stores explanations with positive feedback signals.
    """
    # Only store explanations with good feedback
    if feedback:
        rating = feedback.get("rating", 0)
        comprehension = feedback.get("comprehension_score", 0)

        if rating < 3 and comprehension < 0.6:
            return  # Don't store poor explanations

    memory_entry = {
        "type": "SEMANTIC",
        "index": "successful_explanations",
        "content": {
            "query_pattern": extract_query_pattern(state["query"]),
            "explanation_type": state["explanation_type"],
            "audience_type": state["audience_type"],
            "template_used": state.get("selected_template", {}).get("id"),
            "reasoning_structure": summarize_reasoning(state["reasoning_chain"]),
            "analogies_used": state.get("analogies_used", []),
            "complexity_level": state.get("complexity_applied"),
            "explanation_summary": summarize_explanation(output.explanation),
            "feedback": {
                "rating": feedback.get("rating") if feedback else None,
                "comprehension": feedback.get("comprehension_score") if feedback else None,
                "action_taken": feedback.get("action_taken") if feedback else None
            },
            "brand": state.get("brand"),
            "domain": state.get("domain"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "metadata": {
            "agent": "explainer",
            "batch_id": state.get("batch_id"),
            "ttl_days": 180  # Patterns remain relevant for 6 months
        }
    }

    await memory_service.store(memory_entry)
```

### Integration with CognitiveInput

```python
class ExplainerCognitiveInput(TypedDict):
    """Input structure for cognitive-enhanced explanation generation."""
    query: str
    explanation_type: str  # "causal_chain", "impact_analysis", "concept", "trend"
    audience_type: str  # "executive", "analyst", "technical", "general"
    source_analysis: Dict[str, Any]  # Analysis to explain
    technical_concepts: List[str]  # Concepts that may need simplification
    causal_elements: Optional[Dict[str, Any]]  # Causal chain if applicable
    context_depth: str  # "brief", "detailed", "comprehensive"
    brand: Optional[str]
    user_id: Optional[str]  # For personalization
    cognitive_context: Optional[ExplainerCognitiveContext]
```

### Configuration

```yaml
# config/agents/explainer.yaml
cognitive_integration:
  enabled: true
  context_retrieval:
    explanation_pattern_limit: 30
    narrative_template_limit: 10
    feedback_history_limit: 50
    analogy_limit: 20
    causal_narrative_limit: 15

  memory_contribution:
    type: SEMANTIC
    index: successful_explanations
    ttl_days: 180
    min_rating_threshold: 3  # Only store 3+ rated explanations
    min_comprehension_threshold: 0.6

  training_signals:
    emit: true
    weights:
      user_rating: 0.35
      comprehension_score: 0.25
      action_taken: 0.20
      follow_up_rate: 0.10
      length_appropriateness: 0.10

  personalization:
    enabled: true
    user_preference_ttl_days: 90
    min_interactions_for_personalization: 3
```

### Testing Requirements for DSPy Integration

```python
@pytest.mark.asyncio
async def test_cognitive_context_improves_explanations():
    """Test that cognitive context produces better explanations."""
    agent = ExplainerAgent()

    source_analysis = {
        "type": "causal_chain",
        "treatment": "digital_campaign",
        "outcome": "rx_lift",
        "effect_size": 0.15
    }

    # Without cognitive context
    result_baseline = await agent.explain(
        query="Why did digital campaigns increase prescriptions?",
        explanation_type="causal_chain",
        audience_type="executive",
        source_analysis=source_analysis
    )

    # With cognitive context
    cognitive_context = ExplainerCognitiveContext(
        synthesized_summary="Digital campaigns show 15% lift in Rx",
        explanation_patterns=[
            {
                "pattern_id": "causal_exec_1",
                "structure": "outcome_first_then_mechanism",
                "avg_rating": 4.5
            }
        ],
        audience_preferences={
            "complexity_preference": "low",
            "length_preference": "short",
            "analogy_preference": True
        },
        narrative_templates=[
            {"id": "exec_causal", "structure": ["headline", "mechanism", "evidence"]}
        ],
        feedback_history=[
            {"rating": 5, "explanation_type": "causal_chain", "length": 120}
        ],
        domain_analogies=[
            {"concept": "causal_effect", "analogy": "domino_effect", "audience": "executive"}
        ],
        causal_narratives=[
            {"type": "digital_to_rx", "narrative": "awareness_to_action"}
        ],
        evidence_confidence=0.9
    )

    result_cognitive = await agent.explain(
        query="Why did digital campaigns increase prescriptions?",
        explanation_type="causal_chain",
        audience_type="executive",
        source_analysis=source_analysis,
        cognitive_context=cognitive_context
    )

    # Cognitive version should be more concise for executives
    assert len(result_cognitive.explanation) < len(result_baseline.explanation) * 1.5


@pytest.mark.asyncio
async def test_training_signal_computation():
    """Test training signal computation for various scenarios."""
    # High quality explanation
    signal_good = ExplainerTrainingSignal(
        user_rating=5,
        comprehension_score=0.95,
        follow_up_rate=0.1,
        action_taken=True,
        explanation_length=200,
        target_length="medium",
        time_to_feedback=5.0
    )

    reward_good = signal_good.compute_reward()
    assert reward_good > 0.8

    # Poor explanation
    signal_poor = ExplainerTrainingSignal(
        user_rating=2,
        comprehension_score=0.3,
        follow_up_rate=0.8,
        action_taken=False,
        explanation_length=800,
        target_length="short",
        time_to_feedback=30.0
    )

    reward_poor = signal_poor.compute_reward()
    assert reward_poor < 0.3
    assert reward_good > reward_poor


@pytest.mark.asyncio
async def test_audience_adaptation():
    """Test that explanations adapt to different audiences."""
    agent = ExplainerAgent()

    source = {"metric": "market_share", "change": 0.05, "period": "Q4"}

    # Executive audience
    exec_context = ExplainerCognitiveContext(
        synthesized_summary="Market share up 5%",
        explanation_patterns=[],
        audience_preferences={"complexity_preference": "low", "length_preference": "short"},
        narrative_templates=[],
        feedback_history=[],
        domain_analogies=[],
        causal_narratives=[],
        evidence_confidence=0.85
    )

    exec_result = await agent.explain(
        query="Explain market share performance",
        explanation_type="trend",
        audience_type="executive",
        source_analysis=source,
        cognitive_context=exec_context
    )

    # Technical audience
    tech_context = ExplainerCognitiveContext(
        synthesized_summary="Market share increased 5% in Q4",
        explanation_patterns=[],
        audience_preferences={"complexity_preference": "high", "length_preference": "detailed"},
        narrative_templates=[],
        feedback_history=[],
        domain_analogies=[],
        causal_narratives=[],
        evidence_confidence=0.85
    )

    tech_result = await agent.explain(
        query="Explain market share performance",
        explanation_type="trend",
        audience_type="technical",
        source_analysis=source,
        cognitive_context=tech_context
    )

    # Technical should be longer/more detailed
    assert len(tech_result.explanation) > len(exec_result.explanation)


@pytest.mark.asyncio
async def test_memory_contribution_filtering():
    """Test that only quality explanations are stored."""
    memory_service = MockMemoryService()

    # Good explanation should be stored
    good_output = ExplainerOutput(
        explanation="Clear explanation...",
        status="completed"
    )
    good_feedback = {"rating": 4, "comprehension_score": 0.9}

    await contribute_to_memory(
        state={"query": "test", "explanation_type": "concept"},
        output=good_output,
        feedback=good_feedback
    )

    assert len(memory_service.stored_entries) == 1

    # Poor explanation should not be stored
    memory_service.stored_entries.clear()

    poor_feedback = {"rating": 2, "comprehension_score": 0.4}

    await contribute_to_memory(
        state={"query": "test", "explanation_type": "concept"},
        output=good_output,
        feedback=poor_feedback
    )

    assert len(memory_service.stored_entries) == 0
```
