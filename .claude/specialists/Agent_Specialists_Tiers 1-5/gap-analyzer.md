# Tier 2: Gap Analyzer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 2 (Causal Analytics) |
| **Agent Type** | Standard (Computational) |
| **Model Tier** | Sonnet |
| **Latency Tolerance** | Medium (up to 20s) |
| **Critical Path** | Core E2I mission agent |

## Domain Scope

You are the specialist for the Tier 2 Gap Analyzer Agent:
- `src/agents/gap_analyzer/` - ROI opportunity detection

This is a **Standard Agent** with computational focus:
- Tool execution with structured outputs
- Parallelizable gap detection across segments
- No deep reasoning required

## Design Principles

### Computational Focus
The Gap Analyzer is optimized for throughput:
- Parallel segment analysis
- Efficient database queries
- Structured ROI calculations
- Minimal LLM usage (only for final summary)

### Responsibilities
1. **Gap Detection** - Identify performance gaps across metrics
2. **ROI Calculation** - Estimate value of closing each gap
3. **Prioritization** - Rank opportunities by expected ROI
4. **Segment Analysis** - Break down gaps by HCP segments

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GAP ANALYZER AGENT                          │
│                      (Standard Pattern)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    GAP      │───►│    ROI      │───►│  PRIORITY   │         │
│  │  DETECTOR   │    │ CALCULATOR  │    │   RANKER    │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│         │                                      │                 │
│         ▼                                      ▼                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              PARALLEL SEGMENT ANALYSIS              │        │
│  │   [Region] [Specialty] [Decile] [Territory] ...    │        │
│  └─────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │     OUTPUT      │                          │
│                    │   FORMATTER     │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
gap_analyzer/
├── agent.py              # Main GapAnalyzerAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── gap_detector.py   # Gap identification algorithms
│   ├── roi_calculator.py # ROI estimation logic
│   ├── prioritizer.py    # Opportunity ranking
│   └── formatter.py      # Output formatting
├── gap_types.py          # Gap type definitions
└── benchmarks.py         # Benchmark data management
```

## LangGraph State Definition

```python
# src/agents/gap_analyzer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class PerformanceGap(TypedDict):
    """Individual performance gap"""
    gap_id: str
    metric: str
    segment: str
    segment_value: str
    current_value: float
    target_value: float
    gap_size: float
    gap_percentage: float
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal"]

class ROIEstimate(TypedDict):
    """ROI estimate for closing a gap"""
    gap_id: str
    estimated_revenue_impact: float
    estimated_cost_to_close: float
    expected_roi: float
    payback_period_months: int
    confidence: float
    assumptions: List[str]

class PrioritizedOpportunity(TypedDict):
    """Prioritized gap with action recommendation"""
    rank: int
    gap: PerformanceGap
    roi_estimate: ROIEstimate
    recommended_action: str
    implementation_difficulty: Literal["low", "medium", "high"]
    time_to_impact: str

class GapAnalyzerState(TypedDict):
    """Complete state for Gap Analyzer agent"""
    
    # === INPUT ===
    query: str
    metrics: List[str]  # KPIs to analyze
    segments: List[str]  # Segmentation dimensions
    brand: str
    time_period: str
    filters: Optional[Dict[str, Any]]
    
    # === CONFIGURATION ===
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]
    min_gap_threshold: float  # Minimum gap % to report
    max_opportunities: int  # Maximum opportunities to return
    
    # === DETECTION OUTPUTS ===
    gaps_detected: Optional[List[PerformanceGap]]
    gaps_by_segment: Optional[Dict[str, List[PerformanceGap]]]
    total_gap_value: Optional[float]
    
    # === ROI OUTPUTS ===
    roi_estimates: Optional[List[ROIEstimate]]
    total_addressable_value: Optional[float]
    
    # === PRIORITIZATION OUTPUTS ===
    prioritized_opportunities: Optional[List[PrioritizedOpportunity]]
    quick_wins: Optional[List[PrioritizedOpportunity]]  # Low difficulty, high ROI
    strategic_bets: Optional[List[PrioritizedOpportunity]]  # High impact, high difficulty
    
    # === SUMMARY ===
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]
    
    # === EXECUTION METADATA ===
    detection_latency_ms: int
    roi_latency_ms: int
    total_latency_ms: int
    segments_analyzed: int
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "detecting", "calculating", "prioritizing", "completed", "failed"]
```

## Node Implementations

### Gap Detector Node

```python
# src/agents/gap_analyzer/nodes/gap_detector.py

import asyncio
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ..state import GapAnalyzerState, PerformanceGap

class GapDetectorNode:
    """
    Detect performance gaps across metrics and segments
    Optimized for parallel execution
    """
    
    def __init__(self, data_connector, benchmark_store):
        self.data_connector = data_connector
        self.benchmark_store = benchmark_store
    
    async def execute(self, state: GapAnalyzerState) -> GapAnalyzerState:
        start_time = time.time()
        
        try:
            # Fetch current performance data
            current_data = await self._fetch_performance_data(state)
            
            # Get comparison data based on gap type
            comparison_data = await self._get_comparison_data(state)
            
            # Detect gaps in parallel across segments
            segment_tasks = []
            for segment in state["segments"]:
                segment_tasks.append(
                    self._detect_segment_gaps(
                        current_data, 
                        comparison_data, 
                        segment,
                        state["metrics"],
                        state["gap_type"],
                        state["min_gap_threshold"]
                    )
                )
            
            segment_results = await asyncio.gather(*segment_tasks)
            
            # Flatten and aggregate results
            all_gaps = []
            gaps_by_segment = {}
            
            for segment, gaps in zip(state["segments"], segment_results):
                gaps_by_segment[segment] = gaps
                all_gaps.extend(gaps)
            
            # Calculate total gap value
            total_gap_value = sum(g["gap_size"] for g in all_gaps)
            
            detection_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "gaps_detected": all_gaps,
                "gaps_by_segment": gaps_by_segment,
                "total_gap_value": total_gap_value,
                "segments_analyzed": len(state["segments"]),
                "detection_latency_ms": detection_time,
                "status": "calculating"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "gap_detector", "error": str(e)}],
                "status": "failed"
            }
    
    async def _fetch_performance_data(self, state: GapAnalyzerState) -> pd.DataFrame:
        """Fetch current performance metrics"""
        return await self.data_connector.query(
            source="performance_metrics",
            metrics=state["metrics"],
            segments=state["segments"],
            brand=state["brand"],
            time_period=state["time_period"],
            filters=state.get("filters")
        )
    
    async def _get_comparison_data(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Get comparison data based on gap type"""
        gap_type = state["gap_type"]
        
        if gap_type == "vs_target":
            return await self.benchmark_store.get_targets(
                brand=state["brand"],
                metrics=state["metrics"]
            )
        elif gap_type == "vs_benchmark":
            return await self.benchmark_store.get_peer_benchmarks(
                brand=state["brand"],
                metrics=state["metrics"]
            )
        elif gap_type == "vs_potential":
            return await self._calculate_potential(state)
        elif gap_type == "temporal":
            return await self._get_prior_period(state)
        else:
            # "all" - return all comparison types
            return {
                "target": await self.benchmark_store.get_targets(state["brand"], state["metrics"]),
                "benchmark": await self.benchmark_store.get_peer_benchmarks(state["brand"], state["metrics"]),
                "potential": await self._calculate_potential(state)
            }
    
    async def _detect_segment_gaps(
        self,
        current_data: pd.DataFrame,
        comparison_data: Dict[str, Any],
        segment: str,
        metrics: List[str],
        gap_type: str,
        min_threshold: float
    ) -> List[PerformanceGap]:
        """Detect gaps for a single segment"""
        
        gaps = []
        segment_values = current_data[segment].unique()
        
        for segment_value in segment_values:
            segment_data = current_data[current_data[segment] == segment_value]
            
            for metric in metrics:
                current_value = segment_data[metric].mean()
                target_value = self._get_target_value(
                    comparison_data, metric, segment, segment_value, gap_type
                )
                
                if target_value is None:
                    continue
                
                gap_size = target_value - current_value
                gap_percentage = (gap_size / target_value * 100) if target_value != 0 else 0
                
                # Only report gaps above threshold
                if abs(gap_percentage) >= min_threshold:
                    gaps.append(PerformanceGap(
                        gap_id=f"{segment}_{segment_value}_{metric}",
                        metric=metric,
                        segment=segment,
                        segment_value=str(segment_value),
                        current_value=current_value,
                        target_value=target_value,
                        gap_size=gap_size,
                        gap_percentage=gap_percentage,
                        gap_type=gap_type
                    ))
        
        return gaps
    
    def _get_target_value(
        self,
        comparison_data: Dict,
        metric: str,
        segment: str,
        segment_value: str,
        gap_type: str
    ) -> float:
        """Extract target value from comparison data"""
        try:
            if gap_type in ["vs_target", "vs_benchmark", "vs_potential", "temporal"]:
                return comparison_data.get(metric, {}).get(segment_value, comparison_data.get(metric, {}).get("default"))
            else:
                # For "all", use the most stringent target
                targets = []
                for comparison_type in ["target", "benchmark", "potential"]:
                    if comparison_type in comparison_data:
                        val = comparison_data[comparison_type].get(metric, {}).get(segment_value)
                        if val is not None:
                            targets.append(val)
                return max(targets) if targets else None
        except Exception:
            return None
    
    async def _calculate_potential(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Calculate potential based on top performer analysis"""
        # Top decile performance as potential
        data = await self.data_connector.query(
            source="performance_metrics",
            metrics=state["metrics"],
            segments=state["segments"],
            brand=state["brand"],
            time_period=state["time_period"]
        )
        
        potential = {}
        for metric in state["metrics"]:
            potential[metric] = {"default": data[metric].quantile(0.9)}
        
        return potential
    
    async def _get_prior_period(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Get prior period data for temporal comparison"""
        # Implementation depends on time_period format
        return await self.data_connector.query(
            source="performance_metrics",
            metrics=state["metrics"],
            segments=state["segments"],
            brand=state["brand"],
            time_period=self._get_prior_period_string(state["time_period"])
        )
    
    def _get_prior_period_string(self, current_period: str) -> str:
        """Convert current period to prior period"""
        # Simplified - actual implementation would handle various formats
        if "Q" in current_period:
            year, quarter = current_period.split("-Q")
            quarter = int(quarter)
            if quarter == 1:
                return f"{int(year)-1}-Q4"
            return f"{year}-Q{quarter-1}"
        return current_period
```

### ROI Calculator Node

```python
# src/agents/gap_analyzer/nodes/roi_calculator.py

import asyncio
import time
from typing import List, Dict, Any

from ..state import GapAnalyzerState, ROIEstimate, PerformanceGap

class ROICalculatorNode:
    """
    Calculate ROI for closing identified gaps
    Uses pharmaceutical-specific economics
    """
    
    # Default assumptions for ROI calculation
    DEFAULT_ASSUMPTIONS = {
        "revenue_per_trx": 500,  # Average revenue per prescription
        "cost_per_hcp_visit": 150,
        "cost_per_sample": 25,
        "conversion_rate_improvement": 0.05,  # Expected improvement from intervention
        "time_to_impact_months": 3
    }
    
    def __init__(self, economics_config: Dict[str, Any] = None):
        self.config = {**self.DEFAULT_ASSUMPTIONS, **(economics_config or {})}
    
    async def execute(self, state: GapAnalyzerState) -> GapAnalyzerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            gaps = state["gaps_detected"]
            
            # Calculate ROI for each gap in parallel
            roi_tasks = [self._calculate_gap_roi(gap) for gap in gaps]
            roi_estimates = await asyncio.gather(*roi_tasks)
            
            # Calculate total addressable value
            total_addressable = sum(
                r["estimated_revenue_impact"] for r in roi_estimates if r["expected_roi"] > 0
            )
            
            roi_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "roi_estimates": roi_estimates,
                "total_addressable_value": total_addressable,
                "roi_latency_ms": roi_time,
                "status": "prioritizing"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "roi_calculator", "error": str(e)}],
                "status": "failed"
            }
    
    async def _calculate_gap_roi(self, gap: PerformanceGap) -> ROIEstimate:
        """Calculate ROI for a single gap"""
        
        metric = gap["metric"]
        gap_size = gap["gap_size"]
        
        # Calculate revenue impact based on metric type
        revenue_impact = self._estimate_revenue_impact(metric, gap_size)
        
        # Calculate cost to close
        cost_to_close = self._estimate_cost_to_close(metric, gap_size)
        
        # Calculate ROI
        if cost_to_close > 0:
            roi = (revenue_impact - cost_to_close) / cost_to_close
        else:
            roi = float('inf') if revenue_impact > 0 else 0
        
        # Calculate payback period
        if revenue_impact > 0:
            monthly_revenue = revenue_impact / 12
            payback_months = int(cost_to_close / monthly_revenue) + 1 if monthly_revenue > 0 else 24
        else:
            payback_months = 999
        
        # Determine confidence based on gap type and size
        confidence = self._calculate_confidence(gap)
        
        return ROIEstimate(
            gap_id=gap["gap_id"],
            estimated_revenue_impact=revenue_impact,
            estimated_cost_to_close=cost_to_close,
            expected_roi=roi,
            payback_period_months=min(payback_months, 24),
            confidence=confidence,
            assumptions=[
                f"Revenue per TRx: ${self.config['revenue_per_trx']}",
                f"Expected conversion improvement: {self.config['conversion_rate_improvement']*100}%",
                f"Time to impact: {self.config['time_to_impact_months']} months"
            ]
        )
    
    def _estimate_revenue_impact(self, metric: str, gap_size: float) -> float:
        """Estimate revenue impact of closing gap"""
        
        # Map metrics to revenue impact
        metric_multipliers = {
            "trx": self.config["revenue_per_trx"],
            "nrx": self.config["revenue_per_trx"] * 0.8,
            "conversion_rate": self.config["revenue_per_trx"] * 100,
            "market_share": self.config["revenue_per_trx"] * 1000,
            "reach": self.config["revenue_per_trx"] * 0.1
        }
        
        # Find matching metric
        multiplier = 0
        for key, value in metric_multipliers.items():
            if key in metric.lower():
                multiplier = value
                break
        
        if multiplier == 0:
            multiplier = self.config["revenue_per_trx"] * 0.5  # Default
        
        return abs(gap_size) * multiplier
    
    def _estimate_cost_to_close(self, metric: str, gap_size: float) -> float:
        """Estimate cost to close gap"""
        
        # Different metrics require different interventions
        if "reach" in metric.lower() or "frequency" in metric.lower():
            # Requires more HCP visits
            return abs(gap_size) * self.config["cost_per_hcp_visit"] * 2
        elif "conversion" in metric.lower():
            # Requires training and samples
            return abs(gap_size) * 1000 * (
                self.config["cost_per_hcp_visit"] + self.config["cost_per_sample"] * 10
            )
        else:
            # General intervention cost
            return abs(gap_size) * self.config["cost_per_hcp_visit"]
    
    def _calculate_confidence(self, gap: PerformanceGap) -> float:
        """Calculate confidence in ROI estimate"""
        
        base_confidence = 0.7
        
        # Higher confidence for larger gaps (more room for improvement)
        if abs(gap["gap_percentage"]) > 20:
            base_confidence += 0.1
        
        # Higher confidence for certain gap types
        if gap["gap_type"] == "vs_target":
            base_confidence += 0.1
        elif gap["gap_type"] == "vs_benchmark":
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
```

### Prioritizer Node

```python
# src/agents/gap_analyzer/nodes/prioritizer.py

import time
from typing import List

from ..state import GapAnalyzerState, PrioritizedOpportunity, PerformanceGap, ROIEstimate

class PrioritizerNode:
    """
    Rank and categorize opportunities
    Pure logic - no LLM needed
    """
    
    def __init__(self):
        self.difficulty_thresholds = {
            "low": {"max_cost": 10000, "max_gap_pct": 10},
            "medium": {"max_cost": 50000, "max_gap_pct": 25},
            "high": {"min_cost": 50000}
        }
    
    async def execute(self, state: GapAnalyzerState) -> GapAnalyzerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            gaps = state["gaps_detected"]
            roi_estimates = state["roi_estimates"]
            
            # Create gap-to-roi mapping
            roi_by_gap = {r["gap_id"]: r for r in roi_estimates}
            
            # Create prioritized opportunities
            opportunities = []
            for gap in gaps:
                roi = roi_by_gap.get(gap["gap_id"])
                if roi:
                    opportunity = self._create_opportunity(gap, roi)
                    opportunities.append(opportunity)
            
            # Sort by expected ROI (descending)
            opportunities.sort(key=lambda x: x["roi_estimate"]["expected_roi"], reverse=True)
            
            # Assign ranks
            for i, opp in enumerate(opportunities):
                opp["rank"] = i + 1
            
            # Categorize
            quick_wins = [
                o for o in opportunities 
                if o["implementation_difficulty"] == "low" and o["roi_estimate"]["expected_roi"] > 1
            ][:5]
            
            strategic_bets = [
                o for o in opportunities 
                if o["implementation_difficulty"] == "high" and o["roi_estimate"]["expected_roi"] > 2
            ][:5]
            
            # Limit to max opportunities
            max_opps = state.get("max_opportunities", 10)
            prioritized = opportunities[:max_opps]
            
            total_time = (
                state.get("detection_latency_ms", 0) +
                state.get("roi_latency_ms", 0) +
                int((time.time() - start_time) * 1000)
            )
            
            return {
                **state,
                "prioritized_opportunities": prioritized,
                "quick_wins": quick_wins,
                "strategic_bets": strategic_bets,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "prioritizer", "error": str(e)}],
                "status": "failed"
            }
    
    def _create_opportunity(
        self, 
        gap: PerformanceGap, 
        roi: ROIEstimate
    ) -> PrioritizedOpportunity:
        """Create prioritized opportunity from gap and ROI"""
        
        difficulty = self._assess_difficulty(gap, roi)
        action = self._recommend_action(gap, roi)
        time_to_impact = self._estimate_time_to_impact(gap, difficulty)
        
        return PrioritizedOpportunity(
            rank=0,  # Will be set after sorting
            gap=gap,
            roi_estimate=roi,
            recommended_action=action,
            implementation_difficulty=difficulty,
            time_to_impact=time_to_impact
        )
    
    def _assess_difficulty(self, gap: PerformanceGap, roi: ROIEstimate) -> str:
        """Assess implementation difficulty"""
        
        cost = roi["estimated_cost_to_close"]
        gap_pct = abs(gap["gap_percentage"])
        
        if cost < self.difficulty_thresholds["low"]["max_cost"] and \
           gap_pct < self.difficulty_thresholds["low"]["max_gap_pct"]:
            return "low"
        elif cost >= self.difficulty_thresholds["high"]["min_cost"]:
            return "high"
        else:
            return "medium"
    
    def _recommend_action(self, gap: PerformanceGap, roi: ROIEstimate) -> str:
        """Generate action recommendation"""
        
        metric = gap["metric"].lower()
        segment = gap["segment"]
        segment_value = gap["segment_value"]
        gap_pct = gap["gap_percentage"]
        
        if "reach" in metric:
            return f"Increase HCP contact frequency in {segment}: {segment_value}"
        elif "conversion" in metric:
            return f"Deploy targeted messaging for {segment}: {segment_value}"
        elif "share" in metric:
            return f"Competitive displacement campaign in {segment}: {segment_value}"
        elif "trx" in metric or "nrx" in metric:
            return f"Intensify rep engagement for {segment}: {segment_value}"
        else:
            return f"Address {abs(gap_pct):.1f}% gap in {metric} for {segment}: {segment_value}"
    
    def _estimate_time_to_impact(self, gap: PerformanceGap, difficulty: str) -> str:
        """Estimate time to see impact"""
        
        if difficulty == "low":
            return "1-3 months"
        elif difficulty == "medium":
            return "3-6 months"
        else:
            return "6-12 months"
```

## Graph Assembly

```python
# src/agents/gap_analyzer/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GapAnalyzerState
from .nodes.gap_detector import GapDetectorNode
from .nodes.roi_calculator import ROICalculatorNode
from .nodes.prioritizer import PrioritizerNode
from .nodes.formatter import OutputFormatterNode

def build_gap_analyzer_graph(
    data_connector,
    benchmark_store,
    economics_config: dict = None,
    enable_checkpointing: bool = True
):
    """
    Build the Gap Analyzer agent graph
    
    Architecture:
        [detect] → [calculate_roi] → [prioritize] → [format] → END
    """
    
    # Initialize nodes
    detector = GapDetectorNode(data_connector, benchmark_store)
    roi_calculator = ROICalculatorNode(economics_config)
    prioritizer = PrioritizerNode()
    formatter = OutputFormatterNode()
    
    # Build graph
    workflow = StateGraph(GapAnalyzerState)
    
    # Add nodes
    workflow.add_node("detect", detector.execute)
    workflow.add_node("calculate_roi", roi_calculator.execute)
    workflow.add_node("prioritize", prioritizer.execute)
    workflow.add_node("format", formatter.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("detect")
    
    # Edges
    workflow.add_conditional_edges(
        "detect",
        lambda s: "error" if s.get("status") == "failed" else "calculate_roi",
        {"calculate_roi": "calculate_roi", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "calculate_roi",
        lambda s: "error" if s.get("status") == "failed" else "prioritize",
        {"prioritize": "prioritize", "error": "error_handler"}
    )
    
    workflow.add_edge("prioritize", "format")
    workflow.add_edge("format", END)
    workflow.add_edge("error_handler", END)
    
    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()

async def error_handler_node(state: GapAnalyzerState) -> GapAnalyzerState:
    """Handle errors gracefully"""
    errors = state.get("errors", [])
    
    return {
        **state,
        "executive_summary": "Gap analysis could not be completed.",
        "key_insights": [f"Error: {e.get('error', 'Unknown')}" for e in errors],
        "prioritized_opportunities": [],
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract (from Orchestrator)
```python
class GapAnalyzerInput(BaseModel):
    query: str
    metrics: List[str] = ["trx", "market_share", "conversion_rate"]
    segments: List[str] = ["region", "specialty", "decile"]
    brand: str
    time_period: str = "current_quarter"
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"] = "vs_potential"
    min_gap_threshold: float = 5.0
    max_opportunities: int = 10
```

### Output Contract (to Orchestrator)
```python
class GapAnalyzerOutput(BaseModel):
    prioritized_opportunities: List[PrioritizedOpportunity]
    quick_wins: List[PrioritizedOpportunity]
    strategic_bets: List[PrioritizedOpportunity]
    total_addressable_value: float
    executive_summary: str
    key_insights: List[str]
    total_latency_ms: int
```

## Testing Requirements

```
tests/unit/test_agents/test_gap_analyzer/
├── test_gap_detector.py    # Gap detection logic
├── test_roi_calculator.py  # ROI calculations
├── test_prioritizer.py     # Ranking logic
└── test_integration.py     # End-to-end flow
```

### Performance Requirements
- Gap detection: <10s for 10 segments × 5 metrics
- ROI calculation: <2s per gap
- Total latency: <20s

### Test Cases
1. Gaps detected across all segment types
2. ROI correctly calculated for different metric types
3. Quick wins identified correctly (low difficulty, high ROI)
4. Strategic bets identified correctly (high impact, high difficulty)
5. No gaps scenario handled gracefully

## Handoff Format

```yaml
gap_analyzer_handoff:
  agent: gap_analyzer
  analysis_type: roi_opportunity_detection
  key_findings:
    - total_gaps: <count>
    - total_addressable_value: <currency>
    - top_opportunity: <description>
  quick_wins:
    - action: <recommended action>
      expected_roi: <percentage>
      time_to_impact: <duration>
  strategic_bets:
    - action: <recommended action>
      expected_impact: <revenue>
      implementation_difficulty: <low|medium|high>
  recommendations:
    - <recommendation 1>
    - <recommendation 2>
  requires_further_analysis: <bool>
  suggested_next_agent: <resource_optimizer|experiment_designer>
```

---

## Cognitive RAG DSPy Integration

### Integration Overview

The Gap Analyzer agent integrates with the Cognitive RAG DSPy system to receive enriched context from multi-hop retrieval and contribute gap/opportunity insights back to the semantic memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GAP ANALYZER - COGNITIVE RAG INTEGRATION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2: INVESTIGATOR                    PHASE 3: AGENT                    │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ EvidenceSynthesis    │                │      GAP ANALYZER           │    │
│  │ Signature            │                │                             │    │
│  │  - evidence_items    │───────────────►│  1. Parse synthesized       │    │
│  │  - user_query        │                │     evidence                │    │
│  │  - investigation_path│                │  2. Extract benchmark data  │    │
│  │                      │                │  3. Identify gap patterns   │    │
│  │ OUTPUTS:             │                │  4. Calculate ROI estimates │    │
│  │  - synthesized_summary                │                             │    │
│  │  - benchmark_context ─────────────────│──► Used for vs_benchmark    │    │
│  │  - historical_gaps   ─────────────────│──► Similar gap patterns     │    │
│  └──────────────────────┘                └──────────────┬──────────────┘    │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌─────────────────────────────┐    │
│                                          │    TRAINING SIGNAL          │    │
│                                          │  GapAnalyzerTrainingSignal  │    │
│                                          │   - gap_accuracy            │    │
│                                          │   - roi_precision           │    │
│                                          │   - user_action_rate        │    │
│                                          └──────────────┬──────────────┘    │
│                                                         │                   │
│  PHASE 4: REFLECTOR                                     ▼                   │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ MemoryWorthiness     │◄───────────────│   MEMORY CONTRIBUTION       │    │
│  │ Signature            │                │   - gap_opportunities index │    │
│  │   store_gap_patterns │                │   - benchmark data updates  │    │
│  └──────────────────────┘                └─────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signatures Consumed

The Gap Analyzer consumes evidence from the `EvidenceSynthesisSignature`:

```python
# From CognitiveRAG Phase 2 - Consumed by Gap Analyzer

class EvidenceSynthesisSignature(dspy.Signature):
    """Synthesize evidence from multi-hop retrieval into coherent context."""
    evidence_items: List[Evidence] = dspy.InputField(desc="Retrieved evidence items")
    user_query: str = dspy.InputField(desc="Original user query")
    investigation_path: List[Dict] = dspy.InputField(desc="Multi-hop retrieval path")

    synthesized_summary: str = dspy.OutputField(desc="Coherent synthesis of all evidence")
    benchmark_context: str = dspy.OutputField(desc="Extracted benchmark/target data from evidence")
    historical_gaps: List[Dict] = dspy.OutputField(desc="Previously identified similar gaps")
    confidence_score: float = dspy.OutputField(desc="Confidence in synthesis 0.0-1.0")
```

### Integration in Gap Detector Node

```python
# src/agents/gap_analyzer/nodes/gap_detector.py

from typing import TypedDict, List, Dict, Any, Optional

class GapCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for gap analysis."""
    synthesized_summary: str
    benchmark_context: str
    historical_gaps: List[Dict[str, Any]]
    evidence_confidence: float
    retrieved_benchmarks: Dict[str, float]

class GapDetectorNode:
    """Gap Detector with Cognitive RAG integration."""

    async def execute(self, state: GapAnalyzerState) -> GapAnalyzerState:
        start_time = time.time()

        try:
            # Extract cognitive context if available
            cognitive_context = self._extract_cognitive_context(state)

            # Fetch current performance data
            current_data = await self._fetch_performance_data(state)

            # Get comparison data - enriched with cognitive evidence
            comparison_data = await self._get_comparison_data(
                state,
                cognitive_context
            )

            # Detect gaps with historical pattern awareness
            all_gaps = await self._detect_gaps_with_context(
                current_data,
                comparison_data,
                cognitive_context,
                state
            )

            return {
                **state,
                "gaps_detected": all_gaps,
                "cognitive_enrichment_applied": cognitive_context is not None,
                "status": "calculating"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "gap_detector", "error": str(e)}],
                "status": "failed"
            }

    def _extract_cognitive_context(
        self,
        state: GapAnalyzerState
    ) -> Optional[GapCognitiveContext]:
        """Extract cognitive context from state if CognitiveRAG enriched."""

        cognitive_input = state.get("cognitive_enrichment")
        if not cognitive_input:
            return None

        return GapCognitiveContext(
            synthesized_summary=cognitive_input.get("synthesized_summary", ""),
            benchmark_context=cognitive_input.get("benchmark_context", ""),
            historical_gaps=cognitive_input.get("historical_gaps", []),
            evidence_confidence=cognitive_input.get("confidence_score", 0.5),
            retrieved_benchmarks=self._parse_benchmarks(
                cognitive_input.get("benchmark_context", "")
            )
        )

    def _parse_benchmarks(self, benchmark_context: str) -> Dict[str, float]:
        """Parse benchmark values from cognitive context."""
        benchmarks = {}
        # Parse structured benchmark data from evidence
        # Example: "TRx benchmark: 150, Market Share benchmark: 12.5%"
        import re
        patterns = re.findall(r'(\w+)\s*benchmark[:\s]+([0-9.]+)', benchmark_context, re.I)
        for metric, value in patterns:
            benchmarks[metric.lower()] = float(value)
        return benchmarks

    async def _get_comparison_data(
        self,
        state: GapAnalyzerState,
        cognitive_context: Optional[GapCognitiveContext]
    ) -> Dict[str, Any]:
        """Get comparison data enriched with cognitive evidence."""

        gap_type = state["gap_type"]

        # Start with base comparison data
        base_comparison = await self._get_base_comparison(state, gap_type)

        # Enrich with cognitive benchmarks if available
        if cognitive_context and cognitive_context["retrieved_benchmarks"]:
            for metric, value in cognitive_context["retrieved_benchmarks"].items():
                if metric in base_comparison:
                    # Blend cognitive evidence with stored benchmarks
                    confidence = cognitive_context["evidence_confidence"]
                    base_comparison[metric]["cognitive_adjusted"] = (
                        base_comparison[metric].get("default", value) * (1 - confidence * 0.3) +
                        value * confidence * 0.3
                    )

        return base_comparison

    async def _detect_gaps_with_context(
        self,
        current_data: pd.DataFrame,
        comparison_data: Dict[str, Any],
        cognitive_context: Optional[GapCognitiveContext],
        state: GapAnalyzerState
    ) -> List[PerformanceGap]:
        """Detect gaps with awareness of historical patterns."""

        gaps = await self._detect_segment_gaps(
            current_data, comparison_data, state
        )

        # Enrich with historical pattern matching
        if cognitive_context and cognitive_context["historical_gaps"]:
            for gap in gaps:
                similar_historical = self._find_similar_historical_gap(
                    gap, cognitive_context["historical_gaps"]
                )
                if similar_historical:
                    gap["historical_pattern"] = similar_historical
                    gap["pattern_recurrence"] = True

        return gaps
```

### Training Signal for MIPROv2

```python
# src/agents/gap_analyzer/training_signals.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class GapAnalyzerTrainingSignal:
    """
    Training signal for CognitiveRAG's EvidenceSynthesisSignature.

    Emitted after gap analysis to inform MIPROv2 optimization
    about the quality of synthesized benchmark context.
    """

    # Identifiers
    batch_id: str
    analysis_timestamp: datetime

    # Gap Detection Quality
    gaps_identified: int
    gaps_with_cognitive_enrichment: int
    benchmark_context_used: bool

    # ROI Estimation Quality
    roi_estimates_count: int
    high_confidence_estimates: int  # confidence > 0.8

    # User Validation (collected post-analysis)
    user_action_rate: Optional[float] = None  # % of recommendations acted on
    gap_accuracy_feedback: Optional[float] = None  # user rating 1-5
    roi_precision_feedback: Optional[float] = None  # actual vs estimated ROI

    # Evidence Quality Metrics
    benchmark_retrieval_helpful: bool = False
    historical_gaps_relevant: bool = False

    def compute_reward(self) -> float:
        """Compute reward signal for MIPROv2 optimization."""

        base_reward = 0.5

        # Reward for using cognitive enrichment effectively
        if self.benchmark_context_used and self.gaps_with_cognitive_enrichment > 0:
            enrichment_ratio = self.gaps_with_cognitive_enrichment / max(self.gaps_identified, 1)
            base_reward += 0.2 * enrichment_ratio

        # Reward for high-confidence ROI estimates
        if self.roi_estimates_count > 0:
            confidence_ratio = self.high_confidence_estimates / self.roi_estimates_count
            base_reward += 0.1 * confidence_ratio

        # User feedback rewards (if available)
        if self.user_action_rate is not None:
            base_reward += 0.15 * self.user_action_rate

        if self.gap_accuracy_feedback is not None:
            base_reward += 0.05 * (self.gap_accuracy_feedback / 5.0)

        return min(base_reward, 1.0)

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to DSPy training example format."""
        return {
            "input": {
                "query_type": "gap_analysis",
                "metrics_analyzed": self.gaps_identified,
                "cognitive_enrichment": self.benchmark_context_used,
            },
            "output": {
                "gaps_found": self.gaps_identified,
                "enrichment_effectiveness": self.gaps_with_cognitive_enrichment / max(self.gaps_identified, 1),
            },
            "reward": self.compute_reward(),
            "metadata": {
                "batch_id": self.batch_id,
                "timestamp": self.analysis_timestamp.isoformat(),
            }
        }


async def emit_gap_analyzer_signal(
    result: GapAnalyzerOutput,
    state: GapAnalyzerState,
    signal_store: Any
) -> None:
    """Emit training signal after gap analysis completion."""

    cognitive_enrichment = state.get("cognitive_enrichment")
    gaps = result.prioritized_opportunities

    signal = GapAnalyzerTrainingSignal(
        batch_id=result.batch_id if hasattr(result, 'batch_id') else str(uuid.uuid4()),
        analysis_timestamp=datetime.now(),
        gaps_identified=len(gaps),
        gaps_with_cognitive_enrichment=len([
            g for g in gaps if g.get("historical_pattern")
        ]),
        benchmark_context_used=cognitive_enrichment is not None,
        roi_estimates_count=len(result.roi_estimates) if hasattr(result, 'roi_estimates') else 0,
        high_confidence_estimates=len([
            r for r in (result.roi_estimates or [])
            if r.get("confidence", 0) > 0.8
        ]),
        benchmark_retrieval_helpful=cognitive_enrichment is not None and len(gaps) > 0,
        historical_gaps_relevant=any(g.get("pattern_recurrence") for g in gaps),
    )

    await signal_store.store_signal(
        agent="gap_analyzer",
        signal=signal.to_training_example()
    )
```

### Memory Contribution

```python
# src/agents/gap_analyzer/memory_contribution.py

from typing import Any
from datetime import datetime

async def contribute_to_memory(
    result: GapAnalyzerOutput,
    state: GapAnalyzerState,
    memory_backend: Any
) -> None:
    """
    Contribute gap analysis results to CognitiveRAG's semantic memory.

    Stores in 'gap_opportunities' index for future retrieval.
    """

    for opportunity in result.prioritized_opportunities[:10]:  # Top 10
        gap_entry = {
            "gap_id": opportunity["gap"]["gap_id"],
            "metric": opportunity["gap"]["metric"],
            "segment": opportunity["gap"]["segment"],
            "segment_value": opportunity["gap"]["segment_value"],
            "gap_size": opportunity["gap"]["gap_size"],
            "gap_percentage": opportunity["gap"]["gap_percentage"],
            "expected_roi": opportunity["roi_estimate"]["expected_roi"],
            "recommended_action": opportunity["recommended_action"],
            "difficulty": opportunity["implementation_difficulty"],
            "brand": state.get("brand"),
            "time_period": state.get("time_period"),
            "analysis_date": datetime.now().isoformat(),
            "status": "identified"  # Track lifecycle: identified -> actioned -> resolved
        }

        await memory_backend.store(
            memory_type="SEMANTIC",
            content=gap_entry,
            metadata={
                "agent": "gap_analyzer",
                "index": "gap_opportunities",
                "embedding_fields": ["metric", "segment", "recommended_action"]
            }
        )
```

### Cognitive-Enriched Input TypedDict

```python
# src/agents/gap_analyzer/state.py (additions)

class GapAnalyzerCognitiveInput(TypedDict):
    """Extended input with cognitive enrichment from CognitiveRAG."""

    # Standard inputs
    query: str
    metrics: List[str]
    segments: List[str]
    brand: str
    time_period: str

    # Cognitive enrichment from Phase 2
    cognitive_enrichment: Optional[Dict[str, Any]]
    # Contains:
    #   - synthesized_summary: str
    #   - benchmark_context: str
    #   - historical_gaps: List[Dict]
    #   - confidence_score: float

    # Working memory context
    working_memory: Optional[Dict[str, Any]]
    # Contains:
    #   - current_focus: str
    #   - recent_queries: List[str]
    #   - active_analysis_chain: List[str]
```

### Configuration

```yaml
# config/agents/gap_analyzer.yaml

gap_analyzer:
  # ... existing config ...

  cognitive_rag_integration:
    enabled: true

    # Evidence consumption
    consume_benchmark_context: true
    consume_historical_gaps: true
    blend_cognitive_benchmarks: true
    cognitive_blend_weight: 0.3  # 30% cognitive, 70% stored

    # Training signal emission
    emit_training_signals: true
    signal_destination: "feedback_learner"

    # Memory contribution
    contribute_to_memory: true
    memory_index: "gap_opportunities"
    max_gaps_to_store: 10

    # Pattern matching
    match_historical_patterns: true
    pattern_similarity_threshold: 0.75
```

### Testing Requirements

```python
# tests/unit/test_agents/test_gap_analyzer/test_cognitive_integration.py

import pytest
from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
from src.agents.gap_analyzer.training_signals import GapAnalyzerTrainingSignal

class TestGapAnalyzerCognitiveIntegration:
    """Tests for Gap Analyzer Cognitive RAG integration."""

    @pytest.mark.asyncio
    async def test_cognitive_context_extraction(self):
        """Test extraction of cognitive context from state."""
        pass

    @pytest.mark.asyncio
    async def test_benchmark_parsing(self):
        """Test parsing benchmarks from cognitive evidence."""
        pass

    @pytest.mark.asyncio
    async def test_benchmark_blending(self):
        """Test blending cognitive benchmarks with stored data."""
        pass

    @pytest.mark.asyncio
    async def test_historical_gap_matching(self):
        """Test matching current gaps to historical patterns."""
        pass

    @pytest.mark.asyncio
    async def test_training_signal_emission(self):
        """Test training signal generation and reward computation."""
        signal = GapAnalyzerTrainingSignal(
            batch_id="test",
            analysis_timestamp=datetime.now(),
            gaps_identified=5,
            gaps_with_cognitive_enrichment=3,
            benchmark_context_used=True,
            roi_estimates_count=5,
            high_confidence_estimates=2,
        )

        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_memory_contribution(self):
        """Test contributing gaps to semantic memory."""
        pass

    @pytest.mark.asyncio
    async def test_without_cognitive_enrichment(self):
        """Test graceful fallback without cognitive context."""
        pass
```
