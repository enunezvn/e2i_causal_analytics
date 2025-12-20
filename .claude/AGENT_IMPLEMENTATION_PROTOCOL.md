# Agent Implementation Protocol - 5-Step Systematic Process

**Version**: 1.0
**Last Updated**: 2025-12-18
**Purpose**: Reusable methodology for creating production-ready agents with guaranteed contract compliance

---

## Overview

This protocol defines a systematic 5-step process for implementing production-ready agents in the E2I Causal Analytics system. Following this protocol ensures:

- ✅ 100% contract compliance
- ✅ Comprehensive test coverage (100+ tests per agent)
- ✅ Clear integration blocker documentation
- ✅ Performance validation against SLA targets
- ✅ Consistent code patterns across agents

**When to Use**: When implementing any new agent that has specialist documentation and contracts defined.

**Time Estimate**: 3-6 hours per agent (depending on complexity)

---

## Prerequisites

Before starting, ensure you have:

1. **Specialist Documentation**: Agent specification in `.claude/specialists/`
   - Architecture definition
   - Algorithm specifications
   - Node workflow design
   - Performance requirements

2. **Contract Definitions**: Input/output contracts in `.claude/contracts/`
   - Input model fields and validation rules
   - Output model fields and types
   - State TypedDict structure
   - Any additional TypedDict definitions

3. **Context**: Relevant context files from `.claude/context/`
   - KPI definitions (if applicable)
   - Brand context (if applicable)
   - System architecture context

---

## Step 1: Load Specialist Documentation

**Goal**: Understand agent architecture, algorithms, and requirements

### Actions:

1. **Read specialist file completely**
   ```bash
   # Location pattern:
   .claude/specialists/Agent_Specialists_Tiers 1-5/{agent-name}.md
   # OR
   .claude/specialists/Agent_Specialists_Tier 0/{agent-name}.md
   ```

2. **Extract key information**:
   - **Agent Classification**:
     - Tier (0-5)
     - Type (Standard/Advanced/Strategic)
     - Path (Fast/Reasoning)
     - LLM usage (yes/no)

   - **Performance Requirements**:
     - Latency SLA (e.g., <10s for 50 features)
     - Throughput requirements
     - Resource constraints

   - **Workflow Design**:
     - Node count and names
     - Node execution order (sequential/parallel)
     - Conditional routing (if any)

   - **Algorithms**:
     - Statistical tests (PSI, KS, Chi-square, etc.)
     - Calculation formulas
     - Threshold definitions
     - Severity determination logic

   - **Integration Points**:
     - Data connectors needed
     - External dependencies
     - Repository layer usage

3. **Document findings** in implementation notes

### Success Criteria:
- [ ] Complete understanding of agent architecture
- [ ] All algorithms documented with formulas
- [ ] Node workflow clearly mapped
- [ ] Performance targets identified
- [ ] Dependencies and integration points listed

### Example Output:
```markdown
## drift_monitor Agent Findings

**Classification**: Tier 3 (Monitoring), Standard (Fast Path), No LLM
**Performance**: <10s for 50 features
**Workflow**: 4 nodes sequential (data_drift → model_drift → concept_drift → alert_aggregator)

**Algorithms**:
1. PSI Calculation: sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
2. KS Test: scipy.stats.ks_2samp for continuous distributions
3. Chi-Square Test: scipy.stats.chi2_contingency for categorical distributions
4. Severity Determination:
   - PSI >= 0.25 OR p_value < significance/10 → critical
   - PSI >= threshold OR p_value < significance → high/medium
   - PSI >= 0.05 → low
   - PSI < 0.05 → none

**Integration Points**:
- MockDataConnector (temporary) for data access
- Orchestrator registration required
```

---

## Step 2: Review Contract Definitions

**Goal**: Understand exact input/output contracts and state structure

### Actions:

1. **Read contract file**:
   ```bash
   # Location pattern:
   .claude/contracts/tier{N}-contracts.md
   ```

2. **Extract contract specifications**:

   **A. Input Contract**:
   - Field names and types
   - Required vs optional fields
   - Default values
   - Validation rules (min/max, patterns, enums)
   - Field validators (e.g., time_window format)

   **B. Output Contract**:
   - Field names and types
   - Required fields
   - Nested structures
   - Value constraints (e.g., scores 0.0-1.0)

   **C. State Contract**:
   - All state fields (typically 20-30 fields)
   - Field categories:
     - Input fields
     - Configuration fields
     - Detection/processing outputs
     - Aggregated outputs
     - Summary fields
     - Metadata (timestamps, latency)
     - Error handling (errors, warnings, status)

   **D. Additional TypedDicts**:
   - Domain-specific result structures
   - Alert structures
   - Any other typed dictionaries

3. **Document all validation rules**:
   - Pydantic validators needed
   - Custom validation logic
   - Error messages

### Success Criteria:
- [ ] All input fields documented with types and validation
- [ ] All output fields documented with types
- [ ] Complete state structure mapped (all 20-30 fields)
- [ ] All additional TypedDicts identified
- [ ] Validation rules for Pydantic models documented

### Example Output:
```markdown
## drift_monitor Contracts

**Input (11 fields)**:
- query: str (required)
- features_to_monitor: list[str] (required, min_length=1)
- model_id: Optional[str] (default=None)
- time_window: str (default="7d", pattern=r'^\d+d$', range: 1-365 days)
- brand: Optional[str] (default=None)
- significance_level: float (default=0.05, range: 0.01-0.10)
- psi_threshold: float (default=0.1, range: 0.0-1.0)
- check_data_drift: bool (default=True)
- check_model_drift: bool (default=True)
- check_concept_drift: bool (default=True)

**Validators Required**:
- time_window: Must end with 'd', days 1-365
- features_to_monitor: Non-empty list
- significance_level: 0.01 <= value <= 0.10
- psi_threshold: 0.0 <= value <= 1.0

**Output (11 fields)**:
- data_drift_results: list[DriftResult]
- model_drift_results: list[DriftResult]
- concept_drift_results: list[DriftResult]
- overall_drift_score: float (0.0-1.0)
- features_with_drift: list[str]
- alerts: list[DriftAlert]
- drift_summary: str
- recommended_actions: list[str]
- detection_latency_ms: float
- features_checked: int
- baseline_timestamp: str
- current_timestamp: str
- warnings: list[str]

**State (23 fields)**: [categories documented]

**TypedDicts**:
1. DriftResult (8 fields)
2. DriftAlert (7 fields)
```

---

## Step 3: Create Agent Structure

**Goal**: Implement all agent files with 100% contract compliance

### File Creation Order:

#### 3.1 State Definitions (`state.py`)

Create TypedDict definitions for all contracts.

**Template**:
```python
"""State definitions for {agent_name} agent.

Implements contracts from .claude/contracts/tier{N}-contracts.md
"""
from typing import TypedDict, Literal, Optional

# Define enums as Literal types
DriftType = Literal["data", "model", "concept"]
DriftSeverity = Literal["none", "low", "medium", "high", "critical"]
AlertSeverity = Literal["info", "warning", "critical"]

# Define result TypedDicts
class DriftResult(TypedDict):
    """Individual drift detection result."""
    feature: str
    drift_type: DriftType
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: DriftSeverity
    baseline_period: str
    current_period: str

# Define alert TypedDicts
class DriftAlert(TypedDict):
    """Drift alert structure."""
    alert_id: str
    severity: AlertSeverity
    drift_type: DriftType
    affected_features: list[str]
    message: str
    recommended_action: str
    timestamp: str

# Define state TypedDict
class {AgentName}State(TypedDict):
    """Agent state - implements tier{N} contract."""

    # Input (from DriftMonitorInput)
    query: str
    features_to_monitor: list[str]
    model_id: Optional[str]
    # ... all input fields

    # Configuration
    time_window: str
    significance_level: float
    # ... configuration fields

    # Detection outputs (populate during execution)
    data_drift_results: list[DriftResult]
    model_drift_results: list[DriftResult]
    # ... detection outputs

    # Aggregated outputs (final results)
    overall_drift_score: float
    features_with_drift: list[str]
    # ... aggregated outputs

    # Summary
    drift_summary: str
    recommended_actions: list[str]

    # Metadata
    detection_latency_ms: float
    features_checked: int
    baseline_timestamp: str
    current_timestamp: str

    # Error handling
    errors: list[str]
    warnings: list[str]
    status: Literal["pending", "detecting", "completed", "failed"]
```

**Checklist**:
- [ ] All TypedDicts from contract defined
- [ ] Literal types used for enums
- [ ] State includes ALL contract fields (input + output + metadata)
- [ ] Docstrings reference contract file
- [ ] Type hints are accurate

---

#### 3.2 Node Implementations (`nodes/`)

Create one file per node in the workflow.

**Node File Template**:
```python
"""[Node Name] - [Brief description].

Implements [algorithm name] from specialist documentation.
"""
import time
from datetime import datetime
import numpy as np
from scipy import stats
from ..state import {AgentName}State, DriftResult

class {NodeName}Node:
    """[Node description]."""

    def __init__(self):
        """Initialize node."""
        self.node_name = "{node_name}"

    async def execute(self, state: {AgentName}State) -> {AgentName}State:
        """Execute [node name].

        Args:
            state: Current agent state

        Returns:
            Updated state with [output field] populated
        """
        start_time = time.time()

        # Check if previous node failed
        if state.get("status") == "failed":
            state["{output_field}"] = []
            return state

        # Check if this detection type is enabled
        if not state.get("check_{type}_drift", True):
            state["{output_field}"] = []
            state["warnings"] = state.get("warnings", []) + [
                f"{self.node_name} skipped (check_{type}_drift=False)"
            ]
            return state

        try:
            # Main processing logic
            results = await self._process(state)

            # Update state
            state["{output_field}"] = results
            state["status"] = "detecting"

        except Exception as e:
            state["errors"] = state.get("errors", []) + [
                f"{self.node_name} error: {str(e)}"
            ]
            state["status"] = "failed"
            state["{output_field}"] = []

        # Track latency
        elapsed_ms = (time.time() - start_time) * 1000
        state["detection_latency_ms"] = state.get("detection_latency_ms", 0.0) + elapsed_ms

        return state

    async def _process(self, state: {AgentName}State) -> list[DriftResult]:
        """Main processing logic.

        Returns:
            List of drift results
        """
        # Implementation here
        pass

    def _algorithm_implementation(self, data: np.ndarray, ...) -> float:
        """[Algorithm name] implementation.

        Formula: [mathematical formula]

        Args:
            data: Input data

        Returns:
            Calculated metric
        """
        # Algorithm implementation
        pass
```

**Key Patterns**:

1. **Error Handling**:
   - Wrap main logic in try/except
   - Update state["errors"] on failure
   - Set state["status"] = "failed"
   - Return empty results on error

2. **Status Checks**:
   - Check if previous node failed
   - Check if detection type is enabled
   - Add warnings for skipped operations

3. **Latency Tracking**:
   - Start timer at beginning
   - Add elapsed time to state["detection_latency_ms"]

4. **Algorithm Implementation**:
   - Separate helper methods for algorithms
   - Document formulas in docstrings
   - Handle edge cases (division by zero, empty data, etc.)

**Example Algorithm Implementation**:
```python
def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index.

    PSI Formula: sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change

    Args:
        expected: Baseline distribution
        actual: Current distribution
        bins: Number of bins for histogram

    Returns:
        PSI value (>= 0.0)
    """
    # Create bins from expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)

    # Calculate counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)
```

**Checklist per Node**:
- [ ] Async execute method implemented
- [ ] Error handling with try/except
- [ ] Status checks (failed, detection type enabled)
- [ ] Latency tracking
- [ ] Algorithm helpers with documented formulas
- [ ] Edge case handling (empty data, division by zero)
- [ ] Type hints for all methods
- [ ] Docstrings with formulas and interpretations

---

#### 3.3 Graph Assembly (`graph.py`)

Create LangGraph workflow.

**Template**:
```python
"""LangGraph workflow for {agent_name} agent."""
from langgraph.graph import StateGraph, END
from .state import {AgentName}State
from .nodes import (
    {node1}_node,
    {node2}_node,
    # ... all nodes
)

def create_{agent_name}_graph() -> StateGraph:
    """Create {agent_name} agent graph.

    Workflow: {node1} → {node2} → ... → END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph({AgentName}State)

    # Add nodes
    workflow.add_node("{node1}", {node1}_node.execute)
    workflow.add_node("{node2}", {node2}_node.execute)
    # ... add all nodes

    # Define workflow
    workflow.set_entry_point("{node1}")
    workflow.add_edge("{node1}", "{node2}")
    workflow.add_edge("{node2}", "{node3}")
    # ... connect all nodes
    workflow.add_edge("{last_node}", END)

    return workflow.compile()
```

**Workflow Patterns**:

1. **Sequential**:
   ```python
   workflow.add_edge("node1", "node2")
   workflow.add_edge("node2", "node3")
   ```

2. **Parallel (fan-out/fan-in)**:
   ```python
   # Fan-out
   workflow.add_edge("dispatcher", "worker1")
   workflow.add_edge("dispatcher", "worker2")

   # Fan-in
   workflow.add_edge("worker1", "aggregator")
   workflow.add_edge("worker2", "aggregator")
   ```

3. **Conditional**:
   ```python
   def route_condition(state):
       if state["condition"]:
           return "path_a"
       return "path_b"

   workflow.add_conditional_edges(
       "decision_node",
       route_condition,
       {"path_a": "node_a", "path_b": "node_b"}
   )
   ```

**Checklist**:
- [ ] All nodes added to workflow
- [ ] Entry point set
- [ ] All edges defined (sequential/parallel/conditional)
- [ ] END node connected
- [ ] Workflow compiled and returned

---

#### 3.4 Main Agent Class (`agent.py`)

Create Pydantic input/output models and main agent class.

**Template**:
```python
"""{AgentName} Agent - [Brief description].

Tier {N}: {Category} - {Type} ({Path})
Performance: {SLA requirement}
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from .graph import create_{agent_name}_graph
from .state import {AgentName}State, DriftResult, DriftAlert

class {AgentName}Input(BaseModel):
    """Input model with Pydantic validation.

    Implements input contract from tier{N}-contracts.md
    """
    # Required fields
    query: str
    features_to_monitor: list[str] = Field(..., min_length=1, description="...")

    # Optional fields with defaults
    model_id: Optional[str] = Field(None, description="...")
    time_window: str = Field("7d", description="...")
    brand: Optional[str] = Field(None, description="...")
    significance_level: float = Field(0.05, ge=0.01, le=0.10, description="...")
    psi_threshold: float = Field(0.1, ge=0.0, le=1.0, description="...")
    check_data_drift: bool = Field(True, description="...")
    check_model_drift: bool = Field(True, description="...")
    check_concept_drift: bool = Field(True, description="...")

    @field_validator("time_window")
    @classmethod
    def validate_time_window(cls, v: str) -> str:
        """Validate time window format (e.g., '7d', '30d').

        Args:
            v: Time window string

        Returns:
            Validated time window

        Raises:
            ValueError: If format invalid or out of range
        """
        if not v.endswith("d"):
            raise ValueError("time_window must end with 'd' (e.g., '7d', '30d')")

        try:
            days = int(v[:-1])
        except ValueError:
            raise ValueError("time_window must be a number followed by 'd'")

        if days < 1 or days > 365:
            raise ValueError("time_window must be between 1d and 365d")

        return v

class {AgentName}Output(BaseModel):
    """Output model matching tier{N} contract."""
    # Detection results
    data_drift_results: list[DriftResult]
    model_drift_results: list[DriftResult]
    concept_drift_results: list[DriftResult]

    # Aggregated outputs
    overall_drift_score: float = Field(..., ge=0.0, le=1.0)
    features_with_drift: list[str]
    alerts: list[DriftAlert]

    # Summary
    drift_summary: str
    recommended_actions: list[str]

    # Metadata
    detection_latency_ms: float
    features_checked: int
    baseline_timestamp: str
    current_timestamp: str
    warnings: list[str]

class {AgentName}Agent:
    """Main {agent_name} agent class.

    Tier {N}: {Category} - {Type} ({Path})
    Performance: {SLA requirement}

    Workflow:
    1. {node1}: {description}
    2. {node2}: {description}
    ...
    """

    def __init__(self):
        """Initialize agent."""
        self.graph = create_{agent_name}_graph()

    def run(self, input_data: {AgentName}Input) -> {AgentName}Output:
        """Execute agent workflow.

        Args:
            input_data: Validated input

        Returns:
            Agent output with detection results

        Raises:
            ValueError: If input validation fails
            RuntimeError: If workflow execution fails
        """
        # Create initial state from input
        initial_state = self._create_initial_state(input_data)

        # Execute workflow
        final_state = self.graph.invoke(initial_state)

        # Create output from final state
        output = self._create_output(final_state)

        return output

    def _create_initial_state(self, input_data: {AgentName}Input) -> {AgentName}State:
        """Create initial state from input.

        Args:
            input_data: Validated input

        Returns:
            Initial state with input fields populated
        """
        state: {AgentName}State = {
            # Input fields
            "query": input_data.query,
            "features_to_monitor": input_data.features_to_monitor,
            "model_id": input_data.model_id,
            "time_window": input_data.time_window,
            "brand": input_data.brand,
            "significance_level": input_data.significance_level,
            "psi_threshold": input_data.psi_threshold,
            "check_data_drift": input_data.check_data_drift,
            "check_model_drift": input_data.check_model_drift,
            "check_concept_drift": input_data.check_concept_drift,

            # Initialize empty outputs
            "data_drift_results": [],
            "model_drift_results": [],
            "concept_drift_results": [],
            "overall_drift_score": 0.0,
            "features_with_drift": [],
            "alerts": [],
            "drift_summary": "",
            "recommended_actions": [],

            # Initialize metadata
            "detection_latency_ms": 0.0,
            "features_checked": 0,
            "baseline_timestamp": "",
            "current_timestamp": "",

            # Initialize error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        return state

    def _create_output(self, final_state: {AgentName}State) -> {AgentName}Output:
        """Create output from final state.

        Args:
            final_state: Final workflow state

        Returns:
            Validated output

        Raises:
            ValueError: If state is missing required fields
        """
        # Extract all output fields from state
        output = {AgentName}Output(
            data_drift_results=final_state["data_drift_results"],
            model_drift_results=final_state["model_drift_results"],
            concept_drift_results=final_state["concept_drift_results"],
            overall_drift_score=final_state["overall_drift_score"],
            features_with_drift=final_state["features_with_drift"],
            alerts=final_state["alerts"],
            drift_summary=final_state["drift_summary"],
            recommended_actions=final_state["recommended_actions"],
            detection_latency_ms=final_state["detection_latency_ms"],
            features_checked=final_state["features_checked"],
            baseline_timestamp=final_state["baseline_timestamp"],
            current_timestamp=final_state["current_timestamp"],
            warnings=final_state["warnings"],
        )
        return output
```

**Checklist**:
- [ ] Input model with ALL contract fields
- [ ] Field validators for complex validation (time_window, etc.)
- [ ] Output model with ALL contract fields
- [ ] Output field constraints (ge, le for numeric fields)
- [ ] Main agent class with graph initialization
- [ ] run() method orchestrating workflow
- [ ] _create_initial_state() populating all state fields
- [ ] _create_output() extracting all output fields
- [ ] Comprehensive docstrings

---

#### 3.5 Package Exports (`__init__.py`)

Create clean package API.

**Template**:
```python
"""
{AgentName} Agent

Tier {N}: {Category} - {Type} ({Path})
Performance: {SLA requirement}

Usage:
    from src.agents.{agent_name} import {AgentName}Agent, {AgentName}Input, {AgentName}Output

    agent = {AgentName}Agent()
    result = agent.run({AgentName}Input(
        query="...",
        features_to_monitor=["f1", "f2"]
    ))
"""
from .agent import {AgentName}Agent, {AgentName}Input, {AgentName}Output
from .state import {AgentName}State, DriftResult, DriftAlert

__all__ = [
    "{AgentName}Agent",
    "{AgentName}Input",
    "{AgentName}Output",
    "{AgentName}State",
    "DriftResult",
    "DriftAlert",
]
```

**Also create** `nodes/__init__.py`:
```python
"""Node implementations for {agent_name} agent."""
from .{node1} import {Node1}Node
from .{node2} import {Node2}Node
# ... all nodes

# Create singleton instances
{node1}_node = {Node1}Node()
{node2}_node = {Node2}Node()
# ... all nodes

__all__ = [
    "{node1}_node",
    "{node2}_node",
    # ... all nodes
]
```

**Checklist**:
- [ ] Main __init__.py exports agent, input, output, state
- [ ] nodes/__init__.py exports all node instances
- [ ] Usage example in docstring
- [ ] __all__ defined for explicit exports

---

### Integration Points Documentation

**Create temporary mocks for missing integrations**:

```python
# Example: MockDataConnector
class MockDataConnector:
    """Temporary mock for data access.

    INTEGRATION BLOCKER: Replace with SupabaseDataConnector when available.
    Estimated effort: 1-2 hours

    Required interface:
    - fetch_feature_data(features, time_window, brand) -> DataFrame
    - fetch_predictions(model_id, time_window, brand) -> DataFrame
    """

    async def fetch_feature_data(self, features, time_window, brand=None):
        """Mock fetch - returns random data."""
        # Generate mock data
        return mock_data
```

**Document each mock as integration blocker** with:
- What needs to be replaced
- Estimated effort
- Required interface
- Priority (CRITICAL, HIGH, MEDIUM, LOW)

---

## Step 4: Implement Comprehensive Tests

**Goal**: 100+ tests covering all algorithms, edge cases, and contract requirements

### Test File Structure:

Create 5 test files per agent:

1. `test_{node1}.py` - Tests for first node
2. `test_{node2}.py` - Tests for second node
3. `test_{node3}.py` - Tests for third node (if applicable)
4. `test_{aggregator_node}.py` - Tests for aggregation node
5. `test_{agent_name}_agent.py` - Integration tests

### Test Categories:

#### 4.1 Algorithm Tests

**Test each algorithm implementation**:

```python
"""Tests for {Algorithm Name}."""
import pytest
import numpy as np
from src.agents.{agent_name}.nodes.{node} import {NodeName}Node

class Test{AlgorithmName}:
    """Test {algorithm name} implementation."""

    def test_{algorithm}_identical_distributions(self):
        """Test algorithm with identical distributions."""
        node = {NodeName}Node()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        result = node._{algorithm}(baseline, current)

        # Should detect minimal or no drift
        assert result < 0.1  # Or appropriate threshold

    def test_{algorithm}_shifted_distributions(self):
        """Test algorithm with shifted distributions."""
        node = {NodeName}Node()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Shifted mean

        result = node._{algorithm}(baseline, current)

        # Should detect significant drift
        assert result > 0.1  # Or appropriate threshold

    def test_{algorithm}_insufficient_data(self):
        """Test algorithm with insufficient data."""
        node = {NodeName}Node()

        baseline = np.array([0.1, 0.2])  # Only 2 samples
        current = np.array([0.3, 0.4])

        result = node._{algorithm}(baseline, current)

        # Should handle gracefully (return None or default)
        assert result is None or result == 0.0
```

**Checklist per Algorithm**:
- [ ] Test with identical distributions (no drift)
- [ ] Test with shifted distributions (drift detected)
- [ ] Test with insufficient data
- [ ] Test edge cases (empty arrays, single value, all zeros)
- [ ] Test numerical stability (very large/small values)

---

#### 4.2 Severity Determination Tests

**Test severity classification logic**:

```python
class TestSeverityDetermination:
    """Test severity determination logic."""

    def test_critical_severity(self):
        """Test critical severity threshold."""
        node = {NodeName}Node()

        psi = 0.3  # Above critical threshold
        p_value = 0.001
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_high_severity(self):
        """Test high severity threshold."""
        # Similar pattern for high
        pass

    def test_medium_severity(self):
        """Test medium severity threshold."""
        # Similar pattern for medium
        pass

    def test_low_severity(self):
        """Test low severity threshold."""
        # Similar pattern for low
        pass

    def test_no_drift(self):
        """Test no drift detected."""
        # Similar pattern for none
        pass
```

**Checklist**:
- [ ] Test all severity levels (critical, high, medium, low, none)
- [ ] Test boundary conditions
- [ ] Test combined PSI and p-value logic

---

#### 4.3 Node Execution Tests

**Test each node's execute method**:

```python
class Test{NodeName}Node:
    """Test {NodeName}Node execution."""

    def _create_test_state(self, **overrides):
        """Helper to create test state."""
        state = {
            "query": "test",
            "features_to_monitor": ["f1", "f2"],
            # ... all required state fields
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic node execution."""
        node = {NodeName}Node()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "{output_field}" in result
        assert isinstance(result["{output_field}"], list)
        assert result["status"] in ["detecting", "completed"]

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency is tracked."""
        node = {NodeName}Node()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "detection_latency_ms" in result
        assert result["detection_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = {NodeName}Node()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["{output_field}"] == []

    @pytest.mark.asyncio
    async def test_detection_type_disabled(self):
        """Test detection type can be disabled."""
        node = {NodeName}Node()
        state = self._create_test_state(check_{type}_drift=False)

        result = await node.execute(state)

        assert result["{output_field}"] == []
        assert any("skipped" in w.lower() for w in result["warnings"])
```

**Checklist per Node**:
- [ ] Test basic execution
- [ ] Test latency measurement
- [ ] Test failed status passthrough
- [ ] Test detection type disabled
- [ ] Test error handling
- [ ] Test warnings generation

---

#### 4.4 Input Validation Tests

**Test Pydantic input validation**:

```python
class Test{AgentName}Input:
    """Test input validation."""

    def test_valid_input_minimal(self):
        """Test valid minimal input."""
        input_data = {AgentName}Input(
            query="test",
            features_to_monitor=["f1"]
        )

        assert input_data.query == "test"
        assert input_data.features_to_monitor == ["f1"]
        assert input_data.time_window == "7d"  # Default
        assert input_data.significance_level == 0.05  # Default

    def test_valid_input_full(self):
        """Test valid input with all fields."""
        input_data = {AgentName}Input(
            query="test",
            features_to_monitor=["f1", "f2"],
            model_id="model_v1",
            time_window="14d",
            brand="Remibrutinib",
            significance_level=0.01,
            psi_threshold=0.15,
            check_data_drift=True,
            check_model_drift=False,
            check_concept_drift=True
        )

        assert input_data.model_id == "model_v1"
        assert input_data.time_window == "14d"
        # ... assert all fields

    def test_invalid_empty_features(self):
        """Test invalid empty features list."""
        with pytest.raises(ValidationError):
            {AgentName}Input(
                query="test",
                features_to_monitor=[]  # Empty list
            )

    def test_invalid_time_window_format(self):
        """Test invalid time window format."""
        with pytest.raises(ValidationError) as exc_info:
            {AgentName}Input(
                query="test",
                features_to_monitor=["f1"],
                time_window="7"  # Missing 'd'
            )
        assert "must end with 'd'" in str(exc_info.value)

    def test_invalid_time_window_range(self):
        """Test time window out of range."""
        with pytest.raises(ValidationError):
            {AgentName}Input(
                query="test",
                features_to_monitor=["f1"],
                time_window="500d"  # > 365
            )

    def test_invalid_significance_level(self):
        """Test significance level out of range."""
        with pytest.raises(ValidationError):
            {AgentName}Input(
                query="test",
                features_to_monitor=["f1"],
                significance_level=0.005  # < 0.01
            )
```

**Checklist**:
- [ ] Test valid minimal input (required fields only)
- [ ] Test valid full input (all fields)
- [ ] Test each field validator
- [ ] Test boundary conditions
- [ ] Test error messages are clear

---

#### 4.5 Output Structure Tests

**Test output contract compliance**:

```python
class Test{AgentName}Output:
    """Test output structure."""

    def test_output_structure(self):
        """Test output has all required fields."""
        agent = {AgentName}Agent()
        input_data = {AgentName}Input(
            query="test",
            features_to_monitor=["f1"]
        )

        result = agent.run(input_data)

        # Check all required fields exist
        assert hasattr(result, "data_drift_results")
        assert hasattr(result, "model_drift_results")
        assert hasattr(result, "concept_drift_results")
        assert hasattr(result, "overall_drift_score")
        assert hasattr(result, "features_with_drift")
        assert hasattr(result, "alerts")
        assert hasattr(result, "drift_summary")
        assert hasattr(result, "recommended_actions")
        assert hasattr(result, "detection_latency_ms")
        assert hasattr(result, "features_checked")
        assert hasattr(result, "baseline_timestamp")
        assert hasattr(result, "current_timestamp")
        assert hasattr(result, "warnings")

    def test_output_value_ranges(self):
        """Test output values are in valid ranges."""
        agent = {AgentName}Agent()
        input_data = {AgentName}Input(
            query="test",
            features_to_monitor=["f1"]
        )

        result = agent.run(input_data)

        # Test numeric ranges
        assert 0.0 <= result.overall_drift_score <= 1.0
        assert result.detection_latency_ms >= 0
        assert result.features_checked >= 0

    def test_alert_structure(self):
        """Test alert structure when generated."""
        # Implementation
        pass
```

**Checklist**:
- [ ] Test all output fields exist
- [ ] Test numeric value ranges
- [ ] Test list types are correct
- [ ] Test nested structures (alerts, results)

---

#### 4.6 Integration/End-to-End Tests

**Test complete workflows**:

```python
class TestEndToEndWorkflows:
    """Test complete agent workflows."""

    def test_data_drift_only(self):
        """Test with only data drift enabled."""
        agent = {AgentName}Agent()
        input_data = {AgentName}Input(
            query="Check data drift only",
            features_to_monitor=["f1", "f2"],
            check_data_drift=True,
            check_model_drift=False,
            check_concept_drift=False
        )

        result = agent.run(input_data)

        assert len(result.data_drift_results) >= 0
        assert len(result.model_drift_results) == 0
        assert len(result.concept_drift_results) == 0

    def test_all_drift_types(self):
        """Test with all drift types enabled."""
        # Implementation
        pass

    def test_multiple_brands(self):
        """Test with different brands."""
        agent = {AgentName}Agent()

        for brand in ["Remibrutinib", "Fabhalta", "Kisqali"]:
            input_data = {AgentName}Input(
                query=f"Check drift for {brand}",
                features_to_monitor=["f1"],
                brand=brand
            )

            result = agent.run(input_data)

            assert isinstance(result, {AgentName}Output)

    def test_different_time_windows(self):
        """Test with different time windows."""
        # Implementation
        pass

    def test_latency_under_target(self):
        """Test latency is under SLA target."""
        agent = {AgentName}Agent()
        features = [f"feature_{i}" for i in range(50)]

        input_data = {AgentName}Input(
            query="Latency test",
            features_to_monitor=features
        )

        result = agent.run(input_data)

        # Check against SLA (e.g., <10s for 50 features)
        assert result.detection_latency_ms < 10_000
```

**Checklist**:
- [ ] Test each drift type individually
- [ ] Test all drift types together
- [ ] Test with different brands (if applicable)
- [ ] Test with different time windows
- [ ] Test latency against SLA
- [ ] Test with edge case inputs (many features, extreme parameters)

---

### Test Coverage Goals:

**Minimum test counts per agent**:
- Algorithm tests: 20+ tests
- Severity tests: 10+ tests
- Node execution tests: 20+ tests (4+ per node)
- Input validation tests: 15+ tests
- Output tests: 10+ tests
- Integration tests: 10+ tests

**Total: 100+ tests**

---

## Step 5: Validate Contract Compliance

**Goal**: Document 100% contract compliance and integration blockers

### Create CONTRACT_VALIDATION.md:

**Template**:
```markdown
# {AgentName} Agent - Contract Validation Report

**Agent**: {AgentName}
**Tier**: {N}
**Type**: {Standard/Advanced/Strategic} ({Fast/Reasoning} Path)
**Date**: {YYYY-MM-DD}
**Status**: ✅ VALIDATED - 100% Contract Compliance

---

## Executive Summary

This report validates the {agent_name} agent implementation against the tier{N} contracts
defined in `.claude/contracts/tier{N}-contracts.md`.

**Validation Results**:
- ✅ Input Contract: 100% compliant ({X} fields)
- ✅ Output Contract: 100% compliant ({Y} fields)
- ✅ State Contract: 100% compliant ({Z} fields)
- ✅ TypedDict Contracts: 100% compliant ({N} structures)
- ✅ Test Coverage: {X}+ tests implemented

**Integration Blockers**: {N} documented (see Section 6)

---

## 1. Input Contract Validation

**Contract Location**: `.claude/contracts/tier{N}-contracts.md` (lines {X}-{Y})

### Required Fields (✅ {N}/{N})

| Field | Type | Contract | Implementation | Status |
|-------|------|----------|----------------|--------|
| query | str | Required | agent.py:L{X} | ✅ |
| features_to_monitor | list[str] | Required, min_length=1 | agent.py:L{X} | ✅ |

### Optional Fields (✅ {N}/{N})

| Field | Type | Default | Contract | Implementation | Status |
|-------|------|---------|----------|----------------|--------|
| model_id | Optional[str] | None | Optional | agent.py:L{X} | ✅ |
| time_window | str | "7d" | Pattern: ^\d+d$, range: 1-365 | agent.py:L{X} | ✅ |
| ... | ... | ... | ... | ... | ✅ |

### Field Validators (✅ {N}/{N})

| Validator | Contract Requirement | Implementation | Status |
|-----------|---------------------|----------------|--------|
| time_window | Must end with 'd', 1-365 days | agent.py:L{X}-{Y} | ✅ |
| features_to_monitor | Non-empty list | Pydantic min_length=1 | ✅ |
| significance_level | 0.01 <= value <= 0.10 | Pydantic ge=0.01, le=0.10 | ✅ |
| ... | ... | ... | ✅ |

**VERDICT**: ✅ 100% Input Contract Compliance

---

## 2. Output Contract Validation

**Contract Location**: `.claude/contracts/tier{N}-contracts.md` (lines {X}-{Y})

### Output Fields (✅ {N}/{N})

| Field | Type | Contract | Implementation | Status |
|-------|------|----------|----------------|--------|
| data_drift_results | list[DriftResult] | Required | agent.py:L{X} | ✅ |
| overall_drift_score | float | Required, 0.0-1.0 | agent.py:L{X} | ✅ |
| ... | ... | ... | ... | ✅ |

### Value Constraints (✅ {N}/{N})

| Field | Constraint | Implementation | Status |
|-------|-----------|----------------|--------|
| overall_drift_score | 0.0 <= value <= 1.0 | Pydantic ge=0.0, le=1.0 | ✅ |
| detection_latency_ms | >= 0.0 | Calculated value | ✅ |
| ... | ... | ... | ✅ |

**VERDICT**: ✅ 100% Output Contract Compliance

---

## 3. State Contract Validation

**Contract Location**: `.claude/contracts/tier{N}-contracts.md` (lines {X}-{Y})

### State Fields by Category (✅ {N}/{N})

#### Input Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| query | str | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Configuration Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| time_window | str | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Detection Output Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| data_drift_results | list[DriftResult] | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Aggregated Output Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| overall_drift_score | float | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Summary Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| drift_summary | str | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Metadata Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| detection_latency_ms | float | state.py:L{X} | ✅ |
| ... | ... | ... | ✅ |

#### Error Handling Fields (✅ {X}/{X})
| Field | Type | State Implementation | Status |
|-------|------|---------------------|--------|
| errors | list[str] | state.py:L{X} | ✅ |
| warnings | list[str] | state.py:L{X} | ✅ |
| status | Literal[...] | state.py:L{X} | ✅ |

**VERDICT**: ✅ 100% State Contract Compliance

---

## 4. TypedDict Contract Validation

### DriftResult (✅ {N}/{N} fields)

| Field | Type | Contract | Implementation | Status |
|-------|------|----------|----------------|--------|
| feature | str | Required | state.py:L{X} | ✅ |
| drift_type | DriftType | Required | state.py:L{X} | ✅ |
| ... | ... | ... | ... | ✅ |

### DriftAlert (✅ {N}/{N} fields)

| Field | Type | Contract | Implementation | Status |
|-------|------|----------|----------------|--------|
| alert_id | str | Required | state.py:L{X} | ✅ |
| severity | AlertSeverity | Required | state.py:L{X} | ✅ |
| ... | ... | ... | ... | ✅ |

**VERDICT**: ✅ 100% TypedDict Contract Compliance

---

## 5. Algorithm Documentation

### 5.1 Population Stability Index (PSI)

**Implementation**: `nodes/data_drift.py:L{X}-{Y}`

**Formula**:
```
PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
```

**Interpretation**:
- PSI < 0.1: No significant change
- 0.1 <= PSI < 0.2: Moderate change
- PSI >= 0.2: Significant change

**Edge Cases Handled**:
- Division by zero: Clipping percentages to 0.0001 minimum
- Empty bins: Handled by histogram binning
- Insufficient data: Returns None if < 30 samples

**Tests**: `test_data_drift.py:L{X}-{Y}` ({N} tests)

---

### 5.2 Kolmogorov-Smirnov Test

**Implementation**: `nodes/data_drift.py:L{X}-{Y}`

**Method**: scipy.stats.ks_2samp (two-sample KS test)

**Null Hypothesis**: Two samples drawn from same distribution

**Interpretation**:
- p_value < significance → Reject null (drift detected)
- p_value >= significance → Fail to reject (no drift)

**Tests**: `test_data_drift.py:L{X}-{Y}` ({N} tests)

---

### 5.3 Severity Determination

**Implementation**: `nodes/data_drift.py:L{X}-{Y}`

**Logic**:
```python
if psi >= 0.25 or p_value < significance / 10:
    return "critical", True
elif psi >= psi_threshold or p_value < significance:
    severity = "high" if psi >= 0.2 else "medium"
    return severity, True
elif psi >= 0.05:
    return "low", True
else:
    return "none", False
```

**Thresholds**:
- Critical: PSI >= 0.25 OR p_value < significance/10
- High: PSI >= 0.2 AND (PSI >= threshold OR p_value < significance)
- Medium: PSI >= threshold AND (PSI < 0.2 OR p_value < significance)
- Low: 0.05 <= PSI < threshold
- None: PSI < 0.05

**Tests**: `test_data_drift.py:L{X}-{Y}` ({N} tests)

---

### 5.4 Composite Drift Score

**Implementation**: `nodes/alert_aggregator.py:L{X}-{Y}`

**Formula**:
```python
SEVERITY_WEIGHTS = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0
}

drift_score = sum(SEVERITY_WEIGHTS[r["severity"]] for r in results) / len(results)
```

**Interpretation**:
- 0.0 = No drift detected
- 0.0-0.25 = Low overall drift
- 0.25-0.5 = Medium overall drift
- 0.5-0.75 = High overall drift
- 0.75-1.0 = Critical overall drift

**Tests**: `test_alert_aggregator.py:L{X}-{Y}` ({N} tests)

---

## 6. Integration Blockers

### BLOCKER 1: MockDataConnector (CRITICAL)

**Status**: 🔴 BLOCKING
**Priority**: CRITICAL
**Estimated Effort**: 1-2 hours

**Current State**:
- Using `MockDataConnector` in `nodes/data_drift.py:L{X}-{Y}` and `nodes/model_drift.py:L{X}-{Y}`
- Generates random data for testing
- NOT production-ready

**Required Changes**:
1. Implement `SupabaseDataConnector` in `src/repositories/`
2. Add methods:
   - `fetch_feature_data(features: list[str], time_window: str, brand: Optional[str]) -> pd.DataFrame`
   - `fetch_predictions(model_id: str, time_window: str, brand: Optional[str]) -> pd.DataFrame`
3. Replace MockDataConnector imports in nodes
4. Update integration tests to use real connector

**Interface Contract**:
```python
class DataConnector:
    async def fetch_feature_data(
        self,
        features: list[str],
        time_window: str,
        brand: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch feature data from database.

        Returns DataFrame with columns:
        - timestamp: datetime
        - feature_{name}: float for each feature
        """
        pass
```

**Acceptance Criteria**:
- [ ] SupabaseDataConnector implemented
- [ ] Integration tests pass with real connector
- [ ] MockDataConnector removed from production code

---

### BLOCKER 2: Orchestrator Registration (HIGH)

**Status**: 🟡 NON-BLOCKING (manual testing possible)
**Priority**: HIGH
**Estimated Effort**: 2-3 hours

**Current State**:
- Agent not registered with orchestrator for query routing
- Can be tested manually via direct instantiation

**Required Changes**:
1. Add drift_monitor to `config/agents.yaml`
2. Register with orchestrator agent
3. Define query routing patterns:
   - "drift" keywords
   - "stability" keywords
   - "distribution shift" keywords
4. Update orchestrator integration tests

**Acceptance Criteria**:
- [ ] Agent added to agents.yaml
- [ ] Orchestrator routes drift queries correctly
- [ ] Integration tests demonstrate end-to-end query flow

---

### BLOCKER 3: Concept Drift Detection (LOW)

**Status**: 🟢 NON-BLOCKING (placeholder implemented)
**Priority**: LOW
**Estimated Effort**: 8-12 hours

**Current State**:
- Placeholder implementation in `nodes/concept_drift.py`
- Returns empty results with warning
- Requires ground truth labels and feature importance tracking

**Required Changes**:
1. Implement label collection mechanism
2. Add feature importance tracking
3. Implement drift detection algorithms:
   - ADWIN (Adaptive Windowing)
   - DDM (Drift Detection Method)
   - Page-Hinkley test
4. Update tests to cover real implementation

**Acceptance Criteria**:
- [ ] Ground truth labels available in database
- [ ] Feature importance tracked
- [ ] Concept drift algorithms implemented
- [ ] Tests validate drift detection accuracy

---

## 7. Test Coverage Summary

**Total Tests**: {100+} tests

### By Category:
- Algorithm Tests: {20+} tests
  - PSI calculation: {8} tests
  - KS test: {6} tests
  - Chi-square test: {4} tests
  - Severity determination: {5} tests
  - Drift score: {5} tests

- Node Execution Tests: {20+} tests
  - data_drift node: {6} tests
  - model_drift node: {6} tests
  - concept_drift node: {4} tests
  - alert_aggregator node: {6} tests

- Input Validation Tests: {15+} tests
  - Required fields: {3} tests
  - Optional fields: {3} tests
  - Field validators: {6} tests
  - Edge cases: {5} tests

- Output Tests: {10+} tests
  - Structure validation: {4} tests
  - Value ranges: {3} tests
  - Nested structures: {3} tests

- Integration Tests: {10+} tests
  - Drift type workflows: {4} tests
  - Brand filtering: {3} tests
  - Time windows: {3} tests
  - Performance: {2} tests

### Test Files:
1. `test_data_drift.py`: {40+} tests
2. `test_model_drift.py`: {30+} tests
3. `test_concept_drift.py`: {7} tests
4. `test_alert_aggregator.py`: {40+} tests
5. `test_drift_monitor_agent.py`: {30+} tests

**Test Coverage**: Estimated {85-95}% (all critical paths covered)

---

## 8. Performance Validation

### SLA Compliance

**SLA Target**: <10s for 50 features

**Test Results**:
- Test: `test_drift_monitor_agent.py::test_latency_under_target`
- Features: 50
- Measured Latency: {X}ms
- Status: ✅ PASS (under 10,000ms threshold)

### Optimization Opportunities

1. **Parallel Feature Processing** (FUTURE):
   - Current: Sequential processing in data_drift node
   - Proposed: asyncio.gather for parallel feature drift detection
   - Expected improvement: 40-60% latency reduction

2. **Caching** (FUTURE):
   - Cache baseline statistics for frequently monitored features
   - Expected improvement: 20-30% latency reduction for repeated queries

---

## 9. Deployment Readiness

### Pre-Deployment Checklist

#### Code Quality (✅ {8}/{8})
- [x] All contracts implemented
- [x] 100+ tests passing
- [x] Type hints throughout
- [x] Docstrings for all public methods
- [x] Error handling in place
- [x] Logging implemented
- [x] Performance SLA validated
- [x] Integration blockers documented

#### Integration (🟡 {1}/{3})
- [ ] MockDataConnector replaced (**BLOCKER 1**)
- [ ] Orchestrator registration complete (**BLOCKER 2**)
- [x] Tests passing

#### Documentation (✅ {4}/{4})
- [x] Specialist documentation reviewed
- [x] Contract validation complete
- [x] Integration blockers documented
- [x] Algorithm documentation complete

### Deployment Status

**Overall Status**: 🟡 READY FOR STAGING (with blockers documented)

**Recommended Next Steps**:
1. Resolve BLOCKER 1 (MockDataConnector) - CRITICAL
2. Resolve BLOCKER 2 (Orchestrator registration) - HIGH
3. Deploy to staging environment
4. Run integration tests with real data
5. Monitor performance in staging
6. Address BLOCKER 3 (Concept drift) - LOW (can be deferred)

---

## 10. Summary

✅ **Contract Compliance**: 100% across all components
✅ **Test Coverage**: 100+ tests, {85-95}% coverage
✅ **Performance**: Meets <10s SLA for 50 features
🟡 **Integration**: 3 blockers documented (1 CRITICAL, 1 HIGH, 1 LOW)

The {agent_name} agent implementation is production-ready with documented integration blockers.

**Critical Path to Production**:
1. Implement SupabaseDataConnector (1-2 hours)
2. Register with orchestrator (2-3 hours)
3. Staging validation (2-4 hours)
4. Production deployment

**Total Effort to Production**: 5-9 hours
```

### Validation Checklist:

- [ ] CONTRACT_VALIDATION.md created
- [ ] All input fields validated (100%)
- [ ] All output fields validated (100%)
- [ ] All state fields validated (100%)
- [ ] All TypedDicts validated (100%)
- [ ] All algorithms documented with formulas
- [ ] All integration blockers documented
- [ ] Test coverage summary included
- [ ] Performance validation included
- [ ] Deployment readiness checklist included

---

## Success Criteria - Complete Protocol

### Per-Step Success Criteria:

**Step 1: Load Specialist Docs**
- [ ] Specialist file read completely
- [ ] Agent classification documented
- [ ] Performance requirements identified
- [ ] Workflow design mapped
- [ ] All algorithms documented with formulas
- [ ] Dependencies and integration points listed

**Step 2: Review Contracts**
- [ ] All input fields documented (types, validation, defaults)
- [ ] All output fields documented (types, constraints)
- [ ] Complete state structure mapped (20-30 fields)
- [ ] All additional TypedDicts identified
- [ ] Pydantic validators documented

**Step 3: Create Agent Structure**
- [ ] state.py created with all TypedDicts
- [ ] All node files created (one per workflow node)
- [ ] graph.py created with correct workflow
- [ ] agent.py created with Pydantic models
- [ ] __init__.py files created for clean exports
- [ ] Integration mocks documented as blockers

**Step 4: Implement Tests**
- [ ] 100+ total tests implemented
- [ ] Algorithm tests (20+)
- [ ] Severity tests (10+)
- [ ] Node execution tests (20+)
- [ ] Input validation tests (15+)
- [ ] Output tests (10+)
- [ ] Integration tests (10+)
- [ ] All tests passing

**Step 5: Validate Contracts**
- [ ] CONTRACT_VALIDATION.md created
- [ ] 100% input contract compliance documented
- [ ] 100% output contract compliance documented
- [ ] 100% state contract compliance documented
- [ ] 100% TypedDict compliance documented
- [ ] All algorithms documented with formulas and interpretations
- [ ] All integration blockers documented (priority, effort, interface)
- [ ] Test coverage summary included
- [ ] Performance validation against SLA included
- [ ] Deployment readiness checklist completed

### Overall Success Criteria:

- [ ] Agent fully implements specialist specification
- [ ] 100% contract compliance across all components
- [ ] 100+ comprehensive tests passing
- [ ] Performance meets SLA requirements
- [ ] All integration blockers documented with resolution plans
- [ ] Code follows E2I patterns and conventions
- [ ] Ready for staging deployment (with documented blockers)

---

## Time Estimates

**Step 1**: 20-30 minutes (documentation review)
**Step 2**: 15-20 minutes (contract extraction)
**Step 3**: 90-120 minutes (implementation)
**Step 4**: 60-90 minutes (testing)
**Step 5**: 30-45 minutes (validation documentation)

**Total**: 3-6 hours per agent (depending on complexity)

---

## Common Pitfalls to Avoid

1. **Incomplete Contract Implementation**
   - ❌ Missing optional fields
   - ❌ Incorrect default values
   - ❌ Missing validation rules
   - ✅ Use contract validation tables to verify every field

2. **Inadequate Testing**
   - ❌ Only testing happy path
   - ❌ Missing edge cases (empty data, extreme values)
   - ❌ No performance validation
   - ✅ Follow test category checklist to ensure comprehensive coverage

3. **Undocumented Integration Blockers**
   - ❌ Temporary mocks left undocumented
   - ❌ No resolution plan for blockers
   - ❌ Missing effort estimates
   - ✅ Document each blocker with priority, effort, and interface requirements

4. **Algorithm Implementation Errors**
   - ❌ Not handling edge cases (division by zero, empty arrays)
   - ❌ Incorrect formula implementation
   - ❌ Missing numerical stability checks
   - ✅ Document formulas, test edge cases, validate against known results

5. **Performance Issues**
   - ❌ No latency tracking
   - ❌ Sequential processing when parallel possible
   - ❌ Not validating against SLA
   - ✅ Track latency at every node, test against SLA requirements

---

## Adaptations for Different Agent Types

### Standard (Fast Path) Agents
- Focus on performance optimization
- No LLM calls
- Statistical/algorithmic processing
- <10s latency requirements
- Follow protocol as written

### Advanced (Reasoning Path) Agents
- Include LLM prompt engineering
- May have longer latency SLA (30-60s)
- Additional tests for prompt quality
- Token usage tracking in state
- Add Step 3.6: Prompt templates

### Strategic (Long-Horizon) Agents
- Multi-step planning workflows
- Conditional routing based on intermediate results
- Progress tracking in state
- May span multiple conversation turns
- Add Step 3.6: Planning node implementation

---

## Example Usage

```python
# Step 1: Load specialist docs
specialist_file = ".claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md"
# Read and extract key information...

# Step 2: Review contracts
contract_file = ".claude/contracts/tier3-contracts.md"
# Extract input/output/state contracts...

# Step 3: Create agent structure
# Create state.py, nodes/, graph.py, agent.py, __init__.py

# Step 4: Implement tests
# Create 5 test files with 100+ tests

# Step 5: Validate contracts
# Create CONTRACT_VALIDATION.md with 100% compliance
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-18 | Initial protocol based on drift_monitor implementation |

---

## References

- **Specialist Documentation**: `.claude/specialists/`
- **Contract Definitions**: `.claude/contracts/`
- **Coding Patterns**: `.claude/.agent_docs/coding-patterns.md`
- **Testing Patterns**: `.claude/.agent_docs/testing-patterns.md`
- **Error Handling**: `.claude/.agent_docs/error-handling.md`
- **ML Patterns**: `.claude/.agent_docs/ml-patterns.md`
