"""
Test fixtures for Tool Composer tests.

Provides mock LLM clients, sample data, and shared utilities
for testing the 4-phase composition pipeline.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from src.agents.tool_composer.cache import ToolComposerCacheManager
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    SynthesisInput,
    ToolInput,
    ToolMapping,
    ToolOutput,
)
from src.tool_registry.registry import (
    ToolParameter,
    ToolRegistry,
    ToolSchema,
)

# ============================================================================
# CACHE RESET FIXTURE (Auto-use to prevent singleton pollution)
# ============================================================================


@pytest.fixture(autouse=True)
def reset_cache_singleton():
    """
    Reset the ToolComposerCacheManager singleton before each test.

    This prevents cache state pollution between parallel tests, which can
    cause tests expecting errors (like invalid JSON) to instead get cached
    results from other tests.
    """
    # Reset singleton before test
    ToolComposerCacheManager._instance = None
    yield
    # Reset singleton after test (cleanup)
    ToolComposerCacheManager._instance = None


# ============================================================================
# MOCK LLM CLIENT
# ============================================================================


class MockLLMResponse:
    """Mock response from LLM (Anthropic-style with content list)"""

    def __init__(self, text: str):
        self.content = [Mock(text=text)]


class MockAIMessage:
    """Mock LangChain AIMessage response"""

    def __init__(self, content: str):
        self.content = content


class MockLLMClient:
    """Mock LLM client compatible with both Anthropic and LangChain interfaces"""

    def __init__(self):
        self.messages = MockMessages(self)
        self._decomposition_response: Optional[str] = None
        self._planning_response: Optional[str] = None
        self._synthesis_response: Optional[str] = None
        self._error_to_raise: Optional[Exception] = None
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []

    def set_decomposition_response(self, response: str) -> None:
        """Set the response for decomposition calls"""
        self._decomposition_response = response

    def set_planning_response(self, response: str) -> None:
        """Set the response for planning calls"""
        self._planning_response = response

    def set_synthesis_response(self, response: str) -> None:
        """Set the response for synthesis calls"""
        self._synthesis_response = response

    def set_error(self, error: Exception) -> None:
        """Set an error to be raised on the next ainvoke call"""
        self._error_to_raise = error

    async def ainvoke(self, messages: List[Any]) -> MockAIMessage:
        """
        LangChain-compatible async invoke method.

        Args:
            messages: List of LangChain message objects (SystemMessage, HumanMessage, etc.)

        Returns:
            MockAIMessage with content attribute containing the response

        Raises:
            Exception: If set_error() was called with an exception
        """
        self.call_count += 1

        # Check if we should raise an error
        if self._error_to_raise is not None:
            error = self._error_to_raise
            self._error_to_raise = None  # Reset after raising
            raise error

        # Extract system prompt from messages to determine response type
        system_content = ""
        user_content = ""
        for msg in messages:
            if hasattr(msg, "content"):
                # Check message type by class name
                msg_type = type(msg).__name__
                if "System" in msg_type:
                    system_content = msg.content
                elif "Human" in msg_type:
                    user_content = msg.content

        # Store full content for test verification (not truncated)
        self.call_history.append(
            {
                "system": system_content,
                "user": user_content,
                "messages": messages,  # Store full message objects
            }
        )

        response_text = self.get_response_for_phase(system_content)
        return MockAIMessage(response_text)

    def get_response_for_phase(self, system_prompt: str) -> str:
        """Return appropriate response based on system prompt content"""
        if "decomposition" in system_prompt.lower():
            return self._decomposition_response or self._default_decomposition()
        # Check synth BEFORE tool because synthesis prompt contains "tools"
        elif "synth" in system_prompt.lower():
            return self._synthesis_response or self._default_synthesis()
        elif "planning" in system_prompt.lower() or "tool" in system_prompt.lower():
            return self._planning_response or self._default_planning()
        return "{}"

    def _default_decomposition(self) -> str:
        return json.dumps(
            {
                "reasoning": "Test decomposition reasoning",
                "sub_questions": [
                    {
                        "id": "sq_1",
                        "question": "What is the causal effect of rep visits?",
                        "intent": "CAUSAL",
                        "entities": ["rep_visits", "rx_volume"],
                        "depends_on": [],
                    },
                    {
                        "id": "sq_2",
                        "question": "How does this vary by region?",
                        "intent": "COMPARATIVE",
                        "entities": ["region"],
                        "depends_on": ["sq_1"],
                    },
                ],
            }
        )

    def _default_planning(self) -> str:
        return json.dumps(
            {
                "reasoning": "Test planning reasoning",
                "tool_mappings": [
                    {
                        "sub_question_id": "sq_1",
                        "tool_name": "causal_effect_estimator",
                        "confidence": 0.95,
                        "reasoning": "Matches causal intent",
                    },
                    {
                        "sub_question_id": "sq_2",
                        "tool_name": "cate_analyzer",
                        "confidence": 0.9,
                        "reasoning": "Analyzes regional variation",
                    },
                ],
                "execution_steps": [
                    {
                        "step_id": "step_1",
                        "sub_question_id": "sq_1",
                        "tool_name": "causal_effect_estimator",
                        "input_mapping": {"treatment": "rep_visits", "outcome": "rx_volume"},
                        "depends_on_steps": [],
                    },
                    {
                        "step_id": "step_2",
                        "sub_question_id": "sq_2",
                        "tool_name": "cate_analyzer",
                        "input_mapping": {"effect": "$step_1.effect", "dimension": "region"},
                        "depends_on_steps": ["step_1"],
                    },
                ],
                "parallel_groups": [["step_1"], ["step_2"]],
            }
        )

    def _default_synthesis(self) -> str:
        return json.dumps(
            {
                "answer": "Rep visits show a 15% causal lift in Rx volume with regional variation.",
                "confidence": 0.85,
                "supporting_data": {"effect_size": 0.15, "ci_lower": 0.12, "ci_upper": 0.18},
                "citations": ["step_1", "step_2"],
                "caveats": ["Analysis based on observational data"],
                "failed_components": [],
                "reasoning": "Combined causal analysis with regional breakdown",
            }
        )


class MockMessages:
    """Mock messages endpoint"""

    def __init__(self, client: MockLLMClient):
        self.client = client

    async def create(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: List[Dict[str, str]],
    ) -> MockLLMResponse:
        """Mock create message"""
        self.client.call_count += 1
        self.client.call_history.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system[:100],
                "messages": messages,
            }
        )

        response_text = self.client.get_response_for_phase(system)
        return MockLLMResponse(response_text)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client"""
    return MockLLMClient()


# ============================================================================
# MOCK TOOL REGISTRY
# ============================================================================


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with sample tools"""
    registry = ToolRegistry()
    registry.clear()  # Ensure clean state

    # Register sample tools
    tools = [
        {
            "name": "causal_effect_estimator",
            "description": "Estimate ATE/ATT causal effects using DoWhy",
            "source_agent": "causal_impact",
            "tier": 2,
            "input_parameters": [
                ToolParameter("treatment", "str", "Treatment variable", True),
                ToolParameter("outcome", "str", "Outcome variable", True),
            ],
            "output_schema": "EffectEstimate",
            "avg_execution_ms": 500,
            "callable": lambda treatment, outcome, **kwargs: {
                "effect": 0.15,
                "ci_lower": 0.12,
                "ci_upper": 0.18,
                "p_value": 0.001,
            },
        },
        {
            "name": "cate_analyzer",
            "description": "Analyze conditional average treatment effects by segment",
            "source_agent": "heterogeneous_optimizer",
            "tier": 2,
            "input_parameters": [
                ToolParameter("effect", "float", "Base effect estimate", True),
                ToolParameter("dimension", "str", "Dimension for analysis", True),
            ],
            "output_schema": "CATEResult",
            "avg_execution_ms": 800,
            "callable": lambda effect, dimension, **kwargs: {
                "segments": [
                    {"segment": "Northeast", "cate": 0.12},
                    {"segment": "Midwest", "cate": 0.18},
                ],
                "heterogeneity_score": 0.4,
            },
        },
        {
            "name": "gap_calculator",
            "description": "Calculate opportunity gaps",
            "source_agent": "gap_analyzer",
            "tier": 2,
            "input_parameters": [
                ToolParameter("metric", "str", "Metric to analyze", True),
            ],
            "output_schema": "GapAnalysis",
            "avg_execution_ms": 300,
            "callable": lambda metric, **kwargs: {
                "gap_size": 0.25,
                "top_opportunities": ["region_1", "region_2"],
            },
        },
        {
            "name": "counterfactual_simulator",
            "description": "Simulate counterfactual scenarios",
            "source_agent": "experiment_designer",
            "tier": 3,
            "input_parameters": [
                ToolParameter("scenario", "dict", "Scenario parameters", True),
            ],
            "output_schema": "SimulationResult",
            "avg_execution_ms": 1200,
            "callable": lambda scenario, **kwargs: {
                "predicted_outcome": 0.22,
                "confidence_interval": [0.18, 0.26],
            },
        },
    ]

    for tool_config in tools:
        schema = ToolSchema(
            name=tool_config["name"],
            description=tool_config["description"],
            source_agent=tool_config["source_agent"],
            tier=tool_config["tier"],
            input_parameters=tool_config["input_parameters"],
            output_schema=tool_config["output_schema"],
            avg_execution_ms=tool_config["avg_execution_ms"],
        )
        registry.register(schema=schema, callable=tool_config["callable"])

    yield registry

    # Cleanup
    registry.clear()


@pytest.fixture
def empty_registry():
    """Create an empty tool registry"""
    registry = ToolRegistry()
    registry.clear()
    yield registry
    registry.clear()


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_query():
    """Sample multi-faceted query"""
    return (
        "Compare the causal impact of rep visits vs speaker programs for oncologists, "
        "and predict which approach would work better in the Midwest region"
    )


@pytest.fixture
def simple_query():
    """Simple query for testing"""
    return "What is the causal effect of rep visits on prescription volume?"


@pytest.fixture
def sample_sub_questions():
    """Sample sub-questions"""
    return [
        SubQuestion(
            id="sq_1",
            question="What is the causal effect of rep visits on Rx volume?",
            intent="CAUSAL",
            entities=["rep_visits", "rx_volume"],
            depends_on=[],
        ),
        SubQuestion(
            id="sq_2",
            question="How does this effect vary by region?",
            intent="COMPARATIVE",
            entities=["region"],
            depends_on=["sq_1"],
        ),
    ]


@pytest.fixture
def sample_decomposition(sample_sub_questions):
    """Sample decomposition result"""
    return DecompositionResult(
        original_query="Compare causal impact of rep visits by region",
        sub_questions=sample_sub_questions,
        decomposition_reasoning="Decomposed into causal + regional analysis",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_tool_mappings():
    """Sample tool mappings"""
    return [
        ToolMapping(
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            confidence=0.95,
            reasoning="Matches causal intent",
        ),
        ToolMapping(
            sub_question_id="sq_2",
            tool_name="cate_analyzer",
            source_agent="heterogeneous_optimizer",
            confidence=0.9,
            reasoning="Analyzes heterogeneous effects",
        ),
    ]


@pytest.fixture
def sample_execution_steps():
    """Sample execution steps"""
    return [
        ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
            dependency_type=DependencyType.PARALLEL,
            depends_on_steps=[],
        ),
        ExecutionStep(
            step_id="step_2",
            sub_question_id="sq_2",
            tool_name="cate_analyzer",
            source_agent="heterogeneous_optimizer",
            input_mapping={"effect": "$step_1.effect", "dimension": "region"},
            dependency_type=DependencyType.SEQUENTIAL,
            depends_on_steps=["step_1"],
        ),
    ]


@pytest.fixture
def sample_execution_plan(sample_decomposition, sample_execution_steps, sample_tool_mappings):
    """Sample execution plan"""
    return ExecutionPlan(
        decomposition=sample_decomposition,
        steps=sample_execution_steps,
        tool_mappings=sample_tool_mappings,
        estimated_duration_ms=1300,
        parallel_groups=[["step_1"], ["step_2"]],
        planning_reasoning="Sequential execution with dependency",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_step_result():
    """Sample step result"""
    return StepResult(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="causal_effect_estimator",
        input=ToolInput(
            tool_name="causal_effect_estimator",
            parameters={"treatment": "rep_visits", "outcome": "rx_volume"},
            context={},
        ),
        output=ToolOutput(
            tool_name="causal_effect_estimator",
            success=True,
            result={"effect": 0.15, "ci_lower": 0.12, "ci_upper": 0.18},
            execution_time_ms=450,
        ),
        status=ExecutionStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_ms=450,
    )


@pytest.fixture
def sample_execution_trace(sample_step_result):
    """Sample execution trace"""
    trace = ExecutionTrace(plan_id="plan_test123", started_at=datetime.now(timezone.utc))
    trace.add_result(sample_step_result)
    trace.completed_at = datetime.now(timezone.utc)
    return trace


@pytest.fixture
def sample_synthesis_input(sample_decomposition, sample_execution_trace):
    """Sample synthesis input"""
    return SynthesisInput(
        original_query="What is the causal effect of rep visits?",
        decomposition=sample_decomposition,
        execution_trace=sample_execution_trace,
    )


# ============================================================================
# ASYNC TOOL FIXTURES
# ============================================================================


@pytest.fixture
def async_tool():
    """Create an async tool callable"""

    async def async_tool_fn(treatment: str, outcome: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate async work
        return {"effect": 0.2, "p_value": 0.05}

    return async_tool_fn


@pytest.fixture
def failing_tool():
    """Create a tool that always fails"""

    def failing_tool_fn(**kwargs) -> Dict[str, Any]:
        raise ValueError("Simulated tool failure")

    return failing_tool_fn


@pytest.fixture
def slow_tool():
    """Create a slow tool for timeout testing"""

    async def slow_tool_fn(**kwargs) -> Dict[str, Any]:
        await asyncio.sleep(10)  # Will trigger timeout
        return {"result": "slow"}

    return slow_tool_fn


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def make_decomposition_response(
    sub_questions: List[Dict[str, Any]], reasoning: str = "Test reasoning"
) -> str:
    """Helper to create decomposition JSON response"""
    return json.dumps({"reasoning": reasoning, "sub_questions": sub_questions})


def make_planning_response(
    tool_mappings: List[Dict[str, Any]],
    execution_steps: List[Dict[str, Any]],
    parallel_groups: List[List[str]] = None,
    reasoning: str = "Test planning",
) -> str:
    """Helper to create planning JSON response"""
    return json.dumps(
        {
            "reasoning": reasoning,
            "tool_mappings": tool_mappings,
            "execution_steps": execution_steps,
            "parallel_groups": parallel_groups or [[s["step_id"]] for s in execution_steps],
        }
    )


def make_synthesis_response(
    answer: str,
    confidence: float = 0.85,
    supporting_data: Dict[str, Any] = None,
    citations: List[str] = None,
    caveats: List[str] = None,
    failed_components: List[str] = None,
    reasoning: str = "Test synthesis",
) -> str:
    """Helper to create synthesis JSON response"""
    return json.dumps(
        {
            "answer": answer,
            "confidence": confidence,
            "supporting_data": supporting_data or {},
            "citations": citations or [],
            "caveats": caveats or [],
            "failed_components": failed_components or [],
            "reasoning": reasoning,
        }
    )
