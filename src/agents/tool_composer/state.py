"""
E2I Tool Composer Agent - State Definitions
Version: 4.4
Purpose: LangGraph TypedDict state for contract validation

This module provides TypedDict definitions compatible with the contract validator.
The Tool Composer uses Pydantic models internally (schemas.py, models/), but this
TypedDict state enables consistent contract validation across all agents.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired
from uuid import UUID


# Type aliases
CompositionStatusType = Literal[
    "PENDING",
    "DECOMPOSING",
    "PLANNING",
    "EXECUTING",
    "SYNTHESIZING",
    "SUCCESS",
    "PARTIAL",
    "FAILED",
    "TIMEOUT",
]


class SubQuestion(TypedDict):
    """Atomic sub-question from decomposition."""

    id: str
    question: str
    intent: Literal["CAUSAL", "COMPARATIVE", "PREDICTIVE", "DESCRIPTIVE", "EXPERIMENTAL"]
    entities: NotRequired[List[str]]
    depends_on: NotRequired[List[str]]
    priority: NotRequired[int]


class ToolMapping(TypedDict):
    """Mapping from sub-question to tool."""

    sub_question_id: str
    tool_name: str
    source_agent: str
    confidence: float
    reasoning: NotRequired[str]


class ExecutionStep(TypedDict):
    """Single step in the execution plan."""

    step_id: str
    sub_question_id: str
    tool_name: str
    source_agent: NotRequired[str]
    input_mapping: NotRequired[Dict[str, Any]]
    depends_on_steps: NotRequired[List[str]]
    status: NotRequired[Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]]
    output: NotRequired[Dict[str, Any]]
    error: NotRequired[str]
    duration_ms: NotRequired[int]


class StepResult(TypedDict):
    """Result from executing a single step."""

    step_id: str
    tool_name: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMEOUT", "SKIPPED", "CANCELLED"]
    success: bool
    output: NotRequired[Dict[str, Any]]
    error: NotRequired[str]
    duration_ms: NotRequired[int]
    retry_count: NotRequired[int]


class ToolComposerState(TypedDict):
    """Complete state for Tool Composer agent.

    Contract: .claude/contracts/tier1-tool-composer-contracts.md

    Field Groups:
    - Input (5): query, session_id, user_id, context, timeout_seconds
    - Configuration (3): parallel_limit, max_tools_per_plan, enable_caching
    - Decomposition outputs (4): sub_questions, decomposition_reasoning, question_count, has_dependencies
    - Planning outputs (4): tool_mappings, execution_steps, parallel_groups, planning_reasoning
    - Execution outputs (5): step_results, tools_executed, tools_succeeded, tools_failed, success_rate
    - Synthesis outputs (5): response, confidence, supporting_data, citations, caveats
    - Status (3): status, success, composition_id
    - Timing (6): decompose_latency_ms, plan_latency_ms, execute_latency_ms, synthesize_latency_ms, total_latency_ms, timestamp
    - Error handling (2): errors, error
    - Audit (1): audit_workflow_id
    """

    # ===== INPUT (NotRequired - provided by caller) =====
    query: NotRequired[str]
    session_id: NotRequired[str]
    user_id: NotRequired[str]
    context: NotRequired[Dict[str, Any]]
    timeout_seconds: NotRequired[int]

    # ===== CONFIGURATION (NotRequired - has defaults) =====
    parallel_limit: NotRequired[int]  # Default: 3
    max_tools_per_plan: NotRequired[int]  # Default: 8
    enable_caching: NotRequired[bool]  # Default: True

    # ===== DECOMPOSITION OUTPUTS =====
    sub_questions: NotRequired[List[SubQuestion]]
    decomposition_reasoning: NotRequired[str]
    question_count: NotRequired[int]
    has_dependencies: NotRequired[bool]

    # ===== PLANNING OUTPUTS =====
    tool_mappings: NotRequired[List[ToolMapping]]
    execution_steps: NotRequired[List[ExecutionStep]]
    parallel_groups: NotRequired[List[List[str]]]
    planning_reasoning: NotRequired[str]

    # ===== EXECUTION OUTPUTS =====
    step_results: NotRequired[List[StepResult]]
    tools_executed: NotRequired[int]
    tools_succeeded: NotRequired[int]
    tools_failed: NotRequired[int]
    success_rate: NotRequired[float]

    # ===== SYNTHESIS OUTPUTS (Required for successful completion) =====
    response: NotRequired[str]
    confidence: NotRequired[float]
    supporting_data: NotRequired[Dict[str, Any]]
    citations: NotRequired[List[str]]
    caveats: NotRequired[List[str]]

    # ===== STATUS (Required outputs) =====
    status: CompositionStatusType
    success: bool
    composition_id: NotRequired[str]
    sub_questions_count: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]
    total_duration_ms: NotRequired[int]

    # ===== TIMING (Contract-required output fields) =====
    decompose_latency_ms: NotRequired[int]
    plan_latency_ms: NotRequired[int]
    execute_latency_ms: NotRequired[int]
    synthesize_latency_ms: NotRequired[int]
    total_latency_ms: NotRequired[int]
    timestamp: NotRequired[str]

    # ===== ERROR HANDLING =====
    errors: Annotated[List[Dict[str, Any]], operator.add]
    error: NotRequired[str]

    # ===== AUDIT CHAIN =====
    audit_workflow_id: NotRequired[UUID]
