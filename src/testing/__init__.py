"""E2I Causal Analytics Testing Module.

This module provides testing utilities for validating Tier 1-5 agents
using tier0 synthetic data outputs.

Components:
- Tier0OutputMapper: Maps tier0 outputs to agent-specific inputs
- ContractValidator: Validates agent outputs against TypedDict contracts
- OpikTraceVerifier: Verifies Opik observability traces
- QualityGateValidator: Validates agent outputs against per-agent quality gates
- AGENT_QUALITY_GATES: Per-agent quality threshold definitions
"""

from src.testing.tier0_output_mapper import Tier0OutputMapper
from src.testing.contract_validator import ContractValidator, ValidationResult
from src.testing.opik_trace_verifier import OpikTraceVerifier, TraceVerificationResult
from src.testing.quality_gate_validator import (
    QualityGateValidator,
    QualityGateResult,
    QualityCheckResult,
)
from src.testing.agent_quality_gates import (
    AGENT_QUALITY_GATES,
    AgentQualityGate,
    get_quality_gate,
    list_configured_agents,
)

__all__ = [
    "Tier0OutputMapper",
    "ContractValidator",
    "ValidationResult",
    "OpikTraceVerifier",
    "TraceVerificationResult",
    "QualityGateValidator",
    "QualityGateResult",
    "QualityCheckResult",
    "AGENT_QUALITY_GATES",
    "AgentQualityGate",
    "get_quality_gate",
    "list_configured_agents",
]
