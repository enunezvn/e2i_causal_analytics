"""E2I Causal Analytics Testing Module.

This module provides testing utilities for validating Tier 1-5 agents
using tier0 synthetic data outputs.

Components:
- Tier0OutputMapper: Maps tier0 outputs to agent-specific inputs
- ContractValidator: Validates agent outputs against TypedDict contracts
- OpikTraceVerifier: Verifies Opik observability traces
- QualityGateValidator: Validates agent outputs against per-agent quality gates
- DataSourceValidator: Validates agents use appropriate data sources (real vs mock)
- AGENT_QUALITY_GATES: Per-agent quality threshold definitions
"""

from src.testing.agent_quality_gates import (
    AGENT_QUALITY_GATES,
    AgentQualityGate,
    DataSourceRequirement,
    get_quality_gate,
    list_configured_agents,
)
from src.testing.contract_validator import ContractValidator, ValidationResult
from src.testing.data_source_validator import (
    DataSourceType,
    DataSourceValidationResult,
    DataSourceValidator,
)
from src.testing.opik_trace_verifier import OpikTraceVerifier, TraceVerificationResult
from src.testing.quality_gate_validator import (
    QualityCheckResult,
    QualityGateResult,
    QualityGateValidator,
)
from src.testing.tier0_output_mapper import Tier0OutputMapper

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
    "DataSourceRequirement",
    "get_quality_gate",
    "list_configured_agents",
    # Data source validation
    "DataSourceValidator",
    "DataSourceValidationResult",
    "DataSourceType",
]
