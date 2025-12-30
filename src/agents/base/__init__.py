"""
Base agent utilities and mixins.

This module provides reusable components for agent implementation:
- AuditChainMixin: Tamper-evident logging with hash-linked chains
- audited_traced_node: Decorator combining Opik tracing with audit chain
"""

from src.agents.base.audit_chain_mixin import (
    AgentTier,
    AuditChainEntry,
    AuditChainMixin,
    AuditChainService,
    ChainVerificationResult,
    RefutationResults,
    audited_traced_node,
    create_workflow_initializer,
    get_audit_chain_service,
    init_audit_chain_service,
    set_audit_chain_service,
)

__all__ = [
    "AuditChainMixin",
    "AgentTier",
    "AuditChainEntry",
    "AuditChainService",
    "ChainVerificationResult",
    "RefutationResults",
    "audited_traced_node",
    "create_workflow_initializer",
    "get_audit_chain_service",
    "set_audit_chain_service",
    "init_audit_chain_service",
]
