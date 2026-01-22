"""
E2I Causal Analytics Utilities.

Provides shared utilities across the application:
- Audit chain for ML decision tracking
- LLM factory for consistent LLM client creation
- Structured logging configuration (G14)
"""

from src.utils.audit_chain import (
    AuditChainService,
    AuditChainEntry,
    create_audit_chain_service,
    AgentTier,
)
from src.utils.llm_factory import create_llm_client, get_llm_client, LLMConfig
from src.utils.logging_config import (
    # Configuration
    configure_logging,
    LoggingConfig,
    # Context management
    set_request_context,
    clear_request_context,
    get_request_context,
    # Context variables
    request_id_var,
    trace_id_var,
    span_id_var,
    user_id_var,
    agent_name_var,
    operation_var,
    # Formatters
    JSONFormatter,
    ColoredFormatter,
    ContextFilter,
    # Utilities
    get_logger,
    log_level_context,
    timed_operation,
)

__all__ = [
    # Audit chain
    "AuditChainService",
    "AuditChainEntry",
    "create_audit_chain_service",
    "AgentTier",
    # LLM factory
    "create_llm_client",
    "get_llm_client",
    "LLMConfig",
    # Logging (G14)
    "configure_logging",
    "LoggingConfig",
    "set_request_context",
    "clear_request_context",
    "get_request_context",
    "request_id_var",
    "trace_id_var",
    "span_id_var",
    "user_id_var",
    "agent_name_var",
    "operation_var",
    "JSONFormatter",
    "ColoredFormatter",
    "ContextFilter",
    "get_logger",
    "log_level_context",
    "timed_operation",
]
