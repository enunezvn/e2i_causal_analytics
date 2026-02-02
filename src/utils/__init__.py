"""
E2I Causal Analytics Utilities.

Provides shared utilities across the application:
- Audit chain for ML decision tracking
- LLM factory for consistent LLM client creation
- Structured logging configuration (G14)
"""

from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
    create_audit_chain_service,
)
from src.utils.llm_factory import (
    get_chat_llm,
    get_fast_llm,
    get_llm_provider,
    get_reasoning_llm,
    get_standard_llm,
)
from src.utils.logging_config import (
    ColoredFormatter,
    ContextFilter,
    # Formatters
    JSONFormatter,
    LoggingConfig,
    agent_name_var,
    clear_request_context,
    # Configuration
    configure_logging,
    # Utilities
    get_logger,
    get_request_context,
    log_level_context,
    operation_var,
    # Context variables
    request_id_var,
    # Context management
    set_request_context,
    span_id_var,
    timed_operation,
    trace_id_var,
    user_id_var,
)

__all__ = [
    # Audit chain
    "AuditChainService",
    "AuditChainEntry",
    "create_audit_chain_service",
    "AgentTier",
    # LLM factory
    "get_chat_llm",
    "get_fast_llm",
    "get_standard_llm",
    "get_reasoning_llm",
    "get_llm_provider",
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
