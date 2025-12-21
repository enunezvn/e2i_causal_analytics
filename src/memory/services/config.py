"""
E2I Agentic Memory - Configuration Loader
Loads and validates memory configuration from YAML.

Usage:
    from src.memory.services.config import load_memory_config, get_config

    config = get_config()  # Cached singleton
    print(config.environment)
    print(config.embeddings.model)
"""

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "005_memory_config.yaml"
)


@dataclass
class WorkingMemoryConfig:
    """Configuration for Redis-backed working memory."""

    backend: str = "redis"
    connection: str = "redis://localhost:6382"
    checkpoint_prefix: str = "e2i:checkpoint:"
    session_prefix: str = "e2i:session:"
    evidence_prefix: str = "e2i:evidence:"
    ttl_seconds: int = 86400
    max_sessions_per_user: int = 10
    context_window_messages: int = 10


@dataclass
class EpisodicMemoryConfig:
    """Configuration for Supabase-backed episodic memory."""

    backend: str = "supabase"
    table: str = "episodic_memories"
    vector_column: str = "embedding"
    vector_dims: int = 1536
    retention_days: int = 365


@dataclass
class SemanticMemoryConfig:
    """Configuration for FalkorDB-backed semantic memory."""

    backend: str = "falkordb"
    graph_name: str = "e2i_semantic"
    graphity_enabled: bool = True
    cache_enabled: bool = True
    cache_table: str = "semantic_memory_cache"
    sync_frequency_seconds: int = 300


@dataclass
class ProceduralMemoryConfig:
    """Configuration for Supabase-backed procedural memory."""

    backend: str = "supabase"
    table: str = "procedural_memories"
    vector_column: str = "trigger_embedding"
    vector_dims: int = 1536
    few_shot_max_examples: int = 5
    few_shot_min_similarity: float = 0.7


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""

    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    dimensions: int = 1536
    batch_size: int = 100
    cache_embeddings: bool = True


@dataclass
class LLMConfig:
    """Configuration for LLM service."""

    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class CognitiveWorkflowConfig:
    """Configuration for the cognitive workflow."""

    summarizer_compression_threshold: int = 10
    summarizer_keep_recent: int = 5
    investigator_max_hops: int = 4
    investigator_min_relevance: float = 0.5
    investigator_max_evidence: int = 20
    agent_timeout_seconds: int = 30
    reflector_min_confidence: float = 0.6


@dataclass
class MemoryConfig:
    """
    Main configuration container for the memory system.
    Provides typed access to all configuration sections.
    """

    environment: str = "local_pilot"
    schema_version: str = "1.1"

    # Backend configs
    working: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    episodic: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    semantic: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)
    procedural: ProceduralMemoryConfig = field(default_factory=ProceduralMemoryConfig)

    # Service configs
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Workflow config
    cognitive_workflow: CognitiveWorkflowConfig = field(default_factory=CognitiveWorkflowConfig)

    # Raw config dict for advanced access
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_raw(self, path: str, default: Any = None) -> Any:
        """
        Get a raw config value by dot-separated path.

        Example:
            config.get_raw("memory_backends.working.ttl_seconds", 3600)
        """
        keys = path.split(".")
        value = self._raw
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in config values."""

    def expand_value(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(v) for v in value]
        return value

    return expand_value(config)


def _parse_working_config(raw: Dict[str, Any], env: str) -> WorkingMemoryConfig:
    """Parse working memory configuration."""
    backend_config = raw.get("memory_backends", {}).get("working", {}).get(env, {})
    return WorkingMemoryConfig(
        backend=backend_config.get("backend", "redis"),
        connection=backend_config.get("connection", "redis://localhost:6379"),
        checkpoint_prefix=backend_config.get("checkpoint_prefix", "e2i:checkpoint:"),
        session_prefix=backend_config.get("session_prefix", "e2i:session:"),
        evidence_prefix=backend_config.get("evidence_prefix", "e2i:evidence:"),
        ttl_seconds=backend_config.get("ttl_seconds", 86400),
        max_sessions_per_user=backend_config.get("max_sessions_per_user", 10),
        context_window_messages=backend_config.get("context_window_messages", 10),
    )


def _parse_episodic_config(raw: Dict[str, Any], env: str) -> EpisodicMemoryConfig:
    """Parse episodic memory configuration."""
    backend_config = raw.get("memory_backends", {}).get("episodic", {}).get(env, {})
    return EpisodicMemoryConfig(
        backend=backend_config.get("backend", "supabase"),
        table=backend_config.get("table", "episodic_memories"),
        vector_column=backend_config.get("vector_column", "embedding"),
        vector_dims=backend_config.get("vector_dims", 1536),
        retention_days=backend_config.get("retention_days", 365),
    )


def _parse_semantic_config(raw: Dict[str, Any], env: str) -> SemanticMemoryConfig:
    """Parse semantic memory configuration."""
    backend_config = raw.get("memory_backends", {}).get("semantic", {}).get(env, {})
    graphity = backend_config.get("graphity", {})
    cache = backend_config.get("cache", {})
    return SemanticMemoryConfig(
        backend=backend_config.get("backend", "falkordb"),
        graph_name=backend_config.get("graph_name", "e2i_semantic"),
        graphity_enabled=graphity.get("enabled", True),
        cache_enabled=cache.get("enabled", True),
        cache_table=cache.get("table", "semantic_memory_cache"),
        sync_frequency_seconds=cache.get("sync_frequency_seconds", 300),
    )


def _parse_procedural_config(raw: Dict[str, Any], env: str) -> ProceduralMemoryConfig:
    """Parse procedural memory configuration."""
    backend_config = raw.get("memory_backends", {}).get("procedural", {}).get(env, {})
    few_shot = backend_config.get("few_shot", {})
    return ProceduralMemoryConfig(
        backend=backend_config.get("backend", "supabase"),
        table=backend_config.get("table", "procedural_memories"),
        vector_column=backend_config.get("vector_column", "trigger_embedding"),
        vector_dims=backend_config.get("vector_dims", 1536),
        few_shot_max_examples=few_shot.get("max_examples", 5),
        few_shot_min_similarity=few_shot.get("min_similarity", 0.7),
    )


def _parse_embedding_config(raw: Dict[str, Any], env: str) -> EmbeddingConfig:
    """Parse embedding service configuration."""
    embed_config = raw.get("embeddings", {}).get(env, {})
    return EmbeddingConfig(
        provider=embed_config.get("provider", "openai"),
        model=embed_config.get("model", "text-embedding-ada-002"),
        dimensions=embed_config.get("dimensions", 1536),
        batch_size=embed_config.get("batch_size", 100),
        cache_embeddings=embed_config.get("cache_embeddings", True),
    )


def _parse_llm_config(raw: Dict[str, Any], env: str) -> LLMConfig:
    """Parse LLM service configuration."""
    llm_config = raw.get("llm", {}).get(env, {})
    return LLMConfig(
        provider=llm_config.get("provider", "anthropic"),
        model=llm_config.get("model", "claude-3-5-sonnet-20241022"),
        max_tokens=llm_config.get("max_tokens", 4096),
        temperature=llm_config.get("temperature", 0.3),
    )


def _parse_cognitive_config(raw: Dict[str, Any]) -> CognitiveWorkflowConfig:
    """Parse cognitive workflow configuration."""
    workflow = raw.get("cognitive_workflow", {})
    summarizer = workflow.get("summarizer", {})
    investigator = workflow.get("investigator", {})
    agent = workflow.get("agent", {})
    reflector = workflow.get("reflector", {})

    return CognitiveWorkflowConfig(
        summarizer_compression_threshold=summarizer.get("compression_threshold_messages", 10),
        summarizer_keep_recent=summarizer.get("keep_recent_messages", 5),
        investigator_max_hops=investigator.get("max_hops", 4),
        investigator_min_relevance=investigator.get("min_relevance_threshold", 0.5),
        investigator_max_evidence=investigator.get("max_evidence_items", 20),
        agent_timeout_seconds=agent.get("agent_timeout_seconds", 30),
        reflector_min_confidence=reflector.get("min_confidence_for_learning", 0.6),
    )


def load_memory_config(config_path: Optional[Path] = None) -> MemoryConfig:
    """
    Load memory configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/005_memory_config.yaml

    Returns:
        MemoryConfig with all parsed settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Memory config file not found: {config_path}")

    logger.info(f"Loading memory config from {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Expand environment variables
    raw_config = _expand_env_vars(raw_config)

    # Get environment
    environment = os.environ.get("E2I_ENVIRONMENT", raw_config.get("environment", "local_pilot"))

    # Parse all sections
    config = MemoryConfig(
        environment=environment,
        schema_version=raw_config.get("schema_version", "1.1"),
        working=_parse_working_config(raw_config, environment),
        episodic=_parse_episodic_config(raw_config, environment),
        semantic=_parse_semantic_config(raw_config, environment),
        procedural=_parse_procedural_config(raw_config, environment),
        embeddings=_parse_embedding_config(raw_config, environment),
        llm=_parse_llm_config(raw_config, environment),
        cognitive_workflow=_parse_cognitive_config(raw_config),
        _raw=raw_config,
    )

    logger.info(f"Loaded config for environment: {environment}")
    return config


@lru_cache(maxsize=1)
def get_config() -> MemoryConfig:
    """
    Get the cached memory configuration singleton.

    This function caches the configuration to avoid repeated file reads.
    To reload config, call load_memory_config() directly.

    Returns:
        MemoryConfig singleton
    """
    return load_memory_config()


def clear_config_cache() -> None:
    """Clear the config cache to force a reload on next get_config() call."""
    get_config.cache_clear()
