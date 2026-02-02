"""
Observability Configuration Loader.

Loads and manages configuration from config/observability.yaml with:
- Environment variable substitution
- Environment-specific overrides
- Validation and defaults
- Singleton access pattern

Version: 1.0.0 (Phase 3.4)

Usage:
    from src.agents.ml_foundation.observability_connector.config import (
        ObservabilityConfig,
        get_observability_config,
    )

    # Get configuration
    config = get_observability_config()

    # Access settings
    if config.opik.enabled:
        sample_rate = config.sampling.default_rate

Author: E2I Causal Analytics Team
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATA CLASSES
# ============================================================================


@dataclass
class OpikSettings:
    """Opik SDK settings."""

    enabled: bool = True
    api_key_env: str = "OPIK_API_KEY"
    endpoint_env: str = "OPIK_ENDPOINT"
    workspace: str = "default"
    project_name: str = "e2i-causal-analytics"
    use_local: bool = False
    local_port: int = 5173
    flush_on_shutdown: bool = True
    flush_interval_seconds: int = 10

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.getenv(self.api_key_env)

    @property
    def endpoint(self) -> Optional[str]:
        """Get endpoint from environment."""
        return os.getenv(self.endpoint_env)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "workspace": self.workspace,
            "project_name": self.project_name,
            "use_local": self.use_local,
            "has_api_key": self.api_key is not None,
        }


@dataclass
class SamplingSettings:
    """Trace sampling settings."""

    default_rate: float = 1.0
    production_rate: float = 0.1
    environment_aware: bool = True
    always_sample_errors: bool = True
    always_sample_agents: List[str] = field(default_factory=list)
    agent_overrides: Dict[str, float] = field(default_factory=dict)

    def get_rate(self, agent_name: Optional[str] = None) -> float:
        """Get sample rate for an agent."""
        # Check agent overrides first
        if agent_name and agent_name in self.agent_overrides:
            return self.agent_overrides[agent_name]

        # Check always sample list
        if agent_name and agent_name in self.always_sample_agents:
            return 1.0

        # Environment-aware rate selection
        if self.environment_aware:
            env = os.getenv("ENVIRONMENT", "development").lower()
            if env == "production":
                return self.production_rate

        return self.default_rate


@dataclass
class RetrySettings:
    """Retry configuration for batching."""

    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    max_delay_seconds: float = 10.0


@dataclass
class BatchingSettings:
    """Batch processing settings."""

    enabled: bool = True
    max_batch_size: int = 100
    max_wait_seconds: float = 5.0
    flush_on_shutdown: bool = True
    retry: RetrySettings = field(default_factory=RetrySettings)


@dataclass
class CircuitBreakerFallback:
    """Circuit breaker fallback settings."""

    log_to_database: bool = True
    log_to_file: bool = False
    fallback_log_path: str = "logs/opik_fallback.jsonl"


@dataclass
class CircuitBreakerSettings:
    """Circuit breaker settings."""

    enabled: bool = True
    failure_threshold: int = 5
    reset_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    fallback: CircuitBreakerFallback = field(default_factory=CircuitBreakerFallback)


@dataclass
class CacheTTLSettings:
    """Cache TTL settings."""

    window_1h: int = 60
    window_24h: int = 300
    window_7d: int = 600
    default: int = 120

    def get_ttl(self, window: str) -> int:
        """Get TTL for a time window."""
        ttl_map = {
            "1h": self.window_1h,
            "24h": self.window_24h,
            "7d": self.window_7d,
        }
        return ttl_map.get(window, self.default)


@dataclass
class CacheMemorySettings:
    """Memory cache settings."""

    max_entries: int = 1000
    cleanup_interval_seconds: int = 60


@dataclass
class CacheRedisSettings:
    """Redis cache settings."""

    url_env: str = "REDIS_URL"
    db: int = 1
    connection_timeout_seconds: int = 5
    socket_timeout_seconds: int = 5

    @property
    def url(self) -> Optional[str]:
        """Get Redis URL from environment."""
        return os.getenv(self.url_env)


@dataclass
class CacheSettings:
    """Metrics cache settings."""

    backend: str = "memory"
    fallback_to_memory: bool = True
    key_prefix: str = "obs_metrics"
    ttl: CacheTTLSettings = field(default_factory=CacheTTLSettings)
    memory: CacheMemorySettings = field(default_factory=CacheMemorySettings)
    redis: CacheRedisSettings = field(default_factory=CacheRedisSettings)


@dataclass
class RetentionCleanupSettings:
    """Retention cleanup settings."""

    enabled: bool = True
    batch_size: int = 1000
    schedule: str = "0 2 * * *"
    dry_run: bool = False


@dataclass
class RetentionArchiveSettings:
    """Retention archive settings."""

    enabled: bool = False
    destination: str = ""
    format: str = "parquet"
    compress: bool = True


@dataclass
class RetentionSettings:
    """Data retention settings."""

    span_ttl_days: int = 30
    cleanup: RetentionCleanupSettings = field(default_factory=RetentionCleanupSettings)
    archive: RetentionArchiveSettings = field(default_factory=RetentionArchiveSettings)


@dataclass
class DatabasePoolSettings:
    """Database connection pool settings."""

    min_connections: int = 2
    max_connections: int = 10
    connection_timeout_seconds: int = 30


@dataclass
class DatabaseSettings:
    """Database settings."""

    spans_table: str = "ml_observability_spans"
    latency_summary_view: str = "v_agent_latency_summary"
    batch_insert: bool = True
    insert_batch_size: int = 100
    pool: DatabasePoolSettings = field(default_factory=DatabasePoolSettings)


@dataclass
class SpanSettings:
    """Span creation settings."""

    include_input: bool = True
    include_output: bool = True
    max_input_size_bytes: int = 10240
    max_output_size_bytes: int = 10240
    max_metadata_size_bytes: int = 4096
    truncate_on_overflow: bool = True
    redact_patterns: List[str] = field(
        default_factory=lambda: ["password", "api_key", "secret", "token", "authorization"]
    )


@dataclass
class AgentTierSettings:
    """Per-tier observability settings."""

    sample_rate: float = 1.0
    detailed_logging: bool = False
    agents: List[str] = field(default_factory=list)


@dataclass
class LoggingComponentSettings:
    """Per-component logging settings."""

    opik_connector: str = "INFO"
    batch_processor: str = "INFO"
    circuit_breaker: str = "WARNING"
    metrics_cache: str = "INFO"
    span_repository: str = "INFO"


@dataclass
class LoggingSettings:
    """Logging settings."""

    level: str = "INFO"
    components: LoggingComponentSettings = field(default_factory=LoggingComponentSettings)
    format: str = "json"
    include_timestamp: bool = True
    include_trace_id: bool = True


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================


@dataclass
class ObservabilityConfig:
    """Complete observability configuration."""

    opik: OpikSettings = field(default_factory=OpikSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    batching: BatchingSettings = field(default_factory=BatchingSettings)
    circuit_breaker: CircuitBreakerSettings = field(default_factory=CircuitBreakerSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    retention: RetentionSettings = field(default_factory=RetentionSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    spans: SpanSettings = field(default_factory=SpanSettings)
    agent_tiers: Dict[str, AgentTierSettings] = field(default_factory=dict)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    _config_path: Optional[str] = None

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> "ObservabilityConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file (defaults to config/observability.yaml)
            environment: Environment name for overrides (defaults to ENVIRONMENT env var)

        Returns:
            ObservabilityConfig instance
        """
        # Default path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = str(project_root / "config" / "observability.yaml")

        # Check if file exists
        if not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        # Load YAML
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return cls()

        # Get environment
        env = environment or os.getenv("ENVIRONMENT", "development").lower()

        # Apply environment overrides if present
        if "environments" in data and env in data["environments"]:
            env_overrides = data["environments"][env]
            data = _deep_merge(data, env_overrides)

        # Parse configuration
        config = cls._parse_config(data)
        config._config_path = config_path

        logger.info(f"Loaded observability config from {config_path} (env: {env})")
        return config

    @classmethod
    def _parse_config(cls, data: Dict[str, Any]) -> "ObservabilityConfig":
        """Parse configuration from dictionary."""
        config = cls()

        # Parse opik settings
        if "opik" in data:
            config.opik = _parse_dataclass(OpikSettings, data["opik"])

        # Parse sampling settings
        if "sampling" in data:
            config.sampling = _parse_dataclass(SamplingSettings, data["sampling"])

        # Parse batching settings
        if "batching" in data:
            batching_data = data["batching"]
            if "retry" in batching_data:
                batching_data["retry"] = _parse_dataclass(RetrySettings, batching_data["retry"])
            config.batching = _parse_dataclass(BatchingSettings, batching_data)

        # Parse circuit breaker settings
        if "circuit_breaker" in data:
            cb_data = data["circuit_breaker"]
            if "fallback" in cb_data:
                cb_data["fallback"] = _parse_dataclass(CircuitBreakerFallback, cb_data["fallback"])
            config.circuit_breaker = _parse_dataclass(CircuitBreakerSettings, cb_data)

        # Parse cache settings
        if "cache" in data:
            cache_data = data["cache"]
            if "ttl" in cache_data:
                ttl_data = cache_data["ttl"]
                cache_data["ttl"] = CacheTTLSettings(
                    window_1h=ttl_data.get("1h", 60),
                    window_24h=ttl_data.get("24h", 300),
                    window_7d=ttl_data.get("7d", 600),
                    default=ttl_data.get("default", 120),
                )
            if "memory" in cache_data:
                cache_data["memory"] = _parse_dataclass(CacheMemorySettings, cache_data["memory"])
            if "redis" in cache_data:
                cache_data["redis"] = _parse_dataclass(CacheRedisSettings, cache_data["redis"])
            config.cache = _parse_dataclass(CacheSettings, cache_data)

        # Parse retention settings
        if "retention" in data:
            ret_data = data["retention"]
            if "cleanup" in ret_data:
                ret_data["cleanup"] = _parse_dataclass(
                    RetentionCleanupSettings, ret_data["cleanup"]
                )
            if "archive" in ret_data:
                ret_data["archive"] = _parse_dataclass(
                    RetentionArchiveSettings, ret_data["archive"]
                )
            config.retention = _parse_dataclass(RetentionSettings, ret_data)

        # Parse database settings
        if "database" in data:
            db_data = data["database"]
            if "pool" in db_data:
                db_data["pool"] = _parse_dataclass(DatabasePoolSettings, db_data["pool"])
            config.database = _parse_dataclass(DatabaseSettings, db_data)

        # Parse span settings
        if "spans" in data:
            config.spans = _parse_dataclass(SpanSettings, data["spans"])

        # Parse agent tier settings
        if "agent_tiers" in data:
            config.agent_tiers = {}
            for tier_name, tier_data in data["agent_tiers"].items():
                config.agent_tiers[tier_name] = _parse_dataclass(AgentTierSettings, tier_data)

        # Parse logging settings
        if "logging" in data:
            log_data = data["logging"]
            if "components" in log_data:
                log_data["components"] = _parse_dataclass(
                    LoggingComponentSettings, log_data["components"]
                )
            config.logging = _parse_dataclass(LoggingSettings, log_data)

        return config

    def get_agent_tier(self, agent_name: str) -> Optional[AgentTierSettings]:
        """Get tier settings for an agent."""
        for _tier_name, tier_settings in self.agent_tiers.items():
            if agent_name in tier_settings.agents:
                return tier_settings
        return None

    def get_sample_rate(self, agent_name: str) -> float:
        """Get sample rate for an agent."""
        # Check agent tier settings first
        tier = self.get_agent_tier(agent_name)
        if tier:
            return tier.sample_rate

        # Fall back to sampling settings
        return self.sampling.get_rate(agent_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "opik": self.opik.to_dict(),
            "sampling": {
                "default_rate": self.sampling.default_rate,
                "production_rate": self.sampling.production_rate,
                "always_sample_errors": self.sampling.always_sample_errors,
            },
            "batching": {
                "enabled": self.batching.enabled,
                "max_batch_size": self.batching.max_batch_size,
                "max_wait_seconds": self.batching.max_wait_seconds,
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker.enabled,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "reset_timeout_seconds": self.circuit_breaker.reset_timeout_seconds,
            },
            "cache": {
                "backend": self.cache.backend,
                "key_prefix": self.cache.key_prefix,
            },
            "config_path": self._config_path,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _parse_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Parse dictionary into dataclass, ignoring unknown fields."""
    if data is None:
        return cls()

    # Get valid field names
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to valid fields only
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return cls(**filtered_data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_observability_config: Optional[ObservabilityConfig] = None


def get_observability_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
    force_reload: bool = False,
) -> ObservabilityConfig:
    """Get singleton ObservabilityConfig instance.

    Args:
        config_path: Path to config file (only used on first call or force_reload)
        environment: Environment name (only used on first call or force_reload)
        force_reload: Force reload configuration from file

    Returns:
        ObservabilityConfig singleton instance
    """
    global _observability_config

    if _observability_config is None or force_reload:
        _observability_config = ObservabilityConfig.from_yaml(config_path, environment)

    return _observability_config


def reset_observability_config() -> None:
    """Reset singleton instance (for testing)."""
    global _observability_config
    _observability_config = None
