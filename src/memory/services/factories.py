"""
E2I Agentic Memory - Service Factories
Factory functions for all external service dependencies with connection pooling and error handling.

Usage:
    from src.memory.services.factories import (
        get_redis_client,
        get_supabase_client,
        get_async_supabase_client,
        get_async_supabase_service_client,  # For elevated permissions (bypasses RLS)
        get_falkordb_client,
        get_embedding_service,
        get_llm_service,
        get_graphiti_service,
    )

    # Get clients (cached/pooled)
    redis = get_redis_client()
    supabase = get_supabase_client()
    async_supabase = await get_async_supabase_client()  # For async contexts (anon key)
    async_service = await get_async_supabase_service_client()  # For internal ops (service_role key)
    falkordb = get_falkordb_client()

    # Get services
    embedding_service = get_embedding_service()
    llm_service = get_llm_service()

    # Get Graphiti service (async)
    graphiti = await get_graphiti_service()
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


class ServiceConnectionError(Exception):
    """Raised when a service connection cannot be established."""

    def __init__(self, service: str, message: str, original_error: Optional[Exception] = None):
        self.service = service
        self.original_error = original_error
        super().__init__(f"[{service}] {message}")


# ============================================================================
# EMBEDDING SERVICE ABSTRACTION
# ============================================================================


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embeddings for local pilot environment."""

    def __init__(self, model: str = "text-embedding-ada-002"):
        self._client = None
        self.model = model
        self._cache: Dict[int, List[float]] = {}

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ServiceConnectionError(
                        "OpenAI", "OPENAI_API_KEY environment variable is not set"
                    )
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError as e:
                raise ServiceConnectionError(
                    "OpenAI", "openai package is not installed. Run: pip install openai"
                ) from e
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        client = self._get_client()
        try:
            response = client.embeddings.create(model=self.model, input=text)
            embedding = response.data[0].embedding
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            raise ServiceConnectionError("OpenAI", f"Failed to generate embedding: {e}", e) from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        try:
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ServiceConnectionError("OpenAI", f"Failed to generate batch embeddings: {e}", e) from e


class BedrockEmbeddingService(EmbeddingService):
    """AWS Bedrock embeddings for production environment."""

    def __init__(self, model: str = "amazon.titan-embed-text-v1", region: str = "us-east-1"):
        self._client = None
        self.model = model
        self.region = region

    def _get_client(self):
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("bedrock-runtime", region_name=self.region)
            except ImportError as e:
                raise ServiceConnectionError(
                    "Bedrock", "boto3 package is not installed. Run: pip install boto3"
                ) from e
            except Exception as e:
                raise ServiceConnectionError("Bedrock", f"Failed to create Bedrock client: {e}", e) from e
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Bedrock."""
        client = self._get_client()
        try:
            response = client.invoke_model(modelId=self.model, body=json.dumps({"inputText": text}))
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            raise ServiceConnectionError("Bedrock", f"Failed to generate embedding: {e}", e) from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (sequential for Bedrock)."""
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings


# ============================================================================
# LLM SERVICE ABSTRACTION
# ============================================================================


class LLMService(ABC):
    """Abstract base class for LLM services."""

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a completion for the given prompt."""
        pass


class AnthropicLLMService(LLMService):
    """Anthropic Claude for local pilot environment."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        self._client = None
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ServiceConnectionError(
                        "Anthropic", "ANTHROPIC_API_KEY environment variable is not set"
                    )
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError as e:
                raise ServiceConnectionError(
                    "Anthropic", "anthropic package is not installed. Run: pip install anthropic"
                ) from e
        return self._client

    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate completion using Claude."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise ServiceConnectionError("Anthropic", f"Failed to generate completion: {e}", e) from e


class OpenAILLMService(LLMService):
    """OpenAI GPT for development/testing (cheaper alternative to Claude)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        self._client = None
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ServiceConnectionError(
                        "OpenAI", "OPENAI_API_KEY environment variable is not set"
                    )
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError as e:
                raise ServiceConnectionError(
                    "OpenAI", "openai package is not installed. Run: pip install openai"
                ) from e
        return self._client

    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate completion using OpenAI."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise ServiceConnectionError("OpenAI", f"Failed to generate completion: {e}", e) from e


class BedrockLLMService(LLMService):
    """AWS Bedrock Claude for production environment."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens: int = 4096,
        region: str = "us-east-1",
    ):
        self._client = None
        self.model = model
        self.max_tokens = max_tokens
        self.region = region

    def _get_client(self):
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("bedrock-runtime", region_name=self.region)
            except ImportError as e:
                raise ServiceConnectionError(
                    "Bedrock", "boto3 package is not installed. Run: pip install boto3"
                ) from e
            except Exception as e:
                raise ServiceConnectionError("Bedrock", f"Failed to create Bedrock client: {e}", e) from e
        return self._client

    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate completion using Bedrock Claude."""
        client = self._get_client()
        try:
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens or self.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ),
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            raise ServiceConnectionError("Bedrock", f"Failed to generate completion: {e}", e) from e


# ============================================================================
# CLIENT FACTORIES
# ============================================================================


_redis_client = None


def get_redis_client():
    """
    Get Redis client for working memory.

    Uses REDIS_URL environment variable or defaults to redis://localhost:6379.
    Returns a cached client for connection reuse.

    Returns:
        redis.asyncio.Redis: Async Redis client

    Raises:
        ServiceConnectionError: If Redis connection fails
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as redis
    except ImportError as e:
        raise ServiceConnectionError(
            "Redis", "redis package is not installed. Run: pip install redis"
        ) from e

    url = os.environ.get("REDIS_URL", "redis://localhost:6382")
    logger.info(f"Creating Redis client for: {url.split('@')[-1]}")  # Hide auth in logs

    try:
        _redis_client = redis.from_url(url, decode_responses=True)
        return _redis_client
    except Exception as e:
        raise ServiceConnectionError("Redis", f"Failed to create client: {e}", e) from e


_supabase_client = None
_async_supabase_client = None
_async_supabase_service_client = None


def get_supabase_client():
    """
    Get Supabase client for episodic/procedural memory.

    Requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables.
    Returns a cached client for connection reuse.

    Returns:
        supabase.Client: Supabase client

    Raises:
        ServiceConnectionError: If required environment variables are missing or connection fails
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    try:
        from supabase import create_client
    except ImportError as e:
        raise ServiceConnectionError(
            "Supabase", "supabase package is not installed. Run: pip install supabase"
        ) from e

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url:
        raise ServiceConnectionError("Supabase", "SUPABASE_URL environment variable is not set")
    if not key:
        raise ServiceConnectionError(
            "Supabase", "SUPABASE_ANON_KEY environment variable is not set"
        )

    logger.info(f"Creating Supabase client for: {url}")

    try:
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        raise ServiceConnectionError("Supabase", f"Failed to create client: {e}", e) from e


async def get_async_supabase_client():
    """
    Get async Supabase client for use in async contexts.

    This is the async version of get_supabase_client() for use in async functions
    like LangGraph nodes and tool handlers. Use this when you need to await
    Supabase operations.

    Requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables.
    Returns a cached client for connection reuse.

    Returns:
        AsyncClient: Async Supabase client

    Raises:
        ServiceConnectionError: If required environment variables are missing or connection fails

    Example:
        async def my_async_function():
            client = await get_async_supabase_client()
            result = await client.table("messages").select("*").execute()
    """
    global _async_supabase_client
    if _async_supabase_client is not None:
        return _async_supabase_client

    try:
        from supabase import acreate_client
    except ImportError as e:
        raise ServiceConnectionError(
            "Supabase", "supabase package is not installed. Run: pip install supabase"
        ) from e

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url:
        raise ServiceConnectionError("Supabase", "SUPABASE_URL environment variable is not set")
    if not key:
        raise ServiceConnectionError(
            "Supabase", "SUPABASE_ANON_KEY environment variable is not set"
        )

    logger.info(f"Creating async Supabase client for: {url}")

    try:
        _async_supabase_client = await acreate_client(url, key)
        return _async_supabase_client
    except Exception as e:
        raise ServiceConnectionError("Supabase", f"Failed to create async client: {e}", e) from e


async def get_async_supabase_service_client():
    """
    Get async Supabase client with service role key for elevated permissions.

    This client bypasses Row Level Security (RLS) and should be used for
    internal operations like training signal collection, analytics, and
    background jobs that need full table access.

    Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.
    Falls back to SUPABASE_ANON_KEY if service role key is not available.
    Returns a cached client for connection reuse.

    Returns:
        AsyncClient: Async Supabase client with service role permissions

    Raises:
        ServiceConnectionError: If required environment variables are missing or connection fails

    Example:
        async def persist_training_signal(signal_data):
            client = await get_async_supabase_service_client()
            result = await client.table("chatbot_training_signals").insert(signal_data).execute()
    """
    global _async_supabase_service_client
    if _async_supabase_service_client is not None:
        return _async_supabase_service_client

    try:
        from supabase import acreate_client
    except ImportError as e:
        raise ServiceConnectionError(
            "Supabase", "supabase package is not installed. Run: pip install supabase"
        ) from e

    url = os.environ.get("SUPABASE_URL")
    # Prefer service role key, fall back to anon key
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url:
        raise ServiceConnectionError("Supabase", "SUPABASE_URL environment variable is not set")
    if not key:
        raise ServiceConnectionError(
            "Supabase", "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY environment variable is not set"
        )

    key_type = "service_role" if os.environ.get("SUPABASE_SERVICE_ROLE_KEY") else "anon"
    logger.info(f"Creating async Supabase service client for: {url} (using {key_type} key)")

    try:
        _async_supabase_service_client = await acreate_client(url, key)
        return _async_supabase_service_client
    except Exception as e:
        raise ServiceConnectionError("Supabase", f"Failed to create async service client: {e}", e) from e


_falkordb_client = None


def get_falkordb_client():
    """
    Get FalkorDB client for semantic memory (graph database).

    Uses FALKORDB_HOST and FALKORDB_PORT environment variables or defaults.
    Returns a cached client for connection reuse.

    Returns:
        FalkorDB: FalkorDB client

    Raises:
        ServiceConnectionError: If connection fails
    """
    global _falkordb_client
    if _falkordb_client is not None:
        return _falkordb_client

    try:
        from falkordb import FalkorDB
    except ImportError as e:
        raise ServiceConnectionError(
            "FalkorDB", "falkordb package is not installed. Run: pip install falkordb"
        ) from e

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6381"))  # 6381 external (e2i), 6379 internal

    logger.info(f"Creating FalkorDB client for: {host}:{port}")

    try:
        _falkordb_client = FalkorDB(host=host, port=port)
        return _falkordb_client
    except Exception as e:
        raise ServiceConnectionError("FalkorDB", f"Failed to create client: {e}", e) from e


# ============================================================================
# SERVICE FACTORIES
# ============================================================================


@lru_cache(maxsize=1)
def get_embedding_service(environment: Optional[str] = None) -> EmbeddingService:
    """
    Get embedding service based on environment.

    Args:
        environment: "local_pilot" or "aws_production". If not provided,
                     uses E2I_ENVIRONMENT env var or defaults to "local_pilot".

    Returns:
        EmbeddingService: OpenAI or Bedrock embedding service
    """
    if environment is None:
        environment = os.environ.get("E2I_ENVIRONMENT", "local_pilot")

    logger.info(f"Creating embedding service for environment: {environment}")

    if environment == "aws_production":
        return BedrockEmbeddingService()
    else:
        return OpenAIEmbeddingService()


# Type alias for LLM providers
LLMProvider = Literal["anthropic", "openai", "bedrock"]


def _get_llm_provider() -> LLMProvider:
    """
    Get the configured LLM provider from environment.

    Returns:
        LLMProvider: "anthropic", "openai", or "bedrock"
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider not in ("anthropic", "openai", "bedrock"):
        logger.warning(f"Unknown LLM_PROVIDER '{provider}', defaulting to 'anthropic'")
        return "anthropic"
    return provider  # type: ignore


@lru_cache(maxsize=4)
def _get_llm_service_cached(environment: str, provider: str) -> LLMService:
    """
    Internal cached factory - called with resolved values.

    Cache key includes both environment AND provider to ensure correct
    service is returned when either changes.

    Args:
        environment: Resolved environment string
        provider: Resolved provider string

    Returns:
        LLMService: Appropriate LLM service instance
    """
    logger.info(f"Creating LLM service: environment={environment}, provider={provider}")

    if environment == "aws_production":
        return BedrockLLMService()
    elif provider == "openai":
        return OpenAILLMService()
    else:
        return AnthropicLLMService()


def get_llm_service(
    environment: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
) -> LLMService:
    """
    Get LLM service based on environment and provider preference.

    Args:
        environment: "local_pilot" or "aws_production". Defaults to E2I_ENVIRONMENT env var.
        provider: "anthropic", "openai", or "bedrock". Defaults to LLM_PROVIDER env var.

    Provider Precedence:
        1. Explicit `provider` argument (highest priority)
        2. LLM_PROVIDER environment variable
        3. Default: "anthropic"

    Environment Variables:
        LLM_PROVIDER: "anthropic" (default) or "openai"
            - Set to "openai" for cheaper gpt-4o-mini during development
            - Set to "anthropic" for Claude in production

    Returns:
        LLMService: OpenAI, Anthropic, or Bedrock LLM service

    Example (testing - no patching needed):
        service = get_llm_service(provider="anthropic")

    Example (production - reads from env):
        service = get_llm_service()
    """
    # Resolve environment
    resolved_env = environment or os.environ.get("E2I_ENVIRONMENT", "local_pilot")

    # Resolve provider with precedence: arg > env > default
    resolved_provider = provider or _get_llm_provider()

    # Production override: aws_production always uses Bedrock
    if resolved_env == "aws_production":
        resolved_provider = "bedrock"

    return _get_llm_service_cached(resolved_env, resolved_provider)


# ============================================================================
# GRAPHITI SERVICE FACTORY
# ============================================================================


async def get_graphiti_service():
    """
    Get the Graphiti service for knowledge graph operations.

    This is an async factory that returns the initialized Graphiti service
    singleton. The service provides:
    - Automatic entity/relationship extraction from text
    - Temporal episode tracking
    - Graph search and traversal
    - Integration with FalkorDB semantic memory

    Returns:
        E2IGraphitiService: Initialized Graphiti service

    Raises:
        ServiceConnectionError: If Graphiti initialization fails

    Example:
        service = await get_graphiti_service()
        result = await service.add_episode(
            content="Dr. Smith prescribed Remibrutinib",
            source="orchestrator",
            session_id="session-123"
        )
    """
    try:
        from ..graphiti_service import get_graphiti_service as _get_graphiti_service

        return await _get_graphiti_service()
    except ImportError as e:
        raise ServiceConnectionError("Graphiti", f"Failed to import graphiti_service: {e}") from e
    except Exception as e:
        raise ServiceConnectionError("Graphiti", f"Failed to get Graphiti service: {e}", e) from e


# ============================================================================
# CONNECTION TESTING
# ============================================================================


async def test_redis_connection() -> bool:
    """
    Test Redis connection.

    Returns:
        bool: True if connection successful
    """
    try:
        redis = get_redis_client()
        await redis.ping()
        logger.info("Redis connection: OK")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


async def test_supabase_connection() -> bool:
    """
    Test Supabase connection.

    Returns:
        bool: True if connection successful
    """
    try:
        supabase = get_supabase_client()
        # Try a simple query to verify connection
        supabase.table("episodic_memories").select("id").limit(1).execute()
        logger.info("Supabase connection: OK")
        return True
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return False


def test_falkordb_connection() -> bool:
    """
    Test FalkorDB connection.

    Returns:
        bool: True if connection successful
    """
    try:
        falkordb = get_falkordb_client()
        # Try to get graph list to verify connection
        falkordb.list_graphs()
        logger.info("FalkorDB connection: OK")
        return True
    except Exception as e:
        logger.error(f"FalkorDB connection failed: {e}")
        return False


async def test_all_connections() -> Dict[str, bool]:
    """
    Test all service connections.

    Returns:
        Dict[str, bool]: Status of each connection
    """
    results = {
        "redis": await test_redis_connection(),
        "supabase": await test_supabase_connection(),
        "falkordb": test_falkordb_connection(),
    }
    return results


# ============================================================================
# RESET FUNCTIONS (for testing)
# ============================================================================


def reset_all_clients() -> None:
    """Reset all cached clients. Useful for testing."""
    global _redis_client, _supabase_client, _async_supabase_client, _async_supabase_service_client, _falkordb_client

    _redis_client = None
    _supabase_client = None
    _async_supabase_client = None
    _async_supabase_service_client = None
    _falkordb_client = None

    # Clear LRU caches
    get_embedding_service.cache_clear()
    _get_llm_service_cached.cache_clear()

    logger.info("All service clients have been reset")
