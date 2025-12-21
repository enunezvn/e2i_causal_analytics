"""
E2I Agentic Memory - Service Factories
Factory functions for all external service dependencies with connection pooling and error handling.

Usage:
    from src.memory.services.factories import (
        get_redis_client,
        get_supabase_client,
        get_falkordb_client,
        get_embedding_service,
        get_llm_service,
        get_graphiti_service,
    )

    # Get clients (cached/pooled)
    redis = get_redis_client()
    supabase = get_supabase_client()
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
from typing import Dict, List, Optional

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
            except ImportError:
                raise ServiceConnectionError(
                    "OpenAI", "openai package is not installed. Run: pip install openai"
                )
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
            raise ServiceConnectionError("OpenAI", f"Failed to generate embedding: {e}", e)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        try:
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ServiceConnectionError("OpenAI", f"Failed to generate batch embeddings: {e}", e)


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
            except ImportError:
                raise ServiceConnectionError(
                    "Bedrock", "boto3 package is not installed. Run: pip install boto3"
                )
            except Exception as e:
                raise ServiceConnectionError("Bedrock", f"Failed to create Bedrock client: {e}", e)
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Bedrock."""
        client = self._get_client()
        try:
            response = client.invoke_model(modelId=self.model, body=json.dumps({"inputText": text}))
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            raise ServiceConnectionError("Bedrock", f"Failed to generate embedding: {e}", e)

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
            except ImportError:
                raise ServiceConnectionError(
                    "Anthropic", "anthropic package is not installed. Run: pip install anthropic"
                )
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
            raise ServiceConnectionError("Anthropic", f"Failed to generate completion: {e}", e)


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
            except ImportError:
                raise ServiceConnectionError(
                    "Bedrock", "boto3 package is not installed. Run: pip install boto3"
                )
            except Exception as e:
                raise ServiceConnectionError("Bedrock", f"Failed to create Bedrock client: {e}", e)
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
            raise ServiceConnectionError("Bedrock", f"Failed to generate completion: {e}", e)


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
    except ImportError:
        raise ServiceConnectionError(
            "Redis", "redis package is not installed. Run: pip install redis"
        )

    url = os.environ.get("REDIS_URL", "redis://localhost:6382")
    logger.info(f"Creating Redis client for: {url.split('@')[-1]}")  # Hide auth in logs

    try:
        _redis_client = redis.from_url(url, decode_responses=True)
        return _redis_client
    except Exception as e:
        raise ServiceConnectionError("Redis", f"Failed to create client: {e}", e)


_supabase_client = None


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
    except ImportError:
        raise ServiceConnectionError(
            "Supabase", "supabase package is not installed. Run: pip install supabase"
        )

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
        raise ServiceConnectionError("Supabase", f"Failed to create client: {e}", e)


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
    except ImportError:
        raise ServiceConnectionError(
            "FalkorDB", "falkordb package is not installed. Run: pip install falkordb"
        )

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6381"))  # 6381 external (e2i), 6379 internal

    logger.info(f"Creating FalkorDB client for: {host}:{port}")

    try:
        _falkordb_client = FalkorDB(host=host, port=port)
        return _falkordb_client
    except Exception as e:
        raise ServiceConnectionError("FalkorDB", f"Failed to create client: {e}", e)


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


@lru_cache(maxsize=1)
def get_llm_service(environment: Optional[str] = None) -> LLMService:
    """
    Get LLM service based on environment.

    Args:
        environment: "local_pilot" or "aws_production". If not provided,
                     uses E2I_ENVIRONMENT env var or defaults to "local_pilot".

    Returns:
        LLMService: Anthropic or Bedrock LLM service
    """
    if environment is None:
        environment = os.environ.get("E2I_ENVIRONMENT", "local_pilot")

    logger.info(f"Creating LLM service for environment: {environment}")

    if environment == "aws_production":
        return BedrockLLMService()
    else:
        return AnthropicLLMService()


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
        raise ServiceConnectionError("Graphiti", f"Failed to import graphiti_service: {e}")
    except Exception as e:
        raise ServiceConnectionError("Graphiti", f"Failed to get Graphiti service: {e}", e)


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
    global _redis_client, _supabase_client, _falkordb_client

    _redis_client = None
    _supabase_client = None
    _falkordb_client = None

    # Clear LRU caches
    get_embedding_service.cache_clear()
    get_llm_service.cache_clear()

    logger.info("All service clients have been reset")
