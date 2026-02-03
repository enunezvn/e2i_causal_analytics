"""Rate Limiting Middleware.

Implements request rate limiting to protect the API from abuse.
Supports both in-memory (single instance) and Redis (distributed) backends.
Integrated with security audit logging for compliance.

Author: E2I Causal Analytics Team
Version: 4.2.1
"""

import logging
import os
import time
from collections import defaultdict
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Security audit logging
try:
    from src.utils.security_audit import get_security_audit_service

    _AUDIT_ENABLED = True
except ImportError:
    _AUDIT_ENABLED = False

logger = logging.getLogger(__name__)


class RateLimitBackend:
    """Abstract base for rate limit storage backends."""

    def is_rate_limited(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """Check if request should be rate limited.

        Args:
            key: Unique identifier for the client
            limit: Maximum requests allowed in window
            window: Time window in seconds

        Returns:
            Tuple of (is_limited, remaining_requests)
        """
        raise NotImplementedError


class InMemoryBackend(RateLimitBackend):
    """In-memory rate limit backend for single-instance deployments."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_rate_limited(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.time()
        window_start = now - window

        # Clean old requests
        self._requests[key] = [t for t in self._requests[key] if t > window_start]

        # Check limit
        current_count = len(self._requests[key])
        if current_count >= limit:
            return True, 0

        # Record this request
        self._requests[key].append(now)
        return False, limit - current_count - 1


class RedisBackend(RateLimitBackend):
    """Redis-based rate limit backend for distributed deployments."""

    def __init__(self, redis_url: str | None = None):
        self._redis = None
        self._enabled = False

        try:
            import redis

            url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
            self._redis = redis.from_url(url, decode_responses=True)
            self._redis.ping()
            self._enabled = True
            logger.info(f"Rate limit Redis backend connected: {url}")
        except Exception as e:
            logger.warning(f"Redis rate limit backend unavailable: {e}. Using in-memory fallback.")
            self._fallback = InMemoryBackend()

    def is_rate_limited(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        if not self._enabled:
            return self._fallback.is_rate_limited(key, limit, window)

        try:
            redis_key = f"ratelimit:{key}"
            pipe = self._redis.pipeline()

            # Increment counter
            pipe.incr(redis_key)
            # Set expiry if new key
            pipe.expire(redis_key, window)
            # Get current value
            results = pipe.execute()

            current_count = results[0]
            if current_count > limit:
                return True, 0

            return False, limit - current_count
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}. Allowing request.")
            return False, limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces request rate limits.

    Features:
    - Configurable limits per endpoint pattern
    - Client identification by IP or API key
    - Redis backend for distributed rate limiting
    - Graceful fallback to in-memory when Redis unavailable
    """

    # Default rate limits (requests per minute)
    DEFAULT_LIMITS = {
        "default": (100, 60),  # 100 requests per minute
        "health": (300, 60),  # 300 requests per minute for health checks
        "auth": (20, 60),  # 20 requests per minute for auth endpoints
        "calculate": (30, 60),  # 30 requests per minute for calculations
        "batch": (10, 60),  # 10 requests per minute for batch operations
    }

    # Paths that are exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/api/kpis/health",  # KPI system health check (polled by frontend)
    }

    # Path prefixes that are exempt from rate limiting
    EXEMPT_PREFIXES = (
        "/api/copilotkit",  # CopilotKit AI assistant - needs frequent requests
    )

    # IPs exempt from rate limiting (internal/self traffic)
    EXEMPT_IPS = {
        "127.0.0.1",
        "::1",
        "localhost",
        "138.197.4.36",  # Droplet's own public IP (self-traffic via nginx)
    }

    def __init__(
        self,
        app: Callable,
        use_redis: bool = True,
        redis_url: str | None = None,
        default_limit: int | None = None,
        default_window: int | None = None,
    ):
        """Initialize rate limit middleware.

        Args:
            app: The ASGI application
            use_redis: Whether to use Redis backend (falls back to in-memory if unavailable)
            redis_url: Redis connection URL
            default_limit: Default request limit (overrides DEFAULT_LIMITS["default"])
            default_window: Default time window in seconds
        """
        super().__init__(app)

        if use_redis:
            self._backend = RedisBackend(redis_url)
        else:
            self._backend = InMemoryBackend()

        # Allow override of default limits via environment
        env_limit = os.environ.get("RATE_LIMIT_DEFAULT")
        env_window = os.environ.get("RATE_LIMIT_WINDOW")

        if default_limit or env_limit:
            limit = default_limit or int(env_limit)
            window = default_window or (int(env_window) if env_window else 60)
            self.DEFAULT_LIMITS["default"] = (limit, window)

        logger.info(
            f"Rate limiting enabled: {self.DEFAULT_LIMITS['default'][0]} req/{self.DEFAULT_LIMITS['default'][1]}s"
        )

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request.

        Args:
            request: The incoming request

        Returns:
            Unique client identifier
        """
        # Check for API key in header first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:16]}"

        # Check for authenticated user
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP address — prefer X-Real-IP (set by nginx to $remote_addr, not spoofable)
        ip = request.headers.get("X-Real-IP")
        if not ip:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _get_limit_for_path(self, path: str, method: str) -> tuple[int, int]:
        """Get rate limit for a given path.

        Args:
            path: Request path
            method: HTTP method

        Returns:
            Tuple of (limit, window_seconds)
        """
        # Check for specific patterns
        if "/health" in path:
            return self.DEFAULT_LIMITS["health"]
        if "/auth" in path or "/login" in path:
            return self.DEFAULT_LIMITS["auth"]
        if "/batch" in path:
            return self.DEFAULT_LIMITS["batch"]
        if "/calculate" in path or method == "POST":
            return self.DEFAULT_LIMITS["calculate"]

        return self.DEFAULT_LIMITS["default"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response, or 429 if rate limited
        """
        path = request.url.path

        # Skip rate limiting for exempt paths
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip rate limiting for exempt path prefixes
        if path.startswith(self.EXEMPT_PREFIXES):
            return await call_next(request)

        # Skip rate limiting for internal/self IPs
        client_ip = request.headers.get("X-Real-IP") or (
            request.client.host if request.client else None
        )
        if client_ip in self.EXEMPT_IPS:
            return await call_next(request)

        # Get client identifier and limits
        client_key = self._get_client_key(request)
        limit, window = self._get_limit_for_path(path, request.method)

        # Create a unique key combining client and endpoint category
        rate_key = f"{client_key}:{path.split('/')[1] if '/' in path[1:] else 'root'}"

        # Check rate limit
        is_limited, remaining = self._backend.is_rate_limited(rate_key, limit, window)

        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_key} on {path}")

            # Log security audit event for rate limit exceeded
            if _AUDIT_ENABLED:
                # Extract IP from client_key
                client_ip = client_key.split(":")[-1] if ":" in client_key else client_key
                request_id = getattr(request.state, "request_id", None)
                user_id = getattr(request.state, "user_id", None)

                audit = get_security_audit_service()
                audit.log_rate_limit_blocked(
                    client_ip=client_ip,
                    endpoint=path,
                    block_duration_seconds=window,
                    user_id=user_id,
                    request_id=request_id,
                )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
                    "retry_after": window,
                },
                headers={
                    "Retry-After": str(window),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + window),
                },
            )

        # Process request and add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)

        return response
