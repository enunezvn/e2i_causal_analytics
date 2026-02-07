"""Security Headers Middleware.

Adds security headers to all API responses to protect against common
web vulnerabilities (XSS, clickjacking, MIME sniffing, etc.).

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import os
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking attacks
    - X-XSS-Protection: Legacy XSS protection for older browsers
    - Strict-Transport-Security: Enforces HTTPS (when enabled)
    - Content-Security-Policy: Controls resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    """

    def __init__(
        self,
        app: Callable,
        enable_hsts: bool | None = None,
        hsts_max_age: int = 31536000,  # 1 year
        csp_policy: str | None = None,
    ):
        """Initialize security headers middleware.

        Args:
            app: The ASGI application
            enable_hsts: Enable HSTS header. If None, auto-detects from environment.
            hsts_max_age: HSTS max-age in seconds (default: 1 year)
            csp_policy: Custom Content-Security-Policy. If None, uses default.
        """
        super().__init__(app)

        # Auto-detect HSTS from environment if not specified
        if enable_hsts is None:
            self.enable_hsts = os.environ.get("ENABLE_HSTS", "true").lower() == "true"
        else:
            self.enable_hsts = enable_hsts

        self.hsts_max_age = hsts_max_age

        # Default CSP - restrictive but allows API functionality
        self.csp_policy = csp_policy or self._default_csp()

    def _default_csp(self) -> str:
        """Generate default Content-Security-Policy for API.

        Returns:
            CSP header value
        """
        # API-focused CSP - allows self and common needs
        directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles for API responses
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",  # Prevent framing
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(directives)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers to response.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response with security headers added
        """
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking - deny all framing
        response.headers["X-Frame-Options"] = "DENY"

        # Legacy XSS protection for older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Restrict browser features/permissions
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )

        # Content Security Policy (skip for docs — nginx provides a CDN-permissive CSP)
        docs_paths = ("/api/docs", "/api/redoc", "/api/openapi.json")
        if not request.url.path.startswith(docs_paths):
            response.headers["Content-Security-Policy"] = self.csp_policy

        # HSTS - only enable in production with HTTPS
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        # Prevent caching of sensitive responses
        if request.url.path.startswith("/api/") and request.method != "GET":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response
