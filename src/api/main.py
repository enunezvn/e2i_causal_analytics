"""
E2I Causal Analytics - FastAPI Application
==========================================

Main FastAPI application with 18-agent orchestration layer.

Components:
-----------
- Health checks and monitoring
- API routers for different services
- Tri-memory system integration
- MLOps service connections

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import middleware
from src.api.dependencies.auth import is_auth_enabled

# Import dependencies
from src.api.dependencies.bentoml_client import (
    close_bentoml_client,
    configure_bentoml_endpoints,
    get_bentoml_client,
)
from src.api.dependencies.falkordb_client import (
    close_falkordb,
    falkordb_health_check,
    init_falkordb,
)

# Import OpenTelemetry configuration (Phase 1 G02)
from src.api.dependencies.opentelemetry_config import (
    OTEL_ENABLED,
    init_opentelemetry,
    instrument_fastapi,
    shutdown_opentelemetry,
)
from src.api.dependencies.redis_client import (
    close_redis,
    init_redis,
    redis_health_check,
)
from src.api.dependencies.supabase_client import (
    close_supabase,
    init_supabase,
    supabase_health_check,
)
from src.api.middleware.auth_middleware import JWTAuthMiddleware
from src.api.middleware.rate_limit_middleware import RateLimitMiddleware
from src.api.middleware.security_middleware import SecurityHeadersMiddleware
from src.api.middleware.timing import TimingMiddleware
from src.api.middleware.tracing import TracingMiddleware

# Import routers
from src.api.routes.agents import router as agents_router
from src.api.routes.analytics import router as analytics_router
from src.api.routes.audit import router as audit_router
from src.api.routes.causal import router as causal_router
from src.api.routes.cognitive import router as cognitive_router
from src.api.routes.copilotkit import add_copilotkit_routes
from src.api.routes.copilotkit import router as copilotkit_router
from src.api.routes.digital_twin import router as digital_twin_router
from src.api.routes.experiments import router as experiments_router
from src.api.routes.explain import router as explain_router
from src.api.routes.feedback import router as feedback_router
from src.api.routes.gaps import router as gaps_router
from src.api.routes.graph import router as graph_router
from src.api.routes.health_score import router as health_score_router
from src.api.routes.kpi import router as kpi_router
from src.api.routes.memory import router as memory_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.monitoring import router as monitoring_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.rag import router as rag_router
from src.api.routes.resource_optimizer import router as resource_optimizer_router
from src.api.routes.segments import router as segments_router

# Import MLOps connectors
from src.feature_store.feast_client import FeastClient
from src.mlops.mlflow_connector import get_mlflow_connector
from src.mlops.opik_connector import get_opik_connector

# Configure structured logging (G14 - Observability)
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# SENTRY ERROR TRACKING INITIALIZATION
# =============================================================================

# Initialize Sentry SDK as early as possible to capture all exceptions
# Quick Win QW2 from observability audit remediation plan
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    SENTRY_DSN = os.environ.get("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
    SENTRY_RELEASE = os.environ.get("SENTRY_RELEASE", "e2i-causal-analytics@4.2.0")

    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=SENTRY_ENVIRONMENT,
            release=SENTRY_RELEASE,
            # Performance monitoring
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            # Profile sampling (requires profiling to be enabled)
            profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            # Integrations
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
                LoggingIntegration(
                    level=logging.INFO,  # Capture INFO and above as breadcrumbs
                    event_level=logging.ERROR,  # Send ERROR and above as events
                ),
            ],
            # Capture 100% of exceptions but sample traces/profiles
            sample_rate=1.0,
            # Don't send PII by default
            send_default_pii=False,
            # Set server name for grouping
            server_name=os.environ.get("HOSTNAME", "e2i-api"),
            # Enable request body capture for debugging (be careful with PII)
            max_request_body_size="medium",
            # Attach stacktrace to all messages
            attach_stacktrace=True,
        )
        logger.info(f"Sentry: ENABLED (env={SENTRY_ENVIRONMENT}, release={SENTRY_RELEASE})")
    else:
        logger.info("Sentry: DISABLED (SENTRY_DSN not set)")
        sentry_sdk = None
except ImportError:
    logger.warning("Sentry: sentry-sdk not installed, error tracking disabled")
    sentry_sdk = None

# =============================================================================
# OPENTELEMETRY INITIALIZATION
# =============================================================================
# Initialize OpenTelemetry at module load time to capture traces from first request
# Phase 1 G02 from observability audit remediation plan

OPENTELEMETRY_INITIALIZED = False
try:
    if OTEL_ENABLED:
        OPENTELEMETRY_INITIALIZED = init_opentelemetry()
        if OPENTELEMETRY_INITIALIZED:
            logger.info("OpenTelemetry: Initialized at module load")
        else:
            logger.info("OpenTelemetry: Initialization returned False")
    else:
        logger.info("OpenTelemetry: Disabled via OTEL_ENABLED=false")
except Exception as e:
    logger.warning(f"OpenTelemetry: Module-level initialization failed: {e}")
    OPENTELEMETRY_INITIALIZED = False

# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    # Startup
    logger.info("=" * 60)
    logger.info("E2I Causal Analytics Platform - Starting")
    logger.info("=" * 60)
    logger.info("Version: 4.1.0")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())

    # Initialize BentoML client (optional - for ML model serving)
    try:
        await get_bentoml_client()
        logger.info("BentoML client initialized successfully")

        # Configure model endpoints from environment or defaults
        configure_bentoml_endpoints(
            {
                "churn_model": os.environ.get("CHURN_MODEL_URL", "http://localhost:3000"),
                "conversion_model": os.environ.get("CONVERSION_MODEL_URL", "http://localhost:3001"),
                "causal_model": os.environ.get("CAUSAL_MODEL_URL", "http://localhost:3002"),
            }
        )
    except Exception as e:
        logger.warning(f"BentoML client initialization failed (non-critical): {e}")

    # Initialize Redis connection pool (required for caching/sessions)
    try:
        await init_redis()
        app.state.redis_available = True
        logger.info("Redis connection pool initialized")
    except Exception as e:
        app.state.redis_available = False
        logger.warning(f"Redis initialization failed (degraded mode): {e}")

    # Initialize FalkorDB client (optional - for knowledge graph)
    try:
        await init_falkordb()
        app.state.falkordb_available = True
        logger.info("FalkorDB client initialized")
    except Exception as e:
        app.state.falkordb_available = False
        logger.warning(f"FalkorDB initialization failed (non-critical): {e}")

    # Initialize Supabase client (required for database)
    try:
        supabase = init_supabase()
        app.state.supabase_available = supabase is not None
        if supabase:
            logger.info("Supabase client initialized")
        else:
            logger.warning("Supabase not configured - database features unavailable")
    except Exception as e:
        app.state.supabase_available = False
        logger.warning(f"Supabase initialization failed (degraded mode): {e}")

    # Initialize MLflow client (experiment tracking)
    try:
        mlflow_connector = get_mlflow_connector()
        app.state.mlflow_available = mlflow_connector.enabled
        if mlflow_connector.enabled:
            logger.info(
                f"MLflow client initialized (tracking URI: {mlflow_connector.tracking_uri})"
            )
        else:
            logger.info("MLflow client initialized in disabled mode")
    except Exception as e:
        app.state.mlflow_available = False
        logger.warning(f"MLflow initialization failed (non-critical): {e}")

    # Initialize Feast client (feature store)
    try:
        feast_client = FeastClient()
        await feast_client.initialize()
        app.state.feast_client = feast_client
        app.state.feast_available = feast_client._initialized
        if feast_client._initialized:
            logger.info("Feast feature store client initialized")
        else:
            logger.info("Feast client initialized with fallback mode")
    except Exception as e:
        app.state.feast_available = False
        app.state.feast_client = None
        logger.warning(f"Feast initialization failed (non-critical): {e}")

    # Initialize Opik client (LLM observability)
    try:
        opik_connector = get_opik_connector()
        app.state.opik_available = opik_connector.is_enabled
        if opik_connector.is_enabled:
            logger.info(
                f"Opik observability client initialized (project: {opik_connector.config.project_name})"
            )
        else:
            logger.info("Opik client initialized in disabled mode")
    except Exception as e:
        app.state.opik_available = False
        logger.warning(f"Opik initialization failed (non-critical): {e}")

    # Set OpenTelemetry state (initialized at module load)
    # Phase 1 G02 from observability audit remediation plan
    app.state.opentelemetry_available = OPENTELEMETRY_INITIALIZED
    if OPENTELEMETRY_INITIALIZED:
        logger.info("OpenTelemetry: Tracing available for this instance")
    else:
        logger.info("OpenTelemetry: Tracing not available")

    logger.info("API server ready to accept connections")

    yield  # Application runs here

    # Shutdown
    logger.info("E2I Causal Analytics Platform - Shutting down")

    # Cleanup BentoML client
    try:
        await close_bentoml_client()
        logger.info("BentoML client closed")
    except Exception as e:
        logger.warning(f"BentoML client cleanup failed: {e}")

    # Cleanup Redis connections
    try:
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Redis cleanup failed: {e}")

    # Cleanup FalkorDB connections
    try:
        await close_falkordb()
        logger.info("FalkorDB connection closed")
    except Exception as e:
        logger.warning(f"FalkorDB cleanup failed: {e}")

    # Cleanup Supabase client
    try:
        close_supabase()
        logger.info("Supabase client closed")
    except Exception as e:
        logger.warning(f"Supabase cleanup failed: {e}")

    # Cleanup Feast client
    try:
        if hasattr(app.state, "feast_client") and app.state.feast_client:
            await app.state.feast_client.close()
            logger.info("Feast client closed")
    except Exception as e:
        logger.warning(f"Feast client cleanup failed: {e}")

    # Flush Opik traces before shutdown
    try:
        opik_connector = get_opik_connector()
        opik_connector.flush()
        logger.info("Opik traces flushed")
    except Exception as e:
        logger.warning(f"Opik flush failed: {e}")

    # Shutdown OpenTelemetry (flush pending spans)
    try:
        shutdown_opentelemetry()
        logger.info("OpenTelemetry shutdown complete")
    except Exception as e:
        logger.warning(f"OpenTelemetry shutdown failed: {e}")

    logger.info("Shutdown complete")


# =============================================================================
# OPENAPI TAG METADATA
# =============================================================================

openapi_tags = [
    {
        "name": "Root",
        "description": "Service discovery and API version information.",
    },
    {
        "name": "Health",
        "description": "Liveness, readiness, and dependency health probes for orchestration.",
    },
    {
        "name": "Agent Orchestration",
        "description": "Manage the 18-agent architecture: status, dispatch, and tier information.",
    },
    {
        "name": "Analytics",
        "description": "Cross-agent analytics dashboards, trend summaries, and aggregated insights.",
    },
    {
        "name": "Audit Chain",
        "description": "Immutable audit trail for every causal decision and agent action.",
    },
    {
        "name": "Causal Inference",
        "description": "Multi-library causal analysis: hierarchical CATE, cross-validation, and pipelines across DoWhy, EconML, CausalML, and NetworkX.",
    },
    {
        "name": "Cognitive Workflow",
        "description": "Multi-step cognitive workflows combining causal reasoning with domain knowledge.",
    },
    {
        "name": "copilotkit",
        "description": "CopilotKit AI chat integration for conversational analytics.",
    },
    {
        "name": "Digital Twin",
        "description": "Patient and HCP digital-twin pre-screening simulations.",
    },
    {
        "name": "A/B Testing",
        "description": "Experiment design, execution, and Bayesian analysis for A/B and multi-arm tests.",
    },
    {
        "name": "Model Interpretability",
        "description": "SHAP, LIME, and counterfactual explanations for ML model predictions.",
    },
    {
        "name": "Feedback Learning",
        "description": "Capture user feedback, compute reward signals, and trigger model retraining.",
    },
    {
        "name": "Gap Analysis",
        "description": "Identify care gaps, ROI opportunities, and unmet-need hotspots.",
    },
    {
        "name": "Knowledge Graph",
        "description": "FalkorDB-backed knowledge graph: nodes, relationships, causal chains, and openCypher queries.",
    },
    {
        "name": "Health Score",
        "description": "Composite health scores for brands, territories, and HCPs.",
    },
    {
        "name": "Memory System",
        "description": "Tri-memory (episodic, semantic, procedural) read/write operations.",
    },
    {
        "name": "Metrics",
        "description": "Prometheus-compatible `/metrics` endpoint for platform observability.",
    },
    {
        "name": "Model Monitoring",
        "description": "Data-drift detection, model performance tracking, and alerting.",
    },
    {
        "name": "Hybrid RAG",
        "description": "Retrieval-augmented generation combining vector search with knowledge-graph context.",
    },
    {
        "name": "Resource Optimization",
        "description": "Optimal resource allocation across territories, brands, and channels.",
    },
    {
        "name": "Segment Analysis",
        "description": "Heterogeneous treatment effect segmentation and targeting optimization.",
    },
    {
        "name": "KPIs",
        "description": "Causal KPI calculation, caching, and threshold monitoring across workstreams.",
    },
    {
        "name": "Model Predictions",
        "description": "Churn, conversion, and custom model inference via BentoML serving layer.",
    },
]

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="E2I Causal Analytics Platform",
    description="21-Agent architecture for causal inference, ML interpretability, and digital twin generation",
    version="4.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
)


# =============================================================================
# CUSTOM OPENAPI SCHEMA
# =============================================================================


def custom_openapi():
    """Extend the default OpenAPI schema with servers, contact, license, and ReDoc tag groups."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=openapi_tags,
    )

    # Server list
    schema["servers"] = [
        {
            "url": "https://eznomics.site",
            "description": "Production",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development",
        },
    ]

    # Contact & license
    schema["info"]["contact"] = {
        "name": "E2I Causal Analytics Team",
        "email": "support@eznomics.site",
    }
    schema["info"]["license"] = {
        "name": "Proprietary",
    }

    # ReDoc x-tagGroups for sidebar organization
    schema["x-tagGroups"] = [
        {
            "name": "Core Analytics",
            "tags": [
                "Causal Inference",
                "KPIs",
                "Model Predictions",
                "Model Interpretability",
                "Analytics",
            ],
        },
        {
            "name": "ML Operations",
            "tags": [
                "A/B Testing",
                "Model Monitoring",
                "Feedback Learning",
            ],
        },
        {
            "name": "Intelligence",
            "tags": [
                "Knowledge Graph",
                "Hybrid RAG",
                "Memory System",
                "Cognitive Workflow",
                "Digital Twin",
                "Gap Analysis",
                "Health Score",
                "Segment Analysis",
                "Resource Optimization",
            ],
        },
        {
            "name": "Platform",
            "tags": [
                "copilotkit",
                "Agent Orchestration",
                "Audit Chain",
            ],
        },
        {
            "name": "Operations",
            "tags": [
                "Root",
                "Health",
                "Metrics",
            ],
        },
    ]

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS middleware for frontend integration
# Production-ready configuration with explicit origins, methods, and headers
_DEFAULT_ORIGINS = [
    # Production
    "http://138.197.4.36",
    "http://138.197.4.36:54321",
    "https://138.197.4.36",
    # Development
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:8080",
]

# Allow override via environment variable (comma-separated list)
# Set ALLOWED_ORIGINS="*" only for development/testing, never in production
_env_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _env_origins:
    if _env_origins == "*":
        if os.environ.get("ENVIRONMENT", "development") == "production":
            raise RuntimeError(
                "CORS: Wildcard origin (*) is not allowed when ENVIRONMENT=production. "
                "Set ALLOWED_ORIGINS to a comma-separated list of specific origins."
            )
        logger.warning("CORS: Wildcard origin (*) configured - this is insecure for production!")
        ALLOWED_ORIGINS = ["*"]
    else:
        # Parse and validate origins from environment
        ALLOWED_ORIGINS = [
            origin.strip()
            for origin in _env_origins.split(",")
            if origin.strip()
            and (origin.strip().startswith("http://") or origin.strip().startswith("https://"))
        ]
        if not ALLOWED_ORIGINS:
            logger.warning("CORS: No valid origins in ALLOWED_ORIGINS env var, using defaults")
            ALLOWED_ORIGINS = _DEFAULT_ORIGINS
else:
    ALLOWED_ORIGINS = _DEFAULT_ORIGINS

# Explicitly define allowed methods (not wildcard)
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

# Explicitly define allowed headers
ALLOWED_HEADERS = [
    "Accept",
    "Accept-Language",
    "Authorization",
    "Content-Language",
    "Content-Type",
    "Origin",
    "X-Requested-With",
    "X-Request-ID",
    "X-Correlation-ID",
]

logger.info(f"CORS: Configured with {len(ALLOWED_ORIGINS)} allowed origins")
logger.debug(f"CORS: Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    expose_headers=["X-Request-ID", "X-Correlation-ID"],
)

# JWT Authentication middleware (Supabase)
# Protects all routes except public paths (health, docs, read-only endpoints)
app.add_middleware(JWTAuthMiddleware)
logger.info(
    f"JWT Authentication: {'ENABLED' if is_auth_enabled() else 'DISABLED (Supabase not configured)'}"
)

# Security Headers middleware
# Adds X-Content-Type-Options, X-Frame-Options, CSP, etc.
app.add_middleware(SecurityHeadersMiddleware)
logger.info("Security Headers: ENABLED")

# Rate Limiting middleware
# Protects API from abuse with configurable limits per endpoint
# Can be disabled for testing via DISABLE_RATE_LIMITING env var
if os.environ.get("DISABLE_RATE_LIMITING", "").lower() not in ("1", "true", "yes"):
    app.add_middleware(RateLimitMiddleware, use_redis=True)
    logger.info("Rate Limiting: ENABLED")
else:
    logger.info("Rate Limiting: DISABLED (DISABLE_RATE_LIMITING set)")

# Request Timing middleware
# Records request latency metrics for Prometheus and adds Server-Timing header
# Quick Win QW3 from observability audit remediation plan
slow_threshold = float(os.environ.get("TIMING_SLOW_THRESHOLD_MS", "1000"))
app.add_middleware(TimingMiddleware, slow_threshold_ms=slow_threshold)
logger.info(f"Request Timing: ENABLED (slow threshold: {slow_threshold}ms)")

# Trace Header Extraction middleware
# Extracts X-Request-ID, X-Correlation-ID, traceparent and makes them available
# Quick Win QW5 from observability audit remediation plan
log_trace = os.environ.get("LOG_TRACE_CONTEXT", "").lower() in ("1", "true", "yes")
app.add_middleware(TracingMiddleware, log_trace_context=log_trace)
logger.info("Request Tracing: ENABLED (W3C Trace Context, X-Request-ID support)")

# OpenTelemetry ASGI instrumentation (outermost layer for accurate timing)
# Phase 1 G02 from observability audit remediation plan
# Added LAST so it wraps all other middleware (middleware is LIFO in Starlette)
if OPENTELEMETRY_INITIALIZED:
    if instrument_fastapi(app):
        logger.info("OpenTelemetry: ASGI instrumentation added to FastAPI")
    else:
        logger.warning("OpenTelemetry: ASGI instrumentation not available")

# =============================================================================
# ROOT ENDPOINTS
# =============================================================================


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "E2I Causal Analytics Platform",
        "version": "4.2.0",
        "status": "online",
        "auth_enabled": is_auth_enabled(),
        "docs": "/api/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for container orchestration.

    Used by:
    - Docker Compose healthcheck
    - Kubernetes liveness/readiness probes
    - Load balancers
    """
    # Check BentoML status (non-blocking, with fallback)
    bentoml_status = "unknown"
    bentoml_message = None
    try:
        client = await get_bentoml_client()
        health = await client.health_check()
        bentoml_status = health.get("status", "unknown")
    except Exception:
        bentoml_status = "unavailable"

    # Add explanatory message when BentoML is not healthy
    if bentoml_status in ("unhealthy", "unavailable", "unknown"):
        bentoml_message = (
            "BentoML model serving is not running. Start with: sudo systemctl start e2i-bentoml"
        )

    # Build components dict
    components = {
        "api": "operational",
        "workers": "available",
        "memory_systems": "connected",
        "bentoml": bentoml_status,
    }

    # Build response
    response = {
        "status": "healthy",
        "service": "e2i-causal-analytics-api",
        "version": "4.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }

    # Add notes for non-critical unhealthy components
    if bentoml_message:
        response["notes"] = {
            "bentoml": bentoml_message,
        }

    return response


@app.get("/healthz", tags=["Health"])
async def healthz() -> Dict[str, str]:
    """Kubernetes-style health check (alias for /health)."""
    return {"status": "ok"}


@app.get("/health/bentoml", tags=["Health"])
async def bentoml_health_check() -> Dict[str, Any]:
    """
    Detailed health check for BentoML model serving endpoints.

    Returns health status for each configured model endpoint.
    """
    try:
        client = await get_bentoml_client()
    except Exception as e:
        return {
            "status": "error",
            "message": f"BentoML client not available: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Check base service health
    base_health = await client.health_check()

    # Check individual model endpoints
    model_endpoints = client.config.model_endpoints
    model_health = {}

    for model_name in model_endpoints:
        try:
            health = await client.health_check(model_name)
            model_health[model_name] = health
        except Exception as e:
            model_health[model_name] = {
                "status": "error",
                "error": str(e),
            }

    # Determine overall status
    all_healthy = base_health.get("status") == "healthy"
    if model_health:
        all_healthy = all_healthy and all(
            h.get("status") == "healthy" for h in model_health.values()
        )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "base_service": base_health,
        "models": model_health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - determines if service can accept traffic.

    Checks:
    - Redis connectivity (required)
    - Supabase connectivity (required)
    - FalkorDB connectivity (optional)

    Returns:
        200 if ready to serve requests
        503 if required dependencies are unavailable
    """
    checks = {}
    all_required_ready = True

    # Check Redis (required)
    try:
        redis_status = await redis_health_check()
        checks["redis"] = redis_status
        if redis_status.get("status") != "healthy":
            all_required_ready = False
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
        all_required_ready = False

    # Check Supabase (required)
    try:
        supabase_status = await supabase_health_check()
        checks["supabase"] = supabase_status
        if supabase_status.get("status") not in ("healthy", "unavailable"):
            # "unavailable" means not configured, which is acceptable in some envs
            if supabase_status.get("status") == "unhealthy":
                all_required_ready = False
    except Exception as e:
        checks["supabase"] = {"status": "unhealthy", "error": str(e)}
        all_required_ready = False

    # Check FalkorDB (optional - doesn't affect readiness)
    try:
        falkordb_status = await falkordb_health_check()
        checks["falkordb"] = falkordb_status
    except Exception as e:
        checks["falkordb"] = {"status": "unavailable", "error": str(e)}

    if all_required_ready:
        return {
            "status": "ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks,
            },
        )


# =============================================================================
# ROUTER REGISTRATION
# =============================================================================
# NOTE: All API endpoints are prefixed with /api for consistency.
# Some routers have /api prefix built-in (kpi, predictions, rag), others get it here.

# Model interpretability endpoints (/api/explain/*)
app.include_router(explain_router, prefix="/api")

# Memory system endpoints (/api/memory/*)
app.include_router(memory_router, prefix="/api")

# Cognitive workflow endpoints (/api/cognitive/*)
app.include_router(cognitive_router, prefix="/api")

# Knowledge graph endpoints (/api/graph/*)
app.include_router(graph_router, prefix="/api")

# Hybrid RAG endpoints (already has /api/v1/rag prefix)
app.include_router(rag_router)

# Model monitoring endpoints (/api/monitoring/*)
app.include_router(monitoring_router, prefix="/api")

# A/B testing & experiment execution endpoints (/api/experiments/*)
app.include_router(experiments_router, prefix="/api")

# Gap analysis & ROI opportunities endpoints (/api/gaps/*)
app.include_router(gaps_router, prefix="/api")

# Segment analysis & heterogeneous optimization endpoints (/api/segments/*)
app.include_router(segments_router, prefix="/api")

# Resource optimization endpoints (/api/resources/*)
app.include_router(resource_optimizer_router, prefix="/api")

# Feedback learning endpoints (/api/feedback/*)
app.include_router(feedback_router, prefix="/api")

# Health score monitoring endpoints (/api/health-score/*)
app.include_router(health_score_router, prefix="/api")

# Digital Twin pre-screening endpoints (/api/digital-twin/*)
app.include_router(digital_twin_router, prefix="/api")

# Model prediction endpoints (already has /api/models prefix)
app.include_router(predictions_router)

# KPI endpoints (already has /api/kpis prefix)
app.include_router(kpi_router)

# Causal inference endpoints (/api/causal/*)
app.include_router(causal_router, prefix="/api")

# Audit chain endpoints (/api/audit/*)
app.include_router(audit_router, prefix="/api")

# Analytics & metrics dashboard endpoints (/api/analytics/*)
app.include_router(analytics_router, prefix="/api")

# CopilotKit status endpoints (/api/copilotkit/*)
app.include_router(copilotkit_router, prefix="/api")

# CopilotKit runtime endpoints (for AI chat)
add_copilotkit_routes(app, prefix="/api/copilotkit")

# Agent orchestration endpoints (/api/agents/*)
app.include_router(agents_router, prefix="/api")

# Prometheus metrics endpoint (/metrics) - no /api prefix per convention
app.include_router(metrics_router)

# TODO: Add additional routers as they're developed:
# - Feature engineering: /api/features
# - Model training: /api/models


# =============================================================================
# ERROR HANDLERS
# =============================================================================

# Import error handling module
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.errors import (
    AuthenticationError,
    AuthorizationError,
    DependencyError,
    E2IError,
    EndpointNotFoundError,
    ErrorSeverity,
    RateLimitError,
    error_response,
    wrap_exception,
)
from src.api.errors import (
    TimeoutError as E2ITimeoutError,
)

# Determine if debug mode is enabled
_debug_requested = os.environ.get("E2I_DEBUG_MODE", "").lower() in ("true", "1", "yes")
_environment = os.environ.get("ENVIRONMENT", "development")
DEBUG_MODE = _debug_requested and _environment != "production"
if _debug_requested and _environment == "production":
    logger.warning("E2I_DEBUG_MODE is set but ENVIRONMENT=production -- debug mode DISABLED")


@app.exception_handler(E2IError)
async def e2i_error_handler(request, exc: E2IError):
    """
    Handle all E2IError subclasses with structured responses.

    Provides detailed error context while controlling what's exposed
    based on error severity and debug mode.

    Integrates with Sentry for CRITICAL and HIGH severity errors.
    """
    # Log based on severity
    if exc.severity == ErrorSeverity.CRITICAL:
        logger.critical(
            f"[{exc.error_id}] {exc.category.value}: {exc.message}",
            exc_info=exc.original_error,
            extra={"error_id": exc.error_id, "path": request.url.path},
        )
        # Capture critical errors to Sentry
        if sentry_sdk:
            sentry_sdk.set_context(
                "e2i_error",
                {
                    "error_id": exc.error_id,
                    "category": exc.category.value,
                    "severity": exc.severity.value,
                    "path": request.url.path,
                },
            )
            sentry_sdk.capture_exception(exc.original_error or exc)
    elif exc.severity == ErrorSeverity.HIGH:
        logger.error(
            f"[{exc.error_id}] {exc.category.value}: {exc.message}",
            exc_info=exc.original_error,
            extra={"error_id": exc.error_id, "path": request.url.path},
        )
        # Capture high severity errors to Sentry
        if sentry_sdk:
            sentry_sdk.set_context(
                "e2i_error",
                {
                    "error_id": exc.error_id,
                    "category": exc.category.value,
                    "severity": exc.severity.value,
                    "path": request.url.path,
                },
            )
            sentry_sdk.capture_exception(exc.original_error or exc)
    elif exc.severity == ErrorSeverity.MEDIUM:
        logger.warning(
            f"[{exc.error_id}] {exc.category.value}: {exc.message}",
            extra={"error_id": exc.error_id, "path": request.url.path},
        )
    else:
        logger.info(
            f"[{exc.error_id}] {exc.category.value}: {exc.message}",
            extra={"error_id": exc.error_id, "path": request.url.path},
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc, include_debug=DEBUG_MODE),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """
    Handle Pydantic/FastAPI validation errors with detailed field info.
    """
    # Extract validation errors into structured format
    schema_errors = []
    for error in exc.errors():
        schema_errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    from src.api.errors import SchemaValidationError

    e2i_error = SchemaValidationError(
        "Request validation failed",
        schema_errors=schema_errors,
    )

    logger.info(
        f"[{e2i_error.error_id}] Validation error on {request.url.path}",
        extra={"errors": schema_errors},
    )

    return JSONResponse(
        status_code=422,
        content=error_response(e2i_error, include_debug=DEBUG_MODE),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    """
    Handle standard HTTP exceptions with structured responses.
    """
    # Map status codes to appropriate E2IError types
    if exc.status_code == 404:
        e2i_error = EndpointNotFoundError(request.url.path)
    elif exc.status_code == 401:
        e2i_error = AuthenticationError(
            str(exc.detail) if exc.detail else "Authentication required"
        )
    elif exc.status_code == 403:
        e2i_error = AuthorizationError(str(exc.detail) if exc.detail else "Access denied")
    elif exc.status_code == 429:
        e2i_error = RateLimitError(
            limit=100,  # Default values since we don't have actual limits here
            window_seconds=60,
        )
    elif exc.status_code == 503:
        e2i_error = DependencyError(
            dependency="service",
            original_error=Exception(str(exc.detail)) if exc.detail else None,
        )
    elif exc.status_code == 504:
        e2i_error = E2ITimeoutError(
            operation="request",
            timeout_seconds=30.0,
        )
    else:
        # Generic HTTP error - wrap in E2IError
        e2i_error = E2IError(
            str(exc.detail) if exc.detail else f"HTTP {exc.status_code} error",
            suggested_action="Check the API documentation for valid request format.",
        )
        e2i_error.status_code = exc.status_code

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(e2i_error, include_debug=DEBUG_MODE),
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with structured response."""
    e2i_error = EndpointNotFoundError(request.url.path)

    return JSONResponse(
        status_code=404,
        content=error_response(e2i_error, include_debug=DEBUG_MODE),
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Custom 500 handler with detailed error context.

    Wraps unknown exceptions in E2IError with appropriate categorization.
    """
    # Try to extract agent context from request state if available
    agent_name = (
        getattr(request.state, "current_agent", None) if hasattr(request, "state") else None
    )
    operation = (
        getattr(request.state, "current_operation", None) if hasattr(request, "state") else None
    )

    # Wrap the exception with context
    e2i_error = wrap_exception(
        exc,
        agent_name=agent_name,
        operation=operation,
        include_trace=DEBUG_MODE,
    )

    # Log with full context
    logger.error(
        f"[{e2i_error.error_id}] Internal error on {request.url.path}: {exc}",
        exc_info=True,
        extra={
            "error_id": e2i_error.error_id,
            "path": request.url.path,
            "method": request.method,
            "agent": agent_name,
            "operation": operation,
        },
    )

    return JSONResponse(
        status_code=e2i_error.status_code,
        content=error_response(e2i_error, include_debug=DEBUG_MODE),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """
    Catch-all handler for any unhandled exceptions.

    Ensures all errors return structured responses.
    """
    # Check if already an E2IError (shouldn't happen but safety check)
    if isinstance(exc, E2IError):
        return await e2i_error_handler(request, exc)

    # Extract any available context
    agent_name = (
        getattr(request.state, "current_agent", None) if hasattr(request, "state") else None
    )
    operation = (
        getattr(request.state, "current_operation", None) if hasattr(request, "state") else None
    )

    # Wrap with intelligent categorization
    e2i_error = wrap_exception(
        exc,
        agent_name=agent_name,
        operation=operation,
        include_trace=DEBUG_MODE,
    )

    logger.error(
        f"[{e2i_error.error_id}] Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}",
        exc_info=True,
        extra={
            "error_id": e2i_error.error_id,
            "path": request.url.path,
            "exception_type": type(exc).__name__,
        },
    )

    # Capture unhandled exceptions to Sentry (these are always important)
    if sentry_sdk:
        sentry_sdk.set_context(
            "e2i_error",
            {
                "error_id": e2i_error.error_id,
                "category": e2i_error.category.value,
                "severity": e2i_error.severity.value,
                "path": request.url.path,
                "agent_name": agent_name,
                "operation": operation,
            },
        )
        sentry_sdk.capture_exception(exc)

    return JSONResponse(
        status_code=e2i_error.status_code,
        content=error_response(e2i_error, include_debug=DEBUG_MODE),
    )


# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run with: python -m src.api.main
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
