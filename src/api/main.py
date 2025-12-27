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
Version: 4.1.0
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.cognitive import router as cognitive_router
from src.api.routes.digital_twin import router as digital_twin_router

# Import dependencies
from src.api.dependencies.bentoml_client import (
    close_bentoml_client,
    configure_bentoml_endpoints,
    get_bentoml_client,
)

# Import routers
from src.api.routes.explain import router as explain_router
from src.api.routes.experiments import router as experiments_router
from src.api.routes.graph import router as graph_router
from src.api.routes.memory import router as memory_router
from src.api.routes.monitoring import router as monitoring_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.rag import router as rag_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

    # Initialize BentoML client
    try:
        bentoml_client = await get_bentoml_client()
        logger.info("BentoML client initialized successfully")

        # Configure model endpoints from environment or defaults
        configure_bentoml_endpoints({
            "churn_model": os.environ.get("CHURN_MODEL_URL", "http://localhost:3000"),
            "conversion_model": os.environ.get("CONVERSION_MODEL_URL", "http://localhost:3001"),
            "causal_model": os.environ.get("CAUSAL_MODEL_URL", "http://localhost:3002"),
        })
    except Exception as e:
        logger.warning(f"BentoML client initialization failed (non-critical): {e}")

    # TODO: Initialize other connections
    # - Redis connection pool
    # - FalkorDB client
    # - Supabase client
    # - MLflow client
    # - Feast client
    # - Opik client

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

    # TODO: Cleanup other connections
    # - Close Redis connections
    # - Close database connections
    # - Flush pending logs

    logger.info("Shutdown complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="E2I Causal Analytics Platform",
    description="18-Agent architecture for causal inference, ML interpretability, and digital twin generation",
    version="4.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ROOT ENDPOINTS
# =============================================================================


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "E2I Causal Analytics Platform",
        "version": "4.1.0",
        "status": "online",
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
    try:
        client = await get_bentoml_client()
        health = await client.health_check()
        bentoml_status = health.get("status", "unknown")
    except Exception:
        bentoml_status = "unavailable"

    return {
        "status": "healthy",
        "service": "e2i-causal-analytics-api",
        "version": "4.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "api": "operational",
            "workers": "available",
            "memory_systems": "connected",
            "bentoml": bentoml_status,
        },
    }


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

    Returns:
        200 if ready to serve requests
        503 if dependencies are unavailable
    """
    # TODO: Add actual dependency checks
    # - Redis connectivity
    # - Supabase connectivity
    # - MLflow connectivity

    ready = True  # Placeholder

    if ready:
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"}
        )


# =============================================================================
# ROUTER REGISTRATION
# =============================================================================

# Model interpretability endpoints
app.include_router(explain_router)

# Memory system endpoints
app.include_router(memory_router)

# Cognitive workflow endpoints
app.include_router(cognitive_router)

# Knowledge graph endpoints
app.include_router(graph_router)

# Hybrid RAG endpoints
app.include_router(rag_router)

# Model monitoring endpoints (Phase 14)
app.include_router(monitoring_router)

# A/B testing & experiment execution endpoints (Phase 15)
app.include_router(experiments_router)

# Digital Twin pre-screening endpoints (Phase 15)
app.include_router(digital_twin_router)

# Model prediction endpoints (BentoML integration)
app.include_router(predictions_router)

# TODO: Add additional routers as they're developed:
# - Agent orchestration: /api/agents
# - Causal inference: /api/causal
# - Feature engineering: /api/features
# - Model training: /api/models


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": f"Endpoint {request.url.path} not found",
            "available_docs": "/api/docs",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred. Please contact support.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run with: python -m src.api.main
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
