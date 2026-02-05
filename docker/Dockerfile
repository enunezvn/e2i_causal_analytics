# =============================================================================
# E2I Causal Analytics - Multi-Stage Dockerfile
# =============================================================================
# Stages:
#   1. base        - Common Python setup with uv
#   2. dependencies - Install all Python dependencies  
#   3. development  - Dev tools, debugpy, hot-reload ready
#   4. production   - Slim production image
#
# Build targets:
#   docker build --target development -t e2i-api:dev .
#   docker build --target production -t e2i-api:prod .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # For numpy/scipy
    libopenblas-dev \
    liblapack-dev \
    # For networkx/graphviz
    graphviz \
    libgraphviz-dev \
    # For psycopg2
    libpq-dev \
    # Utilities
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy dependency files
COPY requirements.txt pyproject.toml ./
COPY patches/ ./patches/

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Development
# -----------------------------------------------------------------------------
FROM base AS development

# Install dev dependencies
COPY requirements.txt pyproject.toml ./
COPY patches/ ./patches/

RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov pytest-asyncio

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install debugpy for VS Code remote debugging
RUN pip install debugpy

# Copy source code (will be overridden by bind mount in dev)
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Create directories for shared volumes
RUN mkdir -p /app/data/ml_artifacts \
             /app/data/causal_outputs \
             /app/data/agent_outputs \
             /app/data/experiment_designs \
             /app/data/feature_cache \
             /app/tmp

# Expose ports
EXPOSE 8000 5678

# Default command (overridden in compose)
CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    graphviz \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directories for shared volumes
RUN mkdir -p /app/data/ml_artifacts \
             /app/data/causal_outputs \
             /app/data/agent_outputs \
             /app/data/experiment_designs \
             /app/data/feature_cache

# Create non-root user for security
RUN groupadd --gid 1000 e2i && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home e2i && \
    chown -R e2i:e2i /app

USER e2i

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
