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
FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/app/tmp \
    MPLBACKEND=Agg
# matplotlib (imported transitively via shap) writes a font cache at import time;
# MPLCONFIGDIR points it at the writable /app/tmp tmpfs and MPLBACKEND=Agg forces a
# headless backend so the non-root e2i user cannot crash on an unwritable cache dir.

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
COPY requirements.txt requirements.lock pyproject.toml ./
COPY patches/ ./patches/

# Create virtual environment and install dependencies.
# Two steps: (1) the hash-pinned closure under --require-hashes for
# byte-reproducible, tamper-checked rebuilds (#534); (2) the two local
# ./patches/ packages --no-deps (a local path cannot be hashed, and
# --require-hashes is all-or-nothing — their deps are already satisfied by
# the locked closure). Regenerate the lock with the uv command in its header.
RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --require-hashes -r requirements.lock && \
    pip install --no-cache-dir --no-deps './patches/ag-ui-langgraph[fastapi]' ./patches/copilotkit

# -----------------------------------------------------------------------------
# Stage 3: Development
# -----------------------------------------------------------------------------
FROM base AS development

# Install dev dependencies
COPY requirements.txt requirements.lock pyproject.toml ./
COPY patches/ ./patches/

RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --require-hashes -r requirements.lock && \
    pip install --no-cache-dir --no-deps './patches/ag-ui-langgraph[fastapi]' ./patches/copilotkit

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install debugpy for VS Code remote debugging (pinned for reproducible dev images)
RUN pip install --no-cache-dir debugpy==1.8.20

# Copy source code (will be overridden by bind mount in dev)
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
# Ops/maintenance scripts (e.g. scripts/train_twin_model.py, the reachable
# #705 H4 twin train/persist trigger run via `docker exec … python scripts/…`).
COPY scripts/ ./scripts/

# Create directories for shared volumes. optimized_modules/optimized_prompts
# must exist in the image so a fresh named volume is initialized with e2i
# ownership (the chown -R below) — otherwise the engine creates the
# mountpoint root-owned and non-root worker saves fail silently.
RUN mkdir -p /app/data/ml_artifacts \
             /app/data/causal_outputs \
             /app/data/agent_outputs \
             /app/data/experiment_designs \
             /app/data/feature_cache \
             /app/optimized_modules \
             /app/optimized_prompts \
             /app/tmp

# Create non-root user for security (mirrors production stage UID/GID).
# Bind-mounts in dev compose are owned by host UID 1000, so this aligns.
RUN groupadd --gid 1000 e2i && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home e2i && \
    chown -R e2i:e2i /app

USER e2i

# Expose ports
EXPOSE 8000 5678

# Default command (overridden in compose)
CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Production
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS production

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    # OpenMP runtime: lightgbm (pulled by causalml's UpliftRandomForest, the twin
    # effect estimator) dlopens the SYSTEM libgomp.so.1. The slim base does not
    # ship it and the wheels' bundled copies are not on the loader path, so without
    # this every real /simulate call 422'd "Effect estimation failed: libgomp.so.1".
    libgomp1 \
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
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/app/tmp \
    MPLBACKEND=Agg

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
# Ops/maintenance scripts run inside the container via `docker exec` (e.g.
# scripts/train_twin_model.py, the reachable #705 H4 twin train/persist trigger
# while worker_heavy is dark). The prod rootfs is read-only, so these must be
# baked into the image — they cannot be `docker cp`'d in at runtime.
COPY scripts/ ./scripts/

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
# OOM guard: 2 workers, not 4 — each inline-heavy request (digital-twin
# simulate / SHAP explain) peaks ~1.3 GiB, and 4 concurrent peaked ~5.2 GiB,
# tripping the 5G api cgroup. Paired with the per-worker heavy-compute slot in
# src/api/dependencies/compute.py.
CMD ["gunicorn", "src.api.main:app", \
     "--config", "/app/config/gunicorn.conf.py", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
