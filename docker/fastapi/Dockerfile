# ==============================================================================
# E2I Causal Analytics - FastAPI Backend Dockerfile
# ==============================================================================
# Multi-stage build for optimized production image
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and local patches
COPY requirements.txt .
COPY patches/ ./patches/

# Install CPU-only PyTorch first (saves ~4GB vs CUDA version; no GPU on droplet)
# Strip torch, NVIDIA CUDA libs, and triton from requirements (CUDA-only deps)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-warn-script-location \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|nvidia-|triton)==' requirements.txt > requirements-cpu.txt && \
    pip install --user --no-warn-script-location --no-cache-dir \
    -r requirements-cpu.txt

# ------------------------------------------------------------------------------
# Stage 2: Runtime - Create minimal production image
# ------------------------------------------------------------------------------
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security (don't run as root)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /var/log/e2i && \
    chown -R appuser:appuser /app /var/log/e2i

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /root/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser pyproject.toml ./

# Copy entrypoint script
COPY --chown=appuser:appuser docker/fastapi/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
