# BentoML Model Serving Dockerfile
# This Dockerfile creates a production-ready container for serving ML models
# using BentoML.
#
# Build: docker build -t e2i-model-server -f docker/bentoml/Dockerfile .
# Run: docker run -p 3000:3000 e2i-model-server

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash bentoml
WORKDIR /home/bentoml

# Install BentoML and dependencies
FROM base as dependencies

COPY docker/bentoml/requirements-bentoml.txt ./
RUN pip install --no-cache-dir -r requirements-bentoml.txt

# Production stage
FROM dependencies as production

# Copy real serving service (self-contained, no src.* imports)
COPY --chown=bentoml:bentoml scripts/bentoml/e2i_serving_service.py ./e2i_serving_service.py

# Copy supporting code for reference/packaging workflows
COPY --chown=bentoml:bentoml src/mlops/bentoml_service.py ./src/mlops/
COPY --chown=bentoml:bentoml src/mlops/bentoml_templates/ ./src/mlops/bentoml_templates/
COPY --chown=bentoml:bentoml src/mlops/bentoml_packaging.py ./src/mlops/

# Copy mock service for testing fallback
COPY --chown=bentoml:bentoml docker/bentoml/mock_service.py ./mock_service.py

# Create directories for models and data
RUN mkdir -p /home/bentoml/bentoml /home/bentoml/data \
    && chown -R bentoml:bentoml /home/bentoml

# Switch to non-root user
USER bentoml

# Expose BentoML default port
EXPOSE 3000

# Health check - use BentoML's built-in healthz endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/healthz || exit 1

# Default command - serve the real E2I model service (starts in degraded mode if no models)
CMD ["bentoml", "serve", "e2i_serving_service.py:E2IModelService", "--host", "0.0.0.0", "--port", "3000"]
