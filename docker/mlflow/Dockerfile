# ==============================================================================
# E2I Causal Analytics - MLflow Tracking Server Dockerfile
# ==============================================================================
# MLflow server for experiment tracking, model registry, and artifact storage
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_HOME=/mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create mlflow user and directories
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /mlflow /mlflow/mlflow-artifacts && \
    chown -R mlflow:mlflow /mlflow

# Set working directory
WORKDIR /mlflow

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.14 \
    google-cloud-storage==2.14.0 \
    azure-storage-blob==12.19.0 \
    cryptography==41.0.7

# Switch to mlflow user
USER mlflow

# Expose MLflow UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command (will be overridden in docker-compose)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
     "--default-artifact-root", "./mlflow-artifacts"]
