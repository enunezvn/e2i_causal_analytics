#!/bin/bash
# ==============================================================================
# E2I Causal Analytics - FastAPI Entrypoint Script
# ==============================================================================
# Handles initialization tasks before starting the application
# ==============================================================================

set -e

echo "=========================================="
echo "E2I Causal Analytics - FastAPI Starting"
echo "=========================================="
echo "Environment: ${APP_ENV:-development}"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Wait for database to be ready (if using external database)
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database connection..."

    # Extract host and port from DATABASE_URL
    # Format: postgresql://user:pass@host:port/db
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

    if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
        echo "Checking database at $DB_HOST:$DB_PORT..."

        max_attempts=30
        attempt=1

        while [ $attempt -le $max_attempts ]; do
            if python -c "
import sys
from sqlalchemy import create_engine
try:
    engine = create_engine('$DATABASE_URL', connect_args={'connect_timeout': 5})
    conn = engine.connect()
    conn.close()
    print('Database connection successful!')
    sys.exit(0)
except Exception as e:
    print(f'Attempt $attempt/$max_attempts failed: {e}')
    sys.exit(1)
" 2>&1; then
                echo "Database is ready!"
                break
            fi

            echo "Database not ready yet. Waiting... ($attempt/$max_attempts)"
            sleep 2
            attempt=$((attempt + 1))
        done

        if [ $attempt -gt $max_attempts ]; then
            echo "ERROR: Could not connect to database after $max_attempts attempts"
            echo "Continuing anyway (connection might work later)..."
        fi
    fi
fi

# Wait for Redis to be ready (if using Redis)
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis connection..."

    max_attempts=30
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        if python -c "
import sys
import redis
try:
    # Extract Redis connection details
    redis_url = '$REDIS_URL'
    r = redis.from_url(redis_url, socket_connect_timeout=5, decode_responses=True)
    r.ping()
    print('Redis connection successful!')
    sys.exit(0)
except Exception as e:
    print(f'Attempt $attempt/$max_attempts failed: {e}')
    sys.exit(1)
" 2>&1; then
            echo "Redis is ready!"
            break
        fi

        echo "Redis not ready yet. Waiting... ($attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo "WARNING: Could not connect to Redis after $max_attempts attempts"
        echo "Continuing anyway..."
    fi
fi

# Run database migrations (if applicable)
if [ "$APP_ENV" = "production" ] || [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Checking for database migrations..."

    # Check if alembic is available and migrations exist
    if [ -d "alembic" ] && command -v alembic &> /dev/null; then
        echo "Running database migrations..."
        alembic upgrade head || {
            echo "WARNING: Migration failed, but continuing..."
        }
    else
        echo "No migrations to run (alembic not found or no migrations directory)"
    fi
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p /var/log/e2i
mkdir -p /app/data

# Display startup configuration
echo "=========================================="
echo "Configuration:"
echo "  - Database: $(echo $DATABASE_URL | sed 's/:[^:]*@/:****@/')"
echo "  - Redis: $(echo $REDIS_URL | sed 's/:[^:]*@/:****@/')"
echo "  - MLflow: ${MLFLOW_TRACKING_URI:-not configured}"
echo "  - Debug mode: ${DEBUG:-false}"
echo "  - Log level: ${LOG_LEVEL:-INFO}"
echo "=========================================="

# Execute the main command
echo "Starting application..."
echo "Command: $@"
echo "=========================================="

exec "$@"
