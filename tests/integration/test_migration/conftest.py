"""
Fixtures and configuration for Supabase migration validation tests.

This module provides:
1. Database connection fixtures (psycopg3 and Supabase client)
2. Expected counts and schema definitions
3. Environment variable handling with defaults
4. Skip markers for service availability

Environment Variables:
    SUPABASE_DB_HOST - Database host (default: localhost)
    SUPABASE_DB_PORT - Database port (default: 5432)
    SUPABASE_DB_NAME - Database name (default: postgres)
    SUPABASE_DB_USER - Database user (default: postgres)
    POSTGRES_PASSWORD - Database password (required)
    SUPABASE_URL - Supabase API URL
    SUPABASE_ANON_KEY - Supabase anonymous key
    SUPABASE_SERVICE_KEY - Supabase service role key
"""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest

# =============================================================================
# EXPECTED MIGRATION VALUES
# =============================================================================

# These values represent the expected state after successful migration
EXPECTED_COUNTS = {
    # Table counts
    "public_tables": 84,
    "auth_tables": 20,
    # Row counts
    "auth_users": 6,
    "agent_registry": 11,
    "causal_paths": 50,
    "agent_tier_mapping": 21,
}

# Core tables that must exist in public schema
CORE_TABLES = [
    "agent_registry",
    "causal_paths",
    "agent_tier_mapping",
    "business_metrics",
    "agent_activities",
    "triggers",
    "chatbot_conversations",  # Note: actual table name (not 'conversations')
    "patient_journeys",
    "hcp_profiles",
    "treatment_events",
    "ml_predictions",
    "ml_split_registry",
]

# Required PostgreSQL extensions
REQUIRED_EXTENSIONS = ["uuid-ossp", "pgcrypto"]

# Required enum types
REQUIRED_ENUMS = [
    "data_split_type",
    "brand_type",
    "agent_tier_type",
]

# Split-aware views pattern
SPLIT_VIEW_PREFIXES = ["v_train_", "v_test_", "v_holdout_", "v_validation_"]

# Agent registry expected columns
AGENT_REGISTRY_COLUMNS = [
    "id",
    "agent_name",
    "agent_tier",
    "description",
    "capabilities",
    "created_at",
    "updated_at",
]

# Causal paths expected columns
CAUSAL_PATHS_COLUMNS = [
    "id",
    "causal_chain",
    "source_metric",
    "target_metric",
    "confidence_score",
    "created_at",
]


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================


def get_db_config() -> dict[str, Any]:
    """Get database configuration from environment variables.

    Note: For self-hosted Supabase on droplet, connect directly to the
    supabase-db container via Docker network. The pooler (port 5432)
    requires tenant-aware connection strings.
    """
    # Default to supabase-db container internal IP for self-hosted setup
    # This can be overridden with SUPABASE_DB_HOST env var
    default_host = os.getenv("SUPABASE_DB_HOST", "172.22.0.4")
    return {
        "host": default_host,
        "port": int(os.getenv("SUPABASE_DB_PORT", "5432")),
        "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "user": os.getenv("SUPABASE_DB_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def get_supabase_config() -> dict[str, str]:
    """Get Supabase API configuration from environment variables."""
    return {
        "url": os.getenv("SUPABASE_URL", ""),
        "anon_key": os.getenv("SUPABASE_ANON_KEY", ""),
        "service_key": os.getenv("SUPABASE_SERVICE_KEY", ""),
    }


# =============================================================================
# SERVICE AVAILABILITY CHECKS
# =============================================================================


def is_postgres_available() -> bool:
    """Check if PostgreSQL is accessible."""
    config = get_db_config()
    if not config["password"]:
        return False
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            connect_timeout=5,
        )
        conn.close()
        return True
    except Exception:
        return False


def is_supabase_configured() -> bool:
    """Check if Supabase credentials are configured."""
    config = get_supabase_config()
    return bool(config["url"] and (config["anon_key"] or config["service_key"]))


# =============================================================================
# SKIP MARKERS
# =============================================================================

# Check availability once at module load
_POSTGRES_AVAILABLE = None
_SUPABASE_CONFIGURED = None


def postgres_available() -> bool:
    """Cached check for PostgreSQL availability."""
    global _POSTGRES_AVAILABLE
    if _POSTGRES_AVAILABLE is None:
        _POSTGRES_AVAILABLE = is_postgres_available()
    return _POSTGRES_AVAILABLE


def supabase_configured() -> bool:
    """Cached check for Supabase configuration."""
    global _SUPABASE_CONFIGURED
    if _SUPABASE_CONFIGURED is None:
        _SUPABASE_CONFIGURED = is_supabase_configured()
    return _SUPABASE_CONFIGURED


requires_postgres = pytest.mark.skipif(
    not postgres_available(),
    reason="PostgreSQL not available (check POSTGRES_PASSWORD and connection)",
)

requires_supabase = pytest.mark.skipif(
    not supabase_configured(),
    reason="Supabase not configured (check SUPABASE_URL and keys)",
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def expected_counts() -> dict[str, int]:
    """Get expected row counts for validation."""
    return EXPECTED_COUNTS.copy()


@pytest.fixture(scope="module")
def core_tables() -> list[str]:
    """Get list of core tables that must exist."""
    return CORE_TABLES.copy()


@pytest.fixture(scope="module")
def required_extensions() -> list[str]:
    """Get list of required PostgreSQL extensions."""
    return REQUIRED_EXTENSIONS.copy()


@pytest.fixture(scope="module")
def required_enums() -> list[str]:
    """Get list of required enum types."""
    return REQUIRED_ENUMS.copy()


@pytest.fixture(scope="module")
def db_config() -> dict[str, Any]:
    """Get database configuration."""
    return get_db_config()


@pytest.fixture(scope="module")
def supabase_config() -> dict[str, str]:
    """Get Supabase API configuration."""
    return get_supabase_config()


@pytest.fixture(scope="module")
def pg_connection(db_config: dict[str, Any]) -> Generator:
    """Create a PostgreSQL connection for the test module.

    Yields:
        psycopg2.connection: Database connection

    Skips:
        If PostgreSQL is not available
    """
    if not db_config["password"]:
        pytest.skip("POSTGRES_PASSWORD not set")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            connect_timeout=10,
        )
        yield conn
        conn.close()
    except ImportError:
        pytest.skip("psycopg2 not installed")
    except Exception as e:
        pytest.skip(f"Could not connect to PostgreSQL: {e}")


@pytest.fixture(scope="module")
def supabase_client(supabase_config: dict[str, str]):
    """Create a Supabase client for the test module.

    Returns:
        supabase.Client: Supabase client

    Skips:
        If Supabase is not configured
    """
    if not supabase_config["url"]:
        pytest.skip("SUPABASE_URL not set")

    key = supabase_config["service_key"] or supabase_config["anon_key"]
    if not key:
        pytest.skip("No Supabase key configured")

    try:
        from supabase import create_client

        return create_client(supabase_config["url"], key)
    except ImportError:
        pytest.skip("supabase library not installed")
    except Exception as e:
        pytest.skip(f"Could not create Supabase client: {e}")


@pytest.fixture
def pg_cursor(pg_connection):
    """Create a cursor for a single test.

    Yields:
        psycopg.Cursor: Database cursor

    Note:
        Cursor is closed after each test, connection is reused.
    """
    cursor = pg_connection.cursor()
    yield cursor
    cursor.close()
