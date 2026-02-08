"""Connector factory for drift detection.

This module provides a factory function for creating data connectors
based on configuration. It automatically selects the appropriate
connector based on environment settings.

Example:
    from src.agents.drift_monitor.connectors import get_connector

    # Auto-selects based on environment
    connector = get_connector()

    # Explicitly request mock connector
    connector = get_connector(connector_type="mock")
"""

import logging
import os
from typing import Literal, cast

from src.agents.drift_monitor.connectors.base import BaseDataConnector
from src.agents.drift_monitor.connectors.mock_connector import MockDataConnector
from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

logger = logging.getLogger(__name__)

# Type alias for connector types
ConnectorType = Literal["supabase", "mock", "auto"]


def get_connector(
    connector_type: ConnectorType = "auto",
    **kwargs,
) -> BaseDataConnector:
    """Get a data connector based on configuration.

    This factory function creates the appropriate data connector based on:
    - Explicit connector_type parameter
    - DRIFT_MONITOR_CONNECTOR environment variable
    - Auto-detection based on environment (Supabase credentials available)

    Args:
        connector_type: Type of connector to create
            - "supabase": Production Supabase connector
            - "mock": Mock connector for testing
            - "auto": Auto-detect based on environment
        **kwargs: Additional arguments passed to connector constructor

    Returns:
        BaseDataConnector instance

    Raises:
        ValueError: If unknown connector type specified

    Example:
        # Auto-detect (uses Supabase if credentials available)
        connector = get_connector()

        # Force mock for testing
        connector = get_connector(connector_type="mock")

        # Mock with custom drift magnitude
        connector = get_connector(
            connector_type="mock",
            drift_magnitude=0.5
        )
    """
    # Check environment variable override
    env_connector = os.getenv("DRIFT_MONITOR_CONNECTOR", "").lower()
    if env_connector:
        if env_connector in ("supabase", "mock"):
            connector_type = cast(Literal["supabase", "mock", "auto"], env_connector)
            logger.info(f"Using connector type from env: {connector_type}")

    # Auto-detect based on environment
    if connector_type == "auto":
        connector_type = _auto_detect_connector_type()

    # Create appropriate connector
    if connector_type == "supabase":
        return _create_supabase_connector(**kwargs)
    elif connector_type == "mock":
        return _create_mock_connector(**kwargs)
    else:
        raise ValueError(f"Unknown connector type: {connector_type}")


def _auto_detect_connector_type() -> Literal["supabase", "mock"]:
    """Auto-detect the appropriate connector type.

    Checks for Supabase credentials in environment. If available,
    uses Supabase connector; otherwise falls back to mock.

    Returns:
        "supabase" if credentials available, "mock" otherwise
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if supabase_url and supabase_key:
        logger.info("Auto-detected Supabase credentials, using SupabaseDataConnector")
        return "supabase"
    else:
        logger.warning("No Supabase credentials found, falling back to MockDataConnector")
        return "mock"


def _create_supabase_connector(**kwargs) -> SupabaseDataConnector:
    """Create a Supabase connector.

    Args:
        **kwargs: Arguments passed to SupabaseDataConnector

    Returns:
        SupabaseDataConnector instance
    """
    return SupabaseDataConnector(
        supabase_url=kwargs.get("supabase_url"),
        supabase_key=kwargs.get("supabase_key"),
    )


def _create_mock_connector(**kwargs) -> MockDataConnector:
    """Create a mock connector.

    Args:
        **kwargs: Arguments passed to MockDataConnector

    Returns:
        MockDataConnector instance
    """
    return MockDataConnector(
        drift_magnitude=kwargs.get("drift_magnitude", 0.2),
        sample_size=kwargs.get("sample_size", 1000),
        seed=kwargs.get("seed", 42),
    )
