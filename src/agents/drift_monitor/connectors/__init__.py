"""Data connectors for drift monitoring.

This module provides data connectors for fetching feature and prediction data
for drift detection. The connectors abstract data access, allowing seamless
switching between mock data for testing and real Supabase data for production.

Example:
    from src.agents.drift_monitor.connectors import get_connector

    # Get configured connector (Supabase in production, Mock in testing)
    connector = get_connector()

    # Query feature data
    data = await connector.query_features(
        feature_names=["age", "income"],
        time_window="7d",
        filters={"brand": "remibrutinib"}
    )
"""

from src.agents.drift_monitor.connectors.base import BaseDataConnector
from src.agents.drift_monitor.connectors.factory import get_connector
from src.agents.drift_monitor.connectors.mock_connector import MockDataConnector
from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

__all__ = [
    "BaseDataConnector",
    "MockDataConnector",
    "SupabaseDataConnector",
    "get_connector",
]
