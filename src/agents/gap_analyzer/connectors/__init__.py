"""
Gap Analyzer Connectors - Factory Pattern.

Provides factory functions to get appropriate data connectors
based on environment (mock for testing, Supabase for production).
"""

from typing import Union

# Type hints for connector classes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark_store import BenchmarkStore
    from .supabase_connector import SupabaseDataConnector
    from ..nodes.gap_detector import MockBenchmarkStore, MockDataConnector


def get_data_connector(use_mock: bool = False) -> Union["SupabaseDataConnector", "MockDataConnector"]:
    """
    Factory to get appropriate data connector.

    Args:
        use_mock: If True, return MockDataConnector for testing

    Returns:
        Data connector instance
    """
    if use_mock:
        from ..nodes.gap_detector import MockDataConnector

        return MockDataConnector()

    from .supabase_connector import SupabaseDataConnector

    return SupabaseDataConnector()


def get_benchmark_store(use_mock: bool = False) -> Union["BenchmarkStore", "MockBenchmarkStore"]:
    """
    Factory to get appropriate benchmark store.

    Args:
        use_mock: If True, return MockBenchmarkStore for testing

    Returns:
        Benchmark store instance
    """
    if use_mock:
        from ..nodes.gap_detector import MockBenchmarkStore

        return MockBenchmarkStore()

    from .benchmark_store import BenchmarkStore

    return BenchmarkStore()


__all__ = ["get_data_connector", "get_benchmark_store"]
