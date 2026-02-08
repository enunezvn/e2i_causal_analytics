"""Supabase data connector for Heterogeneous Optimizer Agent.

This module provides the production implementation that queries real data
from Supabase for CATE estimation and treatment effect heterogeneity analysis.

The connector uses MLDataLoader for split-aware querying to prevent data leakage.

Example:
    connector = HeterogeneousOptimizerDataConnector()
    df = await connector.query(
        source="business_metrics",
        columns=["hcp_specialty", "treatment", "outcome", "hcp_tenure"],
        filters={"brand": "remibrutinib"}
    )
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from src.repositories.ml_data_loader import MLDataLoader

logger = logging.getLogger(__name__)


class HeterogeneousOptimizerDataConnector:
    """Production data connector using Supabase via MLDataLoader.

    This connector queries real data from Supabase for CATE estimation:
    - Business metrics with treatment/outcome variables
    - Feature data for effect modifier analysis
    - Segment variables for heterogeneity detection

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        _loader: MLDataLoader instance
    """

    # Supported tables for CATE analysis
    SUPPORTED_TABLES = [
        "business_metrics",
        "predictions",
        "patient_journeys",
        "triggers",
        "causal_paths",
        "agent_activities",
    ]

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        """Initialize Supabase connector.

        Args:
            supabase_url: Supabase project URL (defaults to env var)
            supabase_key: Supabase API key (defaults to env var)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = (
            supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        )

        self._loader: Optional["MLDataLoader"] = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazily initialize MLDataLoader with Supabase client."""
        if not self._initialized:
            try:
                from supabase import create_client

                from src.repositories.ml_data_loader import MLDataLoader

                if not self.supabase_url or not self.supabase_key:
                    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

                client = create_client(self.supabase_url, self.supabase_key)
                self._loader = MLDataLoader(client)
                self._initialized = True
                logger.info("HeterogeneousOptimizerDataConnector initialized successfully")

            except ImportError as e:
                raise ImportError(f"Required package not installed: {e}. Run: pip install supabase")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise

    async def query(
        self,
        source: str,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Query data from Supabase for CATE estimation.

        This method provides the same interface as MockDataConnector for
        easy swapping between test and production environments.

        Args:
            source: Table name to query (must be in SUPPORTED_TABLES)
            columns: List of columns to retrieve
            filters: Optional column-value filters

        Returns:
            DataFrame with requested data

        Raises:
            ValueError: If source table not supported
        """
        await self._ensure_initialized()

        # Validate source table
        if source not in self.SUPPORTED_TABLES:
            logger.warning(
                f"Table '{source}' not in supported tables. "
                f"Attempting query anyway. Supported: {self.SUPPORTED_TABLES}"
            )

        try:
            # Use MLDataLoader to load a sample (no temporal splitting for CATE)
            assert self._loader is not None, "Loader not initialized"
            df = await self._loader.load_table_sample(
                table=source,
                filters=filters,
                limit=100000,  # Large limit for CATE analysis
                columns=columns if columns else None,
            )

            if df.empty:
                logger.warning(f"No data found in table '{source}' with filters {filters}")

            return df

        except Exception as e:
            logger.error(f"Failed to query data from '{source}': {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

    async def query_with_splits(
        self,
        source: str,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
        split: str = "train",
    ) -> pd.DataFrame:
        """Query data with ML split filtering.

        For training CATE models with proper train/val/test splits.

        Args:
            source: Table name to query
            columns: List of columns to retrieve
            filters: Optional column-value filters
            split: ML split to retrieve ("train", "validation", "test")

        Returns:
            DataFrame with data from specified split
        """
        await self._ensure_initialized()

        try:
            # Load with temporal splits
            assert self._loader is not None, "Loader not initialized"
            dataset = await self._loader.load_for_training(
                table=source,
                filters=filters,
                columns=columns,
            )

            if split == "train":
                return dataset.train
            elif split == "validation":
                return dataset.val
            elif split == "test":
                return dataset.test
            else:
                logger.warning(f"Unknown split '{split}', returning train data")
                return dataset.train

        except Exception as e:
            logger.error(f"Failed to query split data from '{source}': {e}")
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

    async def health_check(self) -> Dict[str, bool]:
        """Check connector health and connectivity.

        Returns:
            Dictionary with health status for each component
        """
        health = {
            "connected": False,
            "database": False,
            "business_metrics": False,
        }

        try:
            await self._ensure_initialized()
            health["connected"] = True

            # Check database connectivity by loading a sample
            assert self._loader is not None, "Loader not initialized"
            df = await self._loader.load_table_sample("business_metrics", limit=1)
            health["database"] = True
            health["business_metrics"] = not df.empty

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return health

    async def close(self) -> None:
        """Close Supabase client connection."""
        self._loader = None
        self._initialized = False
        logger.info("HeterogeneousOptimizerDataConnector closed")
