"""
Repository for storing and retrieving SHAP analyses.

Handles CRUD operations for ml_shap_analyses table.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, cast

from .base import BaseRepository

logger = logging.getLogger(__name__)


class ShapAnalysisRepository(BaseRepository):
    """Repository for ml_shap_analyses table."""

    table_name = "ml_shap_analyses"
    model_class = None  # Using dict directly

    async def store_analysis(
        self,
        analysis_dict: Dict[str, Any],
        model_registry_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Store a SHAP analysis result in the database.

        Args:
            analysis_dict: Dictionary containing SHAP analysis data
            model_registry_id: Optional model registry ID to link to

        Returns:
            Stored record with database-generated fields
        """
        if not self.client:
            logger.warning("No Supabase client, skipping DB store")
            return None

        try:
            # Build global importance JSONB from feature_importance list
            global_importance = {}
            for fi in analysis_dict.get("feature_importance", []):
                global_importance[fi["feature"]] = fi["importance"]

            # Build top_interactions JSONB
            top_interactions = []
            for interaction in analysis_dict.get("interactions", [])[:10]:
                top_interactions.append(
                    {
                        "feature_1": interaction["features"][0]
                        if interaction.get("features")
                        else None,
                        "feature_2": interaction["features"][1]
                        if len(interaction.get("features", [])) > 1
                        else None,
                        "interaction_strength": interaction.get("interaction_strength"),
                        "interpretation": interaction.get("interpretation"),
                    }
                )

            # Map Python types to database column names (per mlops_tables.sql schema)
            db_record = {
                "id": str(uuid.uuid4()),
                "model_registry_id": model_registry_id,
                "analysis_type": "global",  # Feature analyzer does global analysis
                "global_importance": global_importance,
                "top_interactions": top_interactions,
                "natural_language_explanation": analysis_dict.get("interpretation"),
                "key_drivers": analysis_dict.get("top_features", [])[:5],
                "sample_size": analysis_dict.get("samples_analyzed"),
                # Schema uses computation_duration_seconds (INTEGER), not computation_time_seconds
                "computation_duration_seconds": int(
                    analysis_dict.get("computation_time_seconds", 0)
                ),
                # Schema uses computation_method, not model_type
                "computation_method": analysis_dict.get("explainer_type"),
                # Note: model_version is stored in ml_model_registry, not in this table
            }

            # Remove None values for optional fields
            db_record = {k: v for k, v in db_record.items() if v is not None}

            result = await self.client.table(self.table_name).insert(db_record).execute()

            if result.data:
                logger.info(
                    f"Stored SHAP analysis for experiment {analysis_dict.get('experiment_id')}"
                )
                return cast(Dict[str, Any], result.data[0])
            return None

        except Exception as e:
            logger.error(f"Failed to store SHAP analysis: {e}", exc_info=True)
            return None

    async def get_by_model_registry_id(
        self,
        model_registry_id: str,
        analysis_type: str = "global",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get SHAP analyses for a specific model from the registry.

        Args:
            model_registry_id: Model registry ID (references ml_model_registry.id)
            analysis_type: Type of analysis (global, local, segment)
            limit: Maximum number of records to return

        Returns:
            List of SHAP analysis records
        """
        if not self.client:
            logger.warning("No Supabase client")
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("model_registry_id", model_registry_id)
                .eq("analysis_type", analysis_type)
                .order("computed_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get SHAP analyses: {e}", exc_info=True)
            return []

    async def get_latest_for_model(
        self,
        model_registry_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest SHAP analysis for a model.

        Args:
            model_registry_id: Model registry ID (references ml_model_registry.id)

        Returns:
            Latest SHAP analysis record or None
        """
        if not self.client:
            return None

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("model_registry_id", model_registry_id)
                .order("computed_at", desc=True)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Failed to get latest SHAP analysis: {e}", exc_info=True)
            return None

    async def get_feature_importance_trends(
        self,
        model_registry_id: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get feature importance trends over time for a model.

        Args:
            model_registry_id: Model registry ID (references ml_model_registry.id)
            limit: Maximum number of analyses to return (default 30)

        Returns:
            List of trend records with feature importance over time
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("id, global_importance, computed_at")
                .eq("model_registry_id", model_registry_id)
                .eq("analysis_type", "global")
                .order("computed_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get feature importance trends: {e}", exc_info=True)
            return []


# Singleton instance
_shap_analysis_repository: Optional[ShapAnalysisRepository] = None


def get_shap_analysis_repository() -> ShapAnalysisRepository:
    """Get singleton ShapAnalysisRepository instance.

    Returns:
        ShapAnalysisRepository with Supabase client if available
    """
    global _shap_analysis_repository

    if _shap_analysis_repository is None:
        try:
            from src.memory.services.factories import get_supabase_client

            client = get_supabase_client()
            _shap_analysis_repository = ShapAnalysisRepository(supabase_client=client)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP repository with client: {e}")
            _shap_analysis_repository = ShapAnalysisRepository(supabase_client=None)

    return _shap_analysis_repository
