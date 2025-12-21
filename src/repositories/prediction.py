"""
Prediction Repository.

Handles ML predictions with rank metrics and model performance tracking.
"""

from typing import Any, Dict, List, Optional

from src.repositories.base import SplitAwareRepository


class PredictionRepository(SplitAwareRepository):
    """
    Repository for ml_predictions table.

    Supports:
    - Prediction score queries
    - Model performance metrics
    - Rank-based retrieval

    Table schema:
    - prediction_id (PK)
    - model_version, model_type
    - prediction_value, confidence_score
    - model_pr_auc, brier_score (WS1 Model Performance KPIs)
    - rank_metrics (JSONB): {"recall_at_5": 0.85, "recall_at_10": 0.92}
    """

    table_name = "ml_predictions"
    model_class = None  # Set to Prediction model when available

    async def get_by_model(
        self,
        model_id: str,
        split: Optional[str] = None,
        limit: int = 1000,
    ) -> List:
        """
        Get predictions for a specific model.

        Args:
            model_id: Model identifier (matches model_version)
            split: Optional ML split filter
            limit: Maximum records

        Returns:
            List of Prediction records
        """
        return await self.get_many(
            filters={"model_version": model_id},
            split=split,
            limit=limit,
        )

    async def get_top_predictions(
        self,
        model_id: str,
        top_k: int = 100,
        split: Optional[str] = None,
    ) -> List:
        """
        Get top predictions by score for a model.

        Args:
            model_id: Model identifier (matches model_version)
            top_k: Number of top predictions
            split: Optional ML split filter

        Returns:
            Top predictions sorted by prediction_value descending
        """
        if not self.client:
            return []

        query = self.client.table(self.table_name).select("*").eq("model_version", model_id)

        if split:
            query = query.eq("data_split", split)

        # Order by prediction_value descending to get top predictions
        result = await query.order("prediction_value", desc=True).limit(top_k).execute()

        return [self._to_model(row) for row in result.data]

    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get aggregate performance metrics for a model.

        Aggregates:
        - model_pr_auc: Average PR-AUC across predictions
        - brier_score: Average Brier score (calibration quality)
        - rank_metrics: Aggregated recall@K from JSONB

        Args:
            model_id: Model identifier (matches model_version)

        Returns:
            Dict with pr_auc, avg_brier_score, recall_at_5, recall_at_10 metrics
        """
        if not self.client:
            return {
                "pr_auc": 0.0,
                "avg_brier_score": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "total_predictions": 0,
            }

        # Fetch performance metrics for the model
        result = await (
            self.client.table(self.table_name)
            .select("model_pr_auc, brier_score, rank_metrics")
            .eq("model_version", model_id)
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {
                "pr_auc": 0.0,
                "avg_brier_score": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "total_predictions": 0,
            }

        # Aggregate metrics in Python
        pr_aucs = []
        brier_scores = []
        recall_5_values = []
        recall_10_values = []

        for row in result.data:
            # Collect PR-AUC values (skip nulls)
            if row.get("model_pr_auc") is not None:
                pr_aucs.append(float(row["model_pr_auc"]))

            # Collect Brier scores (skip nulls)
            if row.get("brier_score") is not None:
                brier_scores.append(float(row["brier_score"]))

            # Extract rank metrics from JSONB
            rank_metrics = row.get("rank_metrics") or {}
            if isinstance(rank_metrics, dict):
                if rank_metrics.get("recall_at_5") is not None:
                    recall_5_values.append(float(rank_metrics["recall_at_5"]))
                if rank_metrics.get("recall_at_10") is not None:
                    recall_10_values.append(float(rank_metrics["recall_at_10"]))

        return {
            "pr_auc": sum(pr_aucs) / len(pr_aucs) if pr_aucs else 0.0,
            "avg_brier_score": sum(brier_scores) / len(brier_scores) if brier_scores else 0.0,
            "recall_at_5": sum(recall_5_values) / len(recall_5_values) if recall_5_values else 0.0,
            "recall_at_10": (
                sum(recall_10_values) / len(recall_10_values) if recall_10_values else 0.0
            ),
            "total_predictions": len(result.data),
        }

    async def get_by_patient(
        self,
        patient_id: str,
        prediction_type: Optional[str] = None,
        limit: int = 100,
    ) -> List:
        """
        Get predictions for a specific patient.

        Args:
            patient_id: Patient identifier
            prediction_type: Optional filter by prediction type
            limit: Maximum records

        Returns:
            List of Prediction records ordered by timestamp descending
        """
        if not self.client:
            return []

        query = self.client.table(self.table_name).select("*").eq("patient_id", patient_id)

        if prediction_type:
            query = query.eq("prediction_type", prediction_type)

        result = await query.order("prediction_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]

    async def get_high_confidence_predictions(
        self,
        model_id: str,
        confidence_threshold: float = 0.8,
        limit: int = 500,
    ) -> List:
        """
        Get high-confidence predictions for a model.

        Args:
            model_id: Model identifier (matches model_version)
            confidence_threshold: Minimum confidence score (default 0.8)
            limit: Maximum records

        Returns:
            High-confidence predictions sorted by confidence descending
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("model_version", model_id)
            .gte("confidence_score", confidence_threshold)
            .order("confidence_score", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_calibration_summary(
        self,
        model_id: str,
    ) -> Dict[str, Any]:
        """
        Get calibration summary for a model.

        Analyzes:
        - Average calibration score
        - Brier score distribution
        - Confidence vs actual correlation

        Args:
            model_id: Model identifier

        Returns:
            Dict with calibration statistics
        """
        if not self.client:
            return {
                "avg_calibration": 0.0,
                "avg_brier": 0.0,
                "min_brier": 0.0,
                "max_brier": 0.0,
                "total_evaluated": 0,
            }

        result = await (
            self.client.table(self.table_name)
            .select("calibration_score, brier_score")
            .eq("model_version", model_id)
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {
                "avg_calibration": 0.0,
                "avg_brier": 0.0,
                "min_brier": 0.0,
                "max_brier": 0.0,
                "total_evaluated": 0,
            }

        calibrations = [
            float(r["calibration_score"])
            for r in result.data
            if r.get("calibration_score") is not None
        ]
        briers = [float(r["brier_score"]) for r in result.data if r.get("brier_score") is not None]

        return {
            "avg_calibration": sum(calibrations) / len(calibrations) if calibrations else 0.0,
            "avg_brier": sum(briers) / len(briers) if briers else 0.0,
            "min_brier": min(briers) if briers else 0.0,
            "max_brier": max(briers) if briers else 0.0,
            "total_evaluated": len(result.data),
        }
