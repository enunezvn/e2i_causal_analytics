"""Data Preparer Agent implementation.

This agent validates data quality, computes baseline metrics,
and enforces a QC gate that blocks downstream training if quality fails.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from .graph import create_data_preparer_graph
from .state import DataPreparerState

logger = logging.getLogger(__name__)


class DataPreparerAgent:
    """Data Preparer: Validate data quality and establish baselines.

    CRITICAL: This agent acts as a GATE. If QC fails, training CANNOT proceed.

    Tier: 0 (ML Foundation)
    Type: Standard (no LLM usage)
    SLA: <60 seconds

    Responsibilities:
    1. Run Great Expectations validation
    2. Detect data leakage (temporal, target, train-test)
    3. Compute baseline metrics from TRAIN split only
    4. Register features in Feast feature store
    5. Generate QC report
    6. Enforce QC gate (blocks training if quality fails)
    """

    def __init__(self):
        """Initialize the data_preparer agent."""
        self.tier = 0
        self.tier_name = "ml_foundation"
        self.agent_type = "standard"
        self.sla_seconds = 60

        # Create the LangGraph
        self.graph = create_data_preparer_graph().compile()

        logger.info("DataPreparerAgent initialized")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data preparation pipeline.

        Args:
            input_data: Input dictionary containing:
                - scope_spec: Scope specification from scope_definer
                - data_source: Data source table/view name
                - split_id: Optional ML split ID
                - validation_suite: Optional GE suite name
                - skip_leakage_check: Whether to skip leakage detection

        Returns:
            Dictionary containing:
                - qc_report: QC report with status and scores
                - baseline_metrics: Baseline metrics for drift detection
                - data_readiness: Data readiness summary
                - gate_passed: CRITICAL - blocks model_trainer if False

        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If data preparation fails
        """
        start_time = datetime.now()
        logger.info("Starting data preparation pipeline")

        # Validate inputs
        if "scope_spec" not in input_data:
            raise ValueError("scope_spec is required")
        if "data_source" not in input_data:
            raise ValueError("data_source is required")

        # Extract scope spec
        scope_spec = input_data["scope_spec"]
        experiment_id = scope_spec.get("experiment_id")
        if not experiment_id:
            raise ValueError("experiment_id missing from scope_spec")

        # Prepare initial state
        initial_state: DataPreparerState = {
            "experiment_id": experiment_id,
            "scope_spec": scope_spec,
            "data_source": input_data["data_source"],
            "split_id": input_data.get("split_id"),
            "validation_suite": input_data.get("validation_suite"),
            "skip_leakage_check": input_data.get("skip_leakage_check", False),
        }

        # TODO: Load data from data_source and split into train/val/test/holdout
        # For now, placeholder - in production this should:
        # 1. Query data from Supabase using data_source
        # 2. Apply split_id if provided, or create new split
        # 3. Load into pandas DataFrames
        # This is a critical gap that needs to be filled

        # Execute the graph
        try:
            final_state = await self.graph.ainvoke(initial_state)

            # Check for errors
            if final_state.get("error"):
                raise RuntimeError(
                    f"Data preparation failed: {final_state['error']} "
                    f"({final_state.get('error_type', 'unknown')})"
                )

            # Construct output
            output = {
                "qc_report": {
                    "report_id": final_state["report_id"],
                    "experiment_id": experiment_id,
                    "status": final_state["qc_status"],
                    "overall_score": final_state["overall_score"],
                    "completeness_score": final_state["completeness_score"],
                    "validity_score": final_state["validity_score"],
                    "consistency_score": final_state["consistency_score"],
                    "uniqueness_score": final_state["uniqueness_score"],
                    "timeliness_score": final_state["timeliness_score"],
                    "expectation_results": final_state.get("expectation_results", []),
                    "failed_expectations": final_state.get("failed_expectations", []),
                    "warnings": final_state.get("warnings", []),
                    "remediation_steps": final_state.get("remediation_steps", []),
                    "blocking_issues": final_state.get("blocking_issues", []),
                    "row_count": final_state["row_count"],
                    "column_count": final_state["column_count"],
                    "validated_at": final_state["validated_at"],
                },
                "baseline_metrics": {
                    "experiment_id": experiment_id,
                    "split_type": "train",
                    "feature_stats": final_state.get("feature_stats", {}),
                    "target_rate": final_state.get("target_rate"),
                    "target_distribution": final_state.get("target_distribution", {}),
                    "correlation_matrix": final_state.get("correlation_matrix", {}),
                    "computed_at": final_state.get("computed_at"),
                    "training_samples": final_state.get("training_samples", 0),
                },
                "data_readiness": {
                    "experiment_id": experiment_id,
                    "is_ready": final_state["is_ready"],
                    "total_samples": final_state["total_samples"],
                    "train_samples": final_state["train_samples"],
                    "validation_samples": final_state["validation_samples"],
                    "test_samples": final_state["test_samples"],
                    "holdout_samples": final_state["holdout_samples"],
                    "available_features": final_state["available_features"],
                    "missing_required_features": final_state["missing_required_features"],
                    "qc_passed": final_state["qc_passed"],
                    "qc_score": final_state["qc_score"],
                    "blockers": final_state["blockers"],
                },
                "gate_passed": final_state["gate_passed"],
            }

            # Log execution time
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Data preparation completed in {duration:.2f}s " f"(SLA: {self.sla_seconds}s)"
            )

            # Check SLA
            if duration > self.sla_seconds:
                logger.warning(f"SLA violation: {duration:.2f}s > {self.sla_seconds}s")

            # TODO: Persist QC report to ml_data_quality_reports table
            # TODO: Persist baseline metrics to ml_data_quality_reports table
            # TODO: Register features in Feast feature store
            # TODO: Emit observability span via observability_connector

            return output

        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise RuntimeError(f"Data preparation failed: {str(e)}") from e
