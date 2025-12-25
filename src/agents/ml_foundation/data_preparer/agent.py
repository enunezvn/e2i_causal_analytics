"""Data Preparer Agent implementation.

This agent validates data quality, computes baseline metrics,
and enforces a QC gate that blocks downstream training if quality fails.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from .graph import create_data_preparer_graph
from .state import DataPreparerState

logger = logging.getLogger(__name__)


def _get_dq_repository():
    """Get DataQualityReportRepository (lazy import to avoid circular deps)."""
    try:
        from src.repositories.data_quality_report import get_data_quality_report_repository
        return get_data_quality_report_repository()
    except Exception as e:
        logger.warning(f"Could not get DQ repository: {e}")
        return None


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector
        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


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

    # Class attributes per contract
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "data_preparer"
    agent_type = "standard"
    sla_seconds = 60
    tools = ["great_expectations", "pandas", "numpy", "scipy", "feast"]
    primary_model = None  # No LLM usage

    def __init__(self):
        """Initialize the data_preparer agent."""
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
        # Note: Data loading is handled by the data_loader node in the graph
        initial_state: DataPreparerState = {
            "experiment_id": experiment_id,
            "scope_spec": scope_spec,
            "data_source": input_data["data_source"],
            "split_id": input_data.get("split_id"),
            "validation_suite": input_data.get("validation_suite"),
            "skip_leakage_check": input_data.get("skip_leakage_check", False),
        }

        # Execute the graph with optional Opik tracing
        opik = _get_opik_connector()
        try:
            # Wrap execution in Opik trace if available
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="prepare_data",
                    metadata={
                        "experiment_id": experiment_id,
                        "data_source": input_data["data_source"],
                        "tier": self.tier,
                    },
                    tags=[self.agent_name, "tier_0", "qc_gate"],
                    input_data={"scope_spec": scope_spec},
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    span.set_output({
                        "gate_passed": final_state.get("gate_passed"),
                        "qc_status": final_state.get("qc_status"),
                        "overall_score": final_state.get("overall_score"),
                    })
            else:
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
                "feast_registration": {
                    "status": final_state.get("feast_registration_status", "skipped"),
                    "features_registered": final_state.get("feast_features_registered", 0),
                    "freshness_check": final_state.get("feast_freshness_check"),
                    "warnings": final_state.get("feast_warnings", []),
                    "registered_at": final_state.get("feast_registered_at"),
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

            # Persist QC report to database
            await self._persist_qc_report(output["qc_report"], input_data["data_source"])

            return output

        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise RuntimeError(f"Data preparation failed: {str(e)}") from e

    async def _persist_qc_report(
        self, qc_report: Dict[str, Any], data_source: str
    ) -> None:
        """Persist QC report to database.

        Args:
            qc_report: QC report dictionary
            data_source: Data source table name
        """
        try:
            repo = _get_dq_repository()
            if repo is None:
                logger.debug("Skipping QC report persistence (no repository)")
                return

            # Map QC report to database record
            db_record = {
                "id": str(uuid.uuid4()),
                "report_name": f"data_preparer_{qc_report['experiment_id']}",
                "expectation_suite_name": f"data_preparer_{data_source}",
                "table_name": data_source,
                "overall_status": qc_report["status"],
                "expectations_evaluated": len(qc_report.get("expectation_results", [])),
                "expectations_passed": len(qc_report.get("expectation_results", []))
                - len(qc_report.get("failed_expectations", [])),
                "expectations_failed": len(qc_report.get("failed_expectations", [])),
                "success_rate": qc_report["overall_score"],
                "failed_expectations": qc_report.get("failed_expectations", []),
                "completeness_score": qc_report.get("completeness_score"),
                "validity_score": qc_report.get("validity_score"),
                "uniqueness_score": qc_report.get("uniqueness_score"),
                "consistency_score": qc_report.get("consistency_score"),
                "timeliness_score": qc_report.get("timeliness_score"),
                "data_split": "train",  # QC runs on train split
                "training_run_id": None,  # Set by model_trainer if applicable
            }

            await repo.store_result(db_record)
            logger.info(f"Persisted QC report for {qc_report['experiment_id']}")

        except Exception as e:
            logger.warning(f"Failed to persist QC report: {e}")
