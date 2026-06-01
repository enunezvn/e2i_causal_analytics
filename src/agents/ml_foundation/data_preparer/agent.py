"""Data Preparer Agent implementation.

This agent validates data quality, computes baseline metrics,
and enforces a QC gate that blocks downstream training if quality fails.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

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
                - skip_leakage_check: Skip the legacy name-based detect_leakage
                  node ONLY. The data-driven adaptive validity / FDR layer
                  (adaptive_validity_check) ALWAYS runs as the safety net and
                  can still escalate leakage_severity / leaked_features
                  regardless of this flag (#533, Option 2).

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
            # D1.2: thread caller-provided audit_workflow_id (see scope_definer
            # for the rationale). Backlog #1 (closed 2026-05-09) tightened the
            # State to required-no-default to fix the LangGraph channel-reducer
            # bug (default_factory firing on every Schema reconstruction).
            # Caller-provided UUID is preferred; absent that, generate one at
            # the agent boundary. Either way the UUID is set ONCE before
            # graph.ainvoke, so LangGraph's reducer pins it across nodes.
            **(
                {"audit_workflow_id": input_data["audit_workflow_id"]}
                if input_data.get("audit_workflow_id") is not None
                else {"audit_workflow_id": uuid4()}
            ),
            "experiment_id": experiment_id,
            "scope_spec": scope_spec,
            "data_source": input_data["data_source"],
            "split_id": input_data.get("split_id"),
            "validation_suite": input_data.get("validation_suite"),
            "skip_leakage_check": input_data.get("skip_leakage_check", False),
            # Track-2B-v3 D5.0: carry the structural-decider activation flag into
            # the top-level DataPreparerState so adaptive_validity_check's
            # state.get("adaptive_structural_decider_enabled", False) is driven by
            # the per-run PipelineConfig. Dark default (False) when absent.
            "adaptive_structural_decider_enabled": input_data.get(
                "adaptive_structural_decider_enabled", False
            ),
            # #594: carry the per-run FDR firing-driver switch into the state so
            # adaptive_validity_check's state.get("adaptive_fdr_enabled", True) is
            # driven by the caller (the tier0 runner disables it for synthetic
            # FIXTURE regimes). Default True (FDR ON) preserves production behavior.
            "adaptive_fdr_enabled": input_data.get("adaptive_fdr_enabled", True),
            # #604: carry the per-run declared-safe full-immunity switch into the
            # state. Default False (immunity OFF) preserves real-cohort behavior;
            # the tier0 runner sets it True only for legacy synthetic fixtures.
            "adaptive_declared_safe_full_immunity": input_data.get(
                "adaptive_declared_safe_full_immunity", False
            ),
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
                    span.set_output(
                        {
                            "gate_passed": final_state.get("gate_passed"),
                            "qc_status": final_state.get("qc_status"),
                            "overall_score": final_state.get("overall_score"),
                        }
                    )
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
                    # D2.2: consumer-contract fields. Pre-D2.2, downstream
                    # readers (model_trainer/qc_gate_checker.py,
                    # model_selector/agent.py) read these from a
                    # runner-patched qc_report at
                    # scripts/run_tier0_test.py:2295-2300. With D2.2 the
                    # producer writes them directly so consumers can
                    # rely on QCReportSchema's typed contract.
                    "qc_passed": final_state.get("qc_passed", False),
                    "qc_errors": final_state.get("blocking_issues", []),
                    "qc_warnings": final_state.get("warnings", []),
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
                "remediation": {
                    "status": final_state.get("remediation_status", "not_needed"),
                    "attempts": final_state.get("remediation_attempts", 0),
                    "actions_taken": final_state.get("remediation_actions_taken", []),
                    "llm_analysis": final_state.get("llm_analysis"),
                    "root_causes": final_state.get("root_causes", []),
                    "recommended_actions": final_state.get("recommended_actions", []),
                },
                "gate_passed": final_state["gate_passed"],
                # Sampling-frame audit report (Phase-1 Task 1.3 promotes this from
                # advisory to blocking; runners read it to surface a dedicated
                # sampling_frame_audit step result independent of the QC gate).
                "sampling_frame_audit_report": final_state.get("sampling_frame_audit_report"),
                # DataFrames for downstream consumers (Feast registration, model training)
                "train_df": final_state.get("train_df"),
                "validation_df": final_state.get("validation_df"),
                "test_df": final_state.get("test_df"),
                "holdout_df": final_state.get("holdout_df"),
                # Leakage detection results
                "leakage_findings": final_state.get("leakage_findings", []),
                "leakage_severity": final_state.get("leakage_severity", "none"),
                "leaked_features": final_state.get("leaked_features", []),
                # Adaptive validity audit trail (Layer 3 + Layer 4 verdicts).
                # Acceptance criterion #4 of adaptive_temporal_validity_redesign.md:
                # every feature decision has a structured record with layer,
                # evidence, confidence, and remediation.
                "adaptive_verdicts": final_state.get("adaptive_verdicts", []),
                "adaptive_flagged_features": final_state.get("adaptive_flagged_features", []),
                # Leakage remediation results
                "leakage_remediation_status": final_state.get("leakage_remediation_status"),
                "leakage_remediated_features": final_state.get("leakage_remediated_features", []),
                "leakage_dropped_features": final_state.get("leakage_dropped_features", []),
                "leakage_added_features": final_state.get("leakage_added_features", []),
                "leakage_remediation_reasoning": final_state.get("leakage_remediation_reasoning"),
                "leakage_remediation_viable": final_state.get("leakage_remediation_viable"),
                # Phase 1 data-sufficiency pre-flight (sufficiency_check
                # node). DataSufficiencyReport.model_dump() shape; see
                # src/utils/sufficiency_schemas.py. Carries verdict +
                # resolved thresholds + detectable MDE + sensitivity grid.
                "sufficiency_report": final_state.get("sufficiency_report"),
                "power_warnings": final_state.get("power_warnings", []),
            }

            # Log execution time
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data preparation completed in {duration:.2f}s (SLA: {self.sla_seconds}s)")

            # Check SLA
            if duration > self.sla_seconds:
                logger.warning(f"SLA violation: {duration:.2f}s > {self.sla_seconds}s")

            # Persist QC report to database (forward the canonical leakage
            # verdict so the DQ row reflects it — gap G7).
            await self._persist_qc_report(
                output["qc_report"],
                input_data["data_source"],
                leakage_detected=bool(final_state.get("leakage_detected", False)),
            )

            return output

        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise RuntimeError(f"Data preparation failed: {str(e)}") from e

    async def _persist_qc_report(
        self,
        qc_report: Dict[str, Any],
        data_source: str,
        leakage_detected: bool = False,
    ) -> None:
        """Persist QC report to database.

        Args:
            qc_report: QC report dictionary
            data_source: Data source table name
            leakage_detected: The run's canonical leakage verdict (from
                ``leakage_detector``). Persisted to the ml_data_quality_reports
                row so a detected leak is not silently stored as False — the
                row's ``leakage_detected`` column is read by
                ``check_data_quality_gate`` and downstream consumers (gap G7).
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
                # Persist the real leakage verdict (gap G7). store_result defaults
                # this column to False, so omitting it fabricated leakage_detected
                # =False on every row even when the detector flagged a leak.
                "leakage_detected": bool(leakage_detected),
                "data_split": "train",  # QC runs on train split
                "training_run_id": None,  # Set by model_trainer if applicable
            }

            await repo.store_result(db_record)
            logger.info(f"Persisted QC report for {qc_report['experiment_id']}")

        except Exception as e:
            logger.warning(f"Failed to persist QC report: {e}")
