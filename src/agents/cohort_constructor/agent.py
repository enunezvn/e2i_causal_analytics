"""CohortConstructor Agent - Tier 0: ML Foundation.

Patient cohort construction agent that applies FDA/EMA label criteria
to define eligible patient populations for ML analysis.

Position: scope_definer → **cohort_constructor** → data_preparer
Type: Standard (tool-heavy, SLA-bound, no LLM)
SLA: <120 seconds for 100K patients

Outputs:
- Eligible patient IDs matching clinical criteria
- Eligibility statistics and audit trail
- Execution metadata with timing breakdown
- Handoff context for data_preparer

Integration:
- Upstream: scope_definer (receives study parameters)
- Downstream: data_preparer (sends eligible patient list)
- Database: ml_cohort_definitions, ml_cohort_executions
- Observability: MLflow experiment tracking, Opik tracing
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .configs import get_brand_config, list_available_configs
from .constants import AGENT_METADATA, SUPPORTED_BRANDS, Defaults
from .constructor import CohortConstructor
from .graph import create_cohort_constructor_graph
from .memory_hooks import (
    CohortConstructorMemoryHooks,
    contribute_to_memory,
    get_cohort_constructor_memory_hooks,
)
from .state import create_initial_state
from .types import CohortConfig, CohortExecutionResult

logger = logging.getLogger(__name__)


def _get_supabase_client():
    """Get Supabase client (lazy import to avoid circular deps)."""
    try:
        from src.memory.services.factories import get_supabase_client

        return get_supabase_client()
    except Exception as e:
        logger.warning(f"Could not get Supabase client: {e}")
        return None


def _get_cohort_mlflow_logger():
    """Get CohortMLflowLogger (lazy import to avoid circular deps)."""
    try:
        from .observability import get_cohort_mlflow_logger

        return get_cohort_mlflow_logger()
    except Exception as e:
        logger.warning(f"Could not get CohortMLflowLogger: {e}")
        return None


def _get_cohort_opik_tracer():
    """Get CohortOpikTracer (lazy import to avoid circular deps)."""
    try:
        from .observability import get_cohort_opik_tracer

        return get_cohort_opik_tracer()
    except Exception as e:
        logger.warning(f"Could not get CohortOpikTracer: {e}")
        return None


class CohortConstructorAgent:
    """CohortConstructor: Define eligible patient populations for ML analysis.

    This agent is part of the ML Foundation tier (Tier 0). It sits between
    scope_definer and data_preparer, applying FDA/EMA label criteria to
    construct cohorts for downstream analysis.

    The agent supports two execution modes:
    1. Graph mode: Execute via LangGraph workflow (recommended for pipelines)
    2. Direct mode: Execute via CohortConstructor class (faster for standalone use)

    Responsibilities:
    - Apply inclusion criteria (AND logic)
    - Apply exclusion criteria (AND NOT logic)
    - Validate temporal eligibility (lookback/followup periods)
    - Generate comprehensive audit trail
    - Track execution metrics

    Tier: 0 (ML Foundation)
    Type: Standard (tool-heavy, SLA-bound, no LLM)
    SLA: <120 seconds for 100K patients
    """

    def __init__(
        self,
        use_graph: bool = True,
        enable_observability: bool = True,
        enable_memory: bool = True,
        db_client: Optional[Any] = None,
    ):
        """Initialize CohortConstructor agent.

        Args:
            use_graph: If True, use LangGraph workflow. If False, use direct execution.
            enable_observability: If True, enable MLflow/Opik tracking.
            enable_memory: If True, contribute completed constructions to the
                tri-memory architecture (default: True). #883 PR B — the hooks
                existed since the 4-memory rollout but had no caller, so the
                learning-loop readers (get_prior_cohorts /
                get_effective_rules_for_brand) were permanently empty. NOTE:
                ``tier0_integration._store_*`` are SEPARATE direct-DB writes
                (ml_cohort_definitions / ml_cohort_executions), not these hooks.
            db_client: Optional Supabase client for database operations.
        """
        self.use_graph = use_graph
        self.enable_observability = enable_observability
        self.enable_memory = enable_memory
        self._memory_hooks: Optional[CohortConstructorMemoryHooks] = None
        self.db_client = db_client or _get_supabase_client()

        # Initialize cohort-specific observability
        self._mlflow_logger = None
        self._opik_tracer = None
        if enable_observability:
            self._mlflow_logger = _get_cohort_mlflow_logger()
            self._opik_tracer = _get_cohort_opik_tracer()

        # Create graph if needed
        self._graph = None
        if use_graph:
            try:
                self._graph = create_cohort_constructor_graph()
            except Exception as e:
                logger.warning(f"Could not create LangGraph workflow: {e}")
                self.use_graph = False

        logger.info(
            f"CohortConstructorAgent initialized: "
            f"use_graph={self.use_graph}, observability={enable_observability}"
        )

    @property
    def memory_hooks(self) -> Optional[CohortConstructorMemoryHooks]:
        """Lazy-load memory hooks (#883 PR B, mirrors health_score #879)."""
        if self._memory_hooks is None and self.enable_memory:
            try:
                self._memory_hooks = get_cohort_constructor_memory_hooks()
            except Exception as e:
                logger.warning(f"Failed to initialize memory hooks: {e}")
        return self._memory_hooks

    async def _contribute_to_memory(
        self,
        result: CohortExecutionResult,
        state: Dict[str, Any],
        session_id: Optional[str],
    ) -> None:
        """Contribute a completed construction to memory — NON-BLOCKING.

        Caller-side try/except per the settled cross-agent posture (#879 /
        #881; the migration-046 trap lesson): a memory failure must NEVER
        poison the run's status or the returned result.
        ``contribute_to_memory`` itself skips non-success constructions and
        covers all the per-run writes: the working-memory cache, the episodic
        construction record (read back by ``get_prior_cohorts``), and the
        semantic cohort-pattern / eligibility-rule graph nodes (read back by
        ``get_effective_rules_for_brand``) — each individually best-effort
        inside the hook.
        """
        if not self.enable_memory:
            return
        try:
            # Graph mode ends in the CohortConstructorState vocabulary
            # ("completed"); direct mode in the CohortExecutionResult
            # vocabulary ("success"). Normalize to the hook's gate/reader
            # vocabulary ("success"/"partial") so graph-mode runs are neither
            # skipped by the gate nor invisible to get_prior_cohorts'
            # status filter.
            status = "success" if result.status in ("success", "completed") else result.status
            memory_stats = await contribute_to_memory(
                result={
                    "cohort_id": result.cohort_id,
                    "execution_id": result.execution_id,
                    "status": status,
                },
                state=state,
                memory_hooks=self.memory_hooks,
                session_id=session_id,
            )
            logger.debug(
                f"Memory contribution complete: "
                f"episodic={memory_stats.get('episodic_stored', 0)}, "
                f"semantic={memory_stats.get('semantic_stored', 0)}, "
                f"rules={memory_stats.get('rules_stored', 0)}, "
                f"cached={memory_stats.get('working_cached', 0)}"
            )
        except Exception as e:
            logger.warning(f"Memory contribution failed (non-blocking): {e}")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return AGENT_METADATA.copy()

    @property
    def supported_brands(self) -> List[str]:
        """Return list of supported pharmaceutical brands."""
        return list(SUPPORTED_BRANDS.keys())

    def get_brand_config(self, brand: str, indication: Optional[str] = None) -> CohortConfig:
        """Get pre-built configuration for a pharmaceutical brand.

        Args:
            brand: Brand name (remibrutinib, fabhalta, kisqali)
            indication: Optional indication (pnh, c3g, etc.)

        Returns:
            CohortConfig with FDA/EMA label criteria
        """
        return get_brand_config(brand, indication)

    def list_configurations(self) -> Dict[str, Dict[str, str]]:
        """List all available brand configurations."""
        return list_available_configs()

    async def run(
        self,
        patient_df: pd.DataFrame,
        brand: Optional[str] = None,
        indication: Optional[str] = None,
        config: Optional[CohortConfig] = None,
        environment: str = Defaults.DEFAULT_ENVIRONMENT,
        executed_by: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, CohortExecutionResult]:
        """Execute cohort construction.

        This is the main entry point for the agent. It accepts patient data
        and returns the eligible subset along with execution metadata.

        Args:
            patient_df: DataFrame with patient data
            brand: Brand name for pre-built configuration
            indication: Optional indication for brand configuration
            config: Explicit CohortConfig (overrides brand/indication)
            environment: Execution environment (development/staging/production)
            executed_by: User or system identifier
            session_id: Optional session identifier for memory correlation
                (#883 PR B); when omitted the memory hook generates a UUID.

        Returns:
            Tuple of (eligible_df, execution_result)

        Raises:
            ValueError: If neither brand nor config is provided
        """
        if self.use_graph and self._graph:
            return await self._run_graph(
                patient_df, brand, indication, config, environment, executed_by, session_id
            )

        eligible_df, result = self._run_direct(
            patient_df, brand, indication, config, environment, executed_by
        )
        # #883 PR B: direct mode has no graph state — synthesize the minimal
        # state shape the hook reads (config / eligibility_stats /
        # execution_metadata). run() is async, so the contribution awaits
        # here; run_sync() bypasses memory by design (no event loop).
        direct_config = config
        if direct_config is None and brand:
            direct_config = get_brand_config(brand, indication)
        await self._contribute_to_memory(
            result,
            state={
                "config": direct_config.to_dict() if direct_config else {},
                "eligibility_stats": result.eligibility_stats,
                "execution_metadata": result.execution_metadata,
                "session_id": session_id,
            },
            session_id=session_id,
        )
        return eligible_df, result

    async def _run_graph(
        self,
        patient_df: pd.DataFrame,
        brand: Optional[str],
        indication: Optional[str],
        config: Optional[CohortConfig],
        environment: str,
        executed_by: Optional[str],
        session_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, CohortExecutionResult]:
        """Execute via LangGraph workflow."""
        logger.info(f"Executing CohortConstructor via LangGraph: brand={brand}")

        # Resolve configuration
        cohort_config = config
        if cohort_config is None and brand:
            cohort_config = get_brand_config(brand, indication)

        # Create initial state
        initial_state = create_initial_state(
            brand=brand,
            indication=indication,
            config=cohort_config.to_dict() if cohort_config else None,  # type: ignore[arg-type]
            environment=environment,  # type: ignore[arg-type]
        )

        # Add source population with DataFrame
        initial_state["source_population"] = {
            "dataframe": patient_df,
            "columns": list(patient_df.columns),
            "row_count": len(patient_df),
        }
        initial_state["executed_by"] = executed_by

        # Start Opik trace if available
        trace_context = None
        trace_cm = None
        if self._opik_tracer and cohort_config:
            try:
                trace_cm = self._opik_tracer.trace_cohort_construction(
                    config=cohort_config,
                    patient_count=len(patient_df),
                    metadata={
                        "environment": environment,
                        "executed_by": executed_by,
                    },
                )
                trace_context = trace_cm.__enter__()
            except Exception as e:
                logger.warning(f"Could not start Opik trace: {e}")
                trace_context = None
                trace_cm = None

        try:
            # Execute graph
            if self._graph is None:
                raise RuntimeError("Cohort constructor graph not initialized")
            final_state = await self._graph.ainvoke(initial_state)

            # Extract results
            eligible_ids = final_state.get("eligible_patient_ids", [])

            # Filter DataFrame to eligible patients
            patient_id_field = "patient_journey_id"
            if patient_id_field in patient_df.columns:
                eligible_df = patient_df[
                    patient_df[patient_id_field].astype(str).isin(eligible_ids)
                ]
            else:
                # Fall back to index-based filtering
                eligible_indices = final_state.get("_temporal_eligible_indices", [])
                eligible_df = (
                    patient_df.loc[eligible_indices] if eligible_indices else pd.DataFrame()
                )

            # Create execution result
            result = CohortExecutionResult(
                cohort_id=final_state.get("cohort_id", ""),
                execution_id=final_state.get("execution_metadata", {}).get("execution_id", ""),
                eligible_patient_ids=eligible_ids,
                eligibility_stats=final_state.get("eligibility_stats", {}),
                eligibility_log=[],  # Logs are in eligibility_stats
                patient_assignments=[],  # Not tracked in graph mode
                execution_metadata=final_state.get("execution_metadata", {}),
                status=final_state.get("status", "unknown"),
                error_message=final_state.get("error"),
                error_code=final_state.get("error_code"),
            )

            # Log to MLflow if available
            if self._mlflow_logger and result.status == "success":
                try:
                    self._mlflow_logger.log_cohort_execution(
                        result=result,
                        config=cohort_config,
                    )
                    # Log SLA compliance
                    self._mlflow_logger.log_sla_compliance(
                        execution_time_ms=result.execution_metadata.get("execution_time_ms", 0),
                        patient_count=len(patient_df),
                    )
                except Exception as e:
                    logger.warning(f"Could not log to MLflow: {e}")

            # Log completion to Opik
            if trace_context:
                try:
                    trace_context.log_execution_complete(
                        eligible_count=len(eligible_ids),
                        total_count=len(patient_df),
                        execution_time_ms=result.execution_metadata.get("execution_time_ms", 0),
                        status=result.status,
                    )
                except Exception as e:
                    logger.warning(f"Could not log completion to Opik: {e}")

            # #883 PR B: contribute the completed construction to memory —
            # non-blocking, keyed to the graph outcome (the helper and the
            # hook both gate on result status).
            await self._contribute_to_memory(result, dict(final_state), session_id)

            return eligible_df, result

        except Exception as e:
            # Log error to Opik
            if trace_context:
                try:
                    trace_context.log_error(str(e))
                except Exception:
                    pass
            raise

        finally:
            # End Opik trace
            if trace_cm:
                try:
                    trace_cm.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Could not end Opik trace: {e}")

    def _run_direct(
        self,
        patient_df: pd.DataFrame,
        brand: Optional[str],
        indication: Optional[str],
        config: Optional[CohortConfig],
        environment: str,
        executed_by: Optional[str],
    ) -> Tuple[pd.DataFrame, CohortExecutionResult]:
        """Execute via direct CohortConstructor class."""
        logger.info(f"Executing CohortConstructor directly: brand={brand}")

        # Get or create configuration
        cohort_config = config
        if cohort_config is None:
            if brand is None:
                raise ValueError("Either brand or config must be provided")
            cohort_config = get_brand_config(brand, indication)

        # Start Opik trace if available
        trace_context = None
        trace_cm = None
        if self._opik_tracer:
            try:
                trace_cm = self._opik_tracer.trace_cohort_construction(
                    config=cohort_config,
                    patient_count=len(patient_df),
                    metadata={
                        "environment": environment,
                        "executed_by": executed_by,
                        "mode": "direct",
                    },
                )
                trace_context = trace_cm.__enter__()
            except Exception as e:
                logger.warning(f"Could not start Opik trace: {e}")
                trace_context = None
                trace_cm = None

        try:
            # Create constructor and execute
            constructor = CohortConstructor(cohort_config)
            eligible_df, result = constructor.construct_cohort(patient_df)

            # Log to MLflow if available
            if self._mlflow_logger and result.status == "success":
                try:
                    self._mlflow_logger.log_cohort_execution(
                        result=result,
                        config=cohort_config,
                    )
                    self._mlflow_logger.log_sla_compliance(
                        execution_time_ms=result.execution_metadata.get("execution_time_ms", 0),
                        patient_count=len(patient_df),
                    )
                except Exception as e:
                    logger.warning(f"Could not log to MLflow: {e}")

            # Log completion to Opik
            if trace_context:
                try:
                    trace_context.log_execution_complete(
                        eligible_count=len(eligible_df),
                        total_count=len(patient_df),
                        execution_time_ms=result.execution_metadata.get("execution_time_ms", 0),
                        status=result.status,
                    )
                except Exception as e:
                    logger.warning(f"Could not log completion to Opik: {e}")

            return eligible_df, result

        except Exception as e:
            if trace_context:
                try:
                    trace_context.log_error(str(e))
                except Exception:
                    pass
            raise

        finally:
            if trace_cm:
                try:
                    trace_cm.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Could not end Opik trace: {e}")

    def run_sync(
        self,
        patient_df: pd.DataFrame,
        brand: Optional[str] = None,
        indication: Optional[str] = None,
        config: Optional[CohortConfig] = None,
    ) -> Tuple[pd.DataFrame, CohortExecutionResult]:
        """Synchronous execution (bypasses graph, uses direct mode).

        This is a convenience method for non-async contexts.

        Args:
            patient_df: DataFrame with patient data
            brand: Brand name for pre-built configuration
            indication: Optional indication
            config: Explicit CohortConfig

        Returns:
            Tuple of (eligible_df, execution_result)
        """
        return self._run_direct(
            patient_df,
            brand,
            indication,
            config,
            environment=Defaults.DEFAULT_ENVIRONMENT,
            executed_by=None,
        )

    async def validate_config(
        self,
        brand: Optional[str] = None,
        indication: Optional[str] = None,
        config: Optional[CohortConfig] = None,
    ) -> Dict[str, Any]:
        """Validate cohort configuration without executing.

        Useful for pre-flight checks before running full cohort construction.

        Args:
            brand: Brand name
            indication: Optional indication
            config: Explicit configuration

        Returns:
            Validation result with any errors or warnings
        """
        result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config_summary": None,
        }

        try:
            if config:
                cohort_config = config
            elif brand:
                cohort_config = get_brand_config(brand, indication)
            else:
                result["valid"] = False
                result["errors"].append("No brand or configuration provided")
                return result

            # Validate structure
            if not cohort_config.inclusion_criteria:
                result["warnings"].append("No inclusion criteria defined")

            if not cohort_config.exclusion_criteria:
                result["warnings"].append("No exclusion criteria defined")

            if not cohort_config.required_fields:
                result["warnings"].append("No required fields specified")

            # Provide summary
            result["config_summary"] = {
                "cohort_name": cohort_config.cohort_name,
                "brand": cohort_config.brand,
                "indication": cohort_config.indication,
                "inclusion_count": len(cohort_config.inclusion_criteria),
                "exclusion_count": len(cohort_config.exclusion_criteria),
                "required_fields": cohort_config.required_fields,
                "lookback_days": cohort_config.temporal_requirements.lookback_days,
                "followup_days": cohort_config.temporal_requirements.followup_days,
            }

        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))

        return result


# Factory function for easy agent creation
def create_cohort_constructor_agent(
    use_graph: bool = True,
    enable_observability: bool = True,
) -> CohortConstructorAgent:
    """Create a CohortConstructorAgent instance.

    Args:
        use_graph: Use LangGraph workflow
        enable_observability: Enable MLflow/Opik tracking

    Returns:
        Configured CohortConstructorAgent
    """
    return CohortConstructorAgent(
        use_graph=use_graph,
        enable_observability=enable_observability,
    )
