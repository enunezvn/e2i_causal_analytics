"""Scope Definer Agent - Tier 0: ML Foundation.

Transforms business objectives into formal ML problem specifications with
measurable success criteria.

Outputs:
- ScopeSpec: ML problem specification (problem_type, target_variable, features, constraints)
- SuccessCriteria: Performance thresholds (minimum_auc, minimum_precision, etc.)

Integration:
- Downstream: data_preparer, model_selector (consumes ScopeSpec)
- Database: ml_experiments table
- Memory: Procedural memory (successful scope patterns)
- Observability: Opik tracing
"""

import logging
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from .graph import create_scope_definer_graph
from .memory_hooks import ScopeDefinerMemoryHooks
from .state import ScopeDefinerState

logger = logging.getLogger(__name__)


async def _get_experiment_repository():
    """Get MLExperimentRepository with async client (lazy import to avoid circular deps)."""
    try:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.ml_experiment import MLExperimentRepository

        client = await get_async_supabase_client()
        return MLExperimentRepository(supabase_client=client)
    except Exception as e:
        logger.warning(f"Could not get experiment repository: {e}")
        return None


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


def _get_procedural_memory():
    """Get procedural memory client (lazy import with graceful degradation)."""
    try:
        from src.memory.procedural_memory import get_procedural_memory_client

        return get_procedural_memory_client()
    except Exception as e:
        logger.debug(f"Procedural memory not available: {e}")
        return None


class ScopeDefinerAgent:
    """Scope Definer: Transform business requirements into ML specifications.

    This agent is the entry point to the ML Foundation tier. It takes natural
    language business objectives and produces formal ScopeSpec and SuccessCriteria
    that drive all downstream ML agents.

    Responsibilities:
    - Classify ML problem type (classification, regression, causal, etc.)
    - Infer target variable and prediction horizon
    - Define population criteria (inclusion/exclusion)
    - Specify feature requirements and exclusions
    - Set regulatory, ethical, and technical constraints
    - Define measurable success criteria
    - Establish baseline expectations

    Tier: 0 (ML Foundation)
    Type: Standard (no LLM, pure computation)
    SLA: <5 seconds
    """

    # Class attributes per contract
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "scope_definer"
    agent_type = "standard"  # No LLM, pure computation
    sla_seconds = 5
    tools: List[Any] = []  # Standard agent, no external tools

    def __init__(self):
        """Initialize the ScopeDefinerAgent."""
        self.graph = create_scope_definer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scope definition workflow.

        Args:
            input_data: Dictionary with required fields:
                - problem_description (str): Natural language problem description
                - business_objective (str): Business objective this model serves
                - target_outcome (str): Target outcome (e.g., "Increase prescriptions")
                Optional fields:
                - problem_type_hint (str): Hint for problem type
                - target_variable (str): Target variable name if known
                - candidate_features (List[str]): Candidate features
                - time_budget_hours (float): Max training time
                - performance_requirements (Dict): Performance thresholds
                - brand (str): Brand context
                - region (str): Region context
                - use_case (str): Use case category
                - prediction_timestamp (datetime | str | pd.Timestamp): Inference
                  cutoff time the model is meant to predict from. Plumbed
                  through ``scope_spec`` so downstream agents (Tier 1-5) can
                  anchor lookback windows and post-prediction filters. Block
                  1B threads it but does not yet consume it.
                - cost_matrix (Dict[str, float]): Optional business cost
                  matrix mapping confusion-matrix outcomes (``tp``/``fp``/
                  ``fn``/``tn``) to per-prediction monetary value. When
                  provided, propagated onto ``scope_spec["cost_matrix"]`` so
                  the evaluator can compute ``business_utility`` (Block 5,
                  finding #10).

        Returns:
            Dictionary with:
                - scope_spec (Dict): Complete ScopeSpec
                - success_criteria (Dict): Complete SuccessCriteria
                - experiment_id (str): Unique experiment identifier
                - experiment_name (str): Human-readable name
                - validation_warnings (List[str]): Non-blocking warnings
                - error (Optional[str]): Error message if failed
                - error_type (Optional[str]): Error type if failed

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        required_fields = ["problem_description", "business_objective", "target_outcome"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(
                    f"{field} is required. Please provide: {', '.join(required_fields)}"
                )

        # Construct initial state
        initial_state: ScopeDefinerState = {
            # D1.2: thread caller-provided audit_workflow_id. Backlog #1
            # (closed 2026-05-09) tightened the State contract from
            # ``Field(default_factory=uuid4)`` to plain ``UUID`` (required,
            # no default) to fix the LangGraph channel-reducer bug
            # (default_factory firing on every Schema reconstruction).
            # Caller-provided UUID is preferred; absent that, generate one at
            # the agent boundary. Either way the UUID is set ONCE before
            # graph.ainvoke, so LangGraph's reducer pins it across nodes.
            **(
                {"audit_workflow_id": input_data["audit_workflow_id"]}
                if input_data.get("audit_workflow_id") is not None
                else {"audit_workflow_id": uuid4()}
            ),
            # Required inputs
            "problem_description": input_data["problem_description"],
            "business_objective": input_data["business_objective"],
            "target_outcome": input_data["target_outcome"],
            # Optional inputs
            "problem_type_hint": input_data.get("problem_type_hint"),
            "target_variable": input_data.get("target_variable"),
            "candidate_features": input_data.get("candidate_features"),
            "time_budget_hours": input_data.get("time_budget_hours"),
            "performance_requirements": input_data.get("performance_requirements", {}),
            "brand": input_data.get("brand", "unknown"),
            "region": input_data.get("region", "all"),
            "use_case": input_data.get("use_case", "commercial_targeting"),
            # Layer 5 manifest opt-in: forward the resolved cohort manifest
            # source verbatim so scope_builder copies it onto scope_spec. Unset
            # → None (no manifest pass; cross-cohort false-positive guard).
            "feature_manifest_source": input_data.get("feature_manifest_source"),
            # Block 1B: surface prediction_timestamp into the state so the
            # scope_builder node can copy it onto scope_spec verbatim.
            "prediction_timestamp": input_data.get("prediction_timestamp"),
            # Block 5: business cost matrix for utility-weighted evaluation.
            "cost_matrix": input_data.get("cost_matrix"),
            # Adaptive success criteria pre-eval inputs (task 05 of
            # adaptive_success_criteria plan). Forwarded verbatim into the
            # state so ``criteria_validator.define_success_criteria`` can
            # stash them on ``success_criteria['_adaptive_inputs']`` when
            # ADAPTIVE_CRITERIA=true. Each is optional — missing inputs
            # cause the validator to fall back to fixed thresholds with
            # ``criteria_source="adaptive_fallback_to_fixed"``.
            "n_samples": input_data.get("n_samples"),
            "prevalence": input_data.get("prevalence"),
            "feature_count": input_data.get("feature_count"),
            "regime": input_data.get("regime"),
        }

        start_time = datetime.now()
        logger.info("Starting scope definition pipeline")

        # Execute the graph with optional Opik tracing
        opik = _get_opik_connector()
        try:
            # Wrap execution in Opik trace if available
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="define_scope",
                    metadata={
                        "tier": self.tier,
                        "problem_type_hint": initial_state.get("problem_type_hint"),
                        "brand": initial_state.get("brand"),
                        "region": initial_state.get("region"),
                    },
                    tags=[self.agent_name, "tier_0", "scope_definition"],
                    input_data={"business_objective": input_data["business_objective"]},
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    if span:
                        span.set_output(
                            {
                                "experiment_id": final_state.get("experiment_id"),
                                "problem_type": final_state.get("scope_spec", {}).get(
                                    "problem_type"
                                ),
                                "validation_passed": final_state.get("validation_passed"),
                            }
                        )
            else:
                final_state = await self.graph.ainvoke(initial_state)

            # Check for errors
            if final_state.get("error"):
                return {
                    "error": final_state["error"],
                    "error_type": final_state.get("error_type", "unknown_error"),
                }

            # Construct output
            output = {
                # Primary outputs
                "scope_spec": final_state.get("scope_spec", {}),
                "success_criteria": final_state.get("success_criteria", {}),
                # Identifiers
                "experiment_id": final_state.get("experiment_id", ""),
                "experiment_name": final_state.get("experiment_name", ""),
                # Validation results
                "validation_passed": final_state.get("validation_passed", False),
                "validation_warnings": final_state.get("validation_warnings", []),
                "validation_errors": final_state.get("validation_errors", []),
                # Metadata
                "created_at": final_state.get("created_at", ""),
                "created_by": final_state.get("created_by", "scope_definer"),
            }

            # Persist to database (ml_experiments table)
            await self._persist_scope_spec(output)

            # Update procedural memory with successful pattern
            await self._update_procedural_memory(output)

            # Populate the semantic knowledge graph (e2i_causal) with the experiment
            # scope so Tier 0 runs actually grow it and the read-hooks return real
            # context (#749 — store_experiment_pattern was defined but never called).
            await self._update_semantic_memory(output)

            # Record the scope definition to episodic memory (#749 — store_scope_definition
            # was defined but never called from run() and used a non-existent insert API).
            await self._update_episodic_memory(output)

            # Log execution time
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Scope definition completed in {duration:.2f}s (SLA: {self.sla_seconds}s)")

            # Check SLA
            if duration > self.sla_seconds:
                logger.warning(f"SLA violation: {duration:.2f}s > {self.sla_seconds}s")

            return output

        except Exception as e:
            logger.error(f"Scope definition failed: {e}", exc_info=True)
            return {
                "error": f"Graph execution failed: {str(e)}",
                "error_type": "graph_execution_error",
            }

    async def _persist_scope_spec(self, output: Dict[str, Any]) -> None:
        """Persist ScopeSpec to ml_experiments table.

        Graceful degradation: If repository is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing scope_spec and success_criteria
        """
        try:
            repo = await _get_experiment_repository()
            if repo is None:
                logger.debug("Skipping experiment persistence (no repository)")
                return

            scope_spec = output.get("scope_spec", {})
            success_criteria = output.get("success_criteria", {})

            # Coalesce sentinel values to NULL — the in-memory state uses
            # "all" / "unknown" for "no specific filter", but the DB enums
            # (region_type, brand_type) don't include those values, so NULL
            # is the right representation at the persistence boundary.
            raw_region = scope_spec.get("region")
            db_region = None if raw_region in (None, "", "all") else raw_region
            raw_brand = scope_spec.get("brand")
            db_brand = None if raw_brand in (None, "", "unknown") else raw_brand

            # Create experiment record
            result = await repo.create_experiment(
                name=output.get("experiment_name", f"exp_{output.get('experiment_id', 'unknown')}"),
                mlflow_experiment_id=output.get("experiment_id", ""),
                prediction_target=scope_spec.get("target_variable", ""),
                description=scope_spec.get("problem_description", ""),
                brand=db_brand,
                region=db_region,
                created_by="scope_definer",
                success_criteria=success_criteria,
            )

            if result:
                logger.info(f"Persisted experiment: {output.get('experiment_id')}")
            else:
                logger.debug("Experiment not persisted (no result returned)")

        except Exception as e:
            logger.warning(f"Failed to persist experiment: {e}")

    async def _update_procedural_memory(self, output: Dict[str, Any]) -> None:
        """Update procedural memory with successful scope pattern.

        Graceful degradation: If memory is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing scope_spec and success_criteria
        """
        try:
            memory = _get_procedural_memory()
            if memory is None:
                logger.debug("Procedural memory not available, skipping update")
                return

            scope_spec = output.get("scope_spec", {})

            # Store successful scope pattern for future reference
            await memory.store_pattern(
                agent_name=self.agent_name,
                pattern_type="scope_definition",
                pattern_data={
                    "problem_type": scope_spec.get("problem_type"),
                    "brand": scope_spec.get("brand"),
                    "region": scope_spec.get("region"),
                    "target_variable": scope_spec.get("target_variable"),
                    "success_criteria": output.get("success_criteria", {}),
                    "experiment_id": output.get("experiment_id"),
                },
                timestamp=output.get("created_at"),
            )

            logger.info(f"Updated procedural memory for experiment: {output.get('experiment_id')}")

        except Exception as e:
            logger.debug(f"Failed to update procedural memory: {e}")

    async def _update_semantic_memory(self, output: Dict[str, Any]) -> None:
        """Populate the semantic knowledge graph (FalkorDB ``e2i_causal``) with the
        experiment scope (#749).

        Mirrors ``_update_procedural_memory`` — graceful degradation: if semantic
        memory is unavailable the write is skipped, never raised. The
        ``store_experiment_pattern`` hook (Experiment + ProblemType + HAS_TYPE) was
        defined but never invoked, so Tier 0 runs left ``e2i_causal`` unpopulated.

        Args:
            output: Agent output containing scope_spec, success_criteria, experiment_id.
        """
        try:
            experiment_id = output.get("experiment_id")
            if not experiment_id:
                logger.debug("No experiment_id; skipping semantic-graph update")
                return

            scope_spec = output.get("scope_spec", {})
            problem_type = scope_spec.get("problem_type") or "unknown"
            target_variable = scope_spec.get("target_variable") or ""

            hooks = ScopeDefinerMemoryHooks()
            await hooks.store_experiment_pattern(
                experiment_id=str(experiment_id),
                experiment_name=(
                    scope_spec.get("experiment_name")
                    or f"{problem_type}:{target_variable or 'target'}"
                ),
                problem_type=problem_type,
                target_variable=target_variable,
                features=scope_spec.get("features", []) or [],
                success_criteria=output.get("success_criteria", {}) or {},
            )
            logger.info(f"Updated semantic graph (e2i_causal) for experiment: {experiment_id}")

        except Exception as e:
            logger.debug(f"Failed to update semantic memory: {e}")

    async def _update_episodic_memory(self, output: Dict[str, Any]) -> None:
        """Record the scope definition to EPISODIC memory (#749).

        ``store_scope_definition`` was defined but never called from ``run()`` AND
        called a non-existent ``insert_episodic_memory`` signature — both fixed
        (compat shim + migration 039). Graceful degradation. ``session_id`` is the
        ``audit_workflow_id`` (uuid column) or a fresh UUID — never the non-UUID
        ``experiment_id``.
        """
        try:
            experiment_id = output.get("experiment_id")
            if not experiment_id:
                return
            session_id = str(output.get("audit_workflow_id") or uuid4())
            hooks = ScopeDefinerMemoryHooks()
            await hooks.store_scope_definition(session_id=session_id, result=output, state=output)
        except Exception as e:
            logger.debug(f"Failed to update episodic memory: {e}")
