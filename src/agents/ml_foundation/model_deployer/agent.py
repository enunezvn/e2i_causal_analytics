"""Model Deployer Agent - STANDARD.

Manages model lifecycle from development through production.

Responsibilities:
- Model registration in MLflow
- Stage promotions (dev → staging → shadow → production)
- BentoML deployments
- Health checks
- Rollback management

Outputs:
- DeploymentManifest: Deployment configuration and status
- VersionRecord: MLflow version record
- Rollback availability

Integration:
- Upstream: model_trainer, feature_analyzer
- Downstream: Tier 1-5 agents (via prediction endpoints)
- Database: ml_deployments, ml_model_registry
- Memory: Procedural memory (successful deployment patterns)
- Observability: Opik tracing
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from .graph import create_model_deployer_graph
from .memory_hooks import ModelDeployerMemoryHooks
from .state import ModelDeployerState

logger = logging.getLogger(__name__)


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


class ModelDeployerAgent:
    """Model Deployer: Manage model lifecycle and deployments.

    Handles stage promotions, deployments, and rollbacks.
    """

    # Agent metadata
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "model_deployer"
    agent_type = "standard"
    sla_seconds = 30
    tools = ["mlflow", "bentoml"]  # MLflow for registry, BentoML for deployment

    def __init__(self):
        """Initialize model_deployer agent."""
        self.graph = create_model_deployer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment workflow.

        Args:
            input_data: Input data conforming to ModelDeployerInput contract
                Required fields:
                - model_uri: str (MLflow model URI)
                - experiment_id: str
                - validation_metrics: Dict (ValidationMetrics from training)
                - success_criteria_met: bool
                - deployment_name: str
                Optional fields:
                - shap_analysis_id: str
                - target_environment: "staging" | "shadow" | "production"
                - resources: Dict[str, str] ({"cpu": "2", "memory": "4Gi"})
                - max_batch_size: int
                - max_latency_ms: int
                - deployment_action: "register" | "promote" | "deploy"

        Returns:
            Output data conforming to ModelDeployerOutput contract
        """
        # Validate required inputs
        required_fields = [
            "model_uri",
            "experiment_id",
            "validation_metrics",
            "success_criteria_met",
            "deployment_name",
        ]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Prepare initial state
        initial_state: ModelDeployerState = {
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
            # Input fields
            "model_uri": input_data["model_uri"],
            "experiment_id": input_data["experiment_id"],
            "validation_metrics": input_data["validation_metrics"],
            "success_criteria_met": input_data["success_criteria_met"],
            "deployment_name": input_data["deployment_name"],
            # Optional fields
            "shap_analysis_id": input_data.get("shap_analysis_id"),
            "target_environment": input_data.get("target_environment", "staging"),
            "resources": input_data.get("resources", {"cpu": "2", "memory": "4Gi"}),
            "max_batch_size": input_data.get("max_batch_size", 100),
            "max_latency_ms": input_data.get("max_latency_ms", 100),
            "deployment_action": input_data.get("deployment_action", "deploy"),
            # Shadow mode metrics (for production promotion)
            "shadow_mode_duration_hours": input_data.get("shadow_mode_duration_hours", 0),
            "shadow_mode_requests": input_data.get("shadow_mode_requests", 0),
            "shadow_mode_error_rate": input_data.get("shadow_mode_error_rate", 1.0),
            "shadow_mode_latency_p99_ms": input_data.get("shadow_mode_latency_p99_ms", 999),
        }
        # v5 Gate C1 (2026-05-11): thread cohort identity through to
        # the deployer state so validate_promotion's
        # build_regulatory_deployment_manifest can resolve the cohort
        # authorization policy. The two accepted shapes:
        # - state["scope_spec"]["feature_manifest_source"] (matches the
        #   upstream data_preparer / model_trainer state contract).
        # - state["feature_manifest_source"] (flat fallback for
        #   standalone invocations or checkpoint replays).
        # Both are stashed on the initial_state when present so the
        # manifest builder finds them in either shape.
        if "scope_spec" in input_data:
            # Codex pass-3 HIGH: ModelDeployerState declares
            # ``scope_spec: Optional[Dict[str, Any]]`` to match LangGraph's
            # channel-reducer round-trip requirements. Typed
            # ``ScopeSpecSchema`` Pydantic instances are rejected at
            # state validation time, so we normalize to dict here at
            # the agent boundary. Plain dicts pass through unchanged;
            # ``None`` is excluded by the outer ``if "scope_spec" in
            # input_data`` guard.
            scope_spec_input = input_data["scope_spec"]
            if hasattr(scope_spec_input, "model_dump"):
                scope_spec_input = scope_spec_input.model_dump()
            initial_state["scope_spec"] = scope_spec_input
        if "feature_manifest_source" in input_data:
            initial_state["feature_manifest_source"] = input_data["feature_manifest_source"]

        # Execute LangGraph workflow with optional Opik tracing
        start_time = datetime.now(timezone.utc)
        experiment_id = input_data["experiment_id"]
        deployment_name = input_data["deployment_name"]
        target_environment = input_data.get("target_environment", "staging")
        deployment_action = input_data.get("deployment_action", "deploy")

        logger.info(
            f"Starting model deployment for experiment {experiment_id}, "
            f"deployment={deployment_name}, target={target_environment}, action={deployment_action}"
        )

        opik = _get_opik_connector()
        try:
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="deploy_model",
                    metadata={
                        "tier": self.tier,
                        "experiment_id": experiment_id,
                        "deployment_name": deployment_name,
                        "target_environment": target_environment,
                        "deployment_action": deployment_action,
                    },
                    tags=[self.agent_name, "tier_0", "model_deployment"],
                    input_data={
                        "experiment_id": experiment_id,
                        "deployment_name": deployment_name,
                        "target_environment": target_environment,
                    },
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    if span and not final_state.get("error"):
                        span.set_output(
                            {
                                "deployment_id": final_state.get("deployment_id"),
                                "deployment_successful": final_state.get("deployment_successful"),
                                "health_check_passed": final_state.get("health_check_passed"),
                                "current_stage": final_state.get("current_stage"),
                            }
                        )
            else:
                final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.exception(f"Model deployment failed: {e}")
            raise RuntimeError(f"Model deployment workflow failed: {str(e)}") from e

        # Check for errors
        if final_state.get("error"):
            error_msg = final_state["error"]
            error_type = final_state.get("error_type", "unknown")
            raise RuntimeError(f"{error_type}: {error_msg}")

        # Build outputs
        deployment_manifest = self._build_deployment_manifest(final_state)
        version_record = self._build_version_record(final_state)

        # Determine overall status
        promotion_successful = final_state.get("promotion_successful", False)
        deployment_successful = final_state.get("deployment_successful", False)
        deployment_action = final_state.get("deployment_action", "deploy")

        if deployment_action in ("promote", "register"):
            # Registration/promotion only — no packaging or deployment needed
            overall_status = "completed" if promotion_successful else "failed"
            deployment_successful = promotion_successful
        else:
            # Full deployment
            overall_status = (
                "completed" if (promotion_successful and deployment_successful) else "partial"
            )

        # Build output
        output = {
            # Deployment manifest
            "deployment_manifest": deployment_manifest,
            # Version record
            "version_record": version_record,
            # BentoML tag
            "bentoml_tag": final_state.get("final_bento_tag", ""),
            # Status flags
            "deployment_successful": deployment_successful,
            "health_check_passed": final_state.get("health_check_passed", False),
            "rollback_available": final_state.get("rollback_available", False),
            # Overall status
            "status": overall_status,
            # v5 Gate C1 (2026-05-11): surface the cohort-scoped
            # regulatory deployment manifest that validate_promotion
            # produced. This is the load-bearing v5 deliverable —
            # the payload the deployer operator attaches to a deployment
            # PR for T2.6c authorization. Signal-only; does NOT mutate
            # promotion_successful.
            "regulatory_deployment_manifest": final_state.get("regulatory_deployment_manifest"),
        }

        # Store to database (ml_deployments and ml_model_registry)
        await self._store_to_database(output, final_state)

        # Update procedural memory with successful deployment pattern
        if output.get("deployment_successful"):
            await self._update_procedural_memory(output, final_state)
            # Populate the semantic knowledge graph (e2i_causal) with the deployment
            # so Tier 0 runs grow it and read-hooks return real context
            # (#749 — store_deployment_pattern was defined but never called).
            await self._update_semantic_memory(output, final_state)
            # Record the deployment to episodic memory (#749 — store_deployment was
            # defined but never called from run() and used a non-existent insert API).
            await self._update_episodic_memory(output, final_state)

        # Log execution time and SLA check
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Model deployment complete for {experiment_id}: "
            f"status={overall_status}, environment={target_environment} "
            f"in {duration:.2f}s"
        )

        if duration > self.sla_seconds:
            logger.warning(
                f"SLA violation: {duration:.2f}s > {self.sla_seconds}s "
                f"for deployment {deployment_name}"
            )

        return output

    def _build_deployment_manifest(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build DeploymentManifest output structure.

        Args:
            state: Final agent state

        Returns:
            DeploymentManifest dict conforming to contract
        """
        return {
            "deployment_id": state.get("deployment_id", ""),
            "experiment_id": state["experiment_id"],
            "model_version": str(state.get("model_version", 1)),
            # Environment
            "environment": state.get("target_environment", "staging"),
            "endpoint_url": state.get("endpoint_url", ""),
            # Resources
            "resources": state.get("resources", {"cpu": "2", "memory": "4Gi"}),
            # Status
            "status": state.get("deployment_status", "pending"),
            "deployed_at": state.get("deployed_at", ""),
            # Health
            "health_check_url": state.get("health_check_url", ""),
            "metrics_url": state.get("metrics_url", ""),
        }

    def _build_version_record(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build VersionRecord output structure.

        Args:
            state: Final agent state

        Returns:
            VersionRecord dict conforming to contract
        """
        return {
            "registered_model_name": state.get("registered_model_name", ""),
            "version": state.get("model_version", 1),
            "stage": state.get("current_stage", "None"),
            "description": state.get("promotion_reason", "Automated deployment"),
        }

    async def _store_to_database(self, output: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Store deployment to ml_deployments and update ml_model_registry.

        Args:
            output: Agent output to store
            state: Final agent state
        """
        try:
            # Import repositories lazily to avoid circular imports
            from src.repositories.deployment import MLDeploymentRepository
            from src.repositories.ml_experiment import MLModelRegistryRepository

            deployment_repo = MLDeploymentRepository()
            registry_repo = MLModelRegistryRepository()

            # Parse model_registry_id from state if available
            model_registry_id: Optional[UUID] = None
            if state.get("model_registry_id"):
                try:
                    model_registry_id = UUID(str(state["model_registry_id"]))
                except ValueError:
                    logger.warning(f"Invalid model_registry_id: {state.get('model_registry_id')}")

            # F4 (audit, codex round-2): default to NOT-persisted. db_persisted
            # flips to True only after a row is CONFIRMED written below — never
            # before — so a swallowed write failure (the outer except logs but
            # does not fail the deployment) can't leave a fabricated True.
            output["db_persisted"] = False

            # 1. Write to ml_deployments table
            output.get("deployment_manifest", {})
            deployment_config = {
                "resources": state.get("resources", {"cpu": "2", "memory": "4Gi"}),
                "max_batch_size": state.get("max_batch_size", 100),
                "max_latency_ms": state.get("max_latency_ms", 100),
                "bento_tag": output.get("bentoml_tag", ""),
                "deployment_action": state.get("deployment_action", "deploy"),
            }

            # Create deployment record
            if model_registry_id is None:
                # F4 (audit, codex round-1): do NOT silently drop the deployment
                # record. ``model_registry_id`` is not yet produced by any node
                # (register_model writes to MLflow, not to ml_model_registry),
                # so the ml_deployments row cannot be FK-linked and is skipped.
                # Surface it loudly + on the output so this is NOT silent data
                # loss (the audit's "handle gracefully without silent data
                # loss"). Producing model_registry_id requires a real
                # ml_model_registry write — which needs a valid experiment_id
                # UUID FK + algorithm/metrics — a distinct persistence feature
                # tracked as an F4 follow-up.
                logger.error(
                    "Deployment '%s' (status=%s) completed but its ml_deployments "
                    "record was NOT persisted: no model_registry_id "
                    "(ml_model_registry write is unwired). See F4 follow-up.",
                    state.get("deployment_name", ""),
                    output.get("status"),
                )
                output["db_persisted"] = False
                output["db_persist_skipped_reason"] = (
                    "no model_registry_id — ml_model_registry write unwired (F4 follow-up)"
                )
                return

            # F4 (audit, codex round-3): create_deployment() builds an in-memory
            # MLDeployment(id=uuid4()) and returns it EVEN WITH NO DB CLIENT (it
            # only inserts when self.client is set). So a truthy deployment.id is
            # NOT proof of a write. Require a real client before trusting it.
            if not getattr(deployment_repo, "client", None):
                output["db_persist_skipped_reason"] = (
                    "no Supabase client — ml_deployments row not written"
                )
                logger.error(
                    "Deployment '%s' completed but ml_deployments was NOT written "
                    "(no Supabase client) — db_persisted=False",
                    state.get("deployment_name", ""),
                )
                return

            deployment = await deployment_repo.create_deployment(
                model_registry_id=model_registry_id,
                deployment_name=state.get("deployment_name", ""),
                environment=state.get("target_environment", "staging"),
                endpoint_name=state.get("endpoint_name"),
                endpoint_url=state.get("endpoint_url"),
                deployed_by=state.get("deployed_by", "model_deployer_agent"),
                deployment_config=deployment_config,
            )

            # F4 (audit, codex round-4): create_deployment() falls through to a
            # prebuilt in-memory MLDeployment(id=uuid4()) when the insert returns
            # no rows, so a truthy deployment.id is NOT proof of a DB write.
            # Verify the row is actually DB-backed by re-reading it (get_by_id
            # returns None with no client / no row), so db_persisted can never
            # be a fabricated True.
            persisted_row = (
                await deployment_repo.get_by_id(str(deployment.id))
                if deployment and deployment.id
                else None
            )
            if persisted_row is None:
                output["db_persist_skipped_reason"] = (
                    "ml_deployments row not confirmed in DB after create_deployment"
                )
                logger.error(
                    "Deployment '%s' completed but its ml_deployments row was not "
                    "confirmed in the DB — db_persisted=False",
                    state.get("deployment_name", ""),
                )
                return

            # Row CONFIRMED present in the DB -> honest to mark persisted.
            output["db_persisted"] = True

            # Update deployment status based on outcome
            if deployment and deployment.id:
                status = "active" if output.get("deployment_successful") else "pending"
                await deployment_repo.update_status(
                    deployment_id=deployment.id,
                    new_status=status,
                )

                # Update metrics if available
                shadow_metrics = state.get("shadow_mode_metrics", {})
                if shadow_metrics:
                    await deployment_repo.update_metrics(
                        deployment_id=deployment.id,
                        shadow_metrics=shadow_metrics,
                        latency_p99_ms=state.get("shadow_mode_latency_p99_ms"),
                        error_rate=state.get("shadow_mode_error_rate"),
                    )

                logger.info(f"Created deployment record: {deployment.id}")

            # 2. Update ml_model_registry table if promotion occurred
            if model_registry_id and state.get("promotion_successful"):
                new_stage = state.get("current_stage", "staging")
                await registry_repo.transition_stage(
                    model_id=model_registry_id,
                    new_stage=new_stage,
                    archive_existing=(new_stage == "production"),
                )
                logger.info(f"Updated model {model_registry_id} stage to {new_stage}")

        except ImportError as e:
            # Repos unavailable (e.g. offline test env) — no row written.
            output["db_persisted"] = False
            output["db_persist_skipped_reason"] = f"repository import failed: {e}"
            logger.warning(f"Repository import failed (expected in testing): {e}")
        except Exception as e:
            # F4 (audit, codex round-2): a swallowed write failure must NOT leave
            # a fabricated db_persisted=True. Log error but don't fail the
            # deployment; persistence is honestly marked False.
            output["db_persisted"] = False
            output["db_persist_skipped_reason"] = f"database storage failed: {e}"
            logger.error(f"Database storage failed: {e}")

    async def _update_procedural_memory(
        self, output: Dict[str, Any], state: Dict[str, Any]
    ) -> None:
        """Update procedural memory with successful deployment pattern.

        Graceful degradation: If memory is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing deployment result
            state: Final agent state
        """
        try:
            memory = _get_procedural_memory()
            if memory is None:
                logger.debug("Procedural memory not available, skipping update")
                return

            # Store successful deployment pattern for future reference
            await memory.store_pattern(
                agent_name=self.agent_name,
                pattern_type="model_deployment",
                pattern_data={
                    "deployment_name": state.get("deployment_name"),
                    "target_environment": state.get("target_environment"),
                    "deployment_action": state.get("deployment_action"),
                    "deployment_successful": output.get("deployment_successful"),
                    "health_check_passed": output.get("health_check_passed"),
                    "rollback_available": output.get("rollback_available"),
                    "experiment_id": state.get("experiment_id"),
                    "model_version": state.get("model_version"),
                    "current_stage": state.get("current_stage"),
                    "resources": state.get("resources"),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(f"Updated procedural memory for deployment: {state.get('deployment_name')}")

        except Exception as e:
            logger.debug(f"Failed to update procedural memory: {e}")

    async def _update_semantic_memory(self, output: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Populate the semantic knowledge graph (FalkorDB ``e2i_causal``) with the
        deployment (#749).

        Mirrors ``_update_procedural_memory`` — graceful degradation. The
        ``store_deployment_pattern`` hook (Deployment + DEPLOYS_FOR, Rollback +
        ROLLED_BACK) was defined but never invoked, so Tier 0 runs left
        ``e2i_causal`` unpopulated. Called only on a successful deployment (same
        gate as the procedural write).

        Args:
            output: Agent output carrying status / rollback_occurred.
            state: Final agent state carrying experiment_id, deployment_name,
                model_version, target_environment, deployment_action.
        """
        try:
            experiment_id = state.get("experiment_id")
            deployment_id = state.get("deployment_name")
            if not experiment_id or not deployment_id:
                logger.debug(
                    "Missing experiment_id/deployment_name; skipping semantic-graph update"
                )
                return

            hooks = ModelDeployerMemoryHooks()
            await hooks.store_deployment_pattern(
                experiment_id=str(experiment_id),
                deployment_id=str(deployment_id),
                model_version=int(state.get("model_version") or 0),
                target_environment=state.get("target_environment") or "unknown",
                deployment_status=output.get("status") or "deployed",
                deployment_strategy=state.get("deployment_strategy") or "unknown",
                rollback_occurred=bool(output.get("rollback_occurred", False)),
            )
            logger.info(f"Updated semantic graph (e2i_causal) for deployment: {deployment_id}")

        except Exception as e:
            logger.debug(f"Failed to update semantic memory: {e}")

    async def _update_episodic_memory(self, output: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Record the deployment to EPISODIC memory (#749).

        ``store_deployment`` was defined but never called from ``run()`` AND called a
        non-existent ``insert_episodic_memory`` signature — both fixed (compat shim +
        migration 039). Graceful degradation. Called only on a successful deployment
        (same gate as procedural/semantic). ``session_id`` is the
        ``audit_workflow_id`` (uuid column) or a fresh UUID.
        """
        try:
            experiment_id = state.get("experiment_id")
            if not experiment_id:
                return
            session_id = str(state.get("audit_workflow_id") or uuid4())
            hooks = ModelDeployerMemoryHooks()
            await hooks.store_deployment(session_id=session_id, result=output, state=state)
        except Exception as e:
            logger.debug(f"Failed to update episodic memory: {e}")
