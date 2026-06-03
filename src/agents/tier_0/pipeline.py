"""ML Foundation Pipeline - Tier 0 Agent Orchestration.

This module orchestrates the complete ML training pipeline using the 7 Tier 0 agents:
1. ScopeDefiner -> scope_spec, success_criteria
2. DataPreparer -> qc_report, baseline_metrics (QC GATE - blocks if failed)
3. ModelSelector -> model_candidate
4. ModelTrainer -> trained_model, validation_metrics
5. FeatureAnalyzer -> shap_analysis, feature_importance
6. ModelDeployer -> deployment_manifest

The pipeline enforces strict handoff protocols between agents and maintains
observability context throughout the workflow.

Feast Integration:
- Feature freshness validation in QC Gate
- Point-in-time correct training data via Feast
- Feature references logged for reproducibility
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, cast
from uuid import UUID

from src.data.manifests.resolution import resolve_manifest_source
from src.utils.audit_chain import AgentTier, AuditChainService

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    SCOPE_DEFINITION = "scope_definition"
    DATA_PREPARATION = "data_preparation"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    FEATURE_ANALYSIS = "feature_analysis"
    MODEL_DEPLOYMENT = "model_deployment"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the ML Foundation Pipeline."""

    # Stage control
    skip_deployment: bool = False
    skip_feature_analysis: bool = False
    target_environment: str = "staging"

    # Training configuration
    enable_hpo: bool = True
    hpo_trials: int = 50
    hpo_timeout_hours: Optional[float] = None
    early_stopping: bool = False

    # PR #463 Phase 2 — opt-in flag forcing the post-training learning-curve
    # diagnostic to run EVEN WHEN the model passes ``success_criteria_met``.
    # Default False (cost control) — the diagnostic is a 7-bucket cheap proxy
    # fit + power-law extrapolation that runs only on failure cases by
    # default. Setting True flips the gate inside ``learning_curve`` so the
    # full report is produced unconditionally; useful for offline audits or
    # paper-replication runs.
    always_run_learning_curve: bool = False

    # Phase 3 — synthetic-data preview on insufficiency (opt-in; default off).
    # When the post-training learning curve recommends more data
    # (``recommended_additional_samples``), produce a PREVIEW synthetic cohort
    # sized to that recommendation so the operator can inspect it. The preview
    # is written to artifacts and surfaced on ``PipelineResult.synthetic_preview``
    # — it is NEVER auto-mixed into training. ``synthetic_preview_scenario`` is
    # the ``synthetic_v2`` scenario to generate (required for the preview to
    # fire; the operator chooses it — there is no auto-inference).
    synthetic_preview_on_insufficient: bool = False
    synthetic_preview_scenario: Optional[str] = None

    # Opt-in synthetic AUGMENTATION (Phase 3 consumption). When set to a
    # reviewed preview cohort (.npz, e.g. produced by synthetic_preview_*),
    # model_trainer concatenates those rows into the TRAINING split ONLY
    # (never validation/test/holdout), after split-ratio validation and before
    # preprocessing. Strict feature-schema match is enforced — a mismatch is
    # refused (advisory), never silently mixed. Default None = off.
    augmentation_data_path: Optional[str] = None

    # Model selection
    interpretability_required: bool = False
    skip_benchmarks: bool = True
    skip_mlflow: bool = False

    # Data preparation
    # Skips ONLY the legacy name-based detect_leakage node. The data-driven
    # adaptive validity / FDR layer always runs as the safety net and can still
    # escalate leakage findings regardless of this flag (#533, Option 2).
    skip_leakage_check: bool = False
    # Track-2B-v3: activate the deterministic structural causal-role decider
    # (Layer-4) for THIS run's cohort. Dark by default — the decider only
    # decides when this is True AND a feature carries a CausalStructureAttestation
    # (0 attested today). Cohort-scoped via this per-run config rather than a
    # global read-default at adaptive_validity_check.py, so activation is opt-in
    # per pipeline run, not a single un-scoped global flip.
    adaptive_structural_decider_enabled: bool = False
    use_sample_data: bool = False

    # Data-sufficiency pre-flight (Phase 1).
    # ``force_low_power_run`` downgrades blocking SOFT_FAIL verdicts to
    # warnings on the causal_inference path. Safe-by-default (False) to
    # avoid silently producing low-power effect estimates in pharma
    # regulatory contexts. ``sufficiency_strictness_preset`` toggles
    # ``conservative``/``moderate``/``strict`` multipliers on the
    # EPV/regression-ratio thresholds (see sufficiency_defaults.py).
    # When set, both flags propagate into ``scope_spec.sufficiency``
    # so the sufficiency_check node sees them via the resolver
    # hierarchy without a new state-injection path.
    force_low_power_run: bool = False
    sufficiency_strictness_preset: Optional[str] = None

    # Feast Feature Store
    enable_feast: bool = True
    feast_feature_refs: Optional[List[str]] = None  # Feature refs to use
    feast_freshness_check: bool = True  # Check feature freshness in QC gate
    feast_max_staleness_hours: float = 24.0  # Max allowed feature staleness
    feast_fallback_enabled: bool = True  # Fall back to custom store if Feast fails

    # Observability
    enable_observability: bool = True
    sample_rate: float = 1.0

    # Callbacks
    on_stage_complete: Optional[Callable[[PipelineStage, Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[PipelineStage, Exception], None]] = None


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    pipeline_run_id: str
    status: str  # "completed", "failed", "partial"
    current_stage: PipelineStage
    experiment_id: Optional[str] = None

    # Audit chain integration
    audit_workflow_id: Optional[UUID] = None

    # Stage outputs
    scope_spec: Optional[Dict[str, Any]] = None
    success_criteria: Optional[Dict[str, Any]] = None
    qc_report: Optional[Dict[str, Any]] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    sufficiency_report: Optional[Dict[str, Any]] = None  # Phase 1 pre-flight verdict
    # Gate N1 (codex-rescue HIGH-3 / N1-H3): the regulatory_adaptation_entry
    # emitted by data_preparer's leakage_remediation node when it adaptively
    # dropped leaked features. Threaded into the deployer (nested under
    # scope_spec) so the deployer backstop can FAIL CLOSED on an un-ingested
    # adaptation. ``None`` when no leakage remediation occurred.
    regulatory_adaptation_entry: Optional[Any] = None
    model_candidate: Optional[Dict[str, Any]] = None
    training_result: Optional[Dict[str, Any]] = None
    shap_analysis: Optional[Dict[str, Any]] = None
    deployment_result: Optional[Dict[str, Any]] = None

    # Feast feature store outputs
    feature_freshness: Optional[Dict[str, Any]] = None
    feature_refs_used: Optional[List[str]] = None
    feast_enabled: bool = False

    # PR #463 Phase 2 — post-training learning-curve diagnostic emitted by
    # ModelTrainer's ``learning_curve`` node. Shape:
    # ``src.utils.sufficiency_schemas.DataSufficiencyReport``-dict. None when
    # the model met success_criteria (the diagnostic is a no-op) AND the
    # caller didn't set ``PipelineConfig.always_run_learning_curve``.
    # Standalone field rather than a key on a unified ``sufficiency_report``
    # because PR #462 (pre-flight DataPreparer sufficiency check) has not
    # landed on main yet — when it does, this field will be merged into the
    # unified report dict downstream of the merge.
    training_sufficiency_report: Optional[Dict[str, Any]] = None

    # Phase 3 — synthetic-data preview metadata, populated only when
    # ``PipelineConfig.synthetic_preview_on_insufficient`` is set, a scenario
    # is specified, and the learning curve recommended more data. Contains the
    # preview cohort's artifact paths + audit metadata; the cohort itself is on
    # disk and is NOT mixed into training (``auto_mixed_into_training=False``).
    synthetic_preview: Optional[Dict[str, Any]] = None

    # Opt-in synthetic-augmentation audit (Phase 3 consumption). Populated from
    # the trainer's ``training_augmentation`` when augmentation_data_path was
    # set. ``applied`` distinguishes a real augmentation from a refusal (the
    # ``skip_reason`` says why); synthetic rows touch the training split only.
    training_augmentation: Optional[Dict[str, Any]] = None

    # Metadata
    stages_completed: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_seconds: Optional[float] = None


class MLFoundationPipeline:
    """End-to-end ML Foundation pipeline orchestrating all Tier 0 agents.

    This pipeline coordinates the complete ML workflow from business objective
    to deployed model, enforcing strict quality gates and handoff protocols.

    Flow:
        scope_definer -> data_preparer (GATE) -> model_selector -> model_trainer
                                                                      |
                                                   feature_analyzer <-+
                                                                      |
                                                   model_deployer  <--+

    Critical Design Principles:
    - QC Gate: Pipeline STOPS if data_preparer QC fails
    - Handoff Protocols: Strict input/output contracts between agents
    - Observability: All stages wrapped with observability spans
    - Idempotency: Pipeline can be resumed from any stage
    - Error Isolation: Stage failures don't corrupt previous outputs
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the ML Foundation Pipeline.

        Args:
            config: Pipeline configuration. Defaults to standard config.
        """
        self.config = config or PipelineConfig()
        self._agents: Dict[str, Any] = {}
        self._observability = None
        self._feast_adapter = None
        self._feast_initialized = False
        self._audit_service: Optional[AuditChainService] = None

    def _get_audit_service(self) -> Optional[AuditChainService]:
        """Get audit chain service (lazy initialization).

        Returns:
            AuditChainService instance or None if initialization fails
        """
        if self._audit_service is None:
            try:
                from src.repositories import get_supabase_client

                supabase = get_supabase_client()
                if supabase:
                    self._audit_service = AuditChainService(supabase)
                    logger.debug("Audit chain service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize audit service (non-fatal): {e}")
        return self._audit_service

    def _record_audit_entry(
        self,
        workflow_id: Optional[UUID],
        agent_name: str,
        action_type: str,
        duration_ms: int,
        output_data: Dict[str, Any],
        validation_passed: Optional[bool] = None,
        confidence_score: Optional[float] = None,
    ) -> None:
        """Record an audit chain entry for a pipeline stage.

        Args:
            workflow_id: UUID of the current workflow
            agent_name: Name of the agent/stage
            action_type: Type of action performed
            duration_ms: Duration of the stage in milliseconds
            output_data: Stage output data to record
            validation_passed: Optional validation status
            confidence_score: Optional confidence score
        """
        if not workflow_id:
            return

        audit_service = self._get_audit_service()
        if not audit_service:
            return

        try:
            audit_service.add_entry(
                workflow_id=workflow_id,
                agent_name=agent_name,
                agent_tier=AgentTier.ML_FOUNDATION,
                action_type=action_type,
                duration_ms=duration_ms,
                output_data=output_data,
                validation_passed=validation_passed,
                confidence_score=confidence_score,
            )
        except Exception as e:
            logger.warning(f"Failed to record audit entry for {agent_name}: {e}")

    def _get_agent(self, agent_name: str) -> Any:
        """Lazy-load an agent instance.

        Args:
            agent_name: Name of the agent to load

        Returns:
            Agent instance
        """
        if agent_name not in self._agents:
            if agent_name == "scope_definer":
                from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent

                self._agents[agent_name] = ScopeDefinerAgent()
            elif agent_name == "data_preparer":
                from src.agents.ml_foundation.data_preparer.agent import DataPreparerAgent

                self._agents[agent_name] = DataPreparerAgent()
            elif agent_name == "model_selector":
                from src.agents.ml_foundation.model_selector.agent import ModelSelectorAgent

                self._agents[agent_name] = ModelSelectorAgent(mode="conditional")
            elif agent_name == "model_trainer":
                from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

                self._agents[agent_name] = ModelTrainerAgent()
            elif agent_name == "feature_analyzer":
                from src.agents.ml_foundation.feature_analyzer.agent import FeatureAnalyzerAgent

                self._agents[agent_name] = FeatureAnalyzerAgent()
            elif agent_name == "model_deployer":
                from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

                self._agents[agent_name] = ModelDeployerAgent()
            elif agent_name == "observability_connector":
                from src.agents.ml_foundation.observability_connector.agent import (
                    ObservabilityConnectorAgent,
                )

                self._agents[agent_name] = ObservabilityConnectorAgent()
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        return self._agents[agent_name]

    def _get_observability(self):
        """Get observability connector (lazy initialization)."""
        if self._observability is None and self.config.enable_observability:
            try:
                self._observability = self._get_agent("observability_connector")
            except Exception as e:
                logger.warning(f"Failed to initialize observability: {e}")
        return self._observability

    async def _get_feast_adapter(self):
        """Get Feast adapter (lazy initialization).

        Returns:
            FeatureAnalyzerAdapter with Feast enabled, or None if disabled
        """
        if not self.config.enable_feast:
            return None

        if self._feast_initialized:
            return self._feast_adapter

        try:
            from src.feature_store.client import FeatureStoreClient
            from src.feature_store.feature_analyzer_adapter import (
                get_feature_analyzer_adapter,
            )

            # Create feature store client
            fs_client = FeatureStoreClient()

            # Create adapter with Feast enabled
            self._feast_adapter = get_feature_analyzer_adapter(
                feature_store_client=fs_client,
                enable_feast=True,
            )
            self._feast_initialized = True
            logger.info("Feast adapter initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Feast adapter: {e}")
            self._feast_adapter = None
            self._feast_initialized = True  # Don't retry

        return self._feast_adapter

    async def _check_feature_freshness(
        self,
        feature_refs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check feature freshness via Feast.

        Args:
            feature_refs: Feature references to check. If None, uses config.

        Returns:
            Freshness check result dict
        """
        refs = feature_refs or self.config.feast_feature_refs

        if not refs:
            return {
                "fresh": True,
                "stale_features": [],
                "recommendations": ["No feature refs configured - skipping freshness check"],
            }

        adapter = await self._get_feast_adapter()
        if not adapter:
            return {
                "fresh": True,
                "stale_features": [],
                "recommendations": ["Feast adapter not available - skipping freshness check"],
            }

        try:
            result = await adapter.check_feature_freshness(
                feature_refs=refs,
                max_staleness_hours=self.config.feast_max_staleness_hours,
            )
            return cast(Dict[str, Any], result)

        except Exception as e:
            logger.warning(f"Feature freshness check failed: {e}")
            return {
                "fresh": True,  # Don't block on freshness check failures
                "stale_features": [],
                "recommendations": [f"Freshness check failed: {str(e)}"],
                "error": str(e),
            }

    async def run(self, input_data: Dict[str, Any]) -> PipelineResult:
        """Execute the complete ML Foundation pipeline.

        Args:
            input_data: Pipeline input data:
                Required:
                - problem_description (str): Natural language problem description
                - business_objective (str): Business objective
                - target_outcome (str): Target outcome
                - data_source (str): Data source table/view name

                Optional:
                - brand (str): Brand context
                - region (str): Region context
                - problem_type_hint (str): Hint for problem type
                - target_variable (str): Target variable name if known
                - candidate_features (List[str]): Candidate features
                - algorithm_preferences (List[str]): Preferred algorithms
                - target_environment (str): Deployment environment
                - feature_refs (List[str]): Feast feature references for training

        Returns:
            PipelineResult with complete pipeline outputs

        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If critical pipeline error occurs
        """
        # Validate required inputs
        required_fields = [
            "problem_description",
            "business_objective",
            "target_outcome",
            "data_source",
        ]
        for field in required_fields:  # noqa: F402
            if field not in input_data:
                raise ValueError(f"Missing required input: {field}")

        # Initialize pipeline result
        pipeline_run_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)

        # Get feature refs from input or config
        feature_refs = input_data.get("feature_refs") or self.config.feast_feature_refs

        # Initialize audit workflow
        audit_workflow_id: Optional[UUID] = None
        audit_service = self._get_audit_service()
        if audit_service:
            try:
                genesis_entry = audit_service.start_workflow(
                    agent_name="ml_foundation_pipeline",
                    agent_tier=AgentTier.ML_FOUNDATION,
                    action_type="pipeline_start",
                )
                audit_workflow_id = genesis_entry.workflow_id
                logger.debug(f"Started audit workflow: {audit_workflow_id}")
            except Exception as e:
                logger.warning(f"Failed to start audit workflow (non-fatal): {e}")

        # D1.1: Mint a workflow-level UUID even when audit_service is
        # unavailable so the SAME UUID flows from orchestrator → all agent
        # states. Closes the audit-chain integrity bug surfaced by Phase-1
        # D1 investigation: prior to this fix, the orchestrator minted a
        # UUID via audit_service (line above) but never threaded it into
        # any *_input dict, so each per-agent State independently minted
        # a fresh UUID via Field(default_factory=uuid4) and the audit
        # chain fractured 1:N across the workflow.
        if audit_workflow_id is None:
            audit_workflow_id = uuid.uuid4()
            logger.debug(f"Minted local audit_workflow_id (no audit_service): {audit_workflow_id}")

        result = PipelineResult(
            pipeline_run_id=pipeline_run_id,
            status="in_progress",
            current_stage=PipelineStage.SCOPE_DEFINITION,
            started_at=started_at.isoformat(),
            feast_enabled=self.config.enable_feast,
            feature_refs_used=feature_refs,
            audit_workflow_id=audit_workflow_id,
        )

        logger.info(f"Starting ML Foundation Pipeline: {pipeline_run_id}")
        if self.config.enable_feast:
            logger.info(f"Feast enabled with {len(feature_refs or [])} feature refs")

        # Create observability context
        observability = self._get_observability()
        obs_context = None
        if observability:
            obs_context = observability.create_observability_context(
                request_id=pipeline_run_id,
                sample_rate=self.config.sample_rate,
            )

        try:
            # Stage 1: Scope Definition
            await self._run_scope_definition(input_data, result, obs_context)

            # Stage 2: Data Preparation (QC GATE)
            gate_passed = await self._run_data_preparation(input_data, result, obs_context)
            if not gate_passed:
                result.status = "failed"
                result.current_stage = PipelineStage.FAILED
                logger.error(f"Pipeline {pipeline_run_id} STOPPED: QC Gate failed")
                return result

            # Stage 3: Model Selection
            await self._run_model_selection(input_data, result, obs_context)

            # Stage 4: Model Training
            await self._run_model_training(input_data, result, obs_context)

            # Stage 5: Feature Analysis (optional)
            if not self.config.skip_feature_analysis:
                await self._run_feature_analysis(result, obs_context)

            # Stage 6: Model Deployment (optional)
            if not self.config.skip_deployment and result.training_result:
                training_output = result.training_result
                if training_output.get("success_criteria_met", False):
                    await self._run_model_deployment(input_data, result, obs_context)
                else:
                    result.warnings.append("Skipping deployment: success criteria not met")
                    logger.warning(
                        f"Pipeline {pipeline_run_id}: Skipping deployment - "
                        "success criteria not met"
                    )

            # Complete
            result.status = "completed"
            result.current_stage = PipelineStage.COMPLETED

        except Exception as e:
            result.status = "failed"
            result.errors.append(
                {
                    "stage": result.current_stage.value,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            logger.exception(f"Pipeline {pipeline_run_id} failed at {result.current_stage}")

            if self.config.on_error:
                self.config.on_error(result.current_stage, e)

        finally:
            completed_at = datetime.now(timezone.utc)
            result.completed_at = completed_at.isoformat()
            result.total_duration_seconds = (completed_at - started_at).total_seconds()

            logger.info(
                f"Pipeline {pipeline_run_id} {result.status} in "
                f"{result.total_duration_seconds:.2f}s "
                f"(stages: {', '.join(result.stages_completed)})"
            )

        return result

    async def _run_scope_definition(
        self,
        input_data: Dict[str, Any],
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> None:
        """Run scope definition stage.

        Handoff Contract (ScopeDefiner -> DataPreparer):
        - scope_spec: Complete ML problem specification
        - success_criteria: Performance thresholds and metrics
        - experiment_id: Unique experiment identifier
        """
        result.current_stage = PipelineStage.SCOPE_DEFINITION
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 1: Running scope_definer")

        # Layer 5 manifest opt-in (the programmatic / live-ready origin). The
        # step-runner scripts resolve this themselves; the pipeline is the
        # origin for the API and the live retraining trigger. Resolve from the
        # data_source path segment or an explicit feature_manifest_source
        # override (the trigger passes the latter), then thread it through
        # scope_definer → scope_spec so Layer-1 manifest contracts (and PR
        # #544's declared-safe FDR honor) engage downstream. Unknown / unset →
        # None (cross-cohort false-positive guard preserved).
        feature_manifest_source = resolve_manifest_source(
            input_data.get("data_source"),
            input_data.get("feature_manifest_source"),
        )

        # Prepare scope_definer input
        scope_input = {
            "problem_description": input_data["problem_description"],
            "business_objective": input_data["business_objective"],
            "target_outcome": input_data["target_outcome"],
            "brand": input_data.get("brand", "unknown"),
            "region": input_data.get("region", "all"),
            "use_case": input_data.get("use_case", "commercial_targeting"),
            "problem_type_hint": input_data.get("problem_type_hint"),
            "target_variable": input_data.get("target_variable"),
            "candidate_features": input_data.get("candidate_features"),
            "feature_manifest_source": feature_manifest_source,
            # D1.1: thread workflow-level audit_workflow_id so per-agent
            # State doesn't mint a fresh UUID via default_factory.
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Execute scope_definer
        scope_definer = self._get_agent("scope_definer")
        scope_output = await scope_definer.run(scope_input)

        # Check for errors
        if scope_output.get("error"):
            raise RuntimeError(
                f"scope_definer failed: {scope_output['error']} "
                f"({scope_output.get('error_type', 'unknown')})"
            )

        # Store outputs
        result.scope_spec = scope_output.get("scope_spec", {})
        result.success_criteria = scope_output.get("success_criteria", {})
        result.experiment_id = scope_output.get("experiment_id")

        # Add any validation warnings
        if scope_output.get("validation_warnings"):
            result.warnings.extend(scope_output["validation_warnings"])

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["scope_definition"] = stage_duration
        result.stages_completed.append("scope_definition")

        logger.info(
            f"Stage 1 complete: experiment_id={result.experiment_id}, "
            f"problem_type={result.scope_spec.get('problem_type')}, "
            f"duration={stage_duration:.2f}s"
        )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="scope_definer",
            action_type="scope_definition",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "experiment_id": result.experiment_id,
                "problem_type": result.scope_spec.get("problem_type"),
            },
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.SCOPE_DEFINITION, scope_output)

    async def _run_data_preparation(
        self,
        input_data: Dict[str, Any],
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> bool:
        """Run data preparation stage (QC GATE).

        Handoff Contract (DataPreparer -> ModelSelector):
        - qc_report: Data quality report with qc_passed status
        - baseline_metrics: Baseline statistics from training split
        - gate_passed: CRITICAL - if False, pipeline MUST STOP
        - feature_freshness: Feast feature freshness validation (if enabled)

        Returns:
            True if QC gate passed, False otherwise
        """
        result.current_stage = PipelineStage.DATA_PREPARATION
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 2: Running data_preparer (QC GATE)")

        # Check feature freshness via Feast (if enabled)
        if self.config.enable_feast and self.config.feast_freshness_check:
            logger.info("Checking feature freshness via Feast...")
            freshness_result = await self._check_feature_freshness(
                feature_refs=result.feature_refs_used
            )
            result.feature_freshness = freshness_result

            # Log freshness status
            if freshness_result.get("fresh"):
                logger.info("Feature freshness check: PASSED")
            else:
                stale = freshness_result.get("stale_features", [])
                logger.warning(f"Feature freshness check: {len(stale)} stale features")
                for rec in freshness_result.get("recommendations", []):
                    logger.warning(f"  - {rec}")

        # Prepare data_preparer input
        data_prep_input = {
            "scope_spec": result.scope_spec,
            "data_source": input_data["data_source"],
            "split_id": input_data.get("split_id"),
            "validation_suite": input_data.get("validation_suite"),
            "skip_leakage_check": self.config.skip_leakage_check,
            # Track-2B-v3 D5.0: thread the structural-decider activation flag so
            # the per-run PipelineConfig (not a global default) scopes activation.
            "adaptive_structural_decider_enabled": self.config.adaptive_structural_decider_enabled,
            # D1.1: thread workflow-level audit_workflow_id (see scope_input).
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Add sample data config if enabled
        if self.config.use_sample_data:
            data_prep_input["scope_spec"]["use_sample_data"] = True
            data_prep_input["scope_spec"]["sample_size"] = input_data.get("sample_size", 1000)

        # Phase 1 data-sufficiency: propagate pipeline-level overrides into
        # scope_spec.sufficiency so the sufficiency_check node sees them via
        # the standard resolver hierarchy.
        #
        # F3 (PR #462 hotfix) + R2.1 (round-2): three bugs were nested here:
        #   1) Typed-shape drop. When the caller passed a typed
        #      `SufficiencyConfig` instance (e.g. via the new
        #      ScopeSpecSchema.sufficiency field), the old `isinstance(dict)`
        #      branch silently overwrote it with `{}` — losing every
        #      user-supplied override (`target_mde`, `epv_floor`, etc.).
        #      Fix: model_dump() the typed value before merging.
        #   2) Wrong merge policy for `force_low_power_run`. The old code
        #      let caller-supplied scope_spec.sufficiency.force_low_power_run
        #      shadow the pipeline-level default. For a pharma-safety flag
        #      with a deliberate safe-by-default value (D5), the
        #      pipeline-level value MUST win — that's the whole point of
        #      keeping the safety knob at the orchestrator level. Other
        #      keys (`strictness_preset`, anything user-supplied like
        #      `target_mde` / `epv_floor`) keep the original "user-wins"
        #      semantics — those are calibration knobs, not safety gates.
        #   3) R2.1: the round-1 fix guarded the merge with
        #      `if self.config.force_low_power_run or
        #          self.config.sufficiency_strictness_preset:` —
        #      so when BOTH pipeline overrides were at their defaults
        #      (False / None), the merge never ran, and a caller-supplied
        #      `scope_spec.sufficiency.force_low_power_run=True` passed
        #      through untouched. That defeats the very contract bug #2
        #      was meant to enforce: the orchestrator owns
        #      force_low_power_run unconditionally. Fix: ALWAYS enter the
        #      merge block. Strict pipeline-wins contract for
        #      force_low_power_run (the pipeline value, whether True or
        #      False, is authoritative; caller scope_spec value is
        #      discarded with a WARN on the downgrade-attempt path).
        #      strictness_preset and other calibration knobs keep
        #      "caller wins, pipeline fills gap" semantics.
        existing_raw = data_prep_input["scope_spec"].get("sufficiency") or {}
        if isinstance(existing_raw, dict):
            existing: Dict[str, Any] = dict(existing_raw)
        elif hasattr(existing_raw, "model_dump"):
            # Typed SufficiencyConfig (or pydantic-compat object). Dump to
            # dict so we can merge per-key without losing user fields.
            existing = existing_raw.model_dump(exclude_none=True)
        else:
            # Unknown shape — fall back to empty rather than crash, but
            # log a warning so the drift is visible.
            logger.warning(
                "scope_spec.sufficiency is neither dict nor pydantic "
                f"model (got {type(existing_raw).__name__}); ignoring."
            )
            existing = {}
        # R2.1: force_low_power_run — strict pipeline-wins, ALWAYS.
        # The orchestrator owns this safety knob; a caller cannot widen
        # OR narrow it via scope_spec.sufficiency. If pipeline value is
        # False (safe default), result is False even when caller asked
        # for True (loud-warn the downgrade-attempt for auditability).
        # If pipeline value is True (explicit override), result is True
        # even when caller asked for False. The pipeline value is the
        # only authoritative source, period.
        caller_force = existing.get("force_low_power_run")
        if caller_force is True and self.config.force_low_power_run is False:
            logger.warning(
                "Caller passed scope_spec.sufficiency.force_low_power_run=True "
                "but PipelineConfig.force_low_power_run is False (the safe "
                "pharma-default); overriding to False per the strict "
                "pipeline-wins contract. To enable low-power runs, set the "
                "flag at the PipelineConfig boundary, not via scope_spec."
            )
        existing["force_low_power_run"] = self.config.force_low_power_run
        # strictness_preset: caller-supplied wins; pipeline-level only
        # fills the gap. A user that took the trouble to set
        # `strict` (regulatory-submission) should not be silently
        # downgraded to `moderate` by an orchestrator default.
        if self.config.sufficiency_strictness_preset and "strictness_preset" not in existing:
            existing["strictness_preset"] = self.config.sufficiency_strictness_preset
        data_prep_input["scope_spec"]["sufficiency"] = existing

        # Execute data_preparer
        data_preparer = self._get_agent("data_preparer")
        try:
            data_output = await data_preparer.run(data_prep_input)
        except RuntimeError as e:
            # data_preparer raises RuntimeError on failure
            result.errors.append(
                {
                    "stage": "data_preparation",
                    "error": str(e),
                    "error_type": "RuntimeError",
                }
            )
            return False

        # Store outputs
        result.qc_report = data_output.get("qc_report", {})
        result.baseline_metrics = data_output.get("baseline_metrics", {})
        # Phase 1 data-sufficiency: pre-flight verdict + thresholds + MDE
        # report attached to PipelineResult for downstream consumers /
        # audit-chain inspection.
        result.sufficiency_report = data_output.get("sufficiency_report")
        # Gate N1 (codex-rescue HIGH-3 / N1-H3): capture the
        # regulatory_adaptation_entry emitted by leakage_remediation when it
        # adaptively dropped leaked features, so _run_model_deployment can
        # thread it to the deployer's last-line-of-defense backstop. Read from
        # the top-level output key first, falling back to the remediation
        # sub-report; ``None`` when no remediation occurred.
        result.regulatory_adaptation_entry = data_output.get("regulatory_adaptation_entry") or (
            data_output.get("remediation") or {}
        ).get("regulatory_adaptation_entry")

        # Check QC gate
        gate_passed = data_output.get("gate_passed", False)

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["data_preparation"] = stage_duration
        result.stages_completed.append("data_preparation")

        # Build log message with optional freshness info
        freshness_status = ""
        if result.feature_freshness:
            stale_count = len(result.feature_freshness.get("stale_features", []))
            freshness_status = f", features_fresh={result.feature_freshness.get('fresh', True)}"
            if stale_count > 0:
                freshness_status += f" ({stale_count} stale)"

        logger.info(
            f"Stage 2 complete: gate_passed={gate_passed}, "
            f"qc_score={result.qc_report.get('overall_score', 0):.2f}"
            f"{freshness_status}, "
            f"duration={stage_duration:.2f}s"
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.DATA_PREPARATION, data_output)

        if not gate_passed:
            result.errors.append(
                {
                    "stage": "data_preparation",
                    "error": "QC Gate failed - data quality below threshold",
                    "error_type": "QCGateError",
                    "blocking_issues": result.qc_report.get("blocking_issues", []),
                }
            )

        # Add warning for stale features (non-blocking by default)
        if result.feature_freshness and not result.feature_freshness.get("fresh", True):
            stale_features = result.feature_freshness.get("stale_features", [])
            result.warnings.append(
                f"Stale features detected: {', '.join(stale_features)}. "
                "Consider refreshing features before production use."
            )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="data_preparer",
            action_type="data_preparation",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "qc_score": result.qc_report.get("overall_score", 0),
                "gate_passed": gate_passed,
            },
            validation_passed=gate_passed,
        )

        return cast(bool, gate_passed)

    async def _run_model_selection(
        self,
        input_data: Dict[str, Any],
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> None:
        """Run model selection stage.

        Handoff Contract (ModelSelector -> ModelTrainer):
        - model_candidate: Selected algorithm with hyperparameter search space
        - selection_rationale: Explanation of selection decision
        """
        result.current_stage = PipelineStage.MODEL_SELECTION
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 3: Running model_selector")

        # Prepare model_selector input
        selector_input = {
            "scope_spec": result.scope_spec,
            "qc_report": result.qc_report,
            "baseline_metrics": result.baseline_metrics,
            "algorithm_preferences": input_data.get("algorithm_preferences"),
            "excluded_algorithms": input_data.get("excluded_algorithms"),
            "interpretability_required": self.config.interpretability_required,
            "skip_benchmarks": self.config.skip_benchmarks,
            "skip_mlflow": self.config.skip_mlflow,
            # D1.1: thread workflow-level audit_workflow_id (see scope_input).
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Execute model_selector
        model_selector = self._get_agent("model_selector")
        selector_output = await model_selector.run(selector_input)

        # Check for errors
        if selector_output.get("error"):
            raise RuntimeError(
                f"model_selector failed: {selector_output['error']} "
                f"({selector_output.get('error_type', 'unknown')})"
            )

        # Store outputs
        result.model_candidate = selector_output.get("model_candidate", {})

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["model_selection"] = stage_duration
        result.stages_completed.append("model_selection")

        logger.info(
            f"Stage 3 complete: algorithm={result.model_candidate.get('algorithm_name')}, "
            f"score={result.model_candidate.get('selection_score', 0):.3f}, "
            f"duration={stage_duration:.2f}s"
        )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="model_selector",
            action_type="model_selection",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "algorithm_name": result.model_candidate.get("algorithm_name"),
                "selection_score": result.model_candidate.get("selection_score", 0),
            },
            confidence_score=result.model_candidate.get("selection_score"),
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.MODEL_SELECTION, selector_output)

    async def _run_model_training(
        self,
        input_data: Dict[str, Any],
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> None:
        """Run model training stage.

        Handoff Contract (ModelTrainer -> FeatureAnalyzer & ModelDeployer):
        - trained_model: Trained model object
        - model_artifact_uri: MLflow model artifact URI
        - validation_metrics: Performance metrics on validation set
        - test_metrics: Final performance metrics on test set
        - success_criteria_met: Whether performance thresholds were met
        - feature_refs_used: Feast feature references used (for reproducibility)
        """
        result.current_stage = PipelineStage.MODEL_TRAINING
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 4: Running model_trainer")
        if result.feature_refs_used:
            logger.info(f"Using {len(result.feature_refs_used)} Feast feature refs")

        # Prepare model_trainer input
        trainer_input = {
            "model_candidate": result.model_candidate,
            "qc_report": result.qc_report,
            "experiment_id": result.experiment_id,
            "success_criteria": result.success_criteria,
            "problem_type": (result.scope_spec or {}).get("problem_type", "binary_classification"),
            "enable_hpo": self.config.enable_hpo,
            "hpo_trials": self.config.hpo_trials,
            "hpo_timeout_hours": self.config.hpo_timeout_hours,
            "early_stopping": self.config.early_stopping,
            "enable_mlflow": not self.config.skip_mlflow,
            # Optional: Pre-loaded data splits (if provided)
            "train_data": input_data.get("train_data"),
            "validation_data": input_data.get("validation_data"),
            "test_data": input_data.get("test_data"),
            "holdout_data": input_data.get("holdout_data"),
            # Opt-in synthetic augmentation cohort path (Phase 3 consumption).
            # None → the augment_training_data node is a no-op.
            "augmentation_data_path": self.config.augmentation_data_path,
            # Feast feature references for reproducibility
            "feature_refs": result.feature_refs_used,
            "feast_enabled": result.feast_enabled,
            # PR #463 Phase 2: forward the always-run flag into model_trainer
            # state so the ``learning_curve`` node sees it regardless of the
            # success-criteria gate. The node default is "run only on failure".
            "always_run_learning_curve": self.config.always_run_learning_curve,
            # Forward the scope_spec so the learning_curve node can read
            # ``problem_type`` / ``success_criteria.min_auc`` / ``sufficiency``
            # without having to reconstruct them from flattened state.
            "scope_spec": result.scope_spec,
            # D1.1: thread workflow-level audit_workflow_id (see scope_input).
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Execute model_trainer
        model_trainer = self._get_agent("model_trainer")
        trainer_output = await model_trainer.run(trainer_input)

        # Store outputs
        result.training_result = trainer_output

        # Surface the opt-in synthetic-augmentation audit at the pipeline level
        # so the audit chain / UI / export can see whether the model was trained
        # on synthetic-augmented data (and how much) without reaching into
        # training_result. None when the trainer didn't emit it.
        result.training_augmentation = trainer_output.get("training_augmentation")

        # PR #463 Phase 2: surface the learning-curve diagnostic at the
        # pipeline-level result so downstream consumers (audit chain, UI,
        # export) don't have to reach into ``training_result``. When the
        # model passed success_criteria AND ``always_run_learning_curve`` is
        # False, this stays None — consistent with the node's short-circuit.
        result.training_sufficiency_report = trainer_output.get("sufficiency_report")

        # Phase 3: opt-in synthetic-data preview when the learning curve
        # recommended more data. Advisory only — failures here never fail the
        # pipeline, and the preview is never auto-mixed into training.
        self._maybe_build_synthetic_preview(result)

        # Hop 4 of 4 (adaptive_criteria_v3_followup): propagate the
        # trainer's (possibly-overlaid) success_criteria back onto
        # ``PipelineResult.success_criteria``. Without this, the
        # attribute set above this method from scope_definer's pre-
        # overlay stash leaks through ``PipelineResult`` even though
        # the JSON artifact / deployer signal are correct (hops 1-3).
        # Empty-dict guard preserves the scope_definer stash if the
        # trainer (or a stub) returns no overlay — mirrors hop 3's
        # runner-side semantics.
        sc_from_trainer = trainer_output.get("success_criteria")
        if sc_from_trainer:
            result.success_criteria = sc_from_trainer

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["model_training"] = stage_duration
        result.stages_completed.append("model_training")

        # Log summary
        success = trainer_output.get("success_criteria_met", False)
        test_metrics = trainer_output.get("test_metrics", {})
        primary_metric = (
            test_metrics.get("auc_roc") or test_metrics.get("rmse") or test_metrics.get("r2") or 0.0
        )

        feature_refs_info = ""
        if result.feature_refs_used:
            feature_refs_info = f", feature_refs={len(result.feature_refs_used)}"

        logger.info(
            f"Stage 4 complete: training_run_id={trainer_output.get('training_run_id')}, "
            f"success_criteria_met={success}, "
            f"primary_metric={primary_metric:.4f}, "
            f"hpo_trials={trainer_output.get('hpo_trials_run', 0)}"
            f"{feature_refs_info}, "
            f"duration={stage_duration:.2f}s"
        )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="model_trainer",
            action_type="model_training",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "training_run_id": trainer_output.get("training_run_id"),
                "success_criteria_met": success,
                "primary_metric": primary_metric,
                "hpo_trials": trainer_output.get("hpo_trials_run", 0),
            },
            validation_passed=success,
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.MODEL_TRAINING, trainer_output)

    def _maybe_build_synthetic_preview(self, result: PipelineResult) -> None:
        """Phase 3: opt-in synthetic preview when the learning curve asked for more data.

        Fires only when (1) ``synthetic_preview_on_insufficient`` is set, (2) a
        ``synthetic_preview_scenario`` is specified, and (3) the post-training
        learning curve produced a positive ``recommended_additional_samples``.
        Advisory: any failure is recorded on ``result.synthetic_preview`` and
        logged, but never propagated (the training result stands on its own).
        The generated cohort is written to artifacts and is NOT auto-mixed.
        """
        if not self.config.synthetic_preview_on_insufficient:
            return
        scenario = self.config.synthetic_preview_scenario
        if not scenario:
            logger.info(
                "synthetic_preview_on_insufficient is set but no "
                "synthetic_preview_scenario was provided; skipping preview."
            )
            return
        report = result.training_sufficiency_report or {}
        recommended = report.get("recommended_additional_samples")
        if not isinstance(recommended, (int, float)) or recommended <= 0:
            # Learning curve didn't run, or didn't recommend more data → nothing
            # to preview. (Pre-flight HARD_FAIL blocks earlier and never reaches
            # training, so this path is specifically the "trained but needs more
            # data to hit target" case.)
            return
        try:
            from src.agents.ml_foundation.data_preparer.adapters.synthetic_preview import (
                build_synthetic_preview,
            )

            result.synthetic_preview = build_synthetic_preview(
                scenario=scenario,
                recommended_n=int(recommended),
                workflow_id=str(result.audit_workflow_id or result.pipeline_run_id),
            )
        except Exception as exc:  # advisory feature — never fail the pipeline
            logger.warning(
                "Synthetic preview generation failed (advisory; pipeline unaffected): %s",
                exc,
                exc_info=True,
            )
            result.synthetic_preview = {"generated": False, "error": str(exc)}

    async def _run_feature_analysis(
        self,
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> None:
        """Run feature analysis stage.

        Handoff Contract (FeatureAnalyzer -> Downstream):
        - shap_analysis: Global feature importance and interactions
        - feature_importance: Ranked list of features by importance
        - interpretation: Natural language explanation (if LLM enabled)
        """
        result.current_stage = PipelineStage.FEATURE_ANALYSIS
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 5: Running feature_analyzer")

        training_output = result.training_result or {}

        # Prepare feature_analyzer input
        analyzer_input = {
            "model_uri": training_output.get("model_artifact_uri", "simulated://model"),
            "experiment_id": result.experiment_id,
            "training_run_id": training_output.get("training_run_id"),
            "max_samples": 1000,
            "compute_interactions": True,
            "store_in_semantic_memory": True,
            # D1.1: thread workflow-level audit_workflow_id (see scope_input).
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Execute feature_analyzer
        feature_analyzer = self._get_agent("feature_analyzer")
        try:
            analyzer_output = await feature_analyzer.run(analyzer_input)
            result.shap_analysis = analyzer_output
        except Exception as e:
            # Feature analysis is optional - log warning but continue
            logger.warning(f"Feature analysis failed (non-blocking): {e}")
            result.warnings.append(f"Feature analysis skipped: {str(e)}")
            analyzer_output = {}

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["feature_analysis"] = stage_duration
        result.stages_completed.append("feature_analysis")

        top_features = analyzer_output.get("top_features", [])[:3]
        logger.info(
            f"Stage 5 complete: top_features={top_features}, duration={stage_duration:.2f}s"
        )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="feature_analyzer",
            action_type="feature_analysis",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "top_features": top_features,
                "feature_count": len(analyzer_output.get("top_features", [])),
            },
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.FEATURE_ANALYSIS, analyzer_output)

    async def _run_model_deployment(
        self,
        input_data: Dict[str, Any],
        result: PipelineResult,
        obs_context: Optional[Dict[str, Any]],
    ) -> None:
        """Run model deployment stage.

        Handoff Contract (ModelDeployer -> End):
        - deployment_manifest: Deployment configuration and endpoint info
        - version_record: Model registry version record
        - deployment_successful: Whether deployment succeeded
        - health_check_passed: Whether health check passed
        """
        result.current_stage = PipelineStage.MODEL_DEPLOYMENT
        stage_start = datetime.now(timezone.utc)

        logger.info("Stage 6: Running model_deployer")

        training_output = result.training_result or {}

        # Determine target environment
        target_env = input_data.get(
            "target_environment",
            self.config.target_environment,
        )

        # Gate N1 (codex-rescue HIGH-3 / N1-H3): nest the leakage-remediation
        # regulatory_adaptation_entry into scope_spec so it reaches the
        # deployer's last-line-of-defense backstop
        # (registry_manager._detect_leftover_adaptation_entries). scope_spec is
        # the carrier the model_deployer agent already threads onto its initial
        # state, so nesting here survives the agent boundary without splatting
        # arbitrary top-level keys. We shallow-copy so result.scope_spec (shared
        # with other stages) is not mutated in place.
        deployer_scope_spec = dict(result.scope_spec) if result.scope_spec else {}
        if result.regulatory_adaptation_entry is not None:
            deployer_scope_spec["regulatory_adaptation_entry"] = result.regulatory_adaptation_entry

        # Prepare model_deployer input
        deployer_input = {
            "model_uri": training_output.get("model_artifact_uri", "simulated://model"),
            "experiment_id": result.experiment_id,
            "validation_metrics": training_output.get("validation_metrics", {}),
            "success_criteria_met": training_output.get("success_criteria_met", False),
            "deployment_name": f"{result.experiment_id}_deployment",
            "target_environment": target_env,
            # Thread the full scope_spec (carrying feature_manifest_source +,
            # when leakage remediation occurred, the regulatory_adaptation_entry)
            # so the deployer's regulatory_deployment_manifest can record which
            # cohort manifest gated the model at promotion time and the backstop
            # can fail closed on an un-ingested adaptation. Previously omitted,
            # so the deployer fell back to a flat None.
            "scope_spec": deployer_scope_spec,
            # Gate N1: also pass the entry on the top-level deployer channel for
            # standalone / future orchestrator ingestion. (The agent currently
            # forwards scope_spec, not arbitrary top-level keys; this is
            # forward-compatible and harmless.)
            "regulatory_adaptation_entry": result.regulatory_adaptation_entry,
            # D1.1: thread workflow-level audit_workflow_id (see scope_input).
            "audit_workflow_id": result.audit_workflow_id,
        }

        # Add shadow mode metrics if deploying to production
        if target_env == "production":
            deployer_input.update(
                {
                    "shadow_mode_duration_hours": input_data.get("shadow_mode_duration_hours", 24),
                    "shadow_mode_requests": input_data.get("shadow_mode_requests", 1000),
                    "shadow_mode_error_rate": input_data.get("shadow_mode_error_rate", 0.01),
                    "shadow_mode_latency_p99_ms": input_data.get("shadow_mode_latency_p99_ms", 200),
                }
            )

        # Execute model_deployer
        model_deployer = self._get_agent("model_deployer")
        try:
            deployer_output = await model_deployer.run(deployer_input)
            result.deployment_result = deployer_output
        except RuntimeError as e:
            # Deployment failures are not fatal to the pipeline
            logger.error(f"Deployment failed: {e}")
            result.errors.append(
                {
                    "stage": "model_deployment",
                    "error": str(e),
                    "error_type": "DeploymentError",
                }
            )
            deployer_output = {"deployment_successful": False, "error": str(e)}

        # Record timing
        stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
        result.stage_timings["model_deployment"] = stage_duration
        result.stages_completed.append("model_deployment")

        success = deployer_output.get("deployment_successful", False)
        endpoint_url = deployer_output.get("deployment_manifest", {}).get("endpoint_url")

        logger.info(
            f"Stage 6 complete: deployment_successful={success}, "
            f"endpoint_url={endpoint_url}, "
            f"duration={stage_duration:.2f}s"
        )

        # Record audit entry
        self._record_audit_entry(
            workflow_id=result.audit_workflow_id,
            agent_name="model_deployer",
            action_type="model_deployment",
            duration_ms=int(stage_duration * 1000),
            output_data={
                "deployment_successful": success,
                "endpoint_url": endpoint_url,
                "target_environment": target_env,
            },
            validation_passed=success,
        )

        if self.config.on_stage_complete:
            self.config.on_stage_complete(PipelineStage.MODEL_DEPLOYMENT, deployer_output)

    async def run_from_stage(
        self,
        stage: PipelineStage,
        previous_result: PipelineResult,
        input_data: Dict[str, Any],
    ) -> PipelineResult:
        """Resume pipeline from a specific stage.

        Useful for retrying failed stages or skipping completed stages.

        Args:
            stage: Stage to start from
            previous_result: Result from previous partial run
            input_data: Original input data

        Returns:
            Updated PipelineResult
        """
        result = previous_result
        result.status = "in_progress"

        obs_context = None
        if self.config.enable_observability:
            observability = self._get_observability()
            if observability:
                obs_context = observability.create_observability_context(
                    request_id=result.pipeline_run_id,
                    experiment_id=result.experiment_id,
                    sample_rate=self.config.sample_rate,
                )

        stage_order = [
            PipelineStage.SCOPE_DEFINITION,
            PipelineStage.DATA_PREPARATION,
            PipelineStage.MODEL_SELECTION,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.FEATURE_ANALYSIS,
            PipelineStage.MODEL_DEPLOYMENT,
        ]

        start_index = stage_order.index(stage)

        try:
            for current_stage in stage_order[start_index:]:
                if current_stage == PipelineStage.SCOPE_DEFINITION:
                    await self._run_scope_definition(input_data, result, obs_context)
                elif current_stage == PipelineStage.DATA_PREPARATION:
                    gate_passed = await self._run_data_preparation(input_data, result, obs_context)
                    if not gate_passed:
                        result.status = "failed"
                        return result
                elif current_stage == PipelineStage.MODEL_SELECTION:
                    await self._run_model_selection(input_data, result, obs_context)
                elif current_stage == PipelineStage.MODEL_TRAINING:
                    await self._run_model_training(input_data, result, obs_context)
                elif current_stage == PipelineStage.FEATURE_ANALYSIS:
                    if not self.config.skip_feature_analysis:
                        await self._run_feature_analysis(result, obs_context)
                elif current_stage == PipelineStage.MODEL_DEPLOYMENT:
                    if not self.config.skip_deployment:
                        await self._run_model_deployment(input_data, result, obs_context)

            result.status = "completed"
            result.current_stage = PipelineStage.COMPLETED

        except Exception as e:
            result.status = "failed"
            result.errors.append(
                {
                    "stage": result.current_stage.value,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

        return result
