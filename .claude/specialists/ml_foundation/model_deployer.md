# CLAUDE.md - Model Deployer Agent

## Overview

The **Model Deployer** manages the model lifecycle from development through production, handling deployments, promotions, and rollbacks. It integrates with MLflow for registry operations and BentoML for model serving.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | <30 seconds |
| **Primary Output** | DeploymentManifest, VersionRecord |
| **Database Table** | `ml_deployments`, `ml_model_registry` |
| **Memory Types** | Working, Episodic, Procedural |
| **MLOps Tools** | MLflow, BentoML |

## Responsibilities

1. **Model Registration**: Register trained models in MLflow registry
2. **Stage Management**: Promote models through stages (dev→staging→shadow→prod)
3. **Deployment Orchestration**: Deploy models to BentoML endpoints
4. **Canary Deployments**: Run shadow deployments before full production
5. **Rollback Management**: Rollback to previous versions on issues
6. **Endpoint Management**: Create, update, and remove model endpoints

## Position in Pipeline

```
┌──────────────────┐
│ feature_analyzer │
│  (SHAP)          │
└────────┬─────────┘
         │ SHAPAnalysis
         ▼
┌──────────────────┐
│  model_deployer  │ ◀── YOU ARE HERE
│  (Deployment)    │
└────────┬─────────┘
         │ DeploymentManifest, Endpoint URL
         ├──────────────────────▶ prediction_synthesizer (Tier 4)
         ▼
┌──────────────────┐
│  observability   │
│   _connector     │
└──────────────────┘
```

## Model Stage Lifecycle

```
┌────────────┐   promote   ┌────────────┐   promote   ┌────────────┐
│ DEVELOPMENT│ ──────────▶ │  STAGING   │ ──────────▶ │   SHADOW   │
└────────────┘             └────────────┘             └────────────┘
                                                            │
                                                      promote (approval)
                                                            ▼
┌────────────┐   archive   ┌────────────┐   promote   ┌────────────┐
│  ARCHIVED  │ ◀────────── │ PRODUCTION │ ◀────────── │   SHADOW   │
└────────────┘             └────────────┘             └────────────┘
                                 │
                            rollback
                                 ▼
                          Previous PRODUCTION
```

## Outputs

### DeploymentManifest

```python
@dataclass
class DeploymentManifest:
    """Deployment configuration and status."""
    deployment_id: str
    model_version_id: str
    experiment_id: str
    
    # Deployment Target
    target_stage: ModelStage
    endpoint_name: str
    endpoint_url: str
    
    # Configuration
    replicas: int
    cpu_limit: str                   # "2"
    memory_limit: str                # "4Gi"
    autoscaling: Dict                # {"min": 1, "max": 5, "target_cpu": 70}
    
    # Artifacts
    model_artifact_uri: str
    preprocessing_artifact_uri: str
    bento_tag: str                   # "e2i_remib_model:v1.2.3"
    
    # Metadata
    deployed_by: str
    deployed_at: datetime
    deployment_duration_seconds: float
```

### VersionRecord

```python
@dataclass
class VersionRecord:
    """Model version update record."""
    model_version_id: str
    model_name: str
    version: int
    
    # Stage Change
    previous_stage: ModelStage
    current_stage: ModelStage
    
    # Metrics at Stage Change
    metrics_at_promotion: Dict[str, float]
    
    # Promotion Details
    promoted_by: str
    promoted_at: datetime
    promotion_reason: str
```

## Database Schema

### ml_deployments Table

```sql
CREATE TABLE ml_deployments (
    deployment_id TEXT PRIMARY KEY,
    model_version_id TEXT REFERENCES ml_model_registry(model_version_id),
    experiment_id TEXT REFERENCES ml_experiments(experiment_id),
    
    -- Deployment Target
    target_stage model_stage_enum NOT NULL,
    endpoint_name TEXT NOT NULL,
    endpoint_url TEXT,
    
    -- Status
    status deployment_status_enum DEFAULT 'pending',
    -- pending, deploying, active, draining, rolled_back, failed
    
    -- Configuration
    replicas INTEGER DEFAULT 1,
    cpu_limit TEXT,
    memory_limit TEXT,
    autoscaling JSONB,
    
    -- Artifacts
    bento_tag TEXT,
    
    -- Traffic
    traffic_percentage INTEGER DEFAULT 0,  -- For canary
    
    -- Timing
    deployed_by agent_name_enum DEFAULT 'model_deployer',
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    deployment_duration_seconds NUMERIC(10,2),
    drained_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ
);

-- Index for active deployments
CREATE INDEX idx_deployment_active ON ml_deployments(status) 
    WHERE status = 'active';
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.mlops.mlflow_client import MLflowClient
from src.mlops.bentoml_service import BentoMLService
from src.database.repositories.ml_deployment import MLDeploymentRepository
from src.database.repositories.ml_model_registry import MLModelRegistryRepository
from .registry_manager import RegistryManager
from .deployment_orchestrator import DeploymentOrchestrator
from .endpoint_manager import EndpointManager

class ModelDeployerAgent(BaseAgent):
    """
    Model Deployer: Manage model lifecycle and deployments.
    
    Handles stage promotions, deployments, and rollbacks.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 30
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.mlflow_client = MLflowClient()
        self.bentoml_service = BentoMLService()
        self.registry_manager = RegistryManager()
        self.deployment_orchestrator = DeploymentOrchestrator()
        self.endpoint_manager = EndpointManager()
        self.deployment_repo = MLDeploymentRepository()
        self.model_repo = MLModelRegistryRepository()
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Deploy model based on action type.
        
        Actions:
        - promote: Move model to next stage
        - deploy: Deploy to endpoint
        - rollback: Revert to previous version
        """
        action = state.deployment_action
        
        if action == "promote":
            return await self._handle_promotion(state)
        elif action == "deploy":
            return await self._handle_deployment(state)
        elif action == "rollback":
            return await self._handle_rollback(state)
        else:
            raise InvalidDeploymentActionError(f"Unknown action: {action}")
    
    async def _handle_promotion(self, state: AgentState) -> AgentState:
        """Promote model to next stage."""
        model_version_id = state.model_version_id
        target_stage = state.target_stage
        
        # Get current model info
        current_model = await self.model_repo.get(model_version_id)
        
        # Validate promotion path
        self._validate_promotion_path(
            current_stage=current_model.stage,
            target_stage=target_stage
        )
        
        # Check promotion criteria
        if target_stage == ModelStage.PRODUCTION:
            await self._verify_production_criteria(current_model)
        
        # Update MLflow registry
        await self.mlflow_client.transition_model_version_stage(
            name=current_model.model_name,
            version=current_model.version,
            stage=target_stage.value
        )
        
        # Update database
        version_record = VersionRecord(
            model_version_id=model_version_id,
            model_name=current_model.model_name,
            version=current_model.version,
            previous_stage=current_model.stage,
            current_stage=target_stage,
            metrics_at_promotion=current_model.metrics,
            promoted_by="model_deployer",
            promoted_at=datetime.utcnow(),
            promotion_reason=state.promotion_reason or "Automated promotion"
        )
        
        await self.model_repo.update_stage(model_version_id, target_stage)
        
        # Update procedural memory
        await self.procedural_memory.store(
            pattern_type="promotion",
            content={
                "from_stage": current_model.stage.value,
                "to_stage": target_stage.value,
                "model_metrics": current_model.metrics
            }
        )
        
        return state.with_updates(version_record=version_record)
    
    async def _handle_deployment(self, state: AgentState) -> AgentState:
        """Deploy model to endpoint."""
        model_version_id = state.model_version_id
        target_stage = state.target_stage
        
        deployment_id = f"deploy_{model_version_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        start_time = time.time()
        
        # Get model artifacts
        model = await self.model_repo.get(model_version_id)
        
        # Package with BentoML
        bento_tag = await self.bentoml_service.package_model(
            model_uri=model.artifact_uri,
            preprocessing_uri=model.preprocessing_artifact_uri,
            name=f"e2i_{model.experiment_id}",
            version=f"v{model.version}"
        )
        
        # Determine deployment strategy
        if target_stage == ModelStage.SHADOW:
            # Shadow deployment: deploy alongside production
            endpoint = await self._deploy_shadow(model, bento_tag)
        elif target_stage == ModelStage.PRODUCTION:
            # Production deployment: full traffic
            endpoint = await self._deploy_production(model, bento_tag)
        else:
            # Staging: deploy to staging endpoint
            endpoint = await self._deploy_staging(model, bento_tag)
        
        deployment_duration = time.time() - start_time
        
        # Build deployment manifest
        deployment_manifest = DeploymentManifest(
            deployment_id=deployment_id,
            model_version_id=model_version_id,
            experiment_id=model.experiment_id,
            target_stage=target_stage,
            endpoint_name=endpoint.name,
            endpoint_url=endpoint.url,
            replicas=endpoint.replicas,
            cpu_limit=endpoint.cpu_limit,
            memory_limit=endpoint.memory_limit,
            autoscaling=endpoint.autoscaling,
            model_artifact_uri=model.artifact_uri,
            preprocessing_artifact_uri=model.preprocessing_artifact_uri,
            bento_tag=bento_tag,
            deployed_by="model_deployer",
            deployed_at=datetime.utcnow(),
            deployment_duration_seconds=deployment_duration
        )
        
        # Persist deployment
        await self.deployment_repo.create(deployment_manifest)
        
        # Update episodic memory
        await self.episodic_memory.store(
            event_type="deployment",
            content={
                "deployment_id": deployment_id,
                "stage": target_stage.value,
                "endpoint_url": endpoint.url
            }
        )
        
        return state.with_updates(deployment_manifest=deployment_manifest)
    
    async def _handle_rollback(self, state: AgentState) -> AgentState:
        """Rollback to previous model version."""
        current_deployment_id = state.current_deployment_id
        rollback_reason = state.rollback_reason
        
        # Get current deployment
        current = await self.deployment_repo.get(current_deployment_id)
        
        # Find previous production deployment
        previous = await self.deployment_repo.get_previous_production(
            experiment_id=current.experiment_id
        )
        
        if not previous:
            raise RollbackFailedError("No previous production deployment found")
        
        # Deploy previous version
        await self.endpoint_manager.switch_traffic(
            from_deployment=current_deployment_id,
            to_deployment=previous.deployment_id,
            strategy="immediate"
        )
        
        # Update deployment statuses
        await self.deployment_repo.update_status(
            current_deployment_id, 
            DeploymentStatus.ROLLED_BACK
        )
        await self.deployment_repo.update_status(
            previous.deployment_id,
            DeploymentStatus.ACTIVE
        )
        
        # Log rollback event
        await self.episodic_memory.store(
            event_type="rollback",
            content={
                "from_deployment": current_deployment_id,
                "to_deployment": previous.deployment_id,
                "reason": rollback_reason
            }
        )
        
        return state.with_updates(
            rollback_successful=True,
            active_deployment_id=previous.deployment_id
        )
    
    def _validate_promotion_path(
        self,
        current_stage: ModelStage,
        target_stage: ModelStage
    ):
        """Validate promotion follows allowed path."""
        ALLOWED_PROMOTIONS = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING],
            ModelStage.STAGING: [ModelStage.SHADOW],
            ModelStage.SHADOW: [ModelStage.PRODUCTION],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
        }
        
        allowed = ALLOWED_PROMOTIONS.get(current_stage, [])
        if target_stage not in allowed:
            raise InvalidPromotionPathError(
                f"Cannot promote from {current_stage} to {target_stage}"
            )
    
    async def _verify_production_criteria(self, model):
        """Verify model meets production criteria."""
        criteria = await self.experiment_repo.get_success_criteria(model.experiment_id)
        
        # Check AUC threshold
        if model.metrics.get("test_auc", 0) < criteria.minimum_auc:
            raise ProductionCriteriaNotMetError(
                f"AUC {model.metrics['test_auc']} below threshold {criteria.minimum_auc}"
            )
        
        # Check shadow period
        shadow_deployment = await self.deployment_repo.get_shadow(model.experiment_id)
        if not shadow_deployment or shadow_deployment.age_hours < 24:
            raise ProductionCriteriaNotMetError(
                "Model must run in shadow mode for at least 24 hours"
            )
```

### endpoint_manager.py (BentoML Integration)

```python
class EndpointManager:
    """Manage BentoML model endpoints."""
    
    async def deploy(
        self,
        bento_tag: str,
        endpoint_name: str,
        replicas: int = 1,
        cpu: str = "2",
        memory: str = "4Gi"
    ) -> Endpoint:
        """Deploy model to BentoML endpoint."""
        
        # Create BentoML deployment
        deployment = bentoml.deployment.create(
            name=endpoint_name,
            bento=bento_tag,
            scaling={
                "replicas": replicas,
                "min_replicas": 1,
                "max_replicas": 5
            },
            resources={
                "cpu": cpu,
                "memory": memory
            }
        )
        
        # Wait for endpoint to be ready
        endpoint_url = await self._wait_for_ready(deployment)
        
        return Endpoint(
            name=endpoint_name,
            url=endpoint_url,
            replicas=replicas,
            cpu_limit=cpu,
            memory_limit=memory,
            autoscaling={"min": 1, "max": 5, "target_cpu": 70}
        )
    
    async def switch_traffic(
        self,
        from_deployment: str,
        to_deployment: str,
        strategy: str = "gradual"  # "gradual" | "immediate"
    ):
        """Switch traffic between deployments."""
        
        if strategy == "immediate":
            # Immediate cutover
            await self._update_traffic(from_deployment, 0)
            await self._update_traffic(to_deployment, 100)
        else:
            # Gradual rollout (10% increments)
            for pct in range(10, 110, 10):
                await self._update_traffic(from_deployment, 100 - pct)
                await self._update_traffic(to_deployment, pct)
                await asyncio.sleep(60)  # 1 minute between increments
```

## Error Handling

```python
class ModelDeployerError(AgentError):
    """Base error for model_deployer."""
    pass

class InvalidPromotionPathError(ModelDeployerError):
    """Promotion path not allowed."""
    pass

class InvalidDeploymentActionError(ModelDeployerError):
    """Unknown deployment action."""
    pass

class ProductionCriteriaNotMetError(ModelDeployerError):
    """Model doesn't meet production criteria."""
    pass

class RollbackFailedError(ModelDeployerError):
    """Rollback operation failed."""
    pass

class EndpointCreationError(ModelDeployerError):
    """Failed to create BentoML endpoint."""
    pass
```

## Testing

```python
class TestModelDeployer:
    
    async def test_promotion_path_validation(self):
        """Test invalid promotion paths are rejected."""
        state = AgentState(
            model_version_id="mv_123",
            current_stage=ModelStage.DEVELOPMENT,
            target_stage=ModelStage.PRODUCTION  # Invalid: skip staging
        )
        
        with pytest.raises(InvalidPromotionPathError):
            await agent._handle_promotion(state)
    
    async def test_production_criteria_check(self):
        """Test production criteria verification."""
        # Model with low AUC should fail
        state = AgentState(
            model_version_id="mv_low_auc",
            target_stage=ModelStage.PRODUCTION
        )
        
        with pytest.raises(ProductionCriteriaNotMetError):
            await agent._handle_promotion(state)
    
    async def test_rollback(self):
        """Test rollback to previous version."""
        result = await agent._handle_rollback(state)
        
        assert result.rollback_successful is True
        assert result.active_deployment_id != state.current_deployment_id
```

## Key Principles

1. **Stage Gates**: Models must pass through all stages in order
2. **Shadow Required**: 24+ hours in shadow before production
3. **Criteria Check**: Production promotion requires meeting all criteria
4. **Rollback Ready**: Always maintain rollback capability
5. **Traffic Management**: Gradual rollouts for production changes
6. **Full Logging**: All deployments and promotions tracked in episodic memory
