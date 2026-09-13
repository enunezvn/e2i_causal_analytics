"""
E2I Causal Inference API
========================

FastAPI endpoints for causal inference capabilities.

Phase B10: Causal API endpoints for:
- Hierarchical analysis (EconML within CausalML segments)
- Library routing (DoWhy, EconML, CausalML, NetworkX)
- Multi-library pipelines (sequential, parallel)
- Cross-validation between libraries

Endpoints:
- /causal/hierarchical/analyze: Run hierarchical CATE analysis
- /causal/hierarchical/{analysis_id}: Get analysis results
- /causal/route: Route query to appropriate library
- /causal/pipeline/sequential: Run sequential multi-library pipeline
- /causal/pipeline/parallel: Run parallel multi-library analysis
- /causal/validate: Run cross-library validation
- /causal/estimators: List available estimators
- /causal/health: Health check for causal engine

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import contextlib
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query

if TYPE_CHECKING:
    from src.repositories.causal_path import CausalPathRepository

from src.api.dependencies.auth import require_analyst, require_viewer
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.errors import user_safe_503_detail
from src.api.models.graph import (
    CausalChainResponse,
    EntityType,
    GraphNode,
    GraphPath,
    GraphRelationship,
    RelationshipType,
)
from src.api.schemas.causal import (
    AGENT_FORCEABLE_ESTIMATORS,
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    AggregationMethod,
    AnalysisStatus,
    CausalAnalysisHistoryItem,
    CausalAnalysisHistoryResponse,
    CausalBrandsResponse,
    CausalDAGModel,
    CausalHealthResponse,
    CausalLibrary,
    CausalVariablesResponse,
    ClinicalContext,
    CrossValidationRequest,
    CrossValidationResponse,
    DiscoveredEffect,
    DiscoverEffectsRequest,
    DiscoverEffectsResponse,
    DiscoverQuestion,
    DiscoverQuestionSelection,
    DiscoverQuestionsResponse,
    EdgeProvenanceModel,
    EstimationDataResponse,
    EstimatorCandidate,
    EstimatorComparison,
    EstimatorInfo,
    EstimatorListResponse,
    HierarchicalAnalysisRequest,
    HierarchicalAnalysisResponse,
    NestedCIResult,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineMode,
    PipelineStageResult,
    ProposedQuestion,
    ProposeQuestionsResponse,
    QuestionType,
    RefutationSummary,
    RefutationTestDetail,
    RouteQueryRequest,
    RouteQueryResponse,
    SegmentationMethod,
    SegmentCATEResult,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
    TreatmentEffectResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.causal.stats import z_score_for_confidence

# #354 C-8: real-pipeline wiring (replaces 503-default short-circuit in
# non-demo mode). Imported lazily-safely; the LibraryExecutor implementations
# inside ParallelPipeline / SequentialPipeline themselves guard their backend
# dependencies (dowhy/econml/causalml/networkx availability), so importing
# the orchestrator classes is cheap.
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.router import (
    LibraryRouter,
)
from src.causal_engine.pipeline.router import (
    QuestionType as RouterQuestionType,
)
from src.causal_engine.pipeline.sequential import SequentialPipeline
from src.causal_engine.pipeline.state import (
    PipelineInput,
    PipelineOutput,
    PipelineState,
)
from src.insights.robustness_phrase import gate_verdict_phrase

# #931: the health check's analysis-activity fields and the Analysis History tab
# read REAL completed causal-analysis events from episodic_memories (the
# canonical store written by the causal_impact agent's
# ``causal_analysis_completed`` episodic hook). Reuse the episodic repository
# rather than issuing raw SQL from the route. Imported at module level so the
# read functions are patchable in tests as ``causal.count_memories_by_type`` /
# ``causal.get_recent_memories``.
from src.memory.episodic_memory import count_memories_by_type, get_recent_memories
from src.repositories.provenance import apply_provenance_filter, deployment_includes_synthetic
from src.utils.redaction import redact_query

from . import activity, agent, catalog, discovery, hierarchical, pipelines
from ._common import (  # noqa: F401  moved here by the #1991 debt-4 split
    _AGENT_HARD_TIMEOUT_S,
    _CAUSAL_JOB_TTL_SECONDS,
    _CYCLE_IRRELEVANT_WARNING,
    _DATA_REQUIRED_LIBRARIES,
    _GENERIC_500_DETAIL,
    _NO_REAL_DATA_BACKEND_DETAIL,
    _NO_RESOLVABLE_DATA_DETAIL,
    _NON_DAG_STRUCTURAL_WARNING,
    _REFUTATION_COMPUTE_BUDGET_S,
    _ROBUSTNESS_BLOCK_WARNING,
    _ROBUSTNESS_REVIEW_WARNING,
    _ROBUSTNESS_UNVALIDATED_WARNING,
    CAUSAL_COMPLETED_EVENT_TYPE,
    _as_float,
    _as_optional_float,
    _dowhy_interval,
    _opt_float,
    _parse_occurred_at,
    _resolve_pipeline_dataframe,
    _te_pvalue_from_z,
)
from .datasets import (  # noqa: F401  moved here by the #1991 debt-4 split
    _ADVANCED_LINE_STAGES,
    _ALL_CLINICAL_COVARIATES,
    _BRAND_CLINICAL_COVARIATES,
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
    _CAUSAL_PHYSICAL_TABLE,
    _COLUMN_DEFINITIONS,
    _COLUMN_LABELS,
    _DEFAULT_CAUSAL_DATASET,
    _DISCOVERY_ROW_CAP,
    _JOIN_DATASETS,
    _NBA_JOINED_COVARIATES,
    _UNCONTROLLED_UAS7_THRESHOLD,
    _UNIVERSAL_COVARIATES,
    _brand_scoped_covariates,
    _column_label,
    _derive_is_accepted,
    _derive_is_advanced_line,
    _derive_is_prior_c5,
    _derive_is_uncontrolled_csu,
    _derive_presence,
    _is_randomized_treatment,
    _list_dataset_brands,
    _negative_control_outcome,
)
from .loaders import (  # noqa: F401  moved here by the #1991 debt-4 split
    _NBA_BASELINE_CATEGORICALS,
    _NBA_JOIN_MAX_PAGES,
    _TE_MAX_PAGES,
    _TE_PAGE_SIZE,
    _coerce_estimation_row,
    _get_causal_path_repo,
    _load_agent_estimation_frame,
    _load_hcp_adoption_join_frame,
    _load_hcp_profile_centrality,
    _load_nba_triggers_join_frame,
    _load_patient_baseline_rows,
    _load_trigger_question_rows,
    _one_hot_categoricals,
    _require_covariate_role,
    _resolve_requested_baselines,
    _te_paged_select,
    _te_paged_select_all_brands,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

for _sub in (hierarchical, catalog, discovery, agent, pipelines, activity):
    router.include_router(_sub.router)


# =============================================================================
# IN-MEMORY STORAGE (for demo - replace with database in production)
# =============================================================================
