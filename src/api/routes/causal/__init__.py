"""Causal inference routes (/api/causal/*), one module per concern (#1991 debt 4).

Layers, lowest first: ``_common``, ``datasets``, ``loaders``, then the six route
modules. A module may import only from a lower layer, never upward; ``_common``
and ``datasets`` sit at the bottom with no intra-package imports at all, and
``loaders`` imports only ``datasets``. Among the route modules, ``discovery``
imports ``catalog`` and ``agent`` (the job fans out over the agent task) and
``activity`` imports ``pipelines`` (the treatment-effects estimator reuses the
Surface-C sequential pipeline).
"""

from fastapi import APIRouter

from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

from . import activity, agent, catalog, discovery, hierarchical, pipelines

router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Deliberate order: the order these routes had in the flat module, which drives
# OpenAPI presentation. Do not alphabetise.
for _sub in (hierarchical, catalog, discovery, agent, pipelines, activity):
    router.include_router(_sub.router)


# ---------------------------------------------------------------------------
# Transitional root re-exports. src/api/routes/segments.py and
# scripts/calibration/reband_sensitivity_readings.py import these INSIDE their
# functions, so dropping them breaks GET /api/segments/datasets and
# POST /api/segments/analyze at CALL time, not at import time; a dozen test
# modules import them at module level. The last two are the non-private names the
# flat module also exposed incidentally, each read off the root by exactly one
# test module: test_causal_nba_baselines calls ``causal.list_causal_variables``
# and test_causal_geo_encoding raises on ``causal.HTTPException``. Task 5
# repoints every consumer at the owning module and deletes this whole section.
# ---------------------------------------------------------------------------
from fastapi import HTTPException  # noqa: E402, F401  TRANSITIONAL (#1991 debt 4) — see above

from .catalog import list_causal_variables  # noqa: E402, F401  TRANSITIONAL — see above
from .datasets import (  # noqa: E402, F401  TRANSITIONAL (#1991 debt 4) — see above
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
    _UNIVERSAL_COVARIATES,
    _brand_scoped_covariates,
    _column_label,
    _derive_is_accepted,
    _derive_is_advanced_line,
    _derive_is_prior_c5,
    _derive_is_uncontrolled_csu,
    _list_dataset_brands,
)
from .loaders import (  # noqa: E402, F401  TRANSITIONAL (#1991 debt 4) — see above
    _coerce_estimation_row,
    _get_causal_path_repo,
    _load_agent_estimation_frame,
    _load_patient_baseline_rows,
    _load_trigger_question_rows,
    _one_hot_categoricals,
    _resolve_requested_baselines,
)

__all__ = ["router"]
