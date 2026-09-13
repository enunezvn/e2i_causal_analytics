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

# This order reproduces the flat file's route registration order exactly, and it
# is load-bearing: the OpenAPI ``paths`` object preserves insertion order,
# ``openapi-typescript`` emits frontend/src/types/generated/api.ts in that order,
# and CI's verify-types workflow diffs that file byte-for-byte. catalog
# contributes TWO routers because /clinical-context and /estimation-data were
# registered after the discover-effects block. Pinned by
# test_causal_openapi_path_order_unchanged. Do not alphabetise.
for _sub_router in (
    hierarchical.router,
    catalog.router,
    discovery.router,
    catalog.context_router,
    agent.router,
    pipelines.router,
    activity.router,
):
    router.include_router(_sub_router)

__all__ = ["router"]
