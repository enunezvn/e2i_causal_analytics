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

__all__ = ["router"]
