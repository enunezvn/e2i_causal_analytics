"""Causal inference routes (/api/causal/*), one module per concern (#1991 debt 4).

Import direction is one-way: _common <- datasets <- loaders <- route modules;
discovery imports catalog and agent (the job fans out over the agent task),
activity imports pipelines (the treatment-effects estimator), and hierarchical
reads the request-frame helper from _common. Nothing imports upward.
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

for _sub in (hierarchical, catalog, discovery, agent, pipelines, activity):
    router.include_router(_sub.router)

__all__ = ["router"]
