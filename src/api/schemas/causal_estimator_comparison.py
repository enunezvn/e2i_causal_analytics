"""The estimator-comparison block of an agent causal analysis response.

Extracted from ``src/api/schemas/causal.py`` (module-size ratchet). The OpenAPI
component names are the class names, so the contract (``api.ts``) is unchanged.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EstimatorCandidate(BaseModel):
    """One estimator the energy-score selector fit and scored for this analysis."""

    estimator: str = Field(..., description="Estimator type (e.g. causal_forest / linear_dml)")
    success: bool = Field(default=False, description="Did this estimator fit successfully?")
    skipped: bool = Field(
        default=False,
        description=(
            "True if this estimator was NOT run because it is not applicable to this "
            "design (e.g. a covariate-based estimator on a zero-covariate / randomized "
            "question) — distinct from a genuine fit failure"
        ),
    )
    energy_score: Optional[float] = Field(
        default=None, description="Energy score (LOWER is better); None if the fit failed"
    )
    tournament_energy_score: Optional[float] = Field(
        default=None,
        description=(
            "Energy score from the selection tournament whenever this candidate was "
            "scored -- kept even when its served full-frame refit was refused "
            "(LOWER is better); None if it was never scored"
        ),
    )
    served_refit: Optional[bool] = Field(
        default=None,
        description=(
            "Outcome of this candidate's served full-frame refit after a subsampled "
            "tournament: True succeeded, False refused; None when no separate served "
            "refit happened (unsubsampled selection, or a tournament loser)"
        ),
    )
    ate: Optional[float] = Field(default=None, description="This estimator's ATE estimate")
    error: Optional[str] = Field(
        default=None, description="Failure reason (or not-applicable reason if skipped)"
    )
    is_selected: bool = Field(default=False, description="True for the winning estimator")


class EstimatorComparison(BaseModel):
    """Why the agent chose this estimator: the full data-driven evaluation.

    The Auto path fits and energy-scores several estimators (causal_forest,
    linear_dml, drlearner, ols) and picks the lowest energy score with a
    robust-over-fast tie-break. This surfaces that comparison so the analyst can
    see WHAT was evaluated and WHY the winner won — not just the winner's name.
    """

    candidates: List[EstimatorCandidate] = Field(default_factory=list)
    selection_reason: Optional[str] = Field(
        default=None, description="Human-readable rationale for the winning estimator"
    )
    energy_score_gap: Optional[float] = Field(
        default=None, description="Energy-score margin between the winner and runner-up"
    )
    n_evaluated: int = Field(default=0, description="How many estimators were fit")
    n_succeeded: int = Field(default=0, description="How many fit successfully")
    quality_tier: Optional[str] = Field(
        default=None,
        description="Winner's quality tier (excellent/good/acceptable/poor/unreliable)",
    )
    requires_review: bool = Field(
        default=False, description="True if the winner breached the review gate"
    )
