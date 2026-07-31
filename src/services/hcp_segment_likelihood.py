"""HCP segment-likelihood serving (#1354).

Serves the demo-3.3 / benchmark-q14 ask — "which HCP segments are most likely to
increase <brand> prescriptions" — by scoring the platform's promoted per-brand
HCP-adoption champions over the REAL addressable HCP cohort and rolling the
per-HCP propensities up to a per-segment ranking.

Why this shape
--------------
The champions (``hcp_adoption_{brand}_goldstd_lr_v1``, promoted 2026-07-31) score
HCP-grain *adoption propensity* — the platform's operationalisation of
"likelihood to prescribe". They do NOT emit a segment-level score, so this
service adds the missing aggregation LAYER on top of the existing serving path:

    resolve champion (fail-closed) -> load the real HCP feature cohort
    (FeatureBuilder, same rows the model trained/serves on) -> score through the
    DEPLOYED BentoML raw-covariate batch endpoint (the same contract
    ``predictions._score_cohort_chunks`` uses) -> group scored HCPs by a
    covariate-backed segment axis -> rank segments by mean propensity.

Honesty posture (pinned project rules)
--------------------------------------
* NEVER scores a non-promoted model — ``resolve_hcp_adoption_champion`` fails
  closed (``ChampionNotPromotedError``) when the registry has no production
  champion for the brand, self-activating/deactivating with promotion state.
* NO fabricated numbers — every propensity comes from the real model over real
  features; an empty/unreachable substrate raises ``SegmentScoringError`` rather
  than inventing a ranking.
* Segment axes are limited to covariates ACTUALLY served to the model
  (``specialty``, ``geographic_region``) so the grouping key is present in every
  scored row without extra plumbing or a fabricated mapping.
* The default scoring population is the FULL addressable cohort (all splits) for
  tight per-segment estimates; the model's out-of-sample discrimination
  (holdout AUC) rides along as provenance, and thin segments are flagged
  ``low_confidence`` rather than hidden.

Reusability (#1356)
-------------------
This is a service-level function, importable by cohort_profiler (#1356) for
"high-value HCP" cohort ranking — it is deliberately NOT buried in the
prediction_synthesizer agent. ``aggregate_by_segment`` is pure and I/O-free.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    goldstd_model_name,
    make_hcp_spec,
)

logger = logging.getLogger(__name__)

# Segment axes = covariates the champion is actually scored on (present in every
# raw-feature row). Grouping by anything else would need a join we do not do here.
HCP_SEGMENT_AXES: tuple[str, ...] = ("specialty", "geographic_region")
DEFAULT_SEGMENT_AXIS = "specialty"

# Full addressable cohort by default: the ranking is a targeting view over the
# HCPs we could act on, and full-cohort per-segment estimates are far tighter
# than the ~250-row holdout (measured 2026-07-31: SE 0.005-0.02 vs a noisy
# n=11 holdout cell). The model's out-of-sample AUC is carried as provenance.
DEFAULT_SCORING_SPLITS: tuple[str, ...] = ("train", "validation", "test", "holdout")

# Below this per-segment n, the mean propensity is flagged low_confidence (never
# dropped — an honest thin-cell signal, not a hidden one).
DEFAULT_MIN_CONFIDENT_N = 30

# The served quantity, named honestly (an adoption-propensity model, not a
# prescribing-delta model).
_PREDICTION_TARGET = "adoption_propensity (likelihood to prescribe)"
_FEATURE_SOURCE = "hcp_profiles covariates over the synthetic gold-standard adoption cohort"

_CANONICAL_BRAND = {b.lower(): b for b in BRANDS}


class ChampionNotPromotedError(RuntimeError):
    """No production champion is registered for the requested brand — the honest
    fail-closed signal (mirrors the orchestrator resolver's contract)."""


class SegmentScoringError(RuntimeError):
    """The scoring substrate was empty/unreachable, or the model server rejected
    the batch — nothing was fabricated; the caller must surface the gap."""


class SegmentScore(BaseModel):
    """One HCP segment's rolled-up predicted adoption propensity."""

    segment: str = Field(..., description="Segment value (e.g. a specialty or region)")
    n: int = Field(..., description="HCPs scored in this segment")
    mean_propensity: float = Field(..., description="Mean predicted adoption propensity [0,1]")
    std_propensity: float = Field(..., description="Population std of the propensities in-segment")
    se_propensity: float = Field(..., description="Standard error of the mean (std / sqrt(n))")
    min_propensity: float = Field(..., description="Min propensity in-segment")
    max_propensity: float = Field(..., description="Max propensity in-segment")
    low_confidence: bool = Field(
        ..., description="True when n is below the confident-cell floor (mean is noisy)"
    )


class SegmentLikelihoodResult(BaseModel):
    """Ranked per-segment likelihood-to-prescribe view for one brand."""

    brand: str
    model_name: str
    prediction_target: str = _PREDICTION_TARGET
    segment_by: str
    n_scored: int
    overall_mean_propensity: float
    holdout_auc: Optional[float] = Field(
        default=None, description="Champion's measured out-of-sample AUC (provenance)"
    )
    feature_source: str = _FEATURE_SOURCE
    segments: List[SegmentScore] = Field(
        default_factory=list, description="Segments ranked desc by mean propensity"
    )


def _canonical_brand(brand: str) -> str:
    """Case-insensitive brand -> canonical BRANDS name, or ValueError (no silent
    default — the platform's fail-closed philosophy for a missing/unknown brand)."""
    if not brand or not isinstance(brand, str):
        raise ValueError("brand is required")
    canonical = _CANONICAL_BRAND.get(brand.strip().lower())
    if canonical is None:
        raise ValueError(f"unknown brand {brand!r}; expected one of {BRANDS}")
    return canonical


def _to_native(value: Any) -> Any:
    """numpy scalar -> python scalar; NaN (a NULL covariate) -> None so it rides
    the serving contract's designed missingness path (median-impute + isna flag),
    never crashing JSON encoding. Mirrors ``predictions._native``."""
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def aggregate_by_segment(
    covariate_rows: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    segment_by: str,
    *,
    min_confident_n: int = DEFAULT_MIN_CONFIDENT_N,
) -> List[SegmentScore]:
    """PURE: group scored HCPs by ``segment_by`` and roll each cell up to a
    ``SegmentScore``, ranked desc by mean propensity.

    ``segment_by`` must be a served covariate axis (``HCP_SEGMENT_AXES``). A None
    segment value is grouped under ``"unknown"`` (never dropped). Thin cells
    (n < ``min_confident_n``) are flagged ``low_confidence`` but still ranked.
    """
    if segment_by not in HCP_SEGMENT_AXES:
        raise ValueError(
            f"segment_by={segment_by!r} is not a served covariate axis; "
            f"expected one of {HCP_SEGMENT_AXES}"
        )
    if len(covariate_rows) != len(probabilities):
        raise ValueError(
            f"row/probability length mismatch: {len(covariate_rows)} rows vs "
            f"{len(probabilities)} probabilities"
        )

    buckets: Dict[str, List[float]] = {}
    for row, prob in zip(covariate_rows, probabilities, strict=True):
        raw = row.get(segment_by)
        key = str(raw) if raw is not None else "unknown"
        buckets.setdefault(key, []).append(float(prob))

    scores: List[SegmentScore] = []
    for segment, probs in buckets.items():
        n = len(probs)
        mean = sum(probs) / n
        # population std (we score the whole segment, not a sample of it)
        variance = sum((p - mean) ** 2 for p in probs) / n
        std = math.sqrt(variance)
        se = std / math.sqrt(n) if n else 0.0
        scores.append(
            SegmentScore(
                segment=segment,
                n=n,
                mean_propensity=mean,
                std_propensity=std,
                se_propensity=se,
                min_propensity=min(probs),
                max_propensity=max(probs),
                low_confidence=n < min_confident_n,
            )
        )
    scores.sort(key=lambda s: s.mean_propensity, reverse=True)
    return scores


async def resolve_hcp_adoption_champion(brand: str, *, db: Any) -> tuple[str, Optional[float]]:
    """Resolve the promoted HCP-adoption champion for ``brand`` from the LIVE
    registry, or fail closed.

    Returns ``(model_name, holdout_auc)``. Raises ``ValueError`` for an unknown
    brand and ``ChampionNotPromotedError`` when the registry has no row that is
    simultaneously stage='production', is_champion, artifact-backed and
    non-synthetic (the same membership predicate the orchestrator resolver's
    ``_probe_prediction_champions`` enforces — never serve a staging model).
    """
    canonical = _canonical_brand(brand)
    model_name = goldstd_model_name("hcp_adoption", canonical)
    result = await (
        db.table("ml_model_registry")
        .select("model_name, stage, is_champion, artifact_path, is_synthetic, auc")
        .eq("model_name", model_name)
        .eq("is_champion", True)
        .eq("stage", "production")
        .execute()
    )
    rows = getattr(result, "data", None) or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Defense-in-depth re-check of the server-side predicate (mirrors the
        # orchestrator probe): promotion state is the honesty gate.
        if row.get("stage") != "production" or not row.get("is_champion"):
            continue
        if not row.get("artifact_path") or row.get("is_synthetic"):
            continue
        auc = row.get("auc")
        return model_name, (float(auc) if auc is not None else None)

    raise ChampionNotPromotedError(
        f"no production champion registered for {canonical} HCP adoption "
        f"(expected model_name={model_name!r}, is_champion, stage='production', "
        "artifact-backed, non-synthetic) — nothing was scored"
    )


async def _load_scoring_frame(spec: Any, splits: Sequence[str], db: Any) -> Any:
    """Load the raw HCP cohort frame (FeatureBuilder — the same rows the model
    serves on). Isolated so tests can inject a frame without a live DB."""
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    fb = FeatureBuilder(spec)
    return await fb.load_frame(db, splits=list(splits))


async def _score_raw_features(
    model_client: Any,
    model_name: str,
    raw_features: List[Dict[str, Any]],
    *,
    chunk_size: int = 1000,
) -> List[float]:
    """Score raw covariate rows through the DEPLOYED BentoML raw-covariate BATCH
    path in chunks — the same contract ``predictions._score_cohort_chunks`` uses.
    Fails closed on a service error or a per-chunk length mismatch (never
    zero-fills a missing score)."""
    probabilities: List[float] = []
    for start in range(0, len(raw_features), chunk_size):
        chunk = raw_features[start : start + chunk_size]
        try:
            result = await model_client.predict_batch(
                model_name,
                {"batch_id": str(uuid.uuid4()), "raw_features": chunk, "model_name": model_name},
            )
        except SegmentScoringError:
            raise
        except Exception as exc:  # noqa: BLE001 — transport / circuit-breaker / HTTP
            # A model-server transport failure (httpx error, circuit breaker open,
            # stale serving schema) must surface via the typed fail-closed
            # contract so BOTH callers (agent + chat tool) handle it on the
            # documented path — never leak a raw exception nor fabricate a score.
            raise SegmentScoringError(
                f"the model server could not score cohort {model_name!r}: {exc}"
            ) from exc
        err = result.get("error")
        if err:
            raise SegmentScoringError(f"cohort scoring failed for {model_name!r}: {err}")
        chunk_probs = result.get("probabilities") or []
        if len(chunk_probs) != len(chunk):
            raise SegmentScoringError(
                f"model {model_name!r} returned {len(chunk_probs)} probabilities "
                f"for {len(chunk)} rows"
            )
        probabilities.extend(float(p) for p in chunk_probs)
    return probabilities


def build_segment_ranking_narrative(
    result: "SegmentLikelihoodResult",
    *,
    top_n: int = 5,
    horizon: Optional[str] = None,
) -> str:
    """Honest prose summary of a segment ranking — shared by the orchestrator's
    prediction_synthesizer path and the AG-UI chat tool so the honesty framing
    lives in one place.

    States the served quantity (adoption propensity, NOT a horizon-specific
    increase), the champion + out-of-sample AUC, per-segment n, and a
    low-confidence caveat where cells are thin. A stated horizon is echoed but
    explicitly marked as context — the model scores current propensity, not a
    horizon-conditioned delta.
    """
    if not result.segments:
        return (
            f"No HCP {result.segment_by} segments could be scored for {result.brand} "
            "(empty cohort) — nothing was ranked."
        )
    lines = [
        f"Predicted adoption propensity (likelihood to prescribe) for {result.brand}, "
        f"ranked by HCP {result.segment_by}, scored over {result.n_scored} HCPs via "
        f"champion {result.model_name}"
        + (
            f" (out-of-sample AUC {result.holdout_auc:.3f})."
            if result.holdout_auc is not None
            else "."
        )
    ]
    for i, seg in enumerate(result.segments[: max(1, top_n)], start=1):
        caveat = " — thin cell, low confidence" if seg.low_confidence else ""
        lines.append(
            f"{i}. {seg.segment}: mean propensity {seg.mean_propensity:.1%} "
            f"(n={seg.n}, SE {seg.se_propensity:.3f}){caveat}"
        )
    if horizon:
        lines.append(
            f"Requested horizon '{horizon}' is context only: this is a current "
            "adoption-propensity ranking, not a horizon-specific increase forecast."
        )
    return "\n".join(lines)


async def _default_db() -> Any:
    from src.memory.services.factories import get_async_supabase_client

    return await get_async_supabase_client()


async def _default_model_client() -> Any:
    from src.api.dependencies.bentoml_client import get_bentoml_client

    return await get_bentoml_client()


async def score_hcp_segments(
    brand: str,
    *,
    segment_by: str = DEFAULT_SEGMENT_AXIS,
    splits: Sequence[str] = DEFAULT_SCORING_SPLITS,
    db: Any = None,
    model_client: Any = None,
    min_confident_n: int = DEFAULT_MIN_CONFIDENT_N,
) -> SegmentLikelihoodResult:
    """Score a brand's promoted HCP-adoption champion over the real HCP cohort
    and return a per-segment likelihood-to-prescribe ranking.

    Fail-closed: an unknown brand raises ``ValueError``; a brand with no promoted
    champion raises ``ChampionNotPromotedError``; an empty scoring substrate or a
    model-server rejection raises ``SegmentScoringError``. Never fabricates.
    """
    if segment_by not in HCP_SEGMENT_AXES:
        raise ValueError(
            f"segment_by={segment_by!r} is not a served covariate axis; "
            f"expected one of {HCP_SEGMENT_AXES}"
        )
    canonical = _canonical_brand(brand)
    if db is None:
        db = await _default_db()
    if model_client is None:
        model_client = await _default_model_client()

    model_name, holdout_auc = await resolve_hcp_adoption_champion(canonical, db=db)

    spec = make_hcp_spec(canonical)
    frame = await _load_scoring_frame(spec, splits, db)
    if frame is None or getattr(frame, "empty", True):
        raise SegmentScoringError(
            f"no HCP feature rows loaded for {canonical} (splits={list(splits)}) — "
            "the adoption substrate is empty or unreachable; nothing was scored"
        )

    covariate_cols = list(spec.base_covariates)
    records = frame.to_dict("records")
    raw_features = [{c: _to_native(rec.get(c)) for c in covariate_cols} for rec in records]

    probabilities = await _score_raw_features(model_client, model_name, raw_features)
    segments = aggregate_by_segment(
        raw_features, probabilities, segment_by, min_confident_n=min_confident_n
    )
    overall = sum(probabilities) / len(probabilities) if probabilities else 0.0

    logger.info(
        "score_hcp_segments: brand=%s model=%s segment_by=%s n_scored=%d segments=%d",
        canonical,
        model_name,
        segment_by,
        len(probabilities),
        len(segments),
    )
    return SegmentLikelihoodResult(
        brand=canonical,
        model_name=model_name,
        prediction_target=_PREDICTION_TARGET,
        segment_by=segment_by,
        n_scored=len(probabilities),
        overall_mean_propensity=overall,
        holdout_auc=holdout_auc,
        feature_source=_FEATURE_SOURCE,
        segments=segments,
    )
