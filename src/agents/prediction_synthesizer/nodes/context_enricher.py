"""
E2I Prediction Synthesizer Agent - Context Enricher Node
Version: 4.4
Purpose: Enrich prediction with context for interpretation
Feast Integration: Online feature retrieval for real-time predictions

#438 fail-closed semantics (F-012-followup from PR #433 codex iter-3 H3):
- Per-field availability flags expose which of 5 dependencies returned data
- Per-field warnings name the failed dependency + exception class
- Aggregate gate: 0-2 fails -> completed, 3-5 -> degraded, all-5 -> failed
- include_context=False early-return preserved (caller opt-out)
- return_exceptions=True primitive preserved (non-fatal aggregation)
- Sentinel-on-failure values are non-plausible (None / [] / {}) so downstream
  consumers cannot confuse "no signal" with "real zero" / "real stable"
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from ..state import PredictionContext, PredictionSynthesizerState

logger = logging.getLogger(__name__)

# Aggregate-gate thresholds (#438 disambiguation matrix)
_NUM_DEPS = 5  # similar / importance / accuracy / trend / online
_DEGRADED_FAIL_THRESHOLD = 3  # >= 3 failures -> status="degraded"


class ContextStore(Protocol):
    """Protocol for context storage"""

    async def find_similar(
        self, entity_type: str, features: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar historical cases"""
        ...

    async def get_accuracy(self, prediction_target: str, entity_type: str) -> float:
        """Get historical accuracy for prediction type"""
        ...

    async def get_prediction_history(
        self, entity_id: str, prediction_target: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get prediction history for entity"""
        ...


class FeatureStore(Protocol):
    """Protocol for feature store with Feast integration support.

    Implementations may provide:
    - get_importance: Feature importance for models (required)
    - get_online_features: Real-time features from Feast (optional)
    - check_feature_freshness: Validate feature freshness (optional)
    """

    async def get_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for model"""
        ...

    # Optional methods for Feast integration
    async def get_online_features(
        self, entity_id: str, feature_refs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get online features for entity (optional - Feast integration)"""
        ...

    async def check_feature_freshness(
        self, entity_id: str, max_staleness_hours: float = 24.0
    ) -> Dict[str, Any]:
        """Check feature freshness (optional - Feast integration)"""
        ...


def _format_unavailable_warning(field: str, exc: BaseException) -> str:
    """Format a per-field unavailability warning.

    Output shape (#438): "<field> unavailable: <ExceptionClass>: <message>"
    """
    exc_cls = type(exc).__name__
    return f"{field} unavailable: {exc_cls}: {exc}"


class ContextEnricherNode:
    """
    Enrich prediction with context for interpretation.
    Fetches similar cases, feature importance, trends, and online features from Feast.

    Feast Integration:
        When feature_store supports get_online_features(), this node will:
        1. Fetch real-time features for the entity
        2. Validate feature freshness
        3. Merge online features with input features
    """

    def __init__(
        self,
        context_store: Optional[ContextStore] = None,
        feature_store: Optional[FeatureStore] = None,
        enable_online_features: bool = True,
        max_staleness_hours: float = 24.0,
    ):
        """
        Initialize context enricher.

        Args:
            context_store: Store for historical context
            feature_store: Store for feature metadata (may be FeastFeatureStore)
            enable_online_features: Whether to fetch online features from Feast
            max_staleness_hours: Maximum acceptable feature age for freshness check
        """
        self.context_store = context_store
        self.feature_store = feature_store
        self.enable_online_features = enable_online_features
        self.max_staleness_hours = max_staleness_hours

    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        """Enrich prediction with context and online features."""
        start_time = time.time()

        if state.get("status") in ["failed", "completed"]:
            return state

        if not state.get("include_context", False):
            total_time = state.get("orchestration_latency_ms", 0) + state.get(
                "ensemble_latency_ms", 0
            )
            return {
                **state,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        try:
            warnings: List[str] = []

            # Fetch context elements in parallel (including online features).
            # return_exceptions=True is the primitive that makes this aggregation
            # non-fatal; per-dep failures are detected per-field below and
            # converted to availability=False + per-field warning, rather than
            # silently swapped to plausible-real defaults (#438).
            tasks = [
                self._get_similar_cases(state),
                self._get_feature_importance(state),
                self._get_historical_accuracy(state),
                self._get_trend(state),
                self._get_online_features(state),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            similar, importance, accuracy, trend, online_features_result = results

            # Per-field disambiguation:
            # - Returned a non-Exception value -> available=True, populated
            # - Returned an Exception -> available=False, sentinel value,
            #   per-field warning naming the dependency + exception class.
            # Sentinel-on-failure values must be NON-plausible so downstream
            # consumers cannot confuse them with honest-real values:
            #   similar_cases   -> []     (already-honest empty container)
            #   feature_importance -> {}  (already-honest empty container)
            #   historical_accuracy -> None  (sentinel: NOT 0.0 which is plausible)
            #   trend_direction -> None    (sentinel: NOT "stable" which is plausible)
            #   online_features -> {} + online_features_available=False
            similar_available = not isinstance(similar, BaseException)
            importance_available = not isinstance(importance, BaseException)
            accuracy_available = not isinstance(accuracy, BaseException)
            trend_available = not isinstance(trend, BaseException)
            online_features_available = not isinstance(online_features_result, BaseException)

            if not similar_available:
                warnings.append(
                    _format_unavailable_warning("similar_cases", similar)  # type: ignore[arg-type]
                )
            if not importance_available:
                warnings.append(
                    _format_unavailable_warning(
                        "feature_importance",
                        importance,  # type: ignore[arg-type]
                    )
                )
            if not accuracy_available:
                warnings.append(
                    _format_unavailable_warning(
                        "historical_accuracy",
                        accuracy,  # type: ignore[arg-type]
                    )
                )
            if not trend_available:
                warnings.append(
                    _format_unavailable_warning("trend_direction", trend)  # type: ignore[arg-type]
                )
            if not online_features_available:
                warnings.append(
                    _format_unavailable_warning(
                        "online_features",
                        online_features_result,  # type: ignore[arg-type]
                    )
                )

            # Resolve online-features payload only on success
            online_features: Dict[str, Any] = {}
            feast_freshness = None
            if online_features_available and isinstance(online_features_result, dict):
                online_features = online_features_result.get("features", {})
                feast_freshness = online_features_result.get("freshness")
                if freshness_warnings := online_features_result.get("warnings", []):
                    warnings.extend(freshness_warnings)

            context: PredictionContext = {
                "similar_cases": similar if similar_available else [],  # type: ignore[typeddict-item]
                "feature_importance": importance if importance_available else {},  # type: ignore[typeddict-item]
                # Sentinel None on failure (NOT 0.0 / NOT "stable")
                "historical_accuracy": accuracy if accuracy_available else None,  # type: ignore[typeddict-item]
                "trend_direction": trend if trend_available else None,  # type: ignore[typeddict-item]
                "similar_cases_available": similar_available,
                "feature_importance_available": importance_available,
                "historical_accuracy_available": accuracy_available,
                "trend_direction_available": trend_available,
                "online_features_available": online_features_available,
            }

            # Aggregate gate (#438 disambiguation): count of failed deps
            failed_deps = sum(
                1
                for ok in (
                    similar_available,
                    importance_available,
                    accuracy_available,
                    trend_available,
                    online_features_available,
                )
                if not ok
            )

            if failed_deps == _NUM_DEPS:
                # All 5 failed AND include_context=True -> contract broken
                status_value = "failed"
                errors_addition = [
                    {
                        "code": "context_enrichment_total_failure",
                        "message": (
                            "All 5 context-enrichment dependencies failed; "
                            "no enrichment signal recoverable."
                        ),
                    }
                ]
            elif failed_deps >= _DEGRADED_FAIL_THRESHOLD:
                status_value = "degraded"
                warnings.append(f"context_enrichment_degraded: {failed_deps}/5 dependencies failed")
                errors_addition = []
            else:
                status_value = "completed"
                errors_addition = []

            context_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("orchestration_latency_ms", 0)
                + state.get("ensemble_latency_ms", 0)
                + context_time
            )

            # Merge online features with input features (only if available)
            merged_features = {**state.get("features", {}), **online_features}

            logger.info(
                f"Context enrichment complete: "
                f"similar_cases={len(context['similar_cases'])}, "
                f"online_features={len(online_features)}, "
                f"deps_failed={failed_deps}/5, "
                f"status={status_value}, "
                f"duration={context_time}ms"
            )

            result_state: Dict[str, Any] = {
                **state,
                "prediction_context": context,
                "features": merged_features,
                "total_latency_ms": total_time,
                "status": status_value,
            }

            # Add Feast metadata if available
            if online_features:
                result_state["feast_online_features"] = online_features
            if feast_freshness:
                result_state["feast_freshness"] = feast_freshness
            if warnings:
                result_state["warnings"] = warnings
            if errors_addition:
                existing_errors = state.get("errors", []) or []
                result_state["errors"] = list(existing_errors) + errors_addition

            return result_state  # type: ignore[return-value]

        except Exception as e:
            # Catastrophic failure inside the enricher itself (NOT a per-dep
            # failure - those are handled above via return_exceptions=True).
            # Preserve the non-fatal contract here: status='completed' with a
            # warning. A per-dep failure that surfaced as an exception above
            # would not reach this branch.
            logger.warning(f"Context enrichment failed: {e}")
            total_time = state.get("orchestration_latency_ms", 0) + state.get(
                "ensemble_latency_ms", 0
            )
            return {
                **state,
                "warnings": [f"Context enrichment failed: {str(e)}"],
                "total_latency_ms": total_time,
                "status": "completed",  # Non-fatal
            }

    async def _get_similar_cases(self, state: PredictionSynthesizerState) -> List[Dict[str, Any]]:
        """Find similar historical cases."""
        if not self.context_store:
            return []

        return await self.context_store.find_similar(
            entity_type=state.get("entity_type", ""),
            features=state.get("features", {}),
            limit=5,
        )

    async def _get_feature_importance(self, state: PredictionSynthesizerState) -> Dict[str, float]:
        """Get feature importance for prediction."""
        if not self.feature_store:
            return {}

        # Aggregate importance across models
        importances: Dict[str, float] = {}
        predictions = state.get("individual_predictions") or []

        for pred in predictions:
            try:
                model_importance = await self.feature_store.get_importance(pred["model_id"])
                for feature, importance in model_importance.items():
                    if feature in importances:
                        importances[feature] = (importances[feature] + importance) / 2
                    else:
                        importances[feature] = importance
            except Exception as e:
                logger.debug(f"Failed to get importance for {pred['model_id']}: {e}")

        # Return top 10 features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])

    async def _get_historical_accuracy(self, state: PredictionSynthesizerState) -> float:
        """Get historical accuracy for this prediction type."""
        if not self.context_store:
            return 0.0

        return await self.context_store.get_accuracy(
            prediction_target=state.get("prediction_target", ""),
            entity_type=state.get("entity_type", ""),
        )

    async def _get_trend(self, state: PredictionSynthesizerState) -> str:
        """Determine trend direction."""
        if not self.context_store:
            return "stable"

        history = await self.context_store.get_prediction_history(
            entity_id=state.get("entity_id", ""),
            prediction_target=state.get("prediction_target", ""),
            limit=10,
        )

        if not history or len(history) < 3:
            return "stable"

        values = [h.get("prediction", 0) for h in history]
        if len(values) < 2:
            return "stable"

        slope = (values[-1] - values[0]) / len(values)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    async def _get_online_features(self, state: PredictionSynthesizerState) -> Dict[str, Any]:
        """Get online features from Feast for the entity.

        Fetches real-time features from Feast online store and validates
        feature freshness. This enables using the latest feature values
        for predictions rather than potentially stale input features.

        Args:
            state: Current prediction state with entity_id

        Returns:
            Dictionary with:
                - features: Dict of online feature values
                - freshness: Freshness check result (if available)
                - warnings: List of any warnings (e.g., stale features)
        """
        if not self.enable_online_features:
            return {"features": {}, "freshness": None, "warnings": []}

        if not self.feature_store:
            return {"features": {}, "freshness": None, "warnings": []}

        entity_id = state.get("entity_id")
        if not entity_id:
            return {"features": {}, "freshness": None, "warnings": []}

        result: Dict[str, Any] = {"features": {}, "freshness": None, "warnings": []}

        try:
            # Check if feature_store supports online features (Feast)
            if hasattr(self.feature_store, "get_online_features"):
                features = await self.feature_store.get_online_features(
                    entity_id=entity_id,
                    feature_refs=None,  # Get all features from default view
                )
                result["features"] = features if features else {}
                logger.debug(f"Retrieved {len(result['features'])} online features")

            # Check feature freshness if supported
            if hasattr(self.feature_store, "check_feature_freshness"):
                freshness = await self.feature_store.check_feature_freshness(
                    entity_id=entity_id,
                    max_staleness_hours=self.max_staleness_hours,
                )
                result["freshness"] = freshness

                # Add warning if features are stale
                if freshness and not freshness.get("fresh", True):
                    stale_features = freshness.get("stale_features", [])
                    warnings_list: list[Any] = result["warnings"]
                    warnings_list.append(
                        f"Stale features detected: {', '.join(stale_features[:5])}"
                        + (f" (+{len(stale_features) - 5} more)" if len(stale_features) > 5 else "")
                    )

        except Exception as e:
            logger.debug(f"Online feature retrieval failed for {entity_id}: {e}")
            warnings_list2: list[Any] = result["warnings"]
            warnings_list2.append(f"Online feature retrieval failed: {str(e)}")

        return result
