"""
E2I Prediction Synthesizer Agent - Context Enricher Fail-Closed Tests

GH #438 (F-012-followup from PR #433 codex iter-3 H3).

Disambiguation matrix:

| Dependency state         | Per-field result | Availability flag | Per-field warning                 |
|--------------------------|------------------|-------------------|-----------------------------------|
| Returned successfully    | populated        | True              | (none)                            |
| Raised Exception         | empty/sentinel   | False             | "<field> unavailable: <Exc>: <m>" |
| Returned empty-but-valid | empty            | True              | (none)                            |

Aggregate:
- 0-2 fields failed         -> status: "completed" (per-field warnings only)
- 3-5 fields failed         -> status: "degraded"  (+ aggregate notice)
- All 5 failed (incl ctx)   -> status: "failed"    (+ top-level error)

`include_context=False` early-return is preserved unchanged.
`return_exceptions=True` primitive is preserved (non-fatal aggregation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.prediction_synthesizer.nodes.context_enricher import (
    ContextEnricherNode,
)

# ============================================================================
# Helpers: stores with selective failure surfaces
# ============================================================================


class SelectivelyFailingContextStore:
    """ContextStore where each method either succeeds or raises Exception.

    Independently controllable per-method so the 5 dep-fail cases can be
    composed for the disambiguation matrix.
    """

    def __init__(
        self,
        similar_fails: bool = False,
        accuracy_fails: bool = False,
        history_fails: bool = False,
        similar_data: Optional[List[Dict[str, Any]]] = None,
        accuracy_value: float = 0.82,
        history_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.similar_fails = similar_fails
        self.accuracy_fails = accuracy_fails
        self.history_fails = history_fails
        self.similar_data = (
            similar_data
            if similar_data is not None
            else [
                {"entity_id": "hcp_100", "prediction": 0.65, "outcome": 1},
            ]
        )
        self.accuracy_value = accuracy_value
        self.history_data = (
            history_data
            if history_data is not None
            else [
                {"prediction": 0.40, "timestamp": "2024-01-01"},
                {"prediction": 0.50, "timestamp": "2024-02-01"},
                {"prediction": 0.60, "timestamp": "2024-03-01"},
            ]
        )

    async def find_similar(
        self, entity_type: str, features: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        if self.similar_fails:
            raise RuntimeError("similar-store unreachable")
        return self.similar_data[:limit]

    async def get_accuracy(self, prediction_target: str, entity_type: str) -> float:
        if self.accuracy_fails:
            raise ConnectionError("accuracy-store timeout")
        return self.accuracy_value

    async def get_prediction_history(
        self, entity_id: str, prediction_target: str, limit: int
    ) -> List[Dict[str, Any]]:
        if self.history_fails:
            raise ValueError("history-store bad rows")
        return self.history_data[:limit]


class SelectivelyFailingFeatureStore:
    """FeatureStore where get_importance and get_online_features can fail."""

    def __init__(
        self,
        importance_fails: bool = False,
        online_fails: bool = False,
        importance_data: Optional[Dict[str, float]] = None,
        online_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.importance_fails = importance_fails
        self.online_fails = online_fails
        self.importance_data = (
            importance_data
            if importance_data is not None
            else {
                "call_frequency": 0.25,
                "prescription_count": 0.20,
            }
        )
        self.online_data = (
            online_data
            if online_data is not None
            else {
                "call_frequency": 25.0,
            }
        )

    async def get_importance(self, model_id: str) -> Dict[str, float]:
        if self.importance_fails:
            raise RuntimeError("importance-store down")
        return self.importance_data

    async def get_online_features(
        self,
        entity_id: str,
        feature_refs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.online_fails:
            raise RuntimeError("feast-online unreachable")
        return self.online_data

    async def check_feature_freshness(
        self,
        entity_id: str,
        max_staleness_hours: float = 24.0,
    ) -> Dict[str, Any]:
        return {"fresh": True, "stale_features": []}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def state_for_enrichment() -> Dict[str, Any]:
    """Minimal state ready for context enrichment (status = enriching)."""
    return {
        "query": "test query",
        "entity_id": "hcp_123",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "features": {"call_frequency": 12.0},
        "time_horizon": "30d",
        "include_context": True,
        "individual_predictions": [
            {
                "model_id": "churn_xgb",
                "prediction": 0.72,
                "confidence": 0.88,
                "latency_ms": 50,
            },
        ],
        "models_succeeded": 1,
        "models_failed": 0,
        "orchestration_latency_ms": 65,
        "ensemble_latency_ms": 5,
        "ensemble_prediction": {
            "point_estimate": 0.70,
            "prediction_interval_lower": 0.58,
            "prediction_interval_upper": 0.82,
            "confidence": 0.85,
            "ensemble_method": "weighted",
            "model_agreement": 0.95,
        },
        "prediction_summary": "Prediction: 0.700",
        "errors": [],
        "warnings": [],
        "status": "enriching",
        "timestamp": "2026-05-22T00:00:00Z",
    }


# ============================================================================
# Case 1: All-success path
# ============================================================================


class TestAllSuccess:
    """Disambiguation row 1: All 5 dependencies returned data successfully."""

    @pytest.mark.asyncio
    async def test_all_dependencies_succeed_marks_all_available(self, state_for_enrichment) -> None:
        ctx_store = SelectivelyFailingContextStore()
        feat_store = SelectivelyFailingFeatureStore()

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        assert context is not None

        # Populated values
        assert len(context["similar_cases"]) > 0
        assert len(context["feature_importance"]) > 0
        assert context["historical_accuracy"] == 0.82
        assert context["trend_direction"] in ("increasing", "stable", "decreasing")

        # All availability flags True
        assert context["similar_cases_available"] is True
        assert context["feature_importance_available"] is True
        assert context["historical_accuracy_available"] is True
        assert context["trend_direction_available"] is True
        assert context["online_features_available"] is True

        # No per-field availability warning emitted for successful fetches
        warnings = result.get("warnings", []) or []
        for w in warnings:
            assert "unavailable" not in w


# ============================================================================
# Cases 2-6: Single-dependency failure (each of the 5 independently)
# ============================================================================


class TestSingleDependencyFails:
    """Disambiguation row 2: one of the 5 deps raises Exception.

    Per-field sentinel + availability=False + named warning.
    Aggregate status stays 'completed' (1/5 <= 2 threshold).
    """

    @pytest.mark.asyncio
    async def test_similar_cases_failure_marks_unavailable_with_warning(
        self, state_for_enrichment
    ) -> None:
        ctx_store = SelectivelyFailingContextStore(similar_fails=True)
        feat_store = SelectivelyFailingFeatureStore()

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        # Sentinel-shaped empty list (NOT a fabrication)
        assert context["similar_cases"] == []
        assert context["similar_cases_available"] is False
        # Other fields still populated and available
        assert context["historical_accuracy"] == 0.82
        assert context["historical_accuracy_available"] is True
        assert context["trend_direction_available"] is True
        # Per-field warning naming the dep + exception class
        warnings = result.get("warnings", []) or []
        named = [w for w in warnings if "similar_cases" in w and "unavailable" in w]
        assert named, f"expected similar_cases unavailable warning, got: {warnings}"
        assert any("RuntimeError" in w for w in named)

    @pytest.mark.asyncio
    async def test_feature_importance_all_models_fail_marks_unavailable(
        self, state_for_enrichment
    ) -> None:
        """REAL-PATH test (no helper monkeypatch).

        #438 codex iter-0 HIGH-1: previously _get_feature_importance swallowed
        per-model exceptions silently and returned an empty dict, hiding the
        dependency failure from the aggregate gate. Now: when ALL per-model
        get_importance() calls raise, the helper raises
        AllModelsImportanceFailed which propagates via asyncio.gather and the
        aggregate gate counts it as a failure.
        """
        ctx_store = SelectivelyFailingContextStore()

        class TotalFailureFeatureStore:
            async def get_importance(self, model_id: str) -> Dict[str, float]:
                raise RuntimeError("importance-store-down")

            async def get_online_features(
                self, entity_id: str, feature_refs: Optional[List[str]] = None
            ) -> Dict[str, Any]:
                return {}

            async def check_feature_freshness(
                self, entity_id: str, max_staleness_hours: float = 24.0
            ) -> Dict[str, Any]:
                return {"fresh": True, "stale_features": []}

        node = ContextEnricherNode(
            context_store=ctx_store, feature_store=TotalFailureFeatureStore()
        )

        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        assert context["feature_importance"] == {}
        assert context["feature_importance_available"] is False
        warnings = result.get("warnings", []) or []
        named = [w for w in warnings if "feature_importance" in w and "unavailable" in w]
        assert named, f"expected feature_importance unavailable warning: {warnings}"
        assert any("AllModelsImportanceFailed" in w for w in named)

    @pytest.mark.asyncio
    async def test_feature_importance_partial_model_failure_stays_available(
        self, state_for_enrichment
    ) -> None:
        """If at least one model's get_importance() succeeds, the dep stays
        available. Pre-#438 the per-model except was non-fatal by design and
        we preserve that for the partial-success case. Only when ALL models
        fail does the dep flip to unavailable.
        """
        ctx_store = SelectivelyFailingContextStore()

        # Configure state with 2 models; only the FIRST raises, the second
        # returns importance. Expect feature_importance_available=True.
        state_for_enrichment["individual_predictions"] = [
            {
                "model_id": "model_failing",
                "prediction": 0.7,
                "confidence": 0.8,
                "latency_ms": 30,
            },
            {
                "model_id": "model_ok",
                "prediction": 0.6,
                "confidence": 0.75,
                "latency_ms": 40,
            },
        ]

        class PartialFailFStore:
            async def get_importance(self, model_id: str) -> Dict[str, float]:
                if model_id == "model_failing":
                    raise RuntimeError("first model importance failed")
                return {"feat_a": 0.6}

            async def get_online_features(
                self, entity_id: str, feature_refs: Optional[List[str]] = None
            ) -> Dict[str, Any]:
                return {}

            async def check_feature_freshness(
                self, entity_id: str, max_staleness_hours: float = 24.0
            ) -> Dict[str, Any]:
                return {"fresh": True, "stale_features": []}

        node = ContextEnricherNode(context_store=ctx_store, feature_store=PartialFailFStore())

        result = await node.execute(state_for_enrichment)

        # Partial model failure -> dep stays available (some real signal returned)
        assert result["status"] == "completed"
        context = result["prediction_context"]
        assert context["feature_importance_available"] is True
        assert "feat_a" in context["feature_importance"]

    @pytest.mark.asyncio
    async def test_historical_accuracy_failure_uses_sentinel_none(
        self, state_for_enrichment
    ) -> None:
        ctx_store = SelectivelyFailingContextStore(accuracy_fails=True)
        feat_store = SelectivelyFailingFeatureStore()

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        # SENTINEL: None (NOT plausible-real 0.0)
        assert context["historical_accuracy"] is None
        assert context["historical_accuracy_available"] is False
        warnings = result.get("warnings", []) or []
        named = [w for w in warnings if "historical_accuracy" in w and "unavailable" in w]
        assert named, f"expected historical_accuracy warning: {warnings}"
        assert any("ConnectionError" in w for w in named)

    @pytest.mark.asyncio
    async def test_trend_direction_failure_uses_sentinel_none(self, state_for_enrichment) -> None:
        ctx_store = SelectivelyFailingContextStore(history_fails=True)
        feat_store = SelectivelyFailingFeatureStore()

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        # SENTINEL: None (NOT plausible-real "stable")
        assert context["trend_direction"] is None
        assert context["trend_direction_available"] is False
        warnings = result.get("warnings", []) or []
        named = [w for w in warnings if "trend_direction" in w and "unavailable" in w]
        assert named, f"expected trend_direction warning: {warnings}"
        assert any("ValueError" in w for w in named)

    @pytest.mark.asyncio
    async def test_online_features_failure_marks_unavailable_with_warning(
        self, state_for_enrichment
    ) -> None:
        """REAL-PATH test (no helper monkeypatch).

        #438 codex iter-0 HIGH-2: previously _get_online_features wrapped the
        entire Feast call in a broad try/except and returned an honest-empty
        dict with a freeform warning, leaving online_features_available
        falsely True. Now: Feast exceptions propagate to asyncio.gather and
        the per-field availability flag flips to False with a per-field
        warning naming the dependency + exception class.
        """
        ctx_store = SelectivelyFailingContextStore()

        class FailingFStore:
            async def get_importance(self, model_id: str) -> Dict[str, float]:
                return {"f1": 0.5}

            async def get_online_features(
                self, entity_id: str, feature_refs: Optional[List[str]] = None
            ) -> Dict[str, Any]:
                raise TimeoutError("feast timeout")

            async def check_feature_freshness(
                self, entity_id: str, max_staleness_hours: float = 24.0
            ) -> Dict[str, Any]:
                return {"fresh": True, "stale_features": []}

        node = ContextEnricherNode(context_store=ctx_store, feature_store=FailingFStore())

        result = await node.execute(state_for_enrichment)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        assert context["online_features_available"] is False
        warnings = result.get("warnings", []) or []
        named = [w for w in warnings if "online_features" in w and "unavailable" in w]
        assert named, f"expected online_features warning: {warnings}"
        assert any("TimeoutError" in w for w in named)


# ============================================================================
# Case 7: Aggregate-degraded (3-of-5 failures)
# ============================================================================


class TestAggregateDegraded:
    """Disambiguation aggregate: 3-5 fields failed -> status='degraded' +
    aggregate notice."""

    @pytest.mark.asyncio
    async def test_three_of_five_failures_flips_status_to_degraded(
        self, state_for_enrichment
    ) -> None:
        ctx_store = SelectivelyFailingContextStore(
            similar_fails=True,
            accuracy_fails=True,
            history_fails=True,  # similar + accuracy + trend = 3 fails
        )
        feat_store = SelectivelyFailingFeatureStore()

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        assert result["status"] == "degraded"
        context = result["prediction_context"]
        # 3 unavailable, 2 available
        assert context["similar_cases_available"] is False
        assert context["historical_accuracy_available"] is False
        assert context["trend_direction_available"] is False
        assert context["feature_importance_available"] is True
        assert context["online_features_available"] is True
        # Sentinel values for failures (NOT plausible-real defaults)
        assert context["similar_cases"] == []
        assert context["historical_accuracy"] is None
        assert context["trend_direction"] is None
        # Aggregate notice in top-level warnings
        warnings = result.get("warnings", []) or []
        agg = [w for w in warnings if "context_enrichment_degraded" in w]
        assert agg, f"expected aggregate notice in warnings: {warnings}"
        assert any("3/5" in w for w in agg)


# ============================================================================
# Case 8: All-five failures -> status='failed' with top-level error
# ============================================================================


class TestAllFiveFail:
    """Disambiguation aggregate: all 5 failed AND include_context=True ->
    status='failed' with top-level error."""

    @pytest.mark.asyncio
    async def test_all_five_failures_flips_status_to_failed(self, state_for_enrichment) -> None:
        """REAL-PATH test (no helper monkeypatch). All 5 deps fail through
        real code paths: context_store raises on 3 methods, feature_store
        raises on get_importance (AllModelsImportanceFailed) and
        get_online_features (TimeoutError)."""
        ctx_store = SelectivelyFailingContextStore(
            similar_fails=True,
            accuracy_fails=True,
            history_fails=True,
        )

        class TotalFailFStore:
            async def get_importance(self, model_id: str) -> Dict[str, float]:
                raise RuntimeError("importance fully down")

            async def get_online_features(
                self, entity_id: str, feature_refs: Optional[List[str]] = None
            ) -> Dict[str, Any]:
                raise TimeoutError("feast fully down")

            async def check_feature_freshness(
                self, entity_id: str, max_staleness_hours: float = 24.0
            ) -> Dict[str, Any]:
                return {"fresh": True, "stale_features": []}

        node = ContextEnricherNode(context_store=ctx_store, feature_store=TotalFailFStore())

        result = await node.execute(state_for_enrichment)

        assert result["status"] == "failed"
        # Top-level error code
        errors = result.get("errors", []) or []
        codes = [e.get("code") if isinstance(e, dict) else e for e in errors]
        assert any(c == "context_enrichment_total_failure" for c in codes), (
            f"expected total-failure error code, got errors: {errors}"
        )
        context = result["prediction_context"]
        assert context is not None
        # All 5 unavailable; values all sentinel
        assert context["similar_cases_available"] is False
        assert context["feature_importance_available"] is False
        assert context["historical_accuracy_available"] is False
        assert context["trend_direction_available"] is False
        assert context["online_features_available"] is False
        assert context["similar_cases"] == []
        assert context["feature_importance"] == {}
        assert context["historical_accuracy"] is None
        assert context["trend_direction"] is None


# ============================================================================
# Case 9: include_context=False preservation (early-return unchanged)
# ============================================================================


class TestIncludeContextFalseEarlyReturn:
    """The `include_context=False` early-return at :107-115 is the caller's
    explicit opt-out and must be preserved unchanged."""

    @pytest.mark.asyncio
    async def test_include_context_false_skips_enrichment_entirely(
        self, state_for_enrichment
    ) -> None:
        state_for_enrichment["include_context"] = False
        # Even with failing stores, the early-return path must not invoke them
        ctx_store = MagicMock()
        ctx_store.find_similar = AsyncMock(side_effect=RuntimeError("would fail"))
        ctx_store.get_accuracy = AsyncMock(side_effect=RuntimeError("would fail"))
        ctx_store.get_prediction_history = AsyncMock(side_effect=RuntimeError("would fail"))

        feat_store = MagicMock()
        feat_store.get_importance = AsyncMock(side_effect=RuntimeError("would fail"))

        node = ContextEnricherNode(context_store=ctx_store, feature_store=feat_store)
        result = await node.execute(state_for_enrichment)

        # Early-return contract: status completed, no prediction_context populated
        assert result["status"] == "completed"
        # Stores not invoked (early-return at :107)
        ctx_store.find_similar.assert_not_called()
        ctx_store.get_accuracy.assert_not_called()
        ctx_store.get_prediction_history.assert_not_called()
        feat_store.get_importance.assert_not_called()
        # No new availability flags should leak into the early-return path
        # (the early return uses the existing schema fields only)
        assert "prediction_context" not in result or result.get("prediction_context") is None
