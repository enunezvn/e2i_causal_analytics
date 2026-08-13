"""Causal Discovery Tools for Agent Workflows.

This module provides tools for automatic DAG structure learning and
driver ranking (causal vs predictive importance comparison).

Version: 1.0.0

Tools:
- discover_dag: Automatic DAG structure learning using ensemble algorithms
- rank_drivers: Compare causal vs predictive feature importance

Usage:
------
    from src.tool_registry.tools.causal_discovery import discover_dag, rank_drivers

    # Discover DAG structure
    result = await discover_dag(
        data=df,
        algorithms=["ges", "pc"],
        ensemble_threshold=0.5,
    )

    # Rank drivers
    ranking = await rank_drivers(
        dag=result["ensemble_dag"],
        target="outcome",
        shap_values=shap_vals,
        feature_names=features,
    )

Author: E2I Causal Analytics Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT SCHEMAS - DISCOVER DAG
# =============================================================================


class DiscoverDagInput(BaseModel):
    """Input schema for discover_dag tool."""

    data: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        description=(
            "Data as dictionary of column names to values (DataFrame.to_dict('list')). "
            "Optional: when omitted, the real DataFrame is resolved from the canonical "
            "kwargs keys (data / dataframe / estimation_data); the tool fails closed if "
            "neither is supplied (never fabricates)."
        ),
    )
    algorithms: List[str] = Field(
        default=["ges", "pc"],
        description="Algorithms to use: 'ges', 'pc', 'fci', 'lingam', 'direct_lingam', 'ica_lingam'",
    )
    ensemble_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of algorithms that must agree on an edge (0-1)",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Significance level for conditional independence tests",
    )
    max_k: Optional[int] = Field(
        default=None,
        description="Maximum conditioning set size (-1 for unlimited)",
    )
    node_names: Optional[List[str]] = Field(
        default=None,
        description="Custom node names (defaults to column names)",
    )
    trace_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Opik trace context for distributed tracing",
    )


class DiscoverDagOutput(BaseModel):
    """Output schema for discover_dag tool."""

    success: bool = Field(..., description="Whether discovery succeeded")
    n_edges: int = Field(default=0, description="Number of edges discovered")
    n_nodes: int = Field(default=0, description="Number of nodes in the DAG")
    edge_list: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of edges with source, target, confidence, type",
    )
    algorithms_used: List[str] = Field(
        default_factory=list,
        description="Algorithms that were successfully run",
    )
    algorithm_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-algorithm results with n_edges, runtime_seconds, converged",
    )
    ensemble_threshold: float = Field(..., description="Threshold used for ensemble")
    gate_decision: Optional[str] = Field(
        default=None,
        description="Gate evaluation: 'accept', 'review', 'reject', 'augment'",
    )
    gate_confidence: Optional[float] = Field(
        default=None,
        description="Gate confidence score (0-1)",
    )
    gate_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for gate decision",
    )
    total_runtime_seconds: float = Field(
        default=0.0,
        description="Total discovery runtime",
    )
    timestamp: str = Field(..., description="Discovery timestamp (ISO format)")
    trace_id: Optional[str] = Field(
        default=None,
        description="Opik trace ID for this discovery",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during discovery",
    )


# =============================================================================
# INPUT/OUTPUT SCHEMAS - RANK DRIVERS
# =============================================================================


class RankDriversInput(BaseModel):
    """Input schema for rank_drivers tool."""

    dag_edge_list: List[Dict[str, str]] = Field(
        ...,
        description="DAG as list of edges: [{'source': 'A', 'target': 'B'}, ...]",
    )
    target: str = Field(
        ...,
        description="Target variable name for causal importance calculation",
    )
    shap_values: List[List[float]] = Field(
        ...,
        description="SHAP values matrix (n_samples x n_features)",
    )
    feature_names: List[str] = Field(
        ...,
        description="Feature names corresponding to SHAP columns",
    )
    concordance_threshold: int = Field(
        default=2,
        description="Maximum rank difference to consider features concordant",
    )
    importance_percentile: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Top percentile to consider as 'important'",
    )
    trace_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Opik trace context for distributed tracing",
    )


class FeatureRankingItem(BaseModel):
    """Single feature ranking information.

    The causal fields (``causal_rank``, ``rank_difference``, ``causal_score``) are
    ``Optional``: they are ``None`` — not ``0`` — when no causal estimate exists
    (e.g. the predictive-only fallback when the DAG has no edges), so a missing
    causal estimate is never mistaken for a real zero-effect value.
    """

    feature_name: str
    causal_rank: Optional[int] = None
    predictive_rank: int
    rank_difference: Optional[int] = None
    causal_score: Optional[float] = None
    predictive_score: float
    is_direct_cause: bool
    path_length: Optional[int]


class RankDriversOutput(BaseModel):
    """Output schema for rank_drivers tool."""

    success: bool = Field(..., description="Whether ranking succeeded")
    target_variable: str = Field(..., description="Target variable used")
    rankings: List[FeatureRankingItem] = Field(
        default_factory=list,
        description="Feature rankings sorted by causal importance",
    )
    n_features: int = Field(default=0, description="Number of features ranked")
    rank_correlation: Optional[float] = Field(
        default=None,
        description=(
            "Spearman correlation between causal and predictive ranks; None when "
            "no causal ranking exists (e.g. predictive-only fallback)"
        ),
    )
    causal_only_features: List[str] = Field(
        default_factory=list,
        description="Features important causally but not predictively",
    )
    predictive_only_features: List[str] = Field(
        default_factory=list,
        description="Features important predictively but not causally",
    )
    concordant_features: List[str] = Field(
        default_factory=list,
        description="Features with similar causal and predictive rankings",
    )
    timestamp: str = Field(..., description="Ranking timestamp (ISO format)")
    trace_id: Optional[str] = Field(
        default=None,
        description="Opik trace ID for this ranking",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during ranking",
    )


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


@dataclass
class CausalDiscoveryTool:
    """
    Tool for automatic DAG structure learning.

    Uses multiple algorithms (GES, PC, etc.) in an ensemble approach
    to discover causal structure from observational data.

    Attributes:
        opik_enabled: Whether Opik tracing is enabled
    """

    opik_enabled: bool = field(default=True)
    _runner: Any = field(default=None, repr=False)
    _gate: Any = field(default=None, repr=False)

    def _ensure_initialized(self) -> None:
        """Lazy initialize discovery components."""
        if self._runner is None:
            from src.causal_engine.discovery import DiscoveryGate, DiscoveryRunner

            self._runner = DiscoveryRunner()
            self._gate = DiscoveryGate()
            logger.info("CausalDiscoveryTool: Components initialized")

    async def invoke(
        self,
        input_data: Union[Dict[str, Any], DiscoverDagInput],
    ) -> DiscoverDagOutput:
        """
        Discover DAG structure from data.

        Args:
            input_data: Either a dict or DiscoverDagInput with parameters

        Returns:
            DiscoverDagOutput with discovered DAG and metadata
        """
        # Parse input
        if isinstance(input_data, dict):
            params = DiscoverDagInput(**input_data)
        else:
            params = input_data

        # Initialize if needed
        self._ensure_initialized()

        # Start trace
        trace_id = self._start_trace(params) if self.opik_enabled else None

        errors = []
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Convert data dict to DataFrame
            df = pd.DataFrame(params.data)

            # Import discovery types
            from src.causal_engine.discovery import (
                DiscoveryAlgorithmType,
                DiscoveryConfig,
            )

            # Map algorithm strings to enum
            algorithm_map = {
                "ges": DiscoveryAlgorithmType.GES,
                "pc": DiscoveryAlgorithmType.PC,
                "fci": DiscoveryAlgorithmType.FCI,
                "lingam": DiscoveryAlgorithmType.LINGAM,
                "direct_lingam": DiscoveryAlgorithmType.DIRECT_LINGAM,
                "ica_lingam": DiscoveryAlgorithmType.ICA_LINGAM,
            }

            algorithms = []
            for alg in params.algorithms:
                if alg.lower() in algorithm_map:
                    algorithms.append(algorithm_map[alg.lower()])
                else:
                    errors.append(f"Unknown algorithm: {alg}")

            if not algorithms:
                algorithms = [DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]
                errors.append("No valid algorithms specified, using defaults (GES, PC)")

            # Create config
            config = DiscoveryConfig(
                algorithms=algorithms,
                ensemble_threshold=params.ensemble_threshold,
                alpha=params.alpha,
                max_cond_vars=params.max_k if params.max_k is not None else None,
            )

            # Run discovery
            result = await self._runner.discover_dag(df, config)

            # Evaluate with gate
            evaluation = self._gate.evaluate(result)

            # Build edge list
            edge_list = []
            for edge in result.edges:
                edge_list.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "confidence": edge.confidence,
                        "type": edge.edge_type.value,
                        "algorithms": list(edge.algorithms),
                    }
                )

            # Build algorithm results
            algorithm_results = {}
            for alg_result in result.algorithm_results:
                algorithm_results[alg_result.algorithm.value] = {
                    "n_edges": len(alg_result.edge_list),
                    "runtime_seconds": alg_result.runtime_seconds,
                    "converged": alg_result.converged,
                    "score": alg_result.score,
                }

            # Compute total runtime from algorithm results
            total_runtime = sum(ar.runtime_seconds for ar in result.algorithm_results)

            # Get algorithms used from results
            algorithms_used = [ar.algorithm.value for ar in result.algorithm_results]

            output = DiscoverDagOutput(
                success=True,
                n_edges=result.n_edges,
                n_nodes=result.n_nodes,
                edge_list=edge_list,
                algorithms_used=algorithms_used,
                algorithm_results=algorithm_results,
                ensemble_threshold=result.config.ensemble_threshold,
                gate_decision=evaluation.decision.value,
                gate_confidence=evaluation.confidence,
                gate_reasons=evaluation.reasons,
                total_runtime_seconds=total_runtime,
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            output = DiscoverDagOutput(
                success=False,
                n_edges=0,
                n_nodes=0,
                edge_list=[],
                algorithms_used=[],
                algorithm_results={},
                ensemble_threshold=params.ensemble_threshold,
                gate_decision=None,
                gate_confidence=None,
                gate_reasons=[],
                total_runtime_seconds=0.0,
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors + [str(e)],
            )

        # End trace
        if self.opik_enabled and trace_id:
            self._end_trace(trace_id, output)

        return output

    def _start_trace(self, params: DiscoverDagInput) -> Optional[str]:
        """Start an Opik trace."""
        try:
            import uuid

            import opik

            trace_id = str(uuid.uuid4())
            opik.track(  # type: ignore[call-arg]
                name="discover_dag",
                input={
                    "algorithms": params.algorithms,
                    "ensemble_threshold": params.ensemble_threshold,
                    "alpha": params.alpha,
                    "n_columns": len(params.data) if params.data else 0,
                },
                metadata={"trace_id": trace_id},
            )
            return trace_id
        except Exception as e:
            logger.debug(f"Opik tracing not available: {e}")
            return None

    def _end_trace(self, trace_id: str, output: DiscoverDagOutput) -> None:
        """End an Opik trace."""
        try:
            import opik

            opik.track(  # type: ignore[call-arg]
                name="discover_dag.complete",
                output={
                    "success": output.success,
                    "n_edges": output.n_edges,
                    "gate_decision": output.gate_decision,
                    "runtime_seconds": output.total_runtime_seconds,
                },
                metadata={
                    "trace_id": trace_id,
                    "errors_count": len(output.errors),
                },
            )
        except Exception:
            pass


@dataclass
class DriverRankerTool:
    """
    Tool for comparing causal vs predictive feature importance.

    Uses DAG structure and SHAP values to identify:
    - Features that are true causal drivers
    - Features that are predictive due to correlation/confounding
    - Discrepancies between causal and predictive importance

    Attributes:
        opik_enabled: Whether Opik tracing is enabled
    """

    opik_enabled: bool = field(default=True)
    _ranker: Any = field(default=None, repr=False)

    def _ensure_initialized(self) -> None:
        """Lazy initialize ranker component."""
        if self._ranker is None:
            from src.causal_engine.discovery import DriverRanker

            self._ranker = DriverRanker()
            logger.info("DriverRankerTool: Components initialized")

    async def invoke(
        self,
        input_data: Union[Dict[str, Any], RankDriversInput],
    ) -> RankDriversOutput:
        """
        Rank features by causal and predictive importance.

        Args:
            input_data: Either a dict or RankDriversInput with parameters

        Returns:
            RankDriversOutput with rankings and analysis
        """
        # Parse input
        if isinstance(input_data, dict):
            params = RankDriversInput(**input_data)
        else:
            params = input_data

        # Initialize if needed
        self._ensure_initialized()

        # Start trace
        trace_id = self._start_trace(params) if self.opik_enabled else None

        errors: List[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Build DAG from edge list
            dag = nx.DiGraph()
            for edge in params.dag_edge_list:
                dag.add_edge(edge["source"], edge["target"])

            # Convert SHAP values to numpy array
            shap_array = np.array(params.shap_values)

            # Update ranker settings if provided
            if params.concordance_threshold != 2:
                self._ranker.concordance_threshold = params.concordance_threshold
            if params.importance_percentile != 0.25:
                self._ranker.importance_percentile = params.importance_percentile

            # Run ranking
            result = self._ranker.rank_drivers(
                dag=dag,
                target=params.target,
                shap_values=shap_array,
                feature_names=params.feature_names,
            )

            # Convert rankings to output format
            rankings = []
            for r in result.rankings:
                rankings.append(
                    FeatureRankingItem(
                        feature_name=r.feature_name,
                        causal_rank=r.causal_rank,
                        predictive_rank=r.predictive_rank,
                        rank_difference=r.rank_difference,
                        causal_score=r.causal_score,
                        predictive_score=r.predictive_score,
                        is_direct_cause=r.is_direct_cause,
                        path_length=r.path_length,
                    )
                )

            output = RankDriversOutput(
                success=True,
                target_variable=result.target_variable,
                rankings=rankings,
                n_features=len(rankings),
                rank_correlation=result.rank_correlation,
                causal_only_features=result.causal_only_features,
                predictive_only_features=result.predictive_only_features,
                concordant_features=result.concordant_features,
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Driver ranking failed: {e}")
            output = RankDriversOutput(
                success=False,
                target_variable=params.target,
                rankings=[],
                n_features=0,
                rank_correlation=0.0,
                causal_only_features=[],
                predictive_only_features=[],
                concordant_features=[],
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors + [str(e)],
            )

        # End trace
        if self.opik_enabled and trace_id:
            self._end_trace(trace_id, output)

        return output

    def _start_trace(self, params: RankDriversInput) -> Optional[str]:
        """Start an Opik trace."""
        try:
            import uuid

            import opik

            trace_id = str(uuid.uuid4())
            opik.track(  # type: ignore[call-arg]
                name="rank_drivers",
                input={
                    "target": params.target,
                    "n_features": len(params.feature_names),
                    "n_edges": len(params.dag_edge_list),
                },
                metadata={"trace_id": trace_id},
            )
            return trace_id
        except Exception as e:
            logger.debug(f"Opik tracing not available: {e}")
            return None

    def _end_trace(self, trace_id: str, output: RankDriversOutput) -> None:
        """End an Opik trace."""
        try:
            import opik

            opik.track(  # type: ignore[call-arg]
                name="rank_drivers.complete",
                output={
                    "success": output.success,
                    "n_features": output.n_features,
                    "rank_correlation": output.rank_correlation,
                    "causal_only_count": len(output.causal_only_features),
                    "predictive_only_count": len(output.predictive_only_features),
                },
                metadata={
                    "trace_id": trace_id,
                    "errors_count": len(output.errors),
                },
            )
        except Exception:
            pass


# =============================================================================
# SINGLETON AND REGISTRATION
# =============================================================================

_discovery_tool_instance: Optional[CausalDiscoveryTool] = None
_ranker_tool_instance: Optional[DriverRankerTool] = None


def get_discovery_tool() -> CausalDiscoveryTool:
    """Get or create the singleton CausalDiscoveryTool instance."""
    global _discovery_tool_instance
    if _discovery_tool_instance is None:
        _discovery_tool_instance = CausalDiscoveryTool()
    return _discovery_tool_instance


def get_ranker_tool() -> DriverRankerTool:
    """Get or create the singleton DriverRankerTool instance."""
    global _ranker_tool_instance
    if _ranker_tool_instance is None:
        _ranker_tool_instance = DriverRankerTool()
    return _ranker_tool_instance


# =============================================================================
# F7: UNIFIED DATA CONTRACT
# =============================================================================
#
# Background (verified by faithful live runs): the Tool Composer had THREE
# incompatible "data" contracts. The 7 causal tools read a real
# ``pandas.DataFrame`` from kwargs via the canonical keys
# ``_DATAFRAME_KWARGS_KEYS`` and the executor auto-injects
# ``context["estimation_data"]`` for them. But ``discover_dag`` /
# ``rank_drivers`` wanted a ``Dict[str, List]`` (``DataFrame.to_dict('list')``)
# and the executor's auto-inject did NOT serve that contract — so any plan that
# chained ``discover_dag`` failed with a pydantic ValidationError on
# ``data.<col>`` (the planner emits column->reference strings, not lists).
#
# F7 unifies the contract: ``discover_dag`` / ``rank_drivers`` now ALSO accept
# the real DataFrame via the standard ``estimation_data`` kwarg (one of
# ``_DATAFRAME_KWARGS_KEYS``), converting internally to the shape they need —
# WHILE preserving back-compat for an explicitly-supplied ``data: Dict``. The
# executor's auto-inject is extended (mirroring its Gate-1 "caller-explicit-wins"
# logic) so these two tools receive ``context["estimation_data"]`` when the
# planner did not supply explicit data.
#
# Anti-mocking invariant (CLAUDE.md): these tools either compute from a REAL
# frame / dict, or fail closed with a descriptive ``RuntimeError``. They NEVER
# fabricate data.

# Canonical kwargs keys under which a real DataFrame may be threaded into a
# composable tool. Defined LOCALLY here (not imported from the higher-level
# ``tool_composer.tool_registrations``) to keep this low-level registry module
# free of an upward dependency / import cycle. Kept in lock-step with
# ``tool_registrations._DATAFRAME_KWARGS_KEYS``.
_DATAFRAME_KWARGS_KEYS: tuple = ("data", "dataframe", "estimation_data")


def _coerce_dataframe(candidate: Any) -> Optional[pd.DataFrame]:
    """Return ``candidate`` as a DataFrame if it is one (duck-typed), else None.

    Duck-typed so we don't accidentally treat the legacy ``data: Dict[str, List]``
    contract as a frame (a plain dict has no ``.columns``).
    """
    if candidate is None:
        return None
    if isinstance(candidate, pd.DataFrame):
        return candidate
    if hasattr(candidate, "columns") and hasattr(candidate, "__len__"):
        # A pandas-like frame from another pandas build / duck type.
        return candidate  # type: ignore[return-value]
    return None


def _resolve_frame_from_kwargs(extra_kwargs: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return a real DataFrame threaded under any ``_DATAFRAME_KWARGS_KEYS`` key.

    Mirrors ``tool_registrations._extract_dataframe_from_kwargs``. Does NOT
    raise; the caller fail-closes on ``None`` per the anti-mocking contract.
    """
    for key in _DATAFRAME_KWARGS_KEYS:
        frame = _coerce_dataframe(extra_kwargs.get(key))
        if frame is not None:
            return frame
    return None


# Minimum fraction of non-null values a numeric column must have to be kept
# BEFORE the complete-case (drop-any-NaN) row filter. A real cohort frame mixes
# dense signal columns (~94% populated) with a few very sparse ones (e.g.
# adherence_rate ~6%); requiring complete rows across ALL of them annihilates
# the cohort. Dropping sparse columns first preserves the usable signal.
_MIN_NONNULL_FRAC = 0.5


def _numeric_frame(df: pd.DataFrame, min_nonnull_frac: float = _MIN_NONNULL_FRAC) -> pd.DataFrame:
    """Coerce ``df`` to numeric, drop all-NaN + overly-sparse columns, then any-NaN rows.

    The complete-case (``dropna(how="any")``) filter is applied ONLY over columns
    that are at least ``min_nonnull_frac`` populated, so a handful of mostly-empty
    columns cannot reduce a real cohort to zero complete rows. Dropped sparse
    columns are surfaced via a WARNING (no silent caps).

    Raises a descriptive ``RuntimeError`` (fail closed) if no usable numeric
    column or no complete row survives — NEVER fabricates values.
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise RuntimeError(
            "received a DataFrame with no numeric columns; causal discovery / "
            "driver ranking require numeric variables. Supply numeric features "
            "or an explicit contract value."
        )

    # Drop overly-sparse columns BEFORE the complete-case row filter so a few
    # 90%-null columns don't annihilate the usable cohort.
    nonnull_frac = numeric.notna().mean()
    dense = numeric.loc[:, nonnull_frac >= min_nonnull_frac]
    dropped = [str(c) for c in numeric.columns if c not in dense.columns]
    if dropped:
        logger.warning(
            "causal_discovery: dropping %d sparse numeric column(s) "
            "(non-null fraction < %.2f) before complete-case filter: %s",
            len(dropped),
            min_nonnull_frac,
            dropped,
        )
    if dense.shape[1] < 2:
        raise RuntimeError(
            "received a DataFrame with fewer than 2 sufficiently-populated "
            f"numeric columns (non-null fraction >= {min_nonnull_frac}); causal "
            "discovery / driver ranking need at least two dense numeric "
            "variables. Supply denser features or an explicit contract value."
        )

    complete = dense.dropna(axis=0, how="any")
    if complete.shape[0] == 0:
        raise RuntimeError(
            "received a DataFrame whose dense numeric columns share no complete "
            "(non-NaN) rows; cannot run on an empty frame."
        )
    return complete


def _frame_to_numeric_dict(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """Convert a DataFrame to the ``Dict[str, List]`` shape discovery needs."""
    result: Dict[str, List[Any]] = _numeric_frame(df).to_dict("list")
    return result


def _is_valid_data_dict(value: Any) -> bool:
    """True iff ``value`` is the legacy ``data: Dict[str, List]`` contract.

    The planner sometimes emits ``data={'col': '$step.field'}`` (column ->
    reference string) which is NOT a valid discovery dict; such a value must be
    rejected so the injected DataFrame is used instead.
    """
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, (list, tuple)) for v in value.values())


async def discover_dag(
    data: Union[pd.DataFrame, Dict[str, List[Any]], None] = None,
    algorithms: Optional[List[str]] = None,
    ensemble_threshold: float = 0.5,
    alpha: float = 0.05,
    max_k: Optional[int] = None,
    node_names: Optional[List[str]] = None,
    trace_context: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Discover DAG structure from observational data.

    This is the registered tool function that wraps CausalDiscoveryTool.

    Data contract (F7 — unified):
        Accepts the working data via EITHER of two routes, in priority order:

        1. **Explicit ``data``** — a ``pandas.DataFrame`` OR the legacy
           ``Dict[str, List]`` (``DataFrame.to_dict('list')``). Caller-explicit
           always wins (C1 trust-gate parity).
        2. **Injected real DataFrame** — under any canonical
           ``_DATAFRAME_KWARGS_KEYS`` key in ``**kwargs`` (the executor injects
           the in-context frame under ``estimation_data``). Converted internally
           to the ``Dict[str, List]`` shape via ``_frame_to_numeric_dict``.

        If NEITHER yields usable data, the tool FAILS CLOSED with a descriptive
        ``RuntimeError`` — it NEVER fabricates a synthetic frame.

    Args:
        data: DataFrame, dict of column name to values, or None (then the frame
            is resolved from ``**kwargs`` canonical keys).
        algorithms: Algorithms to use (default: ["ges", "pc"])
        ensemble_threshold: Minimum algorithm agreement (0-1)
        alpha: Significance level for CI tests
        max_k: Maximum conditioning set size
        node_names: Custom node names
        trace_context: Opik trace context
        **kwargs: May carry the real DataFrame under a canonical key
            (``data`` / ``dataframe`` / ``estimation_data``).

    Returns:
        Dictionary with discovered DAG and metadata
    """
    tool = get_discovery_tool()

    # Resolve the working data, caller-explicit first (F7 unified contract).
    explicit_frame = _coerce_dataframe(data)
    if explicit_frame is not None:
        # Explicit DataFrame -> numeric Dict (engine consumes numeric vars).
        data_dict = _frame_to_numeric_dict(explicit_frame)
    elif _is_valid_data_dict(data):
        # Legacy explicit Dict[str, List] contract — pass through unchanged.
        data_dict = data  # type: ignore[assignment]
    else:
        # No usable explicit ``data``: try the injected real DataFrame from
        # the canonical kwargs keys (executor auto-inject).
        injected = _resolve_frame_from_kwargs(kwargs)
        if injected is None:
            raise RuntimeError(
                "discover_dag requires a real DataFrame (under one of "
                f"{list(_DATAFRAME_KWARGS_KEYS)!r}) or an explicit "
                "`data: Dict[str, List]`; none was supplied. The tool fails "
                "closed rather than fabricate data (CLAUDE.md anti-mocking)."
            )
        data_dict = _frame_to_numeric_dict(injected)

    result = await tool.invoke(
        DiscoverDagInput(
            data=data_dict,
            algorithms=algorithms or ["ges", "pc"],
            ensemble_threshold=ensemble_threshold,
            alpha=alpha,
            max_k=max_k,
            node_names=node_names,
            trace_context=trace_context,
        )
    )

    return result.model_dump()


def _compute_shap_from_frame(
    df: pd.DataFrame,
    target: str,
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[List[float]], List[str]]:
    """Compute REAL SHAP values + feature names from a DataFrame and target.

    F7: ``rank_drivers`` consumes the SAME real DataFrame as ``discover_dag``.
    Its predictive importance needs SHAP values, which require a fitted model.
    This derives them honestly from the frame:

    1. Coerce to numeric, drop all-NaN cols / any-NaN rows.
    2. ``feature_names`` = numeric columns minus ``target`` (or the explicit
       list, intersected with what's present).
    3. Fit a RandomForest on ``(features, target)`` and compute TreeExplainer
       SHAP values.

    Fails closed (descriptive ``RuntimeError``) if ``target`` is absent or no
    usable features remain. NEVER fabricates SHAP values.
    """
    numeric = _numeric_frame(df)
    if target not in numeric.columns:
        raise RuntimeError(
            f"rank_drivers: target {target!r} is not a numeric column of the "
            f"supplied DataFrame (numeric columns: {list(numeric.columns)}). "
            "Cannot compute predictive importance without the target."
        )

    if feature_names:
        feats = [c for c in feature_names if c in numeric.columns and c != target]
    else:
        feats = [c for c in numeric.columns if c != target]
    if not feats:
        raise RuntimeError(
            "rank_drivers: no usable numeric feature columns remain after "
            f"excluding target {target!r}; cannot compute SHAP values."
        )

    import shap
    from sklearn.ensemble import RandomForestRegressor

    x = numeric[feats].to_numpy(dtype=float)
    y = numeric[target].to_numpy(dtype=float)
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(x, y)
    explainer = shap.TreeExplainer(model)
    shap_array = np.asarray(explainer.shap_values(x), dtype=float)
    # TreeExplainer may return (n_samples, n_features) or, for some models, a
    # list/3D array; collapse to a 2-D (n_samples, n_features) matrix.
    if shap_array.ndim == 3:
        shap_array = shap_array.mean(axis=2)
    return shap_array.tolist(), feats


def _normalize_edge_list(
    dag_edge_list: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Keep only ``source``/``target`` (as strings) from each edge.

    F7: the chain ``discover_dag.edge_list -> rank_drivers.dag_edge_list`` is the
    documented pipeline (see ``planner.py``). ``discover_dag`` emits edges with
    extra keys (``confidence: float``, ``type: str``, ``algorithms: list``) but
    ``RankDriversInput.dag_edge_list`` is typed ``List[Dict[str, str]]`` and the
    ranker only reads ``edge["source"]`` / ``edge["target"]``. Stripping the
    extra keys lets the real ``discover_dag`` output flow into ``rank_drivers``
    on ONE data source without loosening the schema. Edges missing source/target
    are skipped (fail-soft on a malformed edge, never fabricate one).
    """
    normalized: List[Dict[str, str]] = []
    for edge in dag_edge_list or []:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            continue
        normalized.append({"source": str(src), "target": str(tgt)})
    return normalized


def _predictive_only_ranking(
    target: str,
    shap_list: Union[np.ndarray, List[List[float]]],
    feature_names: List[str],
    trace_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Rank drivers by PREDICTIVE (SHAP) importance only — used when no causal DAG
    edges were discovered.

    The DAG-based ranker requires every SHAP feature to be a node of the causal
    graph; an empty DAG cannot satisfy that, so the causal-vs-predictive
    concordance is undefined. The mean |SHAP| importance ranking is still a real,
    honest result (predictive drivers of the target) — return it rather than
    failing the step. Fails closed only when the SHAP matrix is unusable (never
    fabricates importances).
    """
    arr = np.asarray(shap_list, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != len(feature_names):
        raise RuntimeError(
            "rank_drivers: cannot compute a predictive ranking — SHAP matrix shape "
            f"{getattr(arr, 'shape', None)} does not match {len(feature_names)} features."
        )
    mean_abs = np.abs(arr).mean(axis=0)
    order = list(np.argsort(mean_abs)[::-1])
    rankings = [
        FeatureRankingItem(
            feature_name=str(feature_names[idx]),
            # No DAG -> no causal estimate. None (not 0) so the absence is explicit
            # and never read as a real zero-effect causal value.
            causal_rank=None,
            predictive_rank=rank,
            rank_difference=None,
            causal_score=None,
            predictive_score=float(mean_abs[idx]),
            is_direct_cause=False,
            path_length=None,
        )
        for rank, idx in enumerate(order, start=1)
    ]
    out = RankDriversOutput(
        success=True,
        target_variable=target,
        rankings=rankings,
        n_features=len(feature_names),
        # Causal-vs-predictive concordance is undefined without a DAG -> None, not 0.0.
        rank_correlation=None,
        causal_only_features=[],
        predictive_only_features=[str(f) for f in feature_names],
        concordant_features=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=(trace_context or {}).get("trace_id") if isinstance(trace_context, dict) else None,
        errors=["no causal DAG edges discovered; predictive (SHAP) ranking only"],
    )
    return out.model_dump()


async def rank_drivers(
    dag_edge_list: List[Dict[str, Any]],
    target: str,
    shap_values: Union[np.ndarray, List[List[float]], None] = None,
    feature_names: Optional[List[str]] = None,
    concordance_threshold: int = 2,
    importance_percentile: float = 0.25,
    trace_context: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Rank features by causal vs predictive importance.

    This is the registered tool function that wraps DriverRankerTool.

    Data contract (F7 — unified):
        Predictive importance needs ``shap_values`` + ``feature_names``. These
        may be supplied explicitly (caller-explicit wins) OR derived from the
        SAME real DataFrame that ``discover_dag`` consumed:

        * If BOTH ``shap_values`` and ``feature_names`` are supplied → use them.
        * Else, resolve a real DataFrame from ``**kwargs`` canonical keys
          (``data`` / ``dataframe`` / ``estimation_data``; the executor injects
          the in-context frame under ``estimation_data``) and compute REAL SHAP
          values from ``(features, target)`` via ``_compute_shap_from_frame``.
        * If neither path yields data → FAIL CLOSED (``RuntimeError``). Never
          fabricates SHAP values.

    Args:
        dag_edge_list: DAG as list of edges (e.g. ``$step_1.edge_list`` from a
            chained ``discover_dag``).
        target: Target variable name (required — predictive importance is
            computed relative to it).
        shap_values: SHAP values (n_samples x n_features), or None to derive.
        feature_names: Feature names, or None to derive from the frame.
        concordance_threshold: Max rank diff for concordant features
        importance_percentile: Top percentile for "important"
        trace_context: Opik trace context
        **kwargs: May carry the real DataFrame under a canonical key.

    Returns:
        Dictionary with rankings and analysis
    """
    tool = get_ranker_tool()

    if shap_values is not None and feature_names:
        # Caller-explicit wins.
        if isinstance(shap_values, np.ndarray):
            shap_list = shap_values.tolist()
        else:
            shap_list = shap_values
        resolved_features = feature_names
    else:
        # Derive REAL SHAP from the same in-context DataFrame.
        frame = _resolve_frame_from_kwargs(kwargs)
        if frame is None:
            raise RuntimeError(
                "rank_drivers requires either explicit `shap_values` + "
                "`feature_names`, or a real DataFrame (under one of "
                f"{list(_DATAFRAME_KWARGS_KEYS)!r}) to derive them from. None "
                "was supplied. The tool fails closed rather than fabricate "
                "SHAP values (CLAUDE.md anti-mocking)."
            )
        # #1548: run the CPU-bound SHAP derivation (RandomForest fit +
        # TreeExplainer ``shap_values``) OFF the event loop. Measured
        # (2026-08-13 live faulthandler dumps): inline, this starved the
        # uvicorn worker's main loop for >120s on heavy /chat/stream turns;
        # uvicorn's ``callback_notify`` never ran, so gunicorn's arbiter
        # murdered the worker at last-notify+120s → mid-stream tear. The
        # executor's ``asyncio.wait_for`` cannot preempt a sync call that
        # never yields.
        #
        # Seam choice: the EXISTING shared bounded heavy-compute pool
        # (``run_in_bounded_executor``, prod heavy-compute cap prior art) —
        # NOT ``asyncio.to_thread`` (loop's default executor is unbounded: N
        # concurrent turns could fit N RandomForests inside the 5G cgroup,
        # the exact OOM class the bounded pool was built to prevent) and NOT
        # ``shap_explainer_realtime``'s pool (import-time global needing the
        # gunicorn ``post_fork`` reset, eagerly imports mlflow, and is
        # coupled to the /explain model-explainer cache). ``compute.py``'s
        # pool is created lazily at first call — inherently preload/fork-safe.
        # No ``heavy_compute_slot`` here: its reject-fast contract is for API
        # entry points that can answer 503 + Retry-After; a composer step
        # should briefly queue (bounded by the executor's ``wait_for``)
        # rather than instantly fail the chat turn, and in-flight compute is
        # already capped by the pool's worker count.
        #
        # Cancellation semantics: with the block off-loop, the composer's
        # ``wait_for`` timeout CAN now fire — but cancelling the executor
        # future cannot interrupt a compute already running in the pool
        # thread; it runs to completion and its result is discarded. That is
        # acceptable: the compute is pure CPU over an in-memory frame (no
        # side effects to corrupt), the abandoned work occupies only a
        # bounded pool slot (delaying other heavy compute, never the loop),
        # and the step fails with a well-formed error instead of the worker
        # being murdered mid-stream.
        #
        # Function-local import (mirrors ``model_inference.py``): keeps the
        # ``src.api.dependencies`` package out of this module's import path
        # for non-API consumers of the tool registry.
        from src.api.dependencies.compute import run_in_bounded_executor

        # Positive log marker for live verification of #1548 (grep:
        # "off-loading SHAP derivation").
        logger.info(
            "rank_drivers: off-loading SHAP derivation to bounded heavy-compute pool (frame=%dx%d)",
            frame.shape[0],
            frame.shape[1],
        )
        shap_list, resolved_features = await run_in_bounded_executor(
            _compute_shap_from_frame, frame, target, feature_names
        )

    # F7: normalize the edge list so a chained ``discover_dag.edge_list`` (which
    # carries extra ``confidence``/``type``/``algorithms`` keys) validates
    # against ``RankDriversInput.dag_edge_list`` (``List[Dict[str, str]]``).
    normalized_edges = _normalize_edge_list(dag_edge_list)

    # When no causal DAG edges were discovered, the DAG-based ranker cannot run
    # (it requires every SHAP feature to be a graph node). Fall back to a real
    # predictive-only (SHAP) ranking rather than failing the step.
    if not normalized_edges:
        return _predictive_only_ranking(target, shap_list, resolved_features, trace_context)

    result = await tool.invoke(
        RankDriversInput(
            dag_edge_list=normalized_edges,
            target=target,
            shap_values=shap_list,
            feature_names=resolved_features,
            concordance_threshold=concordance_threshold,
            importance_percentile=importance_percentile,
            trace_context=trace_context,
        )
    )

    return result.model_dump()


# =============================================================================
# TOOL REGISTRATION
# =============================================================================


def register_discover_dag_tool() -> None:
    """Register the discover_dag tool in the global registry."""
    schema = ToolSchema(
        name="discover_dag",
        description=(
            "Discover causal DAG structure from observational data using ensemble "
            "algorithms (GES, PC, etc.). Returns edges with confidence scores and "
            "gate evaluation for determining result quality."
        ),
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter(
                name="data",
                type="Dict[str, List[Any]]",
                description="Data as dict of column names to values",
                required=True,
            ),
            ToolParameter(
                name="algorithms",
                type="List[str]",
                description="Algorithms: 'ges', 'pc', 'fci', 'lingam'",
                required=False,
                default=["ges", "pc"],
            ),
            ToolParameter(
                name="ensemble_threshold",
                type="float",
                description="Min algorithm agreement (0-1)",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="alpha",
                type="float",
                description="Significance level for CI tests",
                required=False,
                default=0.05,
            ),
        ],
        output_schema="DiscoverDagOutput",
        avg_execution_ms=5000,
        is_async=True,
        supports_batch=False,
    )

    registry = get_registry()
    registry.register(
        schema=schema,
        callable=discover_dag,
        input_model=DiscoverDagInput,
        output_model=DiscoverDagOutput,
    )

    logger.info("Registered discover_dag tool in ToolRegistry")


def register_rank_drivers_tool() -> None:
    """Register the rank_drivers tool in the global registry."""
    schema = ToolSchema(
        name="rank_drivers",
        description=(
            "Compare causal vs predictive feature importance. Uses DAG structure "
            "for causal importance and SHAP values for predictive importance. "
            "Identifies features that are causally important vs just correlated."
        ),
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter(
                name="dag_edge_list",
                type="List[Dict[str, str]]",
                description="DAG as list of {source, target} edges",
                required=True,
            ),
            ToolParameter(
                name="target",
                type="str",
                description="Target variable name",
                required=True,
            ),
            ToolParameter(
                name="shap_values",
                type="List[List[float]]",
                description="SHAP values matrix (n_samples x n_features)",
                required=True,
            ),
            ToolParameter(
                name="feature_names",
                type="List[str]",
                description="Feature names",
                required=True,
            ),
            ToolParameter(
                name="concordance_threshold",
                type="int",
                description="Max rank diff for concordant features",
                required=False,
                default=2,
            ),
        ],
        output_schema="RankDriversOutput",
        avg_execution_ms=500,
        is_async=True,
        supports_batch=False,
    )

    registry = get_registry()
    registry.register(
        schema=schema,
        callable=rank_drivers,
        input_model=RankDriversInput,
        output_model=RankDriversOutput,
    )

    logger.info("Registered rank_drivers tool in ToolRegistry")


def register_all_discovery_tools() -> None:
    """Register all causal discovery tools."""
    register_discover_dag_tool()
    register_rank_drivers_tool()
    logger.info("All causal discovery tools registered")


# Auto-register on import (can be disabled if needed)
try:
    register_all_discovery_tools()
except Exception as e:
    logger.debug(f"Deferred tool registration: {e}")
