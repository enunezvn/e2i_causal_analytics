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
from typing import Any, Dict, List, Optional, Union

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

    data: Dict[str, List[Any]] = Field(
        ...,
        description="Data as dictionary of column names to values (DataFrame.to_dict('list'))",
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
    """Single feature ranking information."""

    feature_name: str
    causal_rank: int
    predictive_rank: int
    rank_difference: int
    causal_score: float
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
    rank_correlation: float = Field(
        default=0.0,
        description="Spearman correlation between causal and predictive ranks",
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
                max_k=params.max_k if params.max_k is not None else -1,
                node_names=params.node_names,
            )

            # Run discovery
            result = await self._runner.discover_dag(df, config)

            # Evaluate with gate
            evaluation = self._gate.evaluate(result)

            # Build edge list
            edge_list = []
            for edge in result.edges:
                edge_list.append({
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                    "type": edge.edge_type.value,
                    "algorithms": list(edge.algorithms),
                })

            # Build algorithm results
            algorithm_results = {}
            for alg_result in result.algorithm_results:
                algorithm_results[alg_result.algorithm.value] = {
                    "n_edges": alg_result.n_edges,
                    "runtime_seconds": alg_result.runtime_seconds,
                    "converged": alg_result.converged,
                    "score": alg_result.score,
                }

            output = DiscoverDagOutput(
                success=True,
                n_edges=result.n_edges,
                n_nodes=result.n_nodes,
                edge_list=edge_list,
                algorithms_used=[a.value for a in result.algorithms_used],
                algorithm_results=algorithm_results,
                ensemble_threshold=result.ensemble_threshold,
                gate_decision=evaluation.decision.value,
                gate_confidence=evaluation.confidence,
                gate_reasons=evaluation.reasons,
                total_runtime_seconds=result.total_runtime_seconds,
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
            opik.track(
                name="discover_dag",
                input={
                    "algorithms": params.algorithms,
                    "ensemble_threshold": params.ensemble_threshold,
                    "alpha": params.alpha,
                    "n_columns": len(params.data),
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

            opik.track(
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

        errors = []
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
            opik.track(
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

            opik.track(
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


async def discover_dag(
    data: Union[pd.DataFrame, Dict[str, List[Any]]],
    algorithms: Optional[List[str]] = None,
    ensemble_threshold: float = 0.5,
    alpha: float = 0.05,
    max_k: Optional[int] = None,
    node_names: Optional[List[str]] = None,
    trace_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Discover DAG structure from observational data.

    This is the registered tool function that wraps CausalDiscoveryTool.

    Args:
        data: DataFrame or dict of column name to values
        algorithms: Algorithms to use (default: ["ges", "pc"])
        ensemble_threshold: Minimum algorithm agreement (0-1)
        alpha: Significance level for CI tests
        max_k: Maximum conditioning set size
        node_names: Custom node names
        trace_context: Opik trace context

    Returns:
        Dictionary with discovered DAG and metadata
    """
    tool = get_discovery_tool()

    # Convert DataFrame to dict if needed
    if isinstance(data, pd.DataFrame):
        data_dict = data.to_dict("list")
    else:
        data_dict = data

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


async def rank_drivers(
    dag_edge_list: List[Dict[str, str]],
    target: str,
    shap_values: Union[np.ndarray, List[List[float]]],
    feature_names: List[str],
    concordance_threshold: int = 2,
    importance_percentile: float = 0.25,
    trace_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Rank features by causal vs predictive importance.

    This is the registered tool function that wraps DriverRankerTool.

    Args:
        dag_edge_list: DAG as list of edges
        target: Target variable name
        shap_values: SHAP values (n_samples x n_features)
        feature_names: Feature names
        concordance_threshold: Max rank diff for concordant features
        importance_percentile: Top percentile for "important"
        trace_context: Opik trace context

    Returns:
        Dictionary with rankings and analysis
    """
    tool = get_ranker_tool()

    # Convert numpy to list if needed
    if isinstance(shap_values, np.ndarray):
        shap_list = shap_values.tolist()
    else:
        shap_list = shap_values

    result = await tool.invoke(
        RankDriversInput(
            dag_edge_list=dag_edge_list,
            target=target,
            shap_values=shap_list,
            feature_names=feature_names,
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
