"""
Causal Library Router

Routes KPI calculations to appropriate causal inference libraries
based on the KPI Framework documentation's library selection matrix.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

from src.kpi.models import CausalLibrary, KPIMetadata


class CausalEstimator(Protocol):
    """Protocol for causal estimators."""

    def estimate(
        self,
        treatment: Any,
        outcome: Any,
        covariates: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate causal effect.

        Returns dict with keys:
            - effect: float - estimated causal effect
            - ci_lower: float - lower confidence interval bound
            - ci_upper: float - upper confidence interval bound
            - p_value: float - statistical significance
        """
        ...


class CausalLibraryRouter:
    """Routes KPI calculations to appropriate causal libraries.

    Library selection follows the KPI Framework documentation:
    - DoWhy: Causal validation, ATE estimation, backdoor/IV adjustment
    - EconML: Heterogeneous treatment effects (CATE), Double ML
    - CausalML: Uplift modeling, targeting optimization
    - NetworkX: DAG analysis, path analysis, flow optimization
    """

    def __init__(self) -> None:
        """Initialize the router with available libraries."""
        self._dowhy_available = self._check_dowhy()
        self._econml_available = self._check_econml()
        self._causalml_available = self._check_causalml()
        self._networkx_available = self._check_networkx()

        logger.info(
            f"Causal libraries available: DoWhy={self._dowhy_available}, "
            f"EconML={self._econml_available}, CausalML={self._causalml_available}, "
            f"NetworkX={self._networkx_available}"
        )

    @staticmethod
    def _check_dowhy() -> bool:
        """Check if DoWhy is available."""
        try:
            import dowhy  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_econml() -> bool:
        """Check if EconML is available."""
        try:
            import econml  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_causalml() -> bool:
        """Check if CausalML is available."""
        try:
            import causalml  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_networkx() -> bool:
        """Check if NetworkX is available."""
        try:
            import networkx  # noqa: F401

            return True
        except ImportError:
            return False

    def is_available(self, library: CausalLibrary) -> bool:
        """Check if a causal library is available."""
        if library == CausalLibrary.DOWHY:
            return self._dowhy_available
        elif library == CausalLibrary.ECONML:
            return self._econml_available
        elif library == CausalLibrary.CAUSALML:
            return self._causalml_available
        elif library == CausalLibrary.NETWORKX:
            return self._networkx_available
        return True  # CausalLibrary.NONE is always available

    def get_recommended_library(self, kpi: KPIMetadata) -> CausalLibrary:
        """Get the recommended library for a KPI calculation.

        Falls back to alternatives if primary library is unavailable.

        Args:
            kpi: The KPI metadata

        Returns:
            The recommended CausalLibrary to use
        """
        primary = kpi.primary_causal_library

        # If primary is available, use it
        if self.is_available(primary):
            return primary

        # Fallback logic based on KPI type
        if primary == CausalLibrary.ECONML and self._dowhy_available:
            logger.warning(f"EconML unavailable for {kpi.id}, falling back to DoWhy")
            return CausalLibrary.DOWHY

        if primary == CausalLibrary.CAUSALML and self._econml_available:
            logger.warning(f"CausalML unavailable for {kpi.id}, falling back to EconML")
            return CausalLibrary.ECONML

        if primary == CausalLibrary.DOWHY and self._networkx_available:
            logger.warning(f"DoWhy unavailable for {kpi.id}, falling back to NetworkX")
            return CausalLibrary.NETWORKX

        # Last resort
        logger.warning(f"No causal library available for {kpi.id}, using non-causal calculation")
        return CausalLibrary.NONE

    def get_estimator(
        self,
        library: CausalLibrary,
        estimator_type: str = "default",
    ) -> CausalEstimator | None:
        """Get a causal estimator for the specified library.

        Args:
            library: The causal library to use
            estimator_type: Type of estimator (e.g., "dml", "x_learner", "uplift")

        Returns:
            Estimator instance or None if unavailable
        """
        if library == CausalLibrary.NONE or not self.is_available(library):
            return None

        # Import and return appropriate estimator
        # These are stub implementations - will be fleshed out in later phases
        if library == CausalLibrary.DOWHY:
            return self._get_dowhy_estimator(estimator_type)
        elif library == CausalLibrary.ECONML:
            return self._get_econml_estimator(estimator_type)
        elif library == CausalLibrary.CAUSALML:
            return self._get_causalml_estimator(estimator_type)
        elif library == CausalLibrary.NETWORKX:
            return self._get_networkx_estimator(estimator_type)

        return None

    def _get_dowhy_estimator(self, estimator_type: str) -> CausalEstimator | None:
        """Get a DoWhy estimator."""
        # Stub - will be implemented in Phase B4 (IV Support)
        # For now, return None as placeholder
        logger.debug(f"DoWhy estimator requested: {estimator_type}")
        return None

    def _get_econml_estimator(self, estimator_type: str) -> CausalEstimator | None:
        """Get an EconML estimator."""
        # Stub - will be implemented in Phase B1-B3
        logger.debug(f"EconML estimator requested: {estimator_type}")
        return None

    def _get_causalml_estimator(self, estimator_type: str) -> CausalEstimator | None:
        """Get a CausalML estimator."""
        # Stub - will be implemented in Phase B5-B6 (Uplift)
        logger.debug(f"CausalML estimator requested: {estimator_type}")
        return None

    def _get_networkx_estimator(self, estimator_type: str) -> CausalEstimator | None:
        """Get a NetworkX-based estimator."""
        # Stub - NetworkX is for graph analysis, not causal estimation
        logger.debug(f"NetworkX analysis requested: {estimator_type}")
        return None

    def select_library_for_analysis(
        self,
        analysis_type: str,
        data_characteristics: dict[str, Any] | None = None,
    ) -> CausalLibrary:
        """Select the best library for a specific analysis type.

        Based on the KPI Framework library selection decision matrix.

        Args:
            analysis_type: Type of analysis needed
            data_characteristics: Optional dict with data info

        Returns:
            Recommended CausalLibrary
        """
        analysis_lower = analysis_type.lower()

        # ATE/causal validation -> DoWhy
        if any(term in analysis_lower for term in ["ate", "validation", "backdoor", "iv"]):
            return CausalLibrary.DOWHY if self._dowhy_available else CausalLibrary.NONE

        # CATE/heterogeneous effects -> EconML
        if any(term in analysis_lower for term in ["cate", "heterogeneous", "dml", "forest"]):
            return CausalLibrary.ECONML if self._econml_available else CausalLibrary.DOWHY

        # Uplift/targeting -> CausalML
        if any(term in analysis_lower for term in ["uplift", "targeting", "segment"]):
            return CausalLibrary.CAUSALML if self._causalml_available else CausalLibrary.ECONML

        # Graph/flow analysis -> NetworkX
        if any(term in analysis_lower for term in ["graph", "flow", "path", "network"]):
            return CausalLibrary.NETWORKX if self._networkx_available else CausalLibrary.NONE

        # Default based on data characteristics
        if data_characteristics:
            n_samples = data_characteristics.get("n_samples", 0)
            n_features = data_characteristics.get("n_features", 0)

            # High-dimensional data benefits from EconML
            if n_features > 50 and self._econml_available:
                return CausalLibrary.ECONML

            # Large datasets benefit from DoWhy
            if n_samples > 10000 and self._dowhy_available:
                return CausalLibrary.DOWHY

        # Default to DoWhy for general causal analysis
        if self._dowhy_available:
            return CausalLibrary.DOWHY

        return CausalLibrary.NONE
