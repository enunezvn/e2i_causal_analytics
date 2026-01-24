"""
KPI Registry

Loads KPI definitions from YAML and provides lookup functionality.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import yaml
import logging

logger = logging.getLogger(__name__)

from src.kpi.models import (
    CalculationType,
    CausalLibrary,
    KPIMetadata,
    KPIThreshold,
    Workstream,
)


class KPIRegistry:
    """Registry for KPI definitions loaded from YAML configuration."""

    _instance: "KPIRegistry | None" = None
    _kpis: dict[str, KPIMetadata]
    _by_workstream: dict[Workstream, list[KPIMetadata]]

    def __new__(cls) -> "KPIRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._kpis = {}
            cls._instance._by_workstream = {}
            cls._instance._load_definitions()
        return cls._instance

    def _load_definitions(self) -> None:
        """Load KPI definitions from YAML file."""
        config_path = self._find_config_path()
        if not config_path.exists():
            logger.warning(f"KPI definitions file not found: {config_path}")
            return

        logger.info(f"Loading KPI definitions from {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Parse each workstream
        workstream_mapping = {
            "ws1_data_quality": Workstream.WS1_DATA_QUALITY,
            "ws1_model_performance": Workstream.WS1_MODEL_PERFORMANCE,
            "ws2_triggers": Workstream.WS2_TRIGGERS,
            "ws3_business": Workstream.WS3_BUSINESS,
            "brand_specific": Workstream.BRAND_SPECIFIC,
            "causal_metrics": Workstream.CAUSAL_METRICS,
        }

        for ws_key, ws_enum in workstream_mapping.items():
            if ws_key in data:
                self._parse_workstream(data[ws_key], ws_enum)

        logger.info(f"Loaded {len(self._kpis)} KPI definitions")

    def _find_config_path(self) -> Path:
        """Find the KPI definitions config file."""
        # Try multiple possible locations
        candidates = [
            Path("config/kpi_definitions.yaml"),
            Path(__file__).parent.parent.parent / "config" / "kpi_definitions.yaml",
            Path(os.getenv("E2I_CONFIG_PATH", "")) / "kpi_definitions.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Default fallback
        return Path("config/kpi_definitions.yaml")

    def _parse_workstream(
        self, ws_data: dict, workstream: Workstream
    ) -> None:
        """Parse KPI definitions for a workstream."""
        for kpi_key, kpi_data in ws_data.items():
            try:
                kpi = self._parse_kpi(kpi_key, kpi_data, workstream)
                self._kpis[kpi.id] = kpi

                if workstream not in self._by_workstream:
                    self._by_workstream[workstream] = []
                self._by_workstream[workstream].append(kpi)

            except Exception as e:
                logger.warning(f"Failed to parse KPI {kpi_key}: {e}")

    def _parse_kpi(
        self, key: str, data: dict, workstream: Workstream
    ) -> KPIMetadata:
        """Parse a single KPI definition."""
        # Parse threshold
        threshold = None
        if "threshold" in data and data["threshold"]:
            threshold = KPIThreshold(
                target=data["threshold"].get("target"),
                warning=data["threshold"].get("warning"),
                critical=data["threshold"].get("critical"),
            )

        # Parse calculation type
        calc_type = CalculationType.DIRECT
        if data.get("calculation_type") == "derived":
            calc_type = CalculationType.DERIVED

        # Determine primary causal library based on workstream and KPI type
        primary_causal = self._determine_causal_library(workstream, data)

        return KPIMetadata(
            id=data.get("id", key),
            name=data.get("name", key),
            definition=data.get("definition", ""),
            formula=data.get("formula", ""),
            calculation_type=calc_type,
            workstream=workstream,
            tables=data.get("tables", []),
            columns=data.get("columns", []),
            view=data.get("view"),
            threshold=threshold,
            unit=data.get("unit"),
            frequency=data.get("frequency", "daily"),
            primary_causal_library=primary_causal,
            brand=data.get("brand"),
            note=data.get("note"),
        )

    def _determine_causal_library(
        self, workstream: Workstream, data: dict
    ) -> CausalLibrary:
        """Determine the appropriate causal library for a KPI.

        Based on the KPI Framework documentation's library selection matrix.
        """
        kpi_id = data.get("id", "")

        # Model Performance KPIs often use EconML for heterogeneous effects
        if workstream == Workstream.WS1_MODEL_PERFORMANCE:
            if "treatment_effect" in kpi_id.lower() or "cate" in kpi_id.lower():
                return CausalLibrary.ECONML
            return CausalLibrary.NONE

        # Trigger Performance uses DoWhy for causal validation
        if workstream == Workstream.WS2_TRIGGERS:
            if "uplift" in kpi_id.lower() or "action_rate" in data.get("name", "").lower():
                return CausalLibrary.CAUSALML
            return CausalLibrary.DOWHY

        # Business metrics use NetworkX for flow analysis
        if workstream == Workstream.WS3_BUSINESS:
            if "roi" in kpi_id.lower() or "conversion" in data.get("name", "").lower():
                return CausalLibrary.DOWHY
            return CausalLibrary.NETWORKX

        # Causal metrics use their designated libraries
        if workstream == Workstream.CAUSAL_METRICS:
            if "ate" in kpi_id.lower():
                return CausalLibrary.DOWHY
            if "cate" in kpi_id.lower():
                return CausalLibrary.ECONML
            if "mediation" in kpi_id.lower():
                return CausalLibrary.DOWHY
            return CausalLibrary.ECONML

        return CausalLibrary.NONE

    def get(self, kpi_id: str) -> KPIMetadata | None:
        """Get a KPI by its ID."""
        return self._kpis.get(kpi_id)

    def get_all(self) -> list[KPIMetadata]:
        """Get all KPI definitions."""
        return list(self._kpis.values())

    def get_by_workstream(self, workstream: Workstream) -> list[KPIMetadata]:
        """Get all KPIs for a specific workstream."""
        return self._by_workstream.get(workstream, [])

    def get_by_causal_library(self, library: CausalLibrary) -> list[KPIMetadata]:
        """Get all KPIs that use a specific causal library."""
        return [
            kpi for kpi in self._kpis.values()
            if kpi.primary_causal_library == library
        ]

    def __iter__(self) -> Iterator[KPIMetadata]:
        """Iterate over all KPIs."""
        return iter(self._kpis.values())

    def __len__(self) -> int:
        """Return number of KPIs."""
        return len(self._kpis)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


@lru_cache(maxsize=1)
def get_registry() -> KPIRegistry:
    """Get the global KPI registry instance."""
    return KPIRegistry()
