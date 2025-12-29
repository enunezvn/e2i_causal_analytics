"""Tests for KPI Registry."""

import pytest
from unittest.mock import patch, mock_open

from src.kpi.models import CausalLibrary, Workstream
from src.kpi.registry import KPIRegistry, get_registry


# Sample YAML content for testing
SAMPLE_KPI_YAML = """
version: "3.0.0"

ws1_data_quality:
  source_coverage_patients:
    id: "WS1-DQ-001"
    name: "Source Coverage - Patients"
    definition: "Percentage of eligible patients"
    formula: "covered_patients / reference_patients"
    calculation_type: derived
    tables:
      - patient_journeys
      - reference_universe
    threshold:
      target: 0.85
      warning: 0.70
      critical: 0.50
    frequency: daily

ws2_triggers:
  trigger_precision:
    id: "WS2-TR-001"
    name: "Trigger Precision"
    definition: "Percentage of fired triggers resulting in positive outcome"
    formula: "TP / (TP + FP)"
    calculation_type: derived
    tables:
      - triggers
    threshold:
      target: 0.70
      warning: 0.55
      critical: 0.40
    frequency: daily
"""


class TestKPIRegistry:
    """Tests for KPIRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        KPIRegistry.reset()
        yield
        KPIRegistry.reset()

    def test_singleton_pattern(self):
        """Test that registry uses singleton pattern."""
        reg1 = KPIRegistry()
        reg2 = KPIRegistry()
        assert reg1 is reg2

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_definitions(self, mock_exists):
        """Test loading KPI definitions from YAML."""
        registry = KPIRegistry()

        # Should have loaded 2 KPIs
        assert len(registry) == 2

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_kpi_by_id(self, mock_exists):
        """Test getting KPI by ID."""
        registry = KPIRegistry()

        kpi = registry.get("WS1-DQ-001")
        assert kpi is not None
        assert kpi.name == "Source Coverage - Patients"
        assert kpi.workstream == Workstream.WS1_DATA_QUALITY

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_nonexistent_kpi(self, mock_exists):
        """Test getting non-existent KPI returns None."""
        registry = KPIRegistry()

        kpi = registry.get("NONEXISTENT")
        assert kpi is None

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_all_kpis(self, mock_exists):
        """Test getting all KPIs."""
        registry = KPIRegistry()

        all_kpis = registry.get_all()
        assert len(all_kpis) == 2

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_by_workstream(self, mock_exists):
        """Test getting KPIs by workstream."""
        registry = KPIRegistry()

        ws1_kpis = registry.get_by_workstream(Workstream.WS1_DATA_QUALITY)
        assert len(ws1_kpis) == 1
        assert ws1_kpis[0].id == "WS1-DQ-001"

        ws2_kpis = registry.get_by_workstream(Workstream.WS2_TRIGGERS)
        assert len(ws2_kpis) == 1
        assert ws2_kpis[0].id == "WS2-TR-001"

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_iteration(self, mock_exists):
        """Test iterating over registry."""
        registry = KPIRegistry()

        kpi_ids = [kpi.id for kpi in registry]
        assert "WS1-DQ-001" in kpi_ids
        assert "WS2-TR-001" in kpi_ids

    @patch("pathlib.Path.exists", return_value=False)
    def test_missing_config_file(self, mock_exists):
        """Test handling missing config file."""
        registry = KPIRegistry()

        # Should have no KPIs
        assert len(registry) == 0


class TestGetRegistry:
    """Tests for get_registry function."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        KPIRegistry.reset()
        get_registry.cache_clear()
        yield
        KPIRegistry.reset()
        get_registry.cache_clear()

    @patch("builtins.open", mock_open(read_data=SAMPLE_KPI_YAML))
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_registry_caching(self, mock_exists):
        """Test that get_registry returns cached instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
