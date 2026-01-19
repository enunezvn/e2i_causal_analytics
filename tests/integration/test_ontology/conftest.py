"""
Pytest fixtures for ontology integration tests.

This module provides fixtures for testing the ontology layer with real
production YAML files and database connections.

Author: E2I Causal Analytics Team
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Get the config directory path."""
    return project_root / "config"


@pytest.fixture
def ontology_config_dir(config_dir: Path) -> Path:
    """Get the ontology config directory path."""
    return config_dir / "ontology"


@pytest.fixture
def domain_vocabulary_path(config_dir: Path) -> Path:
    """Get the domain vocabulary YAML file path."""
    return config_dir / "domain_vocabulary.yaml"


# =============================================================================
# VOCABULARY REGISTRY FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_vocabulary_cache():
    """Clear the VocabularyRegistry singleton cache before and after each test.

    This ensures test isolation for the LRU-cached singleton.
    """
    from src.ontology.vocabulary_registry import VocabularyRegistry

    VocabularyRegistry.clear_cache()
    yield
    VocabularyRegistry.clear_cache()


@pytest.fixture
def vocabulary_registry():
    """Get a fresh VocabularyRegistry instance with production data."""
    from src.ontology.vocabulary_registry import VocabularyRegistry

    return VocabularyRegistry.load()


# =============================================================================
# SCHEMA COMPILER FIXTURES
# =============================================================================


@pytest.fixture
def schema_compiler(ontology_config_dir: Path):
    """Get a SchemaCompiler instance with production config directory."""
    from src.ontology.schema_compiler import SchemaCompiler

    return SchemaCompiler(ontology_dir=ontology_config_dir)


@pytest.fixture
def compiled_schema(schema_compiler):
    """Get the compiled schema from production YAML files."""
    return schema_compiler.compile()


# =============================================================================
# VALIDATOR FIXTURES
# =============================================================================


@pytest.fixture
def ontology_validator(compiled_schema):
    """Get an OntologyValidator instance with compiled schema.

    The validator requires a compiled schema to validate.
    """
    from src.ontology.validator import OntologyValidator

    return OntologyValidator(compiled_schema)


# =============================================================================
# INFERENCE ENGINE FIXTURES
# =============================================================================


@pytest.fixture
def mock_graph_client():
    """Create a mock FalkorDB graph client for inference engine tests."""
    mock_client = MagicMock()

    # Mock query method to return empty results by default
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_client.query.return_value = mock_result

    return mock_client


@pytest.fixture
def inference_engine(mock_graph_client):
    """Get an InferenceEngine instance with mock graph client."""
    from src.ontology.inference_engine import InferenceEngine

    return InferenceEngine(mock_graph_client)


# =============================================================================
# QUERY EXTRACTOR FIXTURES
# =============================================================================


@pytest.fixture
def query_extractor(domain_vocabulary_path):
    """Get an E2IQueryExtractor instance with production vocabulary path."""
    from src.ontology.query_extractor import E2IQueryExtractor

    return E2IQueryExtractor(vocab_path=str(domain_vocabulary_path))


# =============================================================================
# GRAPHITY CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def e2i_graphity_config():
    """Get the E2I-optimized Graphity configuration."""
    from src.ontology.grafiti_config import create_e2i_graphity_config

    return create_e2i_graphity_config()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client for integration tests.

    This allows testing DB ENUM sync without a real database connection.
    """
    mock_client = MagicMock()

    # Mock the execute method to return mock ENUM values
    mock_response = MagicMock()
    mock_response.data = []
    mock_client.rpc.return_value.execute.return_value = mock_response

    return mock_client


@pytest.fixture
def db_enum_values():
    """Provide expected database ENUM values for sync verification.

    These values should match the ENUMs defined in database migrations.
    Values based on actual vocabulary from domain_vocabulary.yaml.
    """
    return {
        "agent_tier": ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5"],
        "agent_name": [
            # Tier 0: ML Foundation
            "scope_definer", "data_preparer", "model_selector",
            "model_trainer", "model_evaluator", "model_deployer", "model_monitor",
            # Tier 1: Coordination
            "orchestrator", "tool_composer",
            # Tier 2: Causal
            "causal_impact", "heterogeneous_optimizer", "gap_analyzer",
            # Tier 3: Monitoring
            "experiment_designer", "drift_monitor", "data_quality_monitor",
            # Tier 4: Prediction
            "prediction_synthesizer", "risk_assessor",
            # Tier 5: Self-Improvement
            "explainer", "feedback_learner", "health_score", "resource_optimizer"
        ],
        "brand_name": ["remibrutinib", "fabhalta", "kisqali"],
        "region_code": ["northeast", "south", "midwest", "west"],
    }


# =============================================================================
# AGENT ROUTING FIXTURES
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing query routing flow."""
    mock = MagicMock()
    mock.route_query = MagicMock(return_value={
        "agent": "causal_impact",
        "confidence": 0.85,
        "reasoning": "Query involves causal analysis"
    })
    return mock


@pytest.fixture
def sample_routing_queries():
    """Provide sample queries for routing integration tests."""
    return {
        "causal_impact": [
            "What is the causal impact of digital marketing on Remibrutinib TRx?",
            "How does rep visit frequency affect prescription rates for Kisqali?",
            "Analyze the causal effect of conference attendance on brand awareness",
        ],
        "gap_analyzer": [
            "Identify performance gaps in the Northeast region for Fabhalta",
            "What are the opportunity gaps in HCP engagement?",
            "Find underperforming territories for Remibrutinib",
        ],
        "prediction_synthesizer": [
            "Predict Q4 TRx volume for Kisqali",
            "Forecast market share trends for the next quarter",
            "What are the expected prescription numbers for Fabhalta?",
        ],
        "experiment_designer": [
            "Design an A/B test for new messaging campaign",
            "Create an experiment to test email frequency impact",
            "Set up a test comparing two promotional strategies",
        ],
        "cohort_constructor": [
            "Build a cohort of high-prescribing oncologists",
            "Create a patient segment based on treatment history",
            "Construct a cohort of HCPs in urban areas",
        ],
        "explainer": [
            "Explain why TRx dropped in the Southwest region",
            "What factors contributed to the Q3 performance increase?",
            "Help me understand the market share fluctuation",
        ],
    }


# =============================================================================
# YAML FILE VALIDATION FIXTURES
# =============================================================================


@pytest.fixture
def required_ontology_files():
    """List of required YAML files in the ontology config directory."""
    return [
        "node_types.yaml",
        "edge_types.yaml",
        "inference_rules.yaml",
        "validation_rules.yaml",
    ]


@pytest.fixture
def required_vocabulary_sections():
    """List of required sections in domain_vocabulary.yaml."""
    return [
        "brands",
        "regions",
        "agents",
        "kpis",
        "therapeutic_areas",
    ]


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for integration tests."""
    return {
        "schema_compile_ms": 500,  # Schema compilation should be < 500ms
        "validation_ms": 200,  # Validation should be < 200ms
        "query_extraction_ms": 50,  # Query extraction should be < 50ms
        "routing_decision_ms": 100,  # Routing decision should be < 100ms
    }
