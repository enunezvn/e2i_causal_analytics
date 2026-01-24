"""
Pytest fixtures for ontology unit tests.

This module provides fixtures for testing the src/ontology module components:
- VocabularyRegistry
- SchemaCompiler
- OntologyValidator
- InferenceEngine
- E2IQueryExtractor
- GraphityConfig

Key Design:
- Cache isolation: VocabularyRegistry singleton is cleared between tests
- Mock data: All fixtures use minimal mock data, not production files
- Temp files: Use tmp_path for temporary YAML files
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from unittest.mock import MagicMock
import yaml


# =============================================================================
# VOCABULARY REGISTRY FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def clear_vocabulary_cache():
    """
    Autouse fixture to clear VocabularyRegistry cache before/after each test.

    This ensures test isolation for the singleton pattern used by VocabularyRegistry.
    """
    from src.ontology.vocabulary_registry import VocabularyRegistry
    VocabularyRegistry.clear_cache()
    yield
    VocabularyRegistry.clear_cache()


@pytest.fixture
def minimal_vocabulary_data() -> Dict[str, Any]:
    """
    Minimal vocabulary data dictionary for testing.

    Contains essential sections: metadata, brands, regions, agents, kpis, aliases.
    """
    return {
        'metadata': {
            'version': '1.0.0-test',
            'description': 'Test vocabulary'
        },
        'brands': {
            'description': 'Test brands',
            'values': ['TestBrand', 'OtherBrand', 'ThirdBrand']
        },
        'regions': {
            'description': 'Test regions',
            'values': ['northeast', 'south', 'midwest', 'west']
        },
        'agents': {
            'tier_0_foundation': ['scope_definer', 'data_preparer'],
            'tier_1_coordination': ['orchestrator'],
            'tier_2_causal': ['causal_impact', 'gap_analyzer']
        },
        'kpis': {
            'trx_volume': {
                'display_name': 'TRx Volume',
                'description': 'Total prescriptions',
                'aliases': ['total rx', 'prescriptions', 'trx']
            },
            'market_share': {
                'display_name': 'Market Share',
                'description': 'Brand market share',
                'aliases': ['share', 'mkt share']
            }
        },
        'aliases': {
            'TestBrand': ['tb', 'testb', 'test-brand'],
            'OtherBrand': ['ob', 'other']
        },
        'journey_stages': {
            'description': 'Patient journey stages',
            'values': ['diagnosis', 'treatment_naive', 'first_line']
        },
        'hcp_segments': {
            'description': 'HCP segments',
            'values': ['high_volume', 'medium_volume', 'low_volume']
        },
        'time_references': {
            'description': 'Time patterns',
            'relative': ['last week', 'this month', 'ytd'],
            'absolute': ['2024', 'Q1 2024']
        },
        'diagnosis_codes': {
            'pattern': r'[A-Z][0-9]{2}\.?[0-9]*'
        },
        # V5.1.0 additions
        'patient_engagement_stages': {
            'description': '7-stage patient engagement funnel',
            'values': ['aware', 'considering', 'prescribed', 'first_fill', 'adherent', 'discontinued', 'maintained']
        },
        'treatment_line_stages': {
            'description': 'Treatment line progression stages',
            'values': ['diagnosis', 'treatment_naive', 'first_line', 'second_line', 'maintenance', 'discontinuation', 'switch']
        },
        'state_to_region_mapping': {
            'description': 'US state to region mapping',
            'mapping': {
                'northeast': ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'],
                'south': ['AL', 'FL', 'GA', 'TX', 'VA'],
                'midwest': ['IL', 'IN', 'MI', 'OH', 'WI'],
                'west': ['AZ', 'CA', 'CO', 'OR', 'WA']
            }
        },
        'competitor_brands': {
            'description': 'Competitor brands by therapeutic area',
            'by_therapeutic_area': {
                'csu_btk_inhibitors': ['Xolair', 'fenebrutinib'],
                'pnh_complement': ['Soliris', 'Ultomiris'],
                'breast_cancer_cdk46': ['Ibrance', 'Verzenio']
            }
        },
        'marketing_channels': {
            'description': 'Marketing channels',
            'channels': {
                'digital': ['email', 'website', 'webinar'],
                'field': ['in_person', 'phone'],
                'crm': ['crm_alert', 'mobile_app'],
                'print': ['direct_mail', 'samples']
            }
        },
        'payer_categories': {
            'description': 'Payer categories',
            'categories': {
                'commercial': {
                    'description': 'Commercial plans',
                    'subcategories': ['national_plans', 'regional_plans']
                },
                'government': {
                    'description': 'Government programs',
                    'subcategories': ['medicare_part_d', 'medicaid']
                }
            }
        },
        'brand_icd10_mappings': {
            'description': 'ICD-10 codes for brand indications',
            'mappings': {
                'TestBrand': {
                    'primary_indication': 'Test Indication',
                    'icd10_codes': ['L50.1', 'L50.8', 'L50.9']
                },
                'OtherBrand': {
                    'primary_indication': 'Other Indication',
                    'icd10_codes': ['D59.5']
                }
            }
        },
        'brand_ndc_codes': {
            'description': 'NDC codes for brand products',
            'mappings': {
                'TestBrand': {
                    'drug_name': 'testdrug',
                    'ndc_codes': ['00078-0903-51', '00078-0903-21']
                }
            }
        }
    }


@pytest.fixture
def mock_vocabulary_file(tmp_path: Path, minimal_vocabulary_data: Dict[str, Any]) -> Path:
    """
    Create a temporary YAML vocabulary file from minimal_vocabulary_data.

    Args:
        tmp_path: pytest's tmp_path fixture
        minimal_vocabulary_data: The vocabulary data fixture

    Returns:
        Path to the temporary vocabulary file
    """
    vocab_file = tmp_path / "test_vocabulary.yaml"
    with open(vocab_file, 'w') as f:
        yaml.dump(minimal_vocabulary_data, f)
    return vocab_file


@pytest.fixture
def empty_vocabulary_file(tmp_path: Path) -> Path:
    """Create an empty vocabulary file for error testing."""
    vocab_file = tmp_path / "empty_vocabulary.yaml"
    with open(vocab_file, 'w') as f:
        yaml.dump({}, f)
    return vocab_file


# =============================================================================
# SCHEMA COMPILER FIXTURES
# =============================================================================

@pytest.fixture
def sample_node_types_yaml() -> Dict[str, Any]:
    """Sample node_types.yaml content for testing SchemaCompiler."""
    return {
        'entities': [
            {
                'label': 'Patient',
                'description': 'Patient entity',
                'primary_key': 'patient_id',
                'properties': [
                    {
                        'name': 'patient_id',
                        'type': 'string',
                        'required': True,
                        'unique': True,
                        'indexed': True
                    },
                    {
                        'name': 'region',
                        'type': 'string',
                        'required': True,
                        'indexed': True
                    },
                    {
                        'name': 'risk_score',
                        'type': 'float',
                        'required': False,
                        'constraints': {'min': 0.0, 'max': 1.0}
                    },
                    {
                        'name': 'created_at',
                        'type': 'datetime',
                        'required': True
                    }
                ],
                'indexes': ['region']
            },
            {
                'label': 'HCP',
                'description': 'Healthcare provider',
                'primary_key': 'hcp_id',
                'properties': [
                    {
                        'name': 'hcp_id',
                        'type': 'string',
                        'required': True,
                        'unique': True,
                        'indexed': True
                    },
                    {
                        'name': 'specialty',
                        'type': 'string',
                        'required': True,
                        'indexed': True
                    },
                    {
                        'name': 'priority_tier',
                        'type': 'integer',
                        'required': False
                    }
                ]
            },
            {
                'label': 'Brand',
                'description': 'Pharmaceutical brand',
                'primary_key': 'brand_id',
                'properties': [
                    {
                        'name': 'brand_id',
                        'type': 'string',
                        'required': True,
                        'unique': True
                    },
                    {
                        'name': 'name',
                        'type': 'string',
                        'required': True
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_edge_types_yaml() -> Dict[str, Any]:
    """Sample edge_types.yaml content for testing SchemaCompiler."""
    return {
        'relationships': [
            {
                'type': 'TREATED_BY',
                'description': 'Patient treated by HCP',
                'from': 'Patient',
                'to': 'HCP',
                'cardinality': '1:N',
                'properties': [
                    {
                        'name': 'is_primary',
                        'type': 'boolean',
                        'required': False
                    },
                    {
                        'name': 'visit_count',
                        'type': 'integer',
                        'required': False
                    }
                ]
            },
            {
                'type': 'PRESCRIBED',
                'description': 'Patient prescribed brand',
                'from': 'Patient',
                'to': 'Brand',
                'cardinality': 'M:N',
                'properties': [
                    {
                        'name': 'prescription_date',
                        'type': 'datetime',
                        'required': True
                    },
                    {
                        'name': 'status',
                        'type': 'string',
                        'required': True
                    }
                ]
            },
            {
                'type': 'PRESCRIBES',
                'description': 'HCP prescribes brand',
                'from': 'HCP',
                'to': 'Brand',
                'cardinality': '1:N',
                'properties': [
                    {
                        'name': 'volume',
                        'type': 'integer',
                        'required': False
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_ontology_dir(
    tmp_path: Path,
    sample_node_types_yaml: Dict[str, Any],
    sample_edge_types_yaml: Dict[str, Any]
) -> Path:
    """
    Create a temporary ontology directory with mock YAML files.

    Args:
        tmp_path: pytest's tmp_path fixture
        sample_node_types_yaml: Node types fixture
        sample_edge_types_yaml: Edge types fixture

    Returns:
        Path to the temporary ontology directory
    """
    ontology_dir = tmp_path / "ontology"
    ontology_dir.mkdir()

    # Write node types
    node_file = ontology_dir / "node_types.yaml"
    with open(node_file, 'w') as f:
        yaml.dump(sample_node_types_yaml, f)

    # Write edge types
    edge_file = ontology_dir / "edge_types.yaml"
    with open(edge_file, 'w') as f:
        yaml.dump(sample_edge_types_yaml, f)

    return ontology_dir


@pytest.fixture
def empty_ontology_dir(tmp_path: Path) -> Path:
    """Create an empty ontology directory for testing."""
    ontology_dir = tmp_path / "empty_ontology"
    ontology_dir.mkdir()
    return ontology_dir


@pytest.fixture
def invalid_reference_yaml(tmp_path: Path) -> Path:
    """Create YAML with invalid entity references for validation testing."""
    ontology_dir = tmp_path / "invalid_ontology"
    ontology_dir.mkdir()

    # Relationships that reference non-existent entities
    invalid_data = {
        'entities': [
            {
                'label': 'Patient',
                'primary_key': 'id',
                'properties': [
                    {'name': 'id', 'type': 'string', 'required': True}
                ]
            }
        ],
        'relationships': [
            {
                'type': 'INVALID_REL',
                'from': 'Patient',
                'to': 'NonExistentEntity',  # Invalid reference
                'cardinality': '1:N',
                'properties': []
            }
        ]
    }

    yaml_file = ontology_dir / "invalid.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(invalid_data, f)

    return ontology_dir


# =============================================================================
# INFERENCE ENGINE FIXTURES (Mock Graph Client)
# =============================================================================

@dataclass
class MockQueryResult:
    """Mock FalkorDB query result."""
    result_set: List[tuple] = field(default_factory=list)

    def __iter__(self):
        return iter(self.result_set)


class MockGraphClient:
    """
    Mock FalkorDB graph client for testing InferenceEngine.

    Simulates graph queries without requiring a real database connection.
    """

    def __init__(self):
        self.queries_executed: List[str] = []
        self._mock_results: Dict[str, MockQueryResult] = {}
        self._default_result = MockQueryResult()

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> MockQueryResult:
        """
        Execute a mock Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            MockQueryResult with predefined or default results
        """
        self.queries_executed.append(cypher)

        # Return predefined result if available
        for pattern, result in self._mock_results.items():
            if pattern in cypher:
                return result

        return self._default_result

    def set_mock_result(self, pattern: str, result: MockQueryResult):
        """Set a mock result for queries containing a pattern."""
        self._mock_results[pattern] = result

    def clear_mock_results(self):
        """Clear all mock results."""
        self._mock_results.clear()
        self.queries_executed.clear()


@pytest.fixture
def mock_graph_client() -> MockGraphClient:
    """Provide a fresh MockGraphClient instance."""
    return MockGraphClient()


@pytest.fixture
def mock_graph_with_paths(mock_graph_client: MockGraphClient) -> MockGraphClient:
    """Mock graph client with causal path results."""
    # Mock path discovery result
    path_result = MockQueryResult(result_set=[
        (
            None,  # path object (not used directly)
            2,     # path_length
            ['source', 'mediator', 'target'],  # node_ids
            ['CAUSES', 'LEADS_TO'],  # rel_types
            [0.8, 0.7]  # weights
        ),
        (
            None,
            3,
            ['source', 'med1', 'med2', 'target'],
            ['CAUSES', 'INFLUENCES', 'CAUSES'],
            [0.9, 0.6, 0.8]
        )
    ])
    mock_graph_client.set_mock_result('MATCH path =', path_result)

    return mock_graph_client


@pytest.fixture
def mock_graph_with_confounders(mock_graph_client: MockGraphClient) -> MockGraphClient:
    """Mock graph client with confounder detection results."""
    confounder_result = MockQueryResult(result_set=[
        ('conf_1', 'Confounder1', 'AFFECTS', 'INFLUENCES', 0.9, 0.8),
        ('conf_2', 'Confounder2', 'CAUSES', 'AFFECTS', 0.7, 0.6)
    ])
    mock_graph_client.set_mock_result('MATCH (c)-[r1]->(t)', confounder_result)

    return mock_graph_client


# =============================================================================
# QUERY EXTRACTOR FIXTURES
# =============================================================================

@pytest.fixture
def query_extractor(mock_vocabulary_file: Path):
    """Create E2IQueryExtractor with mock vocabulary."""
    from src.ontology.query_extractor import E2IQueryExtractor
    return E2IQueryExtractor(str(mock_vocabulary_file))


@pytest.fixture
def sample_queries() -> Dict[str, Dict[str, Any]]:
    """Sample queries with expected extraction results."""
    return {
        'brand_query': {
            'text': "What is the market share for TestBrand in the northeast?",
            'expected_brand': 'TestBrand',
            'expected_region': 'northeast',
            'expected_agent': None  # Default routing
        },
        'alias_query': {
            'text': "Show me tb sales trends",  # 'tb' is alias for TestBrand
            'expected_brand': 'TestBrand',
            'expected_region': None,
            'expected_agent': None
        },
        'causal_query': {
            'text': "What caused the increase in prescriptions?",
            'expected_brand': None,
            'expected_region': None,
            'expected_agent': 'causal_impact'
        },
        'prediction_query': {
            'text': "Predict the market share for next quarter",
            'expected_brand': None,
            'expected_region': None,
            'expected_agent': 'prediction_synthesizer'
        },
        'explanation_query': {
            'text': "Why did the conversion rate drop?",
            'expected_brand': None,
            'expected_region': None,
            'expected_agent': 'explainer'
        },
        'kpi_query': {
            'text': "Show me TRx Volume by region",
            'expected_brand': None,
            'expected_region': None,
            'expected_kpi': 'TRx Volume'
        }
    }


# =============================================================================
# GRAPHITY CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def default_graphity_config():
    """Create default GraphityConfig instance."""
    from src.ontology.grafiti_config import GraphityConfig
    return GraphityConfig()


@pytest.fixture
def custom_graphity_config():
    """Create custom GraphityConfig using builder."""
    from src.ontology.grafiti_config import GraphityConfigBuilder
    return (GraphityConfigBuilder()
        .enabled(True)
        .with_edge_grouping(strategy='by_type', chunk_size=500)
        .with_caching(ttl_seconds=1800, max_size_mb=128)
        .with_hub_detection(threshold=50)
        .with_e2i_patterns()
        .build())


@pytest.fixture
def graphity_config_dict() -> Dict[str, Any]:
    """Serialized GraphityConfig for roundtrip testing."""
    return {
        'enabled': True,
        'edge_grouping': {
            'strategy': 'by_type',
            'chunk_size': 1000,
            'min_edges_for_grouping': 100,
            'rebalance_threshold': 0.3
        },
        'caching': {
            'enabled': True,
            'eviction_policy': 'lru',
            'max_cache_size_mb': 256,
            'ttl_seconds': 3600,
            'hot_path_caching': True,
            'prefetch_depth': 2
        },
        'hub_detection': {
            'enabled': True,
            'min_degree_threshold': 100,
            'detection_method': 'degree_centrality',
            'update_interval_seconds': 3600
        },
        'traversal_patterns': [],
        'index_hints': {},
        'metadata': {'test': True}
    }


# =============================================================================
# VALIDATOR FIXTURES
# =============================================================================

@pytest.fixture
def compiled_schema_for_validation(mock_ontology_dir: Path):
    """Create a CompiledSchema for validation testing."""
    from src.ontology.schema_compiler import SchemaCompiler
    compiler = SchemaCompiler(mock_ontology_dir)
    return compiler.compile()


@pytest.fixture
def schema_with_missing_pk():
    """Create schema where primary key is not in properties."""
    from src.ontology.schema_compiler import (
        CompiledSchema, EntitySchema, PropertySchema, PropertyType
    )

    entity = EntitySchema(
        label='BadEntity',
        properties=[
            PropertySchema(name='other_field', property_type=PropertyType.STRING)
        ],
        primary_key='missing_pk'  # Not in properties
    )

    return CompiledSchema(
        entities={'BadEntity': entity},
        relationships={},
        constraints=[],
        indexes=[],
        graphity_config={'enabled': False},
        version='1.0',
        metadata={}
    )


@pytest.fixture
def schema_with_invalid_naming():
    """Create schema with non-standard naming conventions."""
    from src.ontology.schema_compiler import (
        CompiledSchema, EntitySchema, RelationshipSchema,
        PropertySchema, PropertyType, CardinalityType
    )

    # Entity with non-PascalCase label
    bad_entity = EntitySchema(
        label='bad_entity_name',  # Should be PascalCase
        properties=[
            PropertySchema(name='id', property_type=PropertyType.STRING)
        ],
        primary_key='id'
    )

    # Relationship with non-UPPER_SNAKE_CASE type
    bad_rel = RelationshipSchema(
        type='badRelName',  # Should be UPPER_SNAKE_CASE
        from_label='bad_entity_name',
        to_label='bad_entity_name',
        properties=[],
        cardinality=CardinalityType.ONE_TO_MANY
    )

    return CompiledSchema(
        entities={'bad_entity_name': bad_entity},
        relationships={'badRelName': bad_rel},
        constraints=[],
        indexes=[],
        graphity_config={'enabled': False},
        version='1.0',
        metadata={}
    )
