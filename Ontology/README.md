# E2I Ontology Implementation System

Comprehensive ontology management suite for the E2I Causal Analytics semantic memory layer. This system compiles YAML ontology definitions into FalkorDB-compatible schemas, validates consistency, configures Graphity optimizations, and provides graph-based reasoning capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Workflow Integration](#workflow-integration)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## 🎯 Overview

The E2I Ontology Implementation System provides four core components:

1. **Schema Compiler** - Transforms YAML ontology definitions into FalkorDB schema
2. **Validator** - Ensures schema consistency and adherence to best practices
3. **Graphity Config** - Optimizes graph traversal performance through edge grouping and caching
4. **Inference Engine** - Discovers causal paths and infers relationships through graph reasoning

### Why This System?

- **Maintainability**: Define ontology in human-readable YAML, not scattered Cypher scripts
- **Validation**: Catch schema errors before deployment
- **Performance**: Graphity optimizations improve query performance by 10-100x
- **Intelligence**: Automatic causal path discovery and relationship inference

---

## 🧩 Components

### 1. Schema Compiler (`schema_compiler.py`)

Compiles YAML ontology definitions into:
- FalkorDB-compatible schema
- Cypher DDL statements
- JSON Schema exports
- Graphity optimization configurations

**Key Features:**
- Entity and relationship schema compilation
- Primary key and unique constraint generation
- Index generation with type optimization
- Automatic Graphity configuration for hub nodes
- Cross-reference validation

### 2. Validator (`validator.py`)

Validates compiled schemas for:
- Consistency (no dangling references, duplicate properties)
- Naming conventions (PascalCase entities, UPPER_SNAKE_CASE relationships)
- Constraint validity
- Cardinality consistency
- E2I-specific patterns (temporal properties, causal entities)

**Output:** Detailed validation reports in text or Markdown format

### 3. Graphity Config (`grafiti_config.py`)

Configures FalkorDB Graphity optimizations:
- Edge grouping strategies (by type, property, label, or hybrid)
- Cache configuration (LRU, LFU, TTL, adaptive)
- Hub node identification and partitioning
- Traversal pattern optimization
- Performance monitoring setup

**Key Benefit:** 10-100x performance improvement on complex graph traversals

### 4. Inference Engine (`inference_engine.py`)

Graph-based reasoning engine providing:
- Causal path discovery between entities
- Transitive relationship inference
- Confounder detection
- Mediator identification
- Path ranking and importance scoring
- Relationship materialization

**Use Cases:** Automatic DAG completion, causal hypothesis generation, knowledge graph enrichment

---

## 📦 Prerequisites

### Required Dependencies

```bash
# Core dependencies
pip install pyyaml>=6.0
pip install falkordb>=1.0.0
pip install redis>=4.5.0

# Optional for enhanced functionality
pip install networkx>=3.0      # For advanced graph algorithms
pip install pandas>=2.0         # For data manipulation
pip install matplotlib>=3.7     # For visualization
```

### System Requirements

- Python 3.9+
- FalkorDB instance (local or remote)
- Redis 7.0+ with Graphity module enabled

### FalkorDB Setup

```bash
# Using Docker
docker run -p 6379:6379 falkordb/falkordb:latest

# Or install locally
# Follow: https://docs.falkordb.com/getting-started.html
```

---

## 🚀 Installation

### 1. Clone or Copy Files

```bash
# Copy the four Python files to your project
cp schema_compiler.py /path/to/your/project/config/ontology/
cp validator.py /path/to/your/project/config/ontology/
cp grafiti_config.py /path/to/your/project/config/
cp inference_engine.py /path/to/your/project/config/ontology/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Create `requirements.txt`:
```
pyyaml>=6.0
falkordb>=1.0.0
redis>=4.5.0
```

### 3. Set Up Ontology Directory

```bash
mkdir -p config/ontology/schemas
```

---

## ⚡ Quick Start

### Step 1: Define Your Ontology (YAML)

Create `config/ontology/schemas/entities.yaml`:

```yaml
entities:
  - label: HCP
    description: Healthcare provider
    primary_key: hcp_id
    properties:
      - name: hcp_id
        type: string
        required: true
        indexed: true
        unique: true
      - name: name
        type: string
        required: true
      - name: specialty
        type: string
        indexed: true
      - name: created_at
        type: datetime
        required: true
    indexes:
      - hcp_id
      - specialty

  - label: Patient
    description: Patient entity
    primary_key: patient_id
    properties:
      - name: patient_id
        type: string
        required: true
        indexed: true
        unique: true
      - name: age
        type: integer
      - name: diagnosis_date
        type: datetime
    indexes:
      - patient_id

relationships:
  - type: TREATED
    from: HCP
    to: Patient
    cardinality: "1:N"
    properties:
      - name: start_date
        type: datetime
        required: true
      - name: confidence
        type: float
        default: 1.0
```

### Step 2: Compile Schema

```python
from schema_compiler import SchemaCompiler

# Compile YAML to schema
compiler = SchemaCompiler("config/ontology/schemas")
compiled = compiler.compile()

# Export as Cypher DDL
cypher_ddl = compiler.export_cypher_ddl(compiled)
print(cypher_ddl)

# Export as JSON Schema
json_schema = compiler.export_json_schema(compiled)
```

Or via CLI:

```bash
# Generate Cypher DDL
python schema_compiler.py config/ontology/schemas --output-format cypher > schema.cypher

# Generate JSON Schema
python schema_compiler.py config/ontology/schemas --output-format json > schema.json
```

### Step 3: Validate Schema

```python
from validator import OntologyValidator

# Validate compiled schema
validator = OntologyValidator(compiled)
report = validator.validate()

# Generate report
print(validator.generate_report(report, format='text'))

# Check if passed
if report.passed:
    print("✅ Validation passed!")
else:
    print(f"❌ Validation failed: {report.error_count} errors")
```

Or via CLI:

```bash
# Text report
python validator.py config/ontology/schemas

# Markdown report
python validator.py config/ontology/schemas --format markdown > validation_report.md
```

### Step 4: Configure Graphity

```python
from grafiti_config import GraphityConfigBuilder

# Build configuration
builder = GraphityConfigBuilder()
config = (builder
    .with_edge_grouping(EdgeGroupingStrategy.BY_TYPE, chunk_size=2000)
    .with_cache(CacheStrategy.ADAPTIVE, ttl_seconds=7200)
    .add_hub_entity('HCP', degree_threshold=100, partition_count=8)
    .build())

# Export to YAML
builder.to_yaml('config/graphity_config.yaml')
```

Or via CLI:

```bash
# Generate default E2I config
python grafiti_config.py config/graphity_config.yaml
```

### Step 5: Use Inference Engine

```python
from inference_engine import InferenceEngine
from falkordb import FalkorDB

# Connect to FalkorDB
client = FalkorDB(host='localhost', port=6379)
graph = client.select_graph('e2i_causal')

# Initialize engine
engine = InferenceEngine(graph)

# Discover causal paths
paths = engine.discover_causal_paths(
    source_id='hcp_123',
    target_id='outcome_456',
    max_depth=5
)

print(f"Found {len(paths)} causal paths")

# Rank paths by importance
ranked = engine.rank_causal_paths(paths, method='combined')
best_path, score = ranked[0]
print(f"Best path score: {score:.3f}")

# Infer new relationships
inferred = engine.infer_relationships(min_confidence=0.7)
print(f"Inferred {len(inferred)} relationships")

# Materialize high-confidence inferences
materialized = engine.materialize_inferred_relationships(
    inferred,
    min_confidence=0.8
)
```

---

## 📖 Detailed Usage

### Schema Compiler

#### Defining Entities

```yaml
entities:
  - label: CausalEstimate
    description: Estimated causal effect
    primary_key: estimate_id
    properties:
      - name: estimate_id
        type: string
        required: true
        unique: true
      - name: effect_size
        type: float
        required: true
      - name: confidence_interval_lower
        type: float
      - name: confidence_interval_upper
        type: float
      - name: p_value
        type: float
      - name: created_at
        type: datetime
        required: true
        indexed: true
```

#### Defining Relationships

```yaml
relationships:
  - type: HAS_IMPACT_ON
    from: Intervention
    to: Outcome
    cardinality: "1:N"
    description: Causal relationship between intervention and outcome
    properties:
      - name: effect_size
        type: float
        required: true
      - name: confidence
        type: float
        constraints:
          min: 0.0
          max: 1.0
      - name: timestamp
        type: datetime
```

#### Property Types

Supported property types:
- `string` - Text data
- `integer` - Whole numbers
- `float` - Decimal numbers
- `boolean` - True/false
- `datetime` - ISO 8601 timestamps
- `json` - Structured JSON data
- `array` - Lists of values

#### Programmatic Usage

```python
from schema_compiler import SchemaCompiler
from pathlib import Path

# Initialize compiler
compiler = SchemaCompiler(Path("config/ontology/schemas"))

# Compile schema
compiled = compiler.compile()

# Access compiled components
print(f"Entities: {len(compiled.entities)}")
print(f"Relationships: {len(compiled.relationships)}")
print(f"Constraints: {len(compiled.constraints)}")
print(f"Indexes: {len(compiled.indexes)}")

# Export formats
cypher_ddl = compiler.export_cypher_ddl(compiled)
json_schema = compiler.export_json_schema(compiled)

# Access Graphity config
graphity_config = compiled.graphity_config
print(f"Hub entities: {graphity_config['hub_entities']}")
```

---

### Validator

#### Validation Levels

- **ERROR**: Must be fixed before deployment
- **WARNING**: Should be reviewed, may impact performance or consistency
- **INFO**: Informational, follows best practices

#### Validation Categories

- `schema` - Schema structure issues
- `references` - Entity reference problems
- `naming` - Naming convention violations
- `constraints` - Invalid constraints
- `cardinality` - Cardinality conflicts
- `performance` - Performance concerns
- `completeness` - Missing recommended patterns

#### Programmatic Usage

```python
from validator import OntologyValidator, ValidationLevel

# Validate
validator = OntologyValidator(compiled)
report = validator.validate()

# Filter by level
errors = [i for i in report.issues if i.level == ValidationLevel.ERROR]
warnings = [i for i in report.issues if i.level == ValidationLevel.WARNING]

print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")

# Get statistics
stats = report.statistics
print(f"Entity count: {stats['entity_count']}")
print(f"Relationship count: {stats['relationship_count']}")

# Generate reports
text_report = validator.generate_report(report, format='text')
md_report = validator.generate_report(report, format='markdown')

# Write to file
with open('validation_report.md', 'w') as f:
    f.write(md_report)
```

---

### Graphity Config

#### Edge Grouping Strategies

```python
from grafiti_config import EdgeGroupingStrategy

# BY_TYPE: Group edges by relationship type (default for E2I)
builder.with_edge_grouping(EdgeGroupingStrategy.BY_TYPE, chunk_size=1000)

# BY_PROPERTY: Group by property value
builder.with_edge_grouping(EdgeGroupingStrategy.BY_PROPERTY, chunk_size=1000)

# BY_LABEL: Group by target node label
builder.with_edge_grouping(EdgeGroupingStrategy.BY_LABEL, chunk_size=1000)

# HYBRID: Combine strategies
builder.with_edge_grouping(EdgeGroupingStrategy.HYBRID, chunk_size=1000)
```

#### Cache Strategies

```python
from grafiti_config import CacheStrategy

# LRU: Least Recently Used
builder.with_cache(CacheStrategy.LRU, ttl_seconds=3600, max_size_mb=512)

# LFU: Least Frequently Used
builder.with_cache(CacheStrategy.LFU, ttl_seconds=3600, max_size_mb=512)

# TTL: Time To Live
builder.with_cache(CacheStrategy.TTL, ttl_seconds=1800, max_size_mb=256)

# ADAPTIVE: Adapt based on usage patterns (recommended)
builder.with_cache(CacheStrategy.ADAPTIVE, ttl_seconds=3600, max_size_mb=512)
```

#### Hub Node Configuration

```python
# Configure high-degree nodes for optimization
builder.add_hub_entity(
    entity_label='HCP',
    degree_threshold=100,      # Min edges to be considered hub
    partition_count=8          # Number of partitions
)

builder.add_hub_entity(
    entity_label='Patient',
    degree_threshold=50,
    partition_count=4
)
```

#### Traversal Patterns

```python
# Add custom traversal patterns to optimize
builder.add_traversal_pattern(
    name='hcp_to_outcome',
    cypher_pattern='(h:HCP)-[:TREATED]->(p:Patient)-[:HAS_OUTCOME]->(o:Outcome)',
    frequency='high',
    edge_types=['TREATED', 'HAS_OUTCOME']
)
```

#### Loading from YAML

```python
from grafiti_config import GraphityConfigBuilder
from pathlib import Path

# Load existing config
builder = GraphityConfigBuilder.from_yaml(Path('config/graphity_config.yaml'))

# Modify and save
builder.add_hub_entity('Experiment', degree_threshold=30, partition_count=4)
builder.to_yaml('config/graphity_config_updated.yaml')
```

---

### Inference Engine

#### Discovering Causal Paths

```python
from inference_engine import InferenceEngine

# Initialize engine
engine = InferenceEngine(graph)

# Find all causal paths
paths = engine.discover_causal_paths(
    source_id='intervention_123',
    target_id='outcome_456',
    max_depth=5,
    relationship_types=['CAUSES', 'LED_TO', 'INFLUENCED']
)

# Examine paths
for path in paths:
    print(f"Path length: {path.path_length}")
    print(f"Path strength: {path.path_strength:.3f}")
    print(f"Mediated by: {path.mediated_by}")
    print(f"Path: {' -> '.join([step[1] for step in path.path])}")
```

#### Ranking Paths

```python
# Rank by combined score (default)
ranked = engine.rank_causal_paths(paths, method='combined')

# Top 5 paths
for i, (path, score) in enumerate(ranked[:5], 1):
    print(f"{i}. Score: {score:.3f}, Length: {path.path_length}")

# Other ranking methods
ranked_by_weight = engine.rank_causal_paths(paths, method='edge_weight')
ranked_by_length = engine.rank_causal_paths(paths, method='length')
ranked_by_mediators = engine.rank_causal_paths(paths, method='mediator_count')
```

#### Inferring Relationships

```python
# Infer new relationships using rules
inferred = engine.infer_relationships(
    max_depth=3,
    min_confidence=0.5
)

# Examine inferred relationships
for rel in inferred:
    print(f"{rel.from_id} -[{rel.relationship_type}]-> {rel.to_id}")
    print(f"  Confidence: {rel.confidence:.3f} ({rel.confidence_level.value})")
    print(f"  Reasoning: {rel.reasoning_path}")
    print(f"  Evidence: {rel.supporting_evidence}")
```

#### Custom Inference Rules

```python
from inference_engine import InferenceRule, InferenceType

# Define custom rule
def custom_confidence_calculator(evidence):
    conf1 = evidence.get('confidence_1', 0.5)
    conf2 = evidence.get('confidence_2', 0.5)
    temporal_valid = evidence.get('temporal_order', False)
    
    base_confidence = min(conf1, conf2)
    if temporal_valid:
        base_confidence *= 1.2  # Boost for temporal validity
    
    return min(1.0, base_confidence)

rule = InferenceRule(
    name='intervention_outcome_chain',
    inference_type=InferenceType.CAUSAL_CHAIN,
    source_pattern='(i:Intervention)-[:APPLIED_TO]->(p:Patient)-[:HAS_OUTCOME]->(o:Outcome)',
    inferred_pattern='(i)-[:LED_TO]->(o)',
    confidence_calculator=custom_confidence_calculator,
    max_depth=2
)

# Add to engine
engine.add_rule(rule)
```

#### Detecting Confounders

```python
# Find potential confounders
confounders = engine.find_confounders(
    treatment_id='intervention_123',
    outcome_id='outcome_456',
    max_depth=3
)

# Examine confounders
for confounder in confounders:
    print(f"Confounder: {confounder['id']} ({confounder['label']})")
    print(f"  Strength: {confounder['confounder_strength']:.3f}")
    print(f"  → Treatment: {confounder['relationship_to_treatment']}")
    print(f"  → Outcome: {confounder['relationship_to_outcome']}")
```

#### Finding Mediators

```python
# Identify mediators
mediators = engine.find_mediators(
    source_id='hcp_123',
    target_id='outcome_456'
)

# Examine mediators
for mediator in mediators:
    print(f"Mediator: {mediator['id']} ({mediator['label']})")
    print(f"  Path count: {mediator['path_count']}")
    print(f"  Mediation strength: {mediator['mediation_strength']:.3f}")
```

#### Materializing Inferences

```python
# Materialize high-confidence inferred relationships into the graph
materialized_count = engine.materialize_inferred_relationships(
    inferred_relationships=inferred,
    min_confidence=0.8
)

print(f"Materialized {materialized_count} relationships into graph")
```

---

## 🔄 Workflow Integration

### Complete Ontology Deployment Pipeline

```python
from pathlib import Path
from schema_compiler import SchemaCompiler
from validator import OntologyValidator
from grafiti_config import GraphityConfigBuilder
from falkordb import FalkorDB

def deploy_ontology(ontology_dir: Path, graph_name: str):
    """Complete ontology deployment pipeline"""
    
    # Step 1: Compile
    print("📦 Compiling ontology...")
    compiler = SchemaCompiler(ontology_dir)
    compiled = compiler.compile()
    
    # Step 2: Validate
    print("✓ Validating schema...")
    validator = OntologyValidator(compiled)
    report = validator.validate()
    
    if not report.passed:
        print(f"❌ Validation failed: {report.error_count} errors")
        print(validator.generate_report(report, format='text'))
        return False
    
    print(f"✅ Validation passed")
    
    # Step 3: Generate Graphity config
    print("⚙️  Generating Graphity config...")
    graphity_builder = GraphityConfigBuilder()
    graphity_builder.to_yaml('config/graphity_config.yaml')
    
    # Step 4: Export DDL
    print("📄 Exporting schema...")
    cypher_ddl = compiler.export_cypher_ddl(compiled)
    with open('schema.cypher', 'w') as f:
        f.write(cypher_ddl)
    
    # Step 5: Deploy to FalkorDB
    print("🚀 Deploying to FalkorDB...")
    client = FalkorDB(host='localhost', port=6379)
    graph = client.select_graph(graph_name)
    
    # Execute DDL statements
    for statement in cypher_ddl.split('\n'):
        if statement.strip() and not statement.startswith('//'):
            try:
                graph.query(statement)
            except Exception as e:
                print(f"Warning: {e}")
    
    print("✅ Deployment complete!")
    return True

# Run deployment
success = deploy_ontology(
    ontology_dir=Path('config/ontology/schemas'),
    graph_name='e2i_causal'
)
```

### Continuous Validation in CI/CD

```python
# ci/validate_ontology.py
import sys
from pathlib import Path
from schema_compiler import SchemaCompiler
from validator import OntologyValidator

def validate_for_ci(ontology_dir: Path) -> int:
    """Validate ontology for CI/CD pipeline"""
    
    # Compile
    compiler = SchemaCompiler(ontology_dir)
    compiled = compiler.compile()
    
    # Validate
    validator = OntologyValidator(compiled)
    report = validator.validate()
    
    # Generate markdown report
    md_report = validator.generate_report(report, format='markdown')
    with open('validation_report.md', 'w') as f:
        f.write(md_report)
    
    # Print summary
    print(f"Entities: {len(compiled.entities)}")
    print(f"Relationships: {len(compiled.relationships)}")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    
    # Return exit code
    return 0 if report.passed else 1

if __name__ == '__main__':
    exit_code = validate_for_ci(Path('config/ontology/schemas'))
    sys.exit(exit_code)
```

Add to `.github/workflows/validate.yml`:

```yaml
name: Validate Ontology
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python ci/validate_ontology.py
      - uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: validation_report.md
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# FalkorDB connection
export FALKORDB_HOST=localhost
export FALKORDB_PORT=6379
export FALKORDB_PASSWORD=your_password

# Ontology paths
export ONTOLOGY_DIR=config/ontology/schemas
export GRAPHITY_CONFIG=config/graphity_config.yaml

# Validation settings
export VALIDATION_MIN_CONFIDENCE=0.7
export VALIDATION_FORMAT=markdown
```

### Configuration Files

#### `config/ontology_config.yaml`

```yaml
ontology:
  version: "1.0"
  namespace: "e2i_causal"
  schema_dir: "config/ontology/schemas"

compilation:
  strict_mode: true
  generate_graphity: true
  export_formats:
    - cypher
    - json

validation:
  error_on_warnings: false
  naming_conventions:
    entities: "PascalCase"
    relationships: "UPPER_SNAKE_CASE"
    properties: "snake_case"

deployment:
  auto_deploy: false
  backup_before_deploy: true
  falkordb:
    host: "localhost"
    port: 6379
    graph_name: "e2i_causal"
```

---

## 📚 Examples

### Example 1: E2I Causal Ontology

See `config/ontology/schemas/e2i_causal.yaml`:

```yaml
entities:
  - label: HCP
    primary_key: hcp_id
    properties:
      - name: hcp_id
        type: string
        required: true
        unique: true
      - name: specialty
        type: string
        indexed: true

  - label: Patient
    primary_key: patient_id
    properties:
      - name: patient_id
        type: string
        required: true
        unique: true
      - name: diagnosis_date
        type: datetime

  - label: Intervention
    primary_key: intervention_id
    properties:
      - name: intervention_id
        type: string
        required: true
        unique: true
      - name: intervention_type
        type: string
        indexed: true

  - label: Outcome
    primary_key: outcome_id
    properties:
      - name: outcome_id
        type: string
        required: true
        unique: true
      - name: outcome_value
        type: float

relationships:
  - type: TREATED
    from: HCP
    to: Patient
    cardinality: "1:N"
    properties:
      - name: start_date
        type: datetime
        required: true

  - type: RECEIVED
    from: Patient
    to: Intervention
    cardinality: "M:N"

  - type: LED_TO
    from: Intervention
    to: Outcome
    cardinality: "1:N"
    properties:
      - name: confidence
        type: float
      - name: effect_size
        type: float
```

### Example 2: Performance Monitoring

```python
from grafiti_config import GraphityOptimizer

# Collect graph statistics
graph_stats = {
    'total_nodes': 50000,
    'total_edges': 200000,
    'node_stats': {
        'HCP': {'avg_degree': 80, 'max_degree': 500},
        'Patient': {'avg_degree': 15, 'max_degree': 100},
        'Intervention': {'avg_degree': 25, 'max_degree': 200}
    }
}

# Get optimization recommendations
optimizer = GraphityOptimizer(graph_stats)

print(f"Recommended chunk size: {optimizer.recommend_chunk_size()}")
print(f"Recommended cache size: {optimizer.recommend_cache_size()} MB")

# Identify hub candidates
hubs = optimizer.identify_hub_candidates(degree_threshold=100)
for hub in hubs:
    print(f"Hub: {hub['entity_label']}")
    print(f"  Max degree: {hub['max_degree']}")
    print(f"  Recommended partitions: {hub['recommended_partitions']}")
```

### Example 3: Automated Knowledge Graph Enrichment

```python
from inference_engine import InferenceEngine
from falkordb import FalkorDB

def enrich_knowledge_graph(graph_name: str, min_confidence: float = 0.7):
    """Automatically enrich knowledge graph with inferred relationships"""
    
    # Connect
    client = FalkorDB(host='localhost', port=6379)
    graph = client.select_graph(graph_name)
    engine = InferenceEngine(graph)
    
    # Infer relationships
    print("🔍 Inferring relationships...")
    inferred = engine.infer_relationships(
        max_depth=3,
        min_confidence=min_confidence
    )
    
    print(f"Found {len(inferred)} inferred relationships")
    
    # Filter by confidence level
    high_confidence = [r for r in inferred if r.confidence >= 0.8]
    medium_confidence = [r for r in inferred if 0.6 <= r.confidence < 0.8]
    
    print(f"  High confidence: {len(high_confidence)}")
    print(f"  Medium confidence: {len(medium_confidence)}")
    
    # Materialize high-confidence inferences
    print("💾 Materializing high-confidence inferences...")
    materialized = engine.materialize_inferred_relationships(
        high_confidence,
        min_confidence=0.8
    )
    
    print(f"✅ Materialized {materialized} relationships")
    
    return {
        'inferred_total': len(inferred),
        'high_confidence': len(high_confidence),
        'medium_confidence': len(medium_confidence),
        'materialized': materialized
    }

# Run enrichment
results = enrich_knowledge_graph('e2i_causal', min_confidence=0.7)
print(f"Enrichment complete: {results}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Entity reference not found"

**Problem:** Relationship references an entity that doesn't exist

**Solution:**
```python
# Check all entity labels
compiler = SchemaCompiler("config/ontology/schemas")
compiled = compiler.compile()
print("Available entities:", list(compiled.entities.keys()))

# Ensure relationship references match exactly
```

#### Issue: "Validation failed with naming convention errors"

**Problem:** Entity/relationship names don't follow conventions

**Solution:**
```yaml
# ❌ Bad
entities:
  - label: hcp_entity  # Should be PascalCase

relationships:
  - type: treated_by  # Should be UPPER_SNAKE_CASE

# ✅ Good
entities:
  - label: HCP

relationships:
  - type: TREATED_BY
```

#### Issue: "FalkorDB connection refused"

**Problem:** FalkorDB not running or wrong connection parameters

**Solution:**
```bash
# Check if FalkorDB is running
redis-cli -p 6379 ping

# Start FalkorDB
docker run -p 6379:6379 falkordb/falkordb:latest

# Verify connection in Python
from falkordb import FalkorDB
client = FalkorDB(host='localhost', port=6379)
print(client.list_graphs())
```

#### Issue: "Graphity optimizations not improving performance"

**Problem:** Hub nodes not properly configured or cache not enabled

**Solution:**
```python
# Check hub node degrees
query = """
MATCH (n:HCP)
WITH n, size((n)--()) as degree
WHERE degree > 100
RETURN count(*) as hub_count
"""

# Verify Graphity is enabled in FalkorDB config
# Check redis.conf or FalkorDB settings
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run your operations
compiler = SchemaCompiler("config/ontology/schemas")
compiled = compiler.compile()
```

---

## 📖 API Reference

### SchemaCompiler

```python
class SchemaCompiler:
    def __init__(self, ontology_dir: Path)
    def compile(self) -> CompiledSchema
    def export_cypher_ddl(self, compiled: CompiledSchema) -> str
    def export_json_schema(self, compiled: CompiledSchema) -> Dict[str, Any]
```

### OntologyValidator

```python
class OntologyValidator:
    def __init__(self, compiled_schema: CompiledSchema)
    def validate(self) -> ValidationReport
    def generate_report(self, report: ValidationReport, format: str = 'text') -> str
```

### GraphityConfigBuilder

```python
class GraphityConfigBuilder:
    def __init__(self)
    def with_edge_grouping(self, strategy: EdgeGroupingStrategy, chunk_size: Optional[int]) -> 'GraphityConfigBuilder'
    def with_cache(self, strategy: CacheStrategy, ttl_seconds: Optional[int], max_size_mb: Optional[int]) -> 'GraphityConfigBuilder'
    def add_hub_entity(self, entity_label: str, degree_threshold: int, partition_count: int) -> 'GraphityConfigBuilder'
    def add_traversal_pattern(self, name: str, cypher_pattern: str, frequency: str, edge_types: List[str]) -> 'GraphityConfigBuilder'
    def build(self) -> GraphityConfig
    def to_yaml(self, filepath: Optional[Path]) -> str
    @classmethod
    def from_yaml(cls, filepath: Path) -> 'GraphityConfigBuilder'
```

### InferenceEngine

```python
class InferenceEngine:
    def __init__(self, graph_client: Any)
    def discover_causal_paths(self, source_id: str, target_id: str, max_depth: int, relationship_types: Optional[List[str]]) -> List[CausalPath]
    def infer_relationships(self, max_depth: int, min_confidence: float) -> List[InferredRelationship]
    def find_confounders(self, treatment_id: str, outcome_id: str, max_depth: int) -> List[Dict[str, Any]]
    def find_mediators(self, source_id: str, target_id: str) -> List[Dict[str, Any]]
    def rank_causal_paths(self, causal_paths: List[CausalPath], method: str) -> List[Tuple[CausalPath, float]]
    def materialize_inferred_relationships(self, inferred_relationships: List[InferredRelationship], min_confidence: float) -> int
```

---

## 📞 Support

For issues, questions, or contributions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Examples](#examples)
3. Enable [debug logging](#debug-mode)
4. Check FalkorDB documentation: https://docs.falkordb.com

---

## 📄 License

Part of the E2I Causal Analytics System.

---

## 🎯 Next Steps

1. **Define your ontology** in YAML format
2. **Run validation** to catch issues early
3. **Deploy to FalkorDB** with Graphity optimizations
4. **Use inference engine** to enrich your knowledge graph
5. **Monitor performance** and adjust Graphity config as needed

Happy graphing! 🚀
