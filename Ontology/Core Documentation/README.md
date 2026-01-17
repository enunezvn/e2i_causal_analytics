# Domain Vocabulary - Modular Structure v6.0.0

## 📁 Directory Structure

```
domain_vocabulary_modular/
├── index.yaml                      # Master index and module registry
├── __init__.py                     # Python loader
│
├── core/                           # Business entities & agents
│   ├── 01_entities.yaml           # Brands, regions, therapeutic areas (~100 lines)
│   ├── 02_agents.yaml             # 18-agent architecture (~150 lines)
│   └── 03_attributes.yaml         # Patient/HCP attributes (~100 lines)
│
├── ontology/                       # Semantic graph schema
│   ├── 01_node_types.yaml         # 8 node types with properties (~300 lines)
│   ├── 02_edge_types.yaml         # 15 edge types with cardinality (~200 lines)
│   ├── 03_inference_rules.yaml    # 5 inference rules with Cypher (~150 lines)
│   ├── 04_validation_rules.yaml   # 6 validation rules (~80 lines)
│   └── 05_falkordb_config.yaml    # FalkorDB compilation config (~100 lines)
│
├── operations/                     # Agent workflows & experiments
│   ├── 01_agent_tools.yaml        # Tools and workflows (~120 lines)
│   ├── 02_experiments.yaml        # Experiment lifecycle (~120 lines)
│   ├── 03_confidence.yaml         # Confidence scoring (~110 lines)
│   └── 04_digital_twin.yaml       # Digital twin simulation (~80 lines)
│
├── infrastructure/                 # Monitoring, viz, time
│   ├── 01_drift.yaml              # Drift monitoring (~70 lines)
│   ├── 02_visualization.yaml      # Charts and KPIs (~100 lines)
│   └── 03_time.yaml               # Time references (~60 lines)
│
├── mlops/                          # MLOps & observability
│   ├── 01_mlops.yaml              # Pipeline stages & feature store (~80 lines)
│   └── 02_observability.yaml      # Tracing & circuit breakers (~70 lines)
│
└── feedback/                       # Feedback loop & improvement
    ├── 01_outcome_truth.yaml      # Outcome labeling (~70 lines)
    └── 02_self_improvement.yaml   # Self-improvement rubrics (~60 lines)
```

**Total**: 18 YAML files, ~1,700 lines (vs. 1 monolithic file)

---

## 🚀 Quick Start

### **Python Usage**

```python
from config.domain_vocabulary import load_vocabulary, load_module

# Load all modules (recommended for startup)
vocab = load_vocabulary()

# Access content
brands = vocab['core_entities']['brands']
patients = vocab['node_types']['node_types']['Patient']

# Load single module (lighter weight)
agents = load_module('agent_architecture')
print(agents['agents']['CausalImpactAgent']['purpose'])

# Load by category
core_modules = load_category('core')
ontology_modules = load_category('ontology')

# Convenience functions
from config.domain_vocabulary import get_brands, get_agents, get_node_types

brands = get_brands()
agents = get_agents()
node_types = get_node_types()
```

### **Validation Usage**

```python
from config.domain_vocabulary import get_loader

loader = get_loader()

# Validate enum value
is_valid = loader.validate_enum_value(
    node_type='Patient',
    property_name='journey_stage',
    value='prescribed'
)
# Returns: True

# Get node schema
patient_schema = loader.get_node_type_schema('Patient')
print(patient_schema['properties']['journey_stage'])
```

---

## 📊 Module Categories

### **1. CORE** (Business Entities & Agents)

| Module | File | Purpose |
|--------|------|---------|
| `core_entities` | `core/01_entities.yaml` | 3 brands, 4 regions, therapeutic areas, data sources |
| `agent_architecture` | `core/02_agents.yaml` | 18 agents across 6 tiers with dependencies |
| `patient_hcp_attributes` | `core/03_attributes.yaml` | Journey stages, specialties, trigger types |

**Use when**: Referencing brands, agents, regions, patient stages, HCP specialties

### **2. ONTOLOGY** (Semantic Graph Schema)

| Module | File | Purpose |
|--------|------|---------|
| `node_types` | `ontology/01_node_types.yaml` | 8 node types (Patient, HCP, Brand, etc.) with complete property schemas |
| `edge_types` | `ontology/02_edge_types.yaml` | 15 edge types (TREATED_BY, PRESCRIBED, etc.) with cardinality |
| `inference_rules` | `ontology/03_inference_rules.yaml` | 5 inference rules with Cypher queries |
| `validation_rules` | `ontology/04_validation_rules.yaml` | 6 validation rules (strict/warning) |
| `falkordb_config` | `ontology/05_falkordb_config.yaml` | FalkorDB constraints, indexes, optimization |

**Use when**: Building graph schema, validating data, running inference rules

### **3. OPERATIONS** (Agent Workflows & Experiments)

| Module | File | Purpose |
|--------|------|---------|
| `agent_tools_workflows` | `operations/01_agent_tools.yaml` | Tools (DoWhy, EconML), workflows, causal methods |
| `experiment_lifecycle` | `operations/02_experiments.yaml` | 14 experiment states, 23 transitions |
| `confidence_scoring` | `operations/03_confidence.yaml` | Confidence tiers, reduction/boost rules |
| `digital_twin_simulation` | `operations/04_digital_twin.yaml` | Twin algorithms, interventions, fidelity grades |

**Use when**: Configuring agents, managing experiments, scoring confidence, simulating interventions

### **4. INFRASTRUCTURE** (Monitoring, Viz, Time)

| Module | File | Purpose |
|--------|------|---------|
| `drift_monitoring` | `infrastructure/01_drift.yaml` | Drift types, detection methods, PSI thresholds |
| `visualization_kpis` | `infrastructure/02_visualization.yaml` | 18 chart types, 4 libraries, KPI categories |
| `time_references` | `infrastructure/03_time.yaml` | Time periods, granularities, observation windows |

**Use when**: Monitoring drift, creating visualizations, handling time-based queries

### **5. MLOPS** (MLOps & Observability)

| Module | File | Purpose |
|--------|------|---------|
| `mlops_feature_store` | `mlops/01_mlops.yaml` | Pipeline stages, feature stores, HPO optimizers |
| `observability_tracing` | `mlops/02_observability.yaml` | Trace spans, circuit breakers, sampling modes |

**Use when**: Managing ML pipelines, tracing agent execution, monitoring system health

### **6. FEEDBACK** (Outcome Truth & Self-Improvement)

| Module | File | Purpose |
|--------|------|---------|
| `outcome_truth` | `feedback/01_outcome_truth.yaml` | Outcome labels, truth sources, edge cases |
| `self_improvement` | `feedback/02_self_improvement.yaml` | Rubric criteria, improvement decisions, risk levels |

**Use when**: Labeling outcomes, validating models, implementing self-improvement

---

## 🔧 Integration Examples

### **Example 1: Schema Compilation**

```python
from config.domain_vocabulary import load_module
from config.ontology.schema_compiler import OntologyCompiler

# Load ontology modules
node_types = load_module('node_types')
edge_types = load_module('edge_types')
falkordb_config = load_module('falkordb_config')

# Compile schema
compiler = OntologyCompiler(
    node_types=node_types['node_types'],
    edge_types=edge_types['edge_types'],
    config=falkordb_config['falkordb_config']
)

# Generate FalkorDB constraints
constraints = compiler.compile_constraints()
for constraint in constraints:
    db.execute(constraint)

# Generate indexes
indexes = compiler.compile_indexes()
for index in indexes:
    db.execute(index)
```

### **Example 2: Runtime Validation**

```python
from config.domain_vocabulary import load_module, get_loader
from config.ontology.validator import GraphValidator

# Load validation rules
validation_rules = load_module('validation_rules')

# Create validator
validator = GraphValidator(
    node_types=load_module('node_types'),
    validation_rules=validation_rules['validation_rules']
)

# Validate node
patient_data = {
    'patient_id': 'PAT12345678',
    'journey_stage': 'prescribed',
    'region': 'northeast'
}

is_valid, errors = validator.validate_node('Patient', patient_data)
if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

### **Example 3: Inference Execution**

```python
from config.domain_vocabulary import load_module
from config.ontology.inference_engine import InferenceEngine

# Load inference rules
inference_rules = load_module('inference_rules')

# Create inference engine
engine = InferenceEngine(
    db=falkordb_client,
    graph_name='e2i_semantic',
    rules=inference_rules['inference_rules']
)

# Run all enabled rules
engine.run_all_rules()

# Run specific rule
engine.run_rule('causal_chain')
```

### **Example 4: Agent Configuration**

```python
from config.domain_vocabulary import load_module, get_agents

# Load agent specs
agents = get_agents()
tools = load_module('agent_tools_workflows')

class CausalImpactAgent:
    def __init__(self):
        # Load configuration from vocabulary
        agent_spec = agents['CausalImpactAgent']
        self.tier = agent_spec['tier']
        self.purpose = agent_spec['purpose']
        
        # Load tools
        self.tools = tools['agent_tools']['causal_impact_tools']
        
        # Load workflow
        self.workflow = tools['agent_workflows']['causal_impact_workflow']
        self.workflow_nodes = self.workflow['nodes']
        self.require_dag_approval = self.workflow['require_dag_approval']
```

---

## 🔄 Migration from Monolithic

### **Step-by-Step Migration**

1. **Backup existing file**
   ```bash
   cp config/domain_vocabulary_enhanced.yaml config/domain_vocabulary_enhanced.yaml.backup
   ```

2. **Deploy modular structure**
   ```bash
   cp -r domain_vocabulary_modular/ config/domain_vocabulary/
   ```

3. **Update imports in code**
   ```python
   # Old (monolithic)
   from config.domain_vocabulary_enhanced import vocabulary
   brands = vocabulary['brands']
   
   # New (modular)
   from config.domain_vocabulary import load_vocabulary
   vocab = load_vocabulary()
   brands = vocab['core_entities']['brands']
   ```

4. **Test vocabulary loading**
   ```bash
   python -m config.domain_vocabulary
   ```

5. **Run validation tests**
   ```bash
   pytest tests/test_domain_vocabulary.py -v
   ```

6. **Deploy to production**

### **Backwards Compatibility**

The modular structure maintains the same data format - only the physical organization changes. You can create a compatibility shim:

```python
# config/domain_vocabulary_enhanced.py (compatibility shim)
from config.domain_vocabulary import load_vocabulary

vocabulary = load_vocabulary()

# Flatten structure for backwards compatibility
vocabulary_flat = {}
for module_name, module_content in vocabulary.items():
    if module_content:
        vocabulary_flat.update(module_content)
```

---

## ✅ Benefits of Modular Structure

### **1. Easier Maintenance**
- Modify one file without affecting others
- Smaller files are easier to review and understand
- Clear separation of concerns

### **2. Better Version Control**
- Smaller diffs when making changes
- Easier to track what changed and why
- Reduced merge conflicts

### **3. Parallel Development**
- Multiple team members can work simultaneously
- Each module can have designated maintainer
- Reduces bottlenecks

### **4. Selective Loading**
- Load only needed modules
- Faster startup for lightweight agents
- Reduced memory footprint

### **5. Clear Ownership**
- Each module has designated maintainer (see index.yaml)
- Clear responsibility for updates
- Better accountability

---

## 📝 Adding New Vocabulary

### **Step 1: Identify Module**

Determine which module should contain the new vocabulary:
- Business entities → `core/01_entities.yaml`
- Agent specifications → `core/02_agents.yaml`
- Node/edge types → `ontology/01_node_types.yaml` or `02_edge_types.yaml`
- Inference rules → `ontology/03_inference_rules.yaml`
- Etc.

### **Step 2: Add Content**

```yaml
# Example: Adding new brand to core/01_entities.yaml
brands:
  new_brand:
    therapeutic_area: "Neurology"
    indication: "Multiple Sclerosis"
    abbreviation: "NEW"
    status: "active"
```

### **Step 3: Update Index** (if adding new module)

```yaml
# index.yaml
modules:
  new_module:
    file: "category/##_module.yaml"
    description: "Description here"
    lines: ~100
    contains:
      - "Item 1"
      - "Item 2"
    maintainer: "Team Name"
```

### **Step 4: Validate**

```bash
# Check YAML syntax
yamllint domain_vocabulary_modular/

# Run tests
pytest tests/test_domain_vocabulary.py -v

# Test loading
python -m config.domain_vocabulary
```

### **Step 5: Update Version**

```yaml
# index.yaml
_metadata:
  version: "6.1.0"  # Increment minor version
  last_updated: "2026-01-13"
```

---

## 🧪 Testing

### **Unit Tests**

```python
import pytest
from config.domain_vocabulary import load_module, get_loader

def test_load_all_modules():
    """Test that all modules load successfully."""
    from config.domain_vocabulary import MODULE_REGISTRY
    
    for module_name in MODULE_REGISTRY.keys():
        module = load_module(module_name)
        assert module is not None, f"Failed to load {module_name}"

def test_validate_enum():
    """Test enum validation."""
    loader = get_loader()
    
    # Valid value
    assert loader.validate_enum_value('Patient', 'journey_stage', 'prescribed')
    
    # Invalid value
    assert not loader.validate_enum_value('Patient', 'journey_stage', 'invalid_stage')

def test_node_schema():
    """Test node schema retrieval."""
    loader = get_loader()
    
    patient_schema = loader.get_node_type_schema('Patient')
    assert patient_schema is not None
    assert 'properties' in patient_schema
    assert 'patient_id' in patient_schema['properties']
```

### **Integration Tests**

```bash
# Run all tests
pytest tests/test_domain_vocabulary.py -v

# Test specific module
pytest tests/test_domain_vocabulary.py::test_load_node_types -v

# Test with coverage
pytest tests/test_domain_vocabulary.py --cov=config.domain_vocabulary
```

---

## 📊 Performance

### **Loading Times** (typical)
- Single module: <10ms
- Category (3-5 modules): <50ms
- All modules (18 modules): <200ms
- Cached access: <1ms

### **Memory Usage**
- Single module: ~100KB
- All modules: ~1.5MB
- Cached: Same (no duplication)

### **Optimization Tips**
1. Use caching for frequently accessed modules
2. Load only needed categories at startup
3. Clear cache periodically to free memory
4. Use selective loading in lightweight agents

---

## 🔗 Related Documentation

- **index.yaml**: Master registry of all modules
- **__init__.py**: Python loader documentation
- **ONTOLOGY_LAYER_README.md**: Ontology integration details
- **DOMAIN_VOCABULARY_V6_README.md**: Original v6.0.0 documentation

---

## 📞 Support

For questions or issues:
1. Check this README first
2. Review module-specific comments in YAML files
3. Check index.yaml for module registry
4. Contact designated maintainer (see index.yaml)
5. Contact E2I Platform Team

---

**Version**: 6.0.0 (Modular)  
**Last Updated**: 2026-01-12  
**Maintainer**: E2I Platform Team  
**Status**: ✅ Production Ready
