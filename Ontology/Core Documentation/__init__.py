"""
Domain Vocabulary Loader
=========================

Python loader for modular domain vocabulary structure.

Usage:
    from config.domain_vocabulary import load_vocabulary, load_module
    
    # Load all modules
    vocab = load_vocabulary()
    
    # Access content
    brands = vocab['core_entities']['brands']
    node_types = vocab['node_types']
    
    # Load single module
    ontology = load_module('node_types')
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Module registry (matches index.yaml)
MODULE_REGISTRY = {
    # Core modules
    'core_entities': 'core/01_entities.yaml',
    'agent_architecture': 'core/02_agents.yaml',
    'patient_hcp_attributes': 'core/03_attributes.yaml',
    
    # Ontology modules
    'node_types': 'ontology/01_node_types.yaml',
    'edge_types': 'ontology/02_edge_types.yaml',
    'inference_rules': 'ontology/03_inference_rules.yaml',
    'validation_rules': 'ontology/04_validation_rules.yaml',
    'falkordb_config': 'ontology/05_falkordb_config.yaml',
    
    # Operations modules
    'agent_tools_workflows': 'operations/01_agent_tools.yaml',
    'experiment_lifecycle': 'operations/02_experiments.yaml',
    'confidence_scoring': 'operations/03_confidence.yaml',
    'digital_twin_simulation': 'operations/04_digital_twin.yaml',
    
    # Infrastructure modules
    'drift_monitoring': 'infrastructure/01_drift.yaml',
    'visualization_kpis': 'infrastructure/02_visualization.yaml',
    'time_references': 'infrastructure/03_time.yaml',
    
    # MLOps modules
    'mlops_feature_store': 'mlops/01_mlops.yaml',
    'observability_tracing': 'mlops/02_observability.yaml',
    
    # Feedback modules
    'outcome_truth': 'feedback/01_outcome_truth.yaml',
    'self_improvement': 'feedback/02_self_improvement.yaml',
}


class VocabularyLoader:
    """Loader for modular domain vocabulary."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize loader.
        
        Args:
            base_path: Base path to vocabulary modules (default: ./config/domain_vocabulary/)
        """
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self._cache = {}
        
    def load_module(self, module_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a single vocabulary module.
        
        Args:
            module_name: Name of module (e.g., 'node_types', 'agent_architecture')
            use_cache: Whether to use cached version if available
            
        Returns:
            Dictionary with module content
            
        Raises:
            ValueError: If module name not found in registry
            FileNotFoundError: If module file doesn't exist
        """
        if module_name not in MODULE_REGISTRY:
            raise ValueError(
                f"Module '{module_name}' not found in registry. "
                f"Available modules: {', '.join(MODULE_REGISTRY.keys())}"
            )
        
        # Check cache
        if use_cache and module_name in self._cache:
            logger.debug(f"Loading {module_name} from cache")
            return self._cache[module_name]
        
        # Load from file
        module_path = self.base_path / MODULE_REGISTRY[module_name]
        
        if not module_path.exists():
            raise FileNotFoundError(
                f"Module file not found: {module_path}. "
                f"Expected at: {self.base_path}"
            )
        
        logger.debug(f"Loading {module_name} from {module_path}")
        
        try:
            with open(module_path, 'r') as f:
                content = yaml.safe_load(f)
                
            # Cache the result
            self._cache[module_name] = content
            
            return content
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in {module_path}: {e}")
            raise
    
    def load_all(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load all vocabulary modules.
        
        Args:
            use_cache: Whether to use cached versions if available
            
        Returns:
            Dictionary with all module content, keyed by module name
        """
        vocabulary = {}
        
        for module_name in MODULE_REGISTRY.keys():
            try:
                vocabulary[module_name] = self.load_module(module_name, use_cache)
            except Exception as e:
                logger.error(f"Failed to load module {module_name}: {e}")
                # Continue loading other modules
                vocabulary[module_name] = None
        
        return vocabulary
    
    def load_category(self, category: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load all modules in a category.
        
        Args:
            category: Category name ('core', 'ontology', 'operations', 'infrastructure', 'mlops', 'feedback')
            use_cache: Whether to use cached versions
            
        Returns:
            Dictionary with all modules in category
        """
        valid_categories = {'core', 'ontology', 'operations', 'infrastructure', 'mlops', 'feedback'}
        
        if category not in valid_categories:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Valid categories: {', '.join(valid_categories)}"
            )
        
        category_modules = {
            name: self.load_module(name, use_cache)
            for name, path in MODULE_REGISTRY.items()
            if path.startswith(f"{category}/")
        }
        
        return category_modules
    
    def get_node_type_schema(self, node_type: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific node type.
        
        Args:
            node_type: Node type name (e.g., 'Patient', 'HCP')
            
        Returns:
            Schema dict or None if not found
        """
        node_types = self.load_module('node_types')
        return node_types.get('node_types', {}).get(node_type)
    
    def get_edge_type_schema(self, edge_type: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific edge type.
        
        Args:
            edge_type: Edge type name (e.g., 'TREATED_BY', 'PRESCRIBED')
            
        Returns:
            Schema dict or None if not found
        """
        edge_types = self.load_module('edge_types')
        return edge_types.get('edge_types', {}).get(edge_type)
    
    def validate_enum_value(self, node_type: str, property_name: str, value: Any) -> bool:
        """
        Validate an enum value against vocabulary.
        
        Args:
            node_type: Node type (e.g., 'Patient')
            property_name: Property name (e.g., 'journey_stage')
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        schema = self.get_node_type_schema(node_type)
        
        if not schema:
            logger.warning(f"Node type {node_type} not found")
            return False
        
        property_schema = schema.get('properties', {}).get(property_name)
        
        if not property_schema:
            logger.warning(f"Property {property_name} not found for {node_type}")
            return False
        
        if property_schema.get('type') != 'enum':
            logger.warning(f"Property {property_name} is not an enum")
            return False
        
        valid_values = property_schema.get('values', [])
        return value in valid_values
    
    def clear_cache(self):
        """Clear cached modules."""
        self._cache = {}
        logger.info("Vocabulary cache cleared")


# Global loader instance
_loader = None


def get_loader(base_path: Optional[Path] = None) -> VocabularyLoader:
    """
    Get global loader instance.
    
    Args:
        base_path: Base path to vocabulary modules
        
    Returns:
        VocabularyLoader instance
    """
    global _loader
    
    if _loader is None:
        _loader = VocabularyLoader(base_path)
    
    return _loader


def load_vocabulary(base_path: Optional[Path] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load all vocabulary modules.
    
    Args:
        base_path: Base path to vocabulary modules
        use_cache: Whether to use cached versions
        
    Returns:
        Dictionary with all module content
    """
    loader = get_loader(base_path)
    return loader.load_all(use_cache)


def load_module(module_name: str, base_path: Optional[Path] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load a single vocabulary module.
    
    Args:
        module_name: Name of module
        base_path: Base path to vocabulary modules
        use_cache: Whether to use cached version
        
    Returns:
        Dictionary with module content
    """
    loader = get_loader(base_path)
    return loader.load_module(module_name, use_cache)


def load_category(category: str, base_path: Optional[Path] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load all modules in a category.
    
    Args:
        category: Category name
        base_path: Base path to vocabulary modules
        use_cache: Whether to use cached versions
        
    Returns:
        Dictionary with all modules in category
    """
    loader = get_loader(base_path)
    return loader.load_category(category, use_cache)


# Convenience functions for common access patterns

def get_brands() -> Dict[str, Any]:
    """Get brand definitions."""
    entities = load_module('core_entities')
    return entities.get('brands', {})


def get_agents() -> Dict[str, Any]:
    """Get agent architecture."""
    agents = load_module('agent_architecture')
    return agents.get('agents', {})


def get_node_types() -> Dict[str, Any]:
    """Get node type schemas."""
    node_types = load_module('node_types')
    return node_types.get('node_types', {})


def get_edge_types() -> Dict[str, Any]:
    """Get edge type schemas."""
    edge_types = load_module('edge_types')
    return edge_types.get('edge_types', {})


def get_inference_rules() -> Dict[str, Any]:
    """Get inference rules."""
    rules = load_module('inference_rules')
    return rules.get('inference_rules', {})


def get_validation_rules() -> Dict[str, Any]:
    """Get validation rules."""
    rules = load_module('validation_rules')
    return rules.get('validation_rules', {})


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=== Loading All Vocabulary ===")
    vocab = load_vocabulary()
    print(f"Loaded {len(vocab)} modules")
    
    print("\n=== Core Entities ===")
    entities = load_module('core_entities')
    print(f"Brands: {list(entities['brands'].keys())}")
    
    print("\n=== Ontology ===")
    node_types = get_node_types()
    print(f"Node types: {list(node_types.keys())}")
    
    print("\n=== Agent Architecture ===")
    agents = get_agents()
    print(f"Agents: {len(agents)} agents loaded")
    
    print("\n=== Validation ===")
    loader = get_loader()
    is_valid = loader.validate_enum_value('Patient', 'journey_stage', 'prescribed')
    print(f"'prescribed' is valid journey_stage: {is_valid}")
