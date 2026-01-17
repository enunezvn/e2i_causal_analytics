#!/usr/bin/env python3
"""
Apply Modular Vocabulary to Ontology
=====================================

This script applies the modular domain vocabulary to the ontology layer,
generating:
1. FalkorDB schema (constraints + indexes)
2. Validation configuration
3. Inference rule scheduler
4. Grafiti wrapper with validation

Usage:
    python apply_vocabulary_to_ontology.py --action all
    python apply_vocabulary_to_ontology.py --action schema
    python apply_vocabulary_to_ontology.py --action validate
    python apply_vocabulary_to_ontology.py --action inference
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from __init__ import (
    load_module,
    load_category,
    get_node_types,
    get_edge_types,
    get_inference_rules,
    get_validation_rules
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OntologyApplicator:
    """Applies modular vocabulary to ontology layer."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize applicator.
        
        Args:
            output_dir: Directory for generated files (default: ./ontology_output/)
        """
        self.output_dir = output_dir or Path('./ontology_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ontology applicator (output: {self.output_dir})")
    
    def apply_schema(self) -> Path:
        """
        Generate FalkorDB schema from vocabulary.
        
        Returns:
            Path to generated schema file
        """
        logger.info("=== Applying Schema ===")
        
        # Load ontology modules
        node_types = get_node_types()
        edge_types = get_edge_types()
        falkordb_config = load_module('falkordb_config')['falkordb_config']
        
        logger.info(f"Loaded {len(node_types)} node types")
        logger.info(f"Loaded {len(edge_types)} edge types")
        
        # Generate constraints
        constraints = self._generate_constraints(node_types, falkordb_config)
        logger.info(f"Generated {len(constraints)} constraints")
        
        # Generate indexes
        indexes = self._generate_indexes(node_types, falkordb_config)
        logger.info(f"Generated {len(indexes)} indexes")
        
        # Write to file
        schema_file = self.output_dir / 'falkordb_schema.cypher'
        
        with open(schema_file, 'w') as f:
            f.write("// =============================================================================\n")
            f.write("// FALKORDB SCHEMA - Generated from Modular Vocabulary\n")
            f.write("// =============================================================================\n")
            f.write(f"// Graph: {falkordb_config['graph_name']}\n")
            f.write(f"// Generated: {self._get_timestamp()}\n")
            f.write("// =============================================================================\n\n")
            
            f.write("// CONSTRAINTS (Unique Properties)\n")
            f.write("// =============================================================================\n\n")
            for constraint in constraints:
                f.write(f"{constraint};\n")
            
            f.write("\n// INDEXES (Performance Optimization)\n")
            f.write("// =============================================================================\n\n")
            for index in indexes:
                f.write(f"{index};\n")
        
        logger.info(f"✅ Schema written to {schema_file}")
        return schema_file
    
    def _generate_constraints(self, node_types: Dict, config: Dict) -> List[str]:
        """Generate unique constraints from config."""
        constraints = []
        
        for constraint_spec in config['constraints']['unique_properties']:
            node_type = constraint_spec['node']
            property_name = constraint_spec['property']
            
            # Validate that property exists in node type
            if node_type in node_types:
                props = node_types[node_type].get('properties', {})
                if property_name not in props:
                    logger.warning(
                        f"Property {property_name} not found in {node_type}, skipping constraint"
                    )
                    continue
            
            constraint = (
                f"GRAPH.CONSTRAINT CREATE FOR (n:{node_type}) "
                f"REQUIRE n.{property_name} IS UNIQUE"
            )
            constraints.append(constraint)
        
        return constraints
    
    def _generate_indexes(self, node_types: Dict, config: Dict) -> List[str]:
        """Generate indexes from config."""
        indexes = []
        
        for node_type, properties in config['indexes'].items():
            if node_type not in node_types:
                logger.warning(f"Node type {node_type} not in vocabulary, skipping indexes")
                continue
            
            for property_name in properties:
                # Validate property exists
                props = node_types[node_type].get('properties', {})
                if property_name not in props:
                    logger.warning(
                        f"Property {property_name} not found in {node_type}, skipping index"
                    )
                    continue
                
                index = f"GRAPH.IDX CREATE FOR (n:{node_type}) ON (n.{property_name})"
                indexes.append(index)
        
        return indexes
    
    def apply_validation(self) -> Path:
        """
        Generate validation configuration from vocabulary.
        
        Returns:
            Path to generated validation config
        """
        logger.info("=== Applying Validation Rules ===")
        
        validation_rules = get_validation_rules()
        logger.info(f"Loaded {len(validation_rules)} validation rules")
        
        # Generate validator.py configuration
        config_file = self.output_dir / 'validator_config.py'
        
        with open(config_file, 'w') as f:
            f.write('"""\n')
            f.write('Validator Configuration - Generated from Modular Vocabulary\n')
            f.write('"""\n\n')
            
            f.write('VALIDATION_RULES = {\n')
            
            for rule_name, rule_spec in validation_rules.items():
                f.write(f'    "{rule_name}": {{\n')
                f.write(f'        "description": "{rule_spec["description"]}",\n')
                f.write(f'        "enforcement": "{rule_spec["enforcement"]}",\n')
                
                # Add rule-specific content
                if 'valid_transitions' in rule_spec:
                    f.write(f'        "valid_transitions": {rule_spec["valid_transitions"]},\n')
                elif 'rules' in rule_spec:
                    f.write(f'        "rules": {rule_spec["rules"]},\n')
                elif 'check' in rule_spec:
                    f.write(f'        "check": """{rule_spec["check"]}""",\n')
                
                f.write('    },\n')
            
            f.write('}\n')
        
        logger.info(f"✅ Validation config written to {config_file}")
        return config_file
    
    def apply_inference(self) -> Path:
        """
        Generate inference rule scheduler from vocabulary.
        
        Returns:
            Path to generated scheduler script
        """
        logger.info("=== Applying Inference Rules ===")
        
        inference_rules = get_inference_rules()
        logger.info(f"Loaded {len(inference_rules)} inference rules")
        
        # Generate scheduler script
        scheduler_file = self.output_dir / 'run_inference_rules.py'
        
        with open(scheduler_file, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('Inference Rule Scheduler - Generated from Modular Vocabulary\n')
            f.write('"""\n\n')
            
            f.write('import logging\n')
            f.write('from typing import Dict, Any\n\n')
            
            f.write('logger = logging.getLogger(__name__)\n\n')
            
            f.write('class InferenceEngine:\n')
            f.write('    """Execute inference rules on FalkorDB graph."""\n\n')
            
            f.write('    def __init__(self, db, graph_name: str):\n')
            f.write('        self.db = db\n')
            f.write('        self.graph_name = graph_name\n\n')
            
            # Generate method for each rule
            for rule_name, rule_spec in inference_rules.items():
                if not rule_spec.get('enabled', False):
                    continue
                
                method_name = rule_name.replace('-', '_')
                
                f.write(f'    def run_{method_name}(self) -> Dict[str, Any]:\n')
                f.write(f'        """\n')
                f.write(f'        {rule_spec["description"]}\n')
                f.write(f'        Frequency: {rule_spec["frequency"]}\n')
                f.write(f'        """\n')
                f.write(f'        logger.info("Running inference rule: {rule_name}")\n\n')
                
                f.write(f'        query = """\n')
                f.write(f'{rule_spec["cypher_query"]}\n')
                f.write(f'        """\n\n')
                
                f.write(f'        try:\n')
                f.write(f'            result = self.db.query(query)\n')
                f.write(f'            logger.info(f"{{rule_name}} completed: {{len(result)}} rows affected")\n')
                f.write(f'            return {{"status": "success", "rows": len(result)}}\n')
                f.write(f'        except Exception as e:\n')
                f.write(f'            logger.error(f"{{rule_name}} failed: {{e}}")\n')
                f.write(f'            return {{"status": "error", "message": str(e)}}\n\n')
            
            f.write('    def run_all_rules(self) -> Dict[str, Any]:\n')
            f.write('        """Run all enabled inference rules."""\n')
            f.write('        results = {}\n\n')
            
            for rule_name in inference_rules.keys():
                if not inference_rules[rule_name].get('enabled', False):
                    continue
                method_name = rule_name.replace('-', '_')
                f.write(f'        results["{rule_name}"] = self.run_{method_name}()\n')
            
            f.write('\n        return results\n')
        
        # Make executable
        scheduler_file.chmod(0o755)
        
        logger.info(f"✅ Inference scheduler written to {scheduler_file}")
        return scheduler_file
    
    def apply_all(self) -> Dict[str, Path]:
        """
        Apply all vocabulary to ontology layer.
        
        Returns:
            Dictionary of generated file paths
        """
        logger.info("=== Applying All Vocabulary to Ontology ===\n")
        
        results = {
            'schema': self.apply_schema(),
            'validation': self.apply_validation(),
            'inference': self.apply_inference(),
        }
        
        logger.info("\n=== Application Complete ===")
        logger.info(f"Generated files in: {self.output_dir}")
        for key, path in results.items():
            logger.info(f"  {key}: {path}")
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Apply modular vocabulary to ontology layer'
    )
    
    parser.add_argument(
        '--action',
        choices=['all', 'schema', 'validation', 'inference'],
        default='all',
        help='Action to perform (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./ontology_output'),
        help='Output directory for generated files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create applicator
    applicator = OntologyApplicator(output_dir=args.output_dir)
    
    # Apply vocabulary
    try:
        if args.action == 'all':
            applicator.apply_all()
        elif args.action == 'schema':
            applicator.apply_schema()
        elif args.action == 'validation':
            applicator.apply_validation()
        elif args.action == 'inference':
            applicator.apply_inference()
        
        logger.info("✅ SUCCESS")
        return 0
        
    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
