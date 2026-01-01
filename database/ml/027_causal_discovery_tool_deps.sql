-- ============================================================================
-- E2I Causal Analytics - Migration 027: Causal Discovery Tool Dependencies
-- ============================================================================
--
-- Version: 4.4.0
-- Date: 2026-01-01
-- Description: Adds causal discovery tools to tool_registry and establishes
--              tool dependencies for Tool Composer integration.
--
-- New Tools (3):
--   - discover_dag: DAG structure learning via GES/PC/FCI/LiNGAM
--   - rank_drivers: Causal vs predictive feature importance ranking
--   - detect_structural_drift: Detect drift in causal graph structure
--
-- New Dependencies (1):
--   - rank_drivers consumes discover_dag.edge_list
--
-- Related Files:
--   - src/tool_registry/tools/causal_discovery.py (discover_dag, rank_drivers)
--   - src/tool_registry/tools/structural_drift.py (detect_structural_drift)
--
-- Dependencies: Requires migration 013_tool_composer_tables.sql
-- ============================================================================

-- ============================================================================
-- SECTION 1: ADD CAUSAL DISCOVERY TOOLS TO REGISTRY
-- ============================================================================

-- Note: Using ON CONFLICT to make this migration idempotent
-- tool_registry already exists from migration 013

-- ----------------------------------------------------------------------------
-- Tool: discover_dag
-- Purpose: Discover causal DAG structure using ensemble algorithms
-- Source: CausalDiscoveryTool in src/tool_registry/tools/causal_discovery.py
-- ----------------------------------------------------------------------------
INSERT INTO tool_registry (
    name,
    description,
    category,
    source_agent,
    input_schema,
    output_schema,
    composable,
    avg_latency_ms,
    success_rate,
    version
) VALUES (
    'discover_dag',
    'Discover causal DAG structure using ensemble algorithms (GES, PC, FCI, LiNGAM). Returns adjacency matrix, edge list, and confidence scores.',
    'CAUSAL',
    'causal_impact',
    '{
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Input data as list of row dictionaries or 2D array"
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column names for the data"
            },
            "algorithms": {
                "type": "array",
                "items": {"type": "string", "enum": ["ges", "pc", "fci", "lingam"]},
                "default": ["ges", "pc"],
                "description": "Algorithms to use for ensemble"
            },
            "target_variable": {
                "type": "string",
                "description": "Optional target variable to focus discovery on"
            },
            "max_conditioning_set_size": {
                "type": "integer",
                "default": 3,
                "description": "Maximum size of conditioning sets for PC/FCI"
            },
            "significance_level": {
                "type": "number",
                "default": 0.05,
                "description": "Significance level for conditional independence tests"
            }
        },
        "required": ["data", "columns"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "adjacency_matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Binary adjacency matrix of discovered DAG"
            },
            "edge_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "edge_type": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                },
                "description": "List of edges with metadata"
            },
            "node_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Node names in order"
            },
            "algorithm_agreements": {
                "type": "object",
                "description": "Per-edge agreement scores across algorithms"
            },
            "metadata": {
                "type": "object",
                "description": "Execution metadata including timing and cache info"
            }
        }
    }'::jsonb,
    true,
    5000.0,  -- 5 seconds average (discovery is computationally intensive)
    0.92,
    '4.4.0'
)
ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    output_schema = EXCLUDED.output_schema,
    avg_latency_ms = EXCLUDED.avg_latency_ms,
    version = EXCLUDED.version,
    updated_at = NOW();


-- ----------------------------------------------------------------------------
-- Tool: rank_drivers
-- Purpose: Compare causal vs predictive feature importance
-- Source: DriverRankerTool in src/tool_registry/tools/causal_discovery.py
-- ----------------------------------------------------------------------------
INSERT INTO tool_registry (
    name,
    description,
    category,
    source_agent,
    input_schema,
    output_schema,
    composable,
    avg_latency_ms,
    success_rate,
    version
) VALUES (
    'rank_drivers',
    'Compare causal vs predictive feature importance. Identifies features that are causally important vs merely predictive, enabling targeted interventions.',
    'CAUSAL',
    'causal_impact',
    '{
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Input data as list of row dictionaries or 2D array"
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column names for the data"
            },
            "target_variable": {
                "type": "string",
                "description": "Target variable for importance ranking"
            },
            "dag_edge_list": {
                "type": "array",
                "description": "Optional: pre-computed DAG edge list from discover_dag"
            },
            "dag_adjacency": {
                "type": "array",
                "description": "Optional: pre-computed DAG adjacency matrix"
            },
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of top features to return"
            }
        },
        "required": ["data", "columns", "target_variable"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "causal_rank": {"type": "integer"},
                        "predictive_rank": {"type": "integer"},
                        "causal_score": {"type": "number"},
                        "predictive_score": {"type": "number"},
                        "rank_difference": {"type": "integer"},
                        "category": {"type": "string"}
                    }
                },
                "description": "Feature rankings with causal vs predictive comparison"
            },
            "causal_only_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features important causally but not predictively"
            },
            "predictive_only_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features important predictively but not causally"
            },
            "both_important": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features important in both rankings"
            },
            "correlation": {
                "type": "number",
                "description": "Spearman correlation between causal and predictive rankings"
            }
        }
    }'::jsonb,
    true,
    500.0,  -- 500ms average (lighter computation)
    0.95,
    '4.4.0'
)
ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    output_schema = EXCLUDED.output_schema,
    avg_latency_ms = EXCLUDED.avg_latency_ms,
    version = EXCLUDED.version,
    updated_at = NOW();


-- ----------------------------------------------------------------------------
-- Tool: detect_structural_drift
-- Purpose: Detect drift in causal DAG structure over time
-- Source: StructuralDriftTool in src/tool_registry/tools/structural_drift.py
-- ----------------------------------------------------------------------------
INSERT INTO tool_registry (
    name,
    description,
    category,
    source_agent,
    input_schema,
    output_schema,
    composable,
    avg_latency_ms,
    success_rate,
    version
) VALUES (
    'detect_structural_drift',
    'Detect drift in causal DAG structure over time. Compares baseline and current DAGs to identify structural changes in causal relationships.',
    'MONITORING',
    'drift_monitor',
    '{
        "type": "object",
        "properties": {
            "baseline_dag_adjacency": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Baseline DAG adjacency matrix (binary)"
            },
            "current_dag_adjacency": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Current DAG adjacency matrix (binary)"
            },
            "dag_nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Node names in adjacency matrix order"
            },
            "baseline_edge_types": {
                "type": "object",
                "description": "Optional: edge types for baseline DAG (format: \"A->B\": \"DIRECTED\")"
            },
            "current_edge_types": {
                "type": "object",
                "description": "Optional: edge types for current DAG"
            }
        },
        "required": ["baseline_dag_adjacency", "current_dag_adjacency", "dag_nodes"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "detected": {"type": "boolean", "description": "Whether structural drift was detected"},
            "drift_score": {
                "type": "number",
                "description": "Drift score (0-1): proportion of edges changed"
            },
            "severity": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "critical"],
                "description": "Severity classification based on drift score and edge type changes"
            },
            "added_edges": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edges present in current but not baseline (format: \"A->B\")"
            },
            "removed_edges": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edges present in baseline but not current"
            },
            "stable_edges": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edges present in both DAGs"
            },
            "edge_type_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "edge": {"type": "string"},
                        "baseline_type": {"type": "string"},
                        "current_type": {"type": "string"}
                    }
                },
                "description": "Edges with changed types (e.g., DIRECTED to BIDIRECTED)"
            },
            "recommendation": {
                "type": "string",
                "description": "Actionable recommendation based on drift analysis"
            }
        }
    }'::jsonb,
    true,
    2000.0,  -- 2 seconds average
    0.98,
    '4.4.0'
)
ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    output_schema = EXCLUDED.output_schema,
    avg_latency_ms = EXCLUDED.avg_latency_ms,
    version = EXCLUDED.version,
    updated_at = NOW();


-- ============================================================================
-- SECTION 2: ADD TOOL DEPENDENCIES
-- ============================================================================

-- rank_drivers can consume edge_list from discover_dag
-- This enables chained execution: discover_dag -> rank_drivers
DO $$
DECLARE
    v_discover_dag_id UUID;
    v_rank_drivers_id UUID;
BEGIN
    -- Get tool IDs
    SELECT tool_id INTO v_discover_dag_id FROM tool_registry WHERE name = 'discover_dag';
    SELECT tool_id INTO v_rank_drivers_id FROM tool_registry WHERE name = 'rank_drivers';

    -- Validate both tools exist
    IF v_discover_dag_id IS NULL THEN
        RAISE EXCEPTION 'discover_dag tool not found in registry';
    END IF;

    IF v_rank_drivers_id IS NULL THEN
        RAISE EXCEPTION 'rank_drivers tool not found in registry';
    END IF;

    -- Insert dependency: rank_drivers consumes discover_dag.edge_list
    INSERT INTO tool_dependencies (
        consumer_tool_id,
        producer_tool_id,
        output_field,
        input_field,
        transform_expression
    )
    VALUES (
        v_rank_drivers_id,
        v_discover_dag_id,
        'edge_list',
        'dag_edge_list',
        NULL  -- No transformation needed, direct mapping
    )
    ON CONFLICT (consumer_tool_id, producer_tool_id) DO UPDATE SET
        output_field = EXCLUDED.output_field,
        input_field = EXCLUDED.input_field;

    RAISE NOTICE 'Successfully added dependency: rank_drivers <- discover_dag.edge_list';
END $$;


-- ============================================================================
-- SECTION 3: VERIFICATION QUERIES
-- ============================================================================

-- Verify tools were added (run these after migration)
-- SELECT name, category, source_agent, avg_latency_ms, version
-- FROM tool_registry
-- WHERE name IN ('discover_dag', 'rank_drivers', 'detect_structural_drift');

-- Verify dependencies (run after migration)
-- SELECT
--     c.name AS consumer_tool,
--     p.name AS producer_tool,
--     td.output_field,
--     td.input_field
-- FROM tool_dependencies td
-- JOIN tool_registry c ON td.consumer_tool_id = c.tool_id
-- JOIN tool_registry p ON td.producer_tool_id = p.tool_id
-- WHERE c.name = 'rank_drivers' OR p.name = 'discover_dag';


-- ============================================================================
-- SECTION 4: ROLLBACK SCRIPT
-- ============================================================================
--
-- To rollback this migration, run:
--
-- BEGIN;
--
-- -- Remove dependencies first (foreign key constraint)
-- DELETE FROM tool_dependencies td
-- USING tool_registry c, tool_registry p
-- WHERE td.consumer_tool_id = c.tool_id
--   AND td.producer_tool_id = p.tool_id
--   AND (c.name = 'rank_drivers' AND p.name = 'discover_dag');
--
-- -- Remove tools
-- DELETE FROM tool_registry
-- WHERE name IN ('discover_dag', 'rank_drivers', 'detect_structural_drift');
--
-- COMMIT;
--
-- ============================================================================


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
--
-- Post-migration steps:
-- 1. Verify: SELECT * FROM tool_registry WHERE name LIKE '%dag%' OR name LIKE '%drift%';
-- 2. Test chaining: SELECT * FROM get_tool_execution_order(ARRAY['discover_dag', 'rank_drivers']);
-- 3. Check dependencies: SELECT * FROM tool_dependencies WHERE output_field = 'edge_list';
--
-- ============================================================================
