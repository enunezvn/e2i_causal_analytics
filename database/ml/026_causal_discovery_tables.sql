-- =============================================================================
-- E2I Causal Analytics - Causal Discovery Tables
-- =============================================================================
-- Version: 1.0.0
-- Created: 2025-12-30
-- Description: Tables for storing causal structure learning results
--
-- Features:
--   - Discovered DAG structures with confidence scores
--   - Algorithm run metadata
--   - Gate evaluation decisions
--   - Driver rankings (causal vs predictive)
--   - Ensemble voting results
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================

-- Discovery algorithm types
DO $$ BEGIN
    CREATE TYPE ml.discovery_algorithm AS ENUM (
        'ges',          -- Greedy Equivalence Search
        'pc',           -- Peter-Clark
        'fci',          -- Fast Causal Inference
        'lingam',       -- LiNGAM
        'direct_lingam',-- DirectLiNGAM
        'ica_lingam'    -- ICA-based LiNGAM
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Gate decision types
DO $$ BEGIN
    CREATE TYPE ml.gate_decision AS ENUM (
        'accept',   -- High confidence, use discovered DAG
        'review',   -- Medium confidence, requires expert validation
        'reject',   -- Low confidence, use manual DAG
        'augment'   -- Supplement manual DAG with high-confidence edges
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Edge types in discovered graphs
DO $$ BEGIN
    CREATE TYPE ml.edge_type AS ENUM (
        'directed',     -- Definite causal direction: X -> Y
        'undirected',   -- Unknown direction: X - Y
        'bidirected'    -- Possible confounder: X <-> Y
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Discovered DAG structures
CREATE TABLE IF NOT EXISTS ml.discovered_dags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES ml.user_sessions(id) ON DELETE SET NULL,

    -- Discovery metadata
    discovery_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    n_samples INTEGER NOT NULL,
    n_features INTEGER NOT NULL,
    feature_names JSONB NOT NULL DEFAULT '[]',

    -- Configuration used
    config JSONB NOT NULL DEFAULT '{}',
    algorithms_used TEXT[] NOT NULL DEFAULT '{}',
    ensemble_threshold FLOAT NOT NULL DEFAULT 0.5,
    alpha FLOAT NOT NULL DEFAULT 0.05,

    -- Results
    n_edges INTEGER NOT NULL DEFAULT 0,
    n_nodes INTEGER NOT NULL DEFAULT 0,
    edge_list JSONB NOT NULL DEFAULT '[]',
    confidence_scores JSONB NOT NULL DEFAULT '{}',
    adjacency_matrix JSONB,

    -- Gate evaluation
    gate_decision ml.gate_decision,
    gate_confidence FLOAT,
    gate_reasons JSONB DEFAULT '[]',

    -- Timing
    total_runtime_seconds FLOAT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Algorithm run results (one per algorithm per discovery)
CREATE TABLE IF NOT EXISTS ml.discovery_algorithm_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID NOT NULL REFERENCES ml.discovered_dags(id) ON DELETE CASCADE,

    -- Algorithm info
    algorithm ml.discovery_algorithm NOT NULL,
    runtime_seconds FLOAT NOT NULL,
    converged BOOLEAN NOT NULL DEFAULT TRUE,

    -- Results
    n_edges INTEGER NOT NULL DEFAULT 0,
    edge_list JSONB NOT NULL DEFAULT '[]',
    adjacency_matrix JSONB,
    score FLOAT,  -- For score-based methods

    -- Metadata
    parameters JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Discovered edges with confidence metadata
CREATE TABLE IF NOT EXISTS ml.discovered_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID NOT NULL REFERENCES ml.discovered_dags(id) ON DELETE CASCADE,

    -- Edge definition
    source_node VARCHAR(255) NOT NULL,
    target_node VARCHAR(255) NOT NULL,
    edge_type ml.edge_type NOT NULL DEFAULT 'directed',

    -- Confidence metrics
    confidence FLOAT NOT NULL DEFAULT 1.0,
    algorithm_votes INTEGER NOT NULL DEFAULT 1,
    algorithms TEXT[] NOT NULL DEFAULT '{}',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique edge per DAG
    CONSTRAINT unique_edge_per_dag UNIQUE (dag_id, source_node, target_node)
);

-- Driver rankings (causal vs predictive comparison)
CREATE TABLE IF NOT EXISTS ml.driver_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID REFERENCES ml.discovered_dags(id) ON DELETE CASCADE,
    session_id UUID REFERENCES ml.user_sessions(id) ON DELETE SET NULL,

    -- Target variable
    target_variable VARCHAR(255) NOT NULL,
    ranking_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Rankings (stored as JSONB array of feature rankings)
    rankings JSONB NOT NULL DEFAULT '[]',

    -- Summary statistics
    n_features INTEGER NOT NULL DEFAULT 0,
    rank_correlation FLOAT,  -- Spearman correlation between causal and predictive

    -- Categorized features
    causal_only_features TEXT[] DEFAULT '{}',
    predictive_only_features TEXT[] DEFAULT '{}',
    concordant_features TEXT[] DEFAULT '{}',

    -- Metadata
    config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Individual feature rankings (detailed view)
CREATE TABLE IF NOT EXISTS ml.feature_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ranking_id UUID NOT NULL REFERENCES ml.driver_rankings(id) ON DELETE CASCADE,

    -- Feature info
    feature_name VARCHAR(255) NOT NULL,

    -- Ranks
    causal_rank INTEGER NOT NULL,
    predictive_rank INTEGER NOT NULL,
    rank_difference INTEGER GENERATED ALWAYS AS (predictive_rank - causal_rank) STORED,

    -- Scores
    causal_score FLOAT NOT NULL DEFAULT 0.0,
    predictive_score FLOAT NOT NULL DEFAULT 0.0,

    -- Causal metadata
    is_direct_cause BOOLEAN NOT NULL DEFAULT FALSE,
    path_length INTEGER,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique feature per ranking
    CONSTRAINT unique_feature_per_ranking UNIQUE (ranking_id, feature_name)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Discovered DAGs indexes
CREATE INDEX IF NOT EXISTS idx_discovered_dags_session
    ON ml.discovered_dags(session_id);
CREATE INDEX IF NOT EXISTS idx_discovered_dags_gate_decision
    ON ml.discovered_dags(gate_decision);
CREATE INDEX IF NOT EXISTS idx_discovered_dags_created
    ON ml.discovered_dags(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_dags_algorithms
    ON ml.discovered_dags USING GIN(algorithms_used);

-- Algorithm runs indexes
CREATE INDEX IF NOT EXISTS idx_discovery_runs_dag
    ON ml.discovery_algorithm_runs(dag_id);
CREATE INDEX IF NOT EXISTS idx_discovery_runs_algorithm
    ON ml.discovery_algorithm_runs(algorithm);

-- Discovered edges indexes
CREATE INDEX IF NOT EXISTS idx_discovered_edges_dag
    ON ml.discovered_edges(dag_id);
CREATE INDEX IF NOT EXISTS idx_discovered_edges_source
    ON ml.discovered_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_discovered_edges_target
    ON ml.discovered_edges(target_node);
CREATE INDEX IF NOT EXISTS idx_discovered_edges_confidence
    ON ml.discovered_edges(confidence DESC);

-- Driver rankings indexes
CREATE INDEX IF NOT EXISTS idx_driver_rankings_dag
    ON ml.driver_rankings(dag_id);
CREATE INDEX IF NOT EXISTS idx_driver_rankings_session
    ON ml.driver_rankings(session_id);
CREATE INDEX IF NOT EXISTS idx_driver_rankings_target
    ON ml.driver_rankings(target_variable);
CREATE INDEX IF NOT EXISTS idx_driver_rankings_created
    ON ml.driver_rankings(created_at DESC);

-- Feature rankings indexes
CREATE INDEX IF NOT EXISTS idx_feature_rankings_ranking
    ON ml.feature_rankings(ranking_id);
CREATE INDEX IF NOT EXISTS idx_feature_rankings_feature
    ON ml.feature_rankings(feature_name);
CREATE INDEX IF NOT EXISTS idx_feature_rankings_causal_rank
    ON ml.feature_rankings(causal_rank);
CREATE INDEX IF NOT EXISTS idx_feature_rankings_direct_cause
    ON ml.feature_rankings(is_direct_cause) WHERE is_direct_cause = TRUE;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION ml.update_discovered_dags_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_discovered_dags_timestamp ON ml.discovered_dags;
CREATE TRIGGER trigger_update_discovered_dags_timestamp
    BEFORE UPDATE ON ml.discovered_dags
    FOR EACH ROW
    EXECUTE FUNCTION ml.update_discovered_dags_timestamp();

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Recent discovery runs with summary
CREATE OR REPLACE VIEW ml.v_recent_discoveries AS
SELECT
    d.id,
    d.session_id,
    d.discovery_timestamp,
    d.n_samples,
    d.n_features,
    d.n_edges,
    d.gate_decision,
    d.gate_confidence,
    d.total_runtime_seconds,
    d.algorithms_used,
    d.ensemble_threshold,
    COUNT(DISTINCT ar.id) as n_algorithm_runs,
    AVG(ar.runtime_seconds) as avg_algorithm_runtime,
    SUM(CASE WHEN ar.converged THEN 1 ELSE 0 END) as n_converged
FROM ml.discovered_dags d
LEFT JOIN ml.discovery_algorithm_runs ar ON ar.dag_id = d.id
GROUP BY d.id
ORDER BY d.created_at DESC;

-- View: High-confidence edges
CREATE OR REPLACE VIEW ml.v_high_confidence_edges AS
SELECT
    e.id,
    e.dag_id,
    e.source_node,
    e.target_node,
    e.edge_type,
    e.confidence,
    e.algorithm_votes,
    e.algorithms,
    d.gate_decision,
    d.session_id
FROM ml.discovered_edges e
JOIN ml.discovered_dags d ON d.id = e.dag_id
WHERE e.confidence >= 0.8
ORDER BY e.confidence DESC;

-- View: Discordant features (large rank differences)
CREATE OR REPLACE VIEW ml.v_discordant_features AS
SELECT
    fr.id,
    fr.ranking_id,
    fr.feature_name,
    fr.causal_rank,
    fr.predictive_rank,
    fr.rank_difference,
    fr.causal_score,
    fr.predictive_score,
    fr.is_direct_cause,
    dr.target_variable,
    dr.session_id
FROM ml.feature_rankings fr
JOIN ml.driver_rankings dr ON dr.id = fr.ranking_id
WHERE ABS(fr.rank_difference) >= 3
ORDER BY ABS(fr.rank_difference) DESC;

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function: Get edges for a DAG with confidence filtering
CREATE OR REPLACE FUNCTION ml.get_dag_edges(
    p_dag_id UUID,
    p_min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    source_node VARCHAR(255),
    target_node VARCHAR(255),
    edge_type ml.edge_type,
    confidence FLOAT,
    algorithm_votes INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.source_node,
        e.target_node,
        e.edge_type,
        e.confidence,
        e.algorithm_votes
    FROM ml.discovered_edges e
    WHERE e.dag_id = p_dag_id
      AND e.confidence >= p_min_confidence
    ORDER BY e.confidence DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Get feature ranking comparison
CREATE OR REPLACE FUNCTION ml.get_feature_ranking_comparison(
    p_ranking_id UUID
)
RETURNS TABLE (
    feature_name VARCHAR(255),
    causal_rank INTEGER,
    predictive_rank INTEGER,
    rank_difference INTEGER,
    causal_score FLOAT,
    predictive_score FLOAT,
    classification TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fr.feature_name,
        fr.causal_rank,
        fr.predictive_rank,
        fr.rank_difference,
        fr.causal_score,
        fr.predictive_score,
        CASE
            WHEN ABS(fr.rank_difference) <= 2 THEN 'concordant'
            WHEN fr.rank_difference > 2 THEN 'predictive_overweighted'
            ELSE 'causal_overweighted'
        END as classification
    FROM ml.feature_rankings fr
    WHERE fr.ranking_id = p_ranking_id
    ORDER BY fr.causal_rank;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ROW LEVEL SECURITY (Optional - Enable if needed)
-- =============================================================================

-- Note: Uncomment if RLS is required for multi-tenant access
-- ALTER TABLE ml.discovered_dags ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml.discovery_algorithm_runs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml.discovered_edges ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml.driver_rankings ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml.feature_rankings ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Grant appropriate permissions (adjust role names as needed)
GRANT SELECT, INSERT, UPDATE ON ml.discovered_dags TO authenticated;
GRANT SELECT, INSERT ON ml.discovery_algorithm_runs TO authenticated;
GRANT SELECT, INSERT ON ml.discovered_edges TO authenticated;
GRANT SELECT, INSERT ON ml.driver_rankings TO authenticated;
GRANT SELECT, INSERT ON ml.feature_rankings TO authenticated;

GRANT SELECT ON ml.v_recent_discoveries TO authenticated;
GRANT SELECT ON ml.v_high_confidence_edges TO authenticated;
GRANT SELECT ON ml.v_discordant_features TO authenticated;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE ml.discovered_dags IS 'Stores discovered causal DAG structures from structure learning algorithms';
COMMENT ON TABLE ml.discovery_algorithm_runs IS 'Individual algorithm run results within a discovery session';
COMMENT ON TABLE ml.discovered_edges IS 'Edges in discovered DAGs with confidence metadata';
COMMENT ON TABLE ml.driver_rankings IS 'Causal vs predictive feature importance rankings';
COMMENT ON TABLE ml.feature_rankings IS 'Detailed per-feature ranking information';

COMMENT ON VIEW ml.v_recent_discoveries IS 'Summary view of recent causal discovery runs';
COMMENT ON VIEW ml.v_high_confidence_edges IS 'Edges with confidence >= 0.8';
COMMENT ON VIEW ml.v_discordant_features IS 'Features with large rank differences between causal and predictive';
