-- ==============================================================================
-- E2I Causal Analytics - Feature Store Schema Migration
-- ==============================================================================
-- Creates lightweight feature store tables for managing features in Supabase
-- Designed to replace Feast with a simpler, integrated solution
-- ==============================================================================

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS feature_values CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS feature_groups CASCADE;
DROP TYPE IF EXISTS feature_value_type CASCADE;
DROP TYPE IF EXISTS feature_freshness_status CASCADE;

-- ==============================================================================
-- Enums
-- ==============================================================================

-- Feature data types
CREATE TYPE feature_value_type AS ENUM (
    'int64',
    'float64',
    'string',
    'bool',
    'timestamp',
    'array_int64',
    'array_float64',
    'array_string'
);

-- Feature freshness status
CREATE TYPE feature_freshness_status AS ENUM (
    'fresh',      -- Within SLA
    'stale',      -- Outside SLA but usable
    'expired'     -- Too old, should not be used
);

-- ==============================================================================
-- Feature Groups Table
-- ==============================================================================
-- Logical grouping of related features (e.g., "hcp_demographics", "brand_metrics")

CREATE TABLE feature_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,

    -- Ownership and organization
    owner VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,

    -- Data source configuration
    source_table VARCHAR(255),           -- Source table in Supabase
    source_query TEXT,                   -- Custom query for feature generation

    -- Freshness configuration
    expected_update_frequency_hours INTEGER DEFAULT 24,
    max_age_hours INTEGER DEFAULT 168,  -- 7 days default

    -- Metadata
    schema_version VARCHAR(50) DEFAULT '1.0.0',
    mlflow_experiment_id VARCHAR(255),   -- Link to MLflow experiment

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes
    CONSTRAINT valid_update_frequency CHECK (expected_update_frequency_hours > 0),
    CONSTRAINT valid_max_age CHECK (max_age_hours > 0)
);

CREATE INDEX idx_feature_groups_name ON feature_groups(name);
CREATE INDEX idx_feature_groups_owner ON feature_groups(owner);
CREATE INDEX idx_feature_groups_tags ON feature_groups USING GIN(tags);

COMMENT ON TABLE feature_groups IS 'Logical grouping of related features with metadata and freshness configuration';

-- ==============================================================================
-- Features Table
-- ==============================================================================
-- Individual feature definitions

CREATE TABLE features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_group_id UUID NOT NULL REFERENCES feature_groups(id) ON DELETE CASCADE,

    -- Feature identification
    name VARCHAR(255) NOT NULL,
    description TEXT,
    value_type feature_value_type NOT NULL,

    -- Entity keys (what this feature describes)
    -- e.g., ["hcp_id"], ["brand_id", "geography_id"]
    entity_keys JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Feature computation
    computation_query TEXT,              -- SQL query to compute this feature
    dependencies JSONB DEFAULT '[]'::jsonb,  -- List of dependent feature IDs

    -- Statistics and monitoring
    statistics JSONB DEFAULT '{}'::jsonb,    -- Min, max, mean, std, etc.
    drift_threshold FLOAT DEFAULT 0.1,       -- Threshold for drift detection

    -- Metadata
    owner VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    version VARCHAR(50) DEFAULT '1.0.0',

    -- MLflow tracking
    mlflow_run_id VARCHAR(255),          -- Link to MLflow run that created this feature

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_feature_per_group UNIQUE(feature_group_id, name),
    CONSTRAINT valid_entity_keys CHECK (jsonb_array_length(entity_keys) > 0)
);

CREATE INDEX idx_features_name ON features(name);
CREATE INDEX idx_features_group ON features(feature_group_id);
CREATE INDEX idx_features_owner ON features(owner);
CREATE INDEX idx_features_tags ON features USING GIN(tags);
CREATE INDEX idx_features_entity_keys ON features USING GIN(entity_keys);

COMMENT ON TABLE features IS 'Individual feature definitions with computation logic and metadata';

-- ==============================================================================
-- Feature Values Table
-- ==============================================================================
-- Time-series storage of actual feature values

CREATE TABLE feature_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_id UUID NOT NULL REFERENCES features(id) ON DELETE CASCADE,

    -- Entity identification
    -- e.g., {"hcp_id": "HCP123", "brand_id": "remibrutinib"}
    entity_values JSONB NOT NULL,

    -- Feature value (stored as JSONB for flexibility)
    value JSONB NOT NULL,

    -- Timestamps
    event_timestamp TIMESTAMPTZ NOT NULL,   -- When the event occurred
    created_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When stored in feature store

    -- Freshness tracking
    freshness_status feature_freshness_status DEFAULT 'fresh',

    -- Metadata
    source_job_id VARCHAR(255),         -- ID of job that created this value
    version INTEGER DEFAULT 1,

    -- Constraints
    CONSTRAINT valid_event_timestamp CHECK (event_timestamp <= NOW()),
    CONSTRAINT feature_entity_timestamp_unique UNIQUE(feature_id, entity_values, event_timestamp)
);

-- Partitioning by event_timestamp for performance (monthly partitions)
-- Note: In production, implement table partitioning strategy

-- Indexes for fast retrieval
CREATE INDEX idx_feature_values_feature ON feature_values(feature_id);
CREATE INDEX idx_feature_values_entity ON feature_values USING GIN(entity_values);
CREATE INDEX idx_feature_values_timestamp ON feature_values(event_timestamp DESC);
CREATE INDEX idx_feature_values_freshness ON feature_values(freshness_status);
CREATE INDEX idx_feature_values_composite ON feature_values(feature_id, event_timestamp DESC);

COMMENT ON TABLE feature_values IS 'Time-series storage of actual feature values with freshness tracking';

-- ==============================================================================
-- Views for Common Queries
-- ==============================================================================

-- Latest feature values per entity (for online serving)
CREATE OR REPLACE VIEW feature_values_latest AS
SELECT DISTINCT ON (feature_id, entity_values)
    fv.*,
    f.name as feature_name,
    f.value_type,
    fg.name as feature_group_name
FROM feature_values fv
JOIN features f ON fv.feature_id = f.id
JOIN feature_groups fg ON f.feature_group_id = fg.id
ORDER BY feature_id, entity_values, event_timestamp DESC;

COMMENT ON VIEW feature_values_latest IS 'Most recent feature value for each entity (online serving)';

-- Feature freshness monitoring
CREATE OR REPLACE VIEW feature_freshness_monitor AS
SELECT
    fg.name as feature_group,
    f.name as feature_name,
    f.id as feature_id,
    COUNT(*) as total_entities,
    COUNT(*) FILTER (WHERE fv.freshness_status = 'fresh') as fresh_count,
    COUNT(*) FILTER (WHERE fv.freshness_status = 'stale') as stale_count,
    COUNT(*) FILTER (WHERE fv.freshness_status = 'expired') as expired_count,
    MAX(fv.event_timestamp) as latest_event,
    MAX(fv.created_timestamp) as latest_created,
    NOW() - MAX(fv.event_timestamp) as time_since_last_event
FROM features f
JOIN feature_groups fg ON f.feature_group_id = fg.id
LEFT JOIN feature_values_latest fv ON f.id = fv.feature_id
GROUP BY fg.name, f.name, f.id
ORDER BY time_since_last_event DESC;

COMMENT ON VIEW feature_freshness_monitor IS 'Monitor feature freshness across all features';

-- ==============================================================================
-- Functions
-- ==============================================================================

-- Function to update feature freshness status
CREATE OR REPLACE FUNCTION update_feature_freshness()
RETURNS void AS $$
BEGIN
    UPDATE feature_values fv
    SET freshness_status =
        CASE
            WHEN NOW() - fv.event_timestamp <= (f.fg_max_age * INTERVAL '1 hour') * 0.5 THEN 'fresh'::feature_freshness_status
            WHEN NOW() - fv.event_timestamp <= (f.fg_max_age * INTERVAL '1 hour') THEN 'stale'::feature_freshness_status
            ELSE 'expired'::feature_freshness_status
        END
    FROM (
        SELECT
            features.id as feature_id,
            feature_groups.max_age_hours as fg_max_age
        FROM features
        JOIN feature_groups ON features.feature_group_id = feature_groups.id
    ) f
    WHERE fv.feature_id = f.feature_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_feature_freshness() IS 'Update freshness status for all feature values based on age';

-- Function to get feature values for a specific entity
CREATE OR REPLACE FUNCTION get_entity_features(
    p_entity_values JSONB,
    p_feature_group_name VARCHAR DEFAULT NULL,
    p_include_stale BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(
    feature_name VARCHAR,
    feature_group VARCHAR,
    value JSONB,
    event_timestamp TIMESTAMPTZ,
    freshness_status feature_freshness_status
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.name::VARCHAR as feature_name,
        fg.name::VARCHAR as feature_group,
        fv.value,
        fv.event_timestamp,
        fv.freshness_status
    FROM feature_values_latest fv
    JOIN features f ON fv.feature_id = f.id
    JOIN feature_groups fg ON f.feature_group_id = fg.id
    WHERE fv.entity_values = p_entity_values
        AND (p_feature_group_name IS NULL OR fg.name = p_feature_group_name)
        AND (p_include_stale OR fv.freshness_status = 'fresh')
    ORDER BY fg.name, f.name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_entity_features IS 'Retrieve all features for a specific entity with freshness filtering';

-- ==============================================================================
-- Triggers
-- ==============================================================================

-- Update updated_at timestamp on feature groups
CREATE OR REPLACE FUNCTION update_feature_group_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_feature_group_timestamp
    BEFORE UPDATE ON feature_groups
    FOR EACH ROW
    EXECUTE FUNCTION update_feature_group_timestamp();

-- Update updated_at timestamp on features
CREATE OR REPLACE FUNCTION update_feature_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_feature_timestamp
    BEFORE UPDATE ON features
    FOR EACH ROW
    EXECUTE FUNCTION update_feature_timestamp();

-- ==============================================================================
-- Seed Data (Example)
-- ==============================================================================

-- Example: HCP Demographics Feature Group
INSERT INTO feature_groups (name, description, owner, source_table, expected_update_frequency_hours, max_age_hours, tags)
VALUES (
    'hcp_demographics',
    'Healthcare provider demographic features',
    'data-team',
    'hcps',
    168,  -- Weekly updates
    720,  -- 30 days max age
    '["demographics", "hcp", "core"]'::jsonb
);

-- Example: Brand Performance Feature Group
INSERT INTO feature_groups (name, description, owner, source_table, expected_update_frequency_hours, max_age_hours, tags)
VALUES (
    'brand_performance',
    'Brand-level performance metrics',
    'analytics-team',
    'brand_metrics',
    24,   -- Daily updates
    168,  -- 7 days max age
    '["brand", "performance", "metrics"]'::jsonb
);

-- Example features for HCP Demographics
INSERT INTO features (feature_group_id, name, description, value_type, entity_keys, owner, tags)
SELECT
    id,
    'specialty',
    'Primary medical specialty of HCP',
    'string'::feature_value_type,
    '["hcp_id"]'::jsonb,
    'data-team',
    '["categorical"]'::jsonb
FROM feature_groups WHERE name = 'hcp_demographics';

INSERT INTO features (feature_group_id, name, description, value_type, entity_keys, owner, tags)
SELECT
    id,
    'years_in_practice',
    'Number of years HCP has been practicing',
    'int64'::feature_value_type,
    '["hcp_id"]'::jsonb,
    'data-team',
    '["numerical"]'::jsonb
FROM feature_groups WHERE name = 'hcp_demographics';

-- Example features for Brand Performance
INSERT INTO features (feature_group_id, name, description, value_type, entity_keys, owner, tags)
SELECT
    id,
    'total_trx_30d',
    'Total prescriptions in last 30 days',
    'int64'::feature_value_type,
    '["brand_id", "geography_id"]'::jsonb,
    'analytics-team',
    '["aggregate", "rx"]'::jsonb
FROM feature_groups WHERE name = 'brand_performance';

-- ==============================================================================
-- Permissions (RLS - Row Level Security)
-- ==============================================================================

-- Enable RLS on all tables
ALTER TABLE feature_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE features ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_values ENABLE ROW LEVEL SECURITY;

-- Policies for authenticated users (adjust based on your auth setup)
-- Allow all operations for authenticated users for now
-- TODO: Implement fine-grained permissions based on roles

CREATE POLICY "Allow authenticated read access to feature_groups"
    ON feature_groups FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated read access to features"
    ON features FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated read access to feature_values"
    ON feature_values FOR SELECT
    TO authenticated
    USING (true);

-- ==============================================================================
-- Indexes for Performance
-- ==============================================================================

-- Additional indexes for common query patterns
CREATE INDEX idx_feature_values_latest_lookup
    ON feature_values(feature_id, entity_values, event_timestamp DESC)
    WHERE freshness_status IN ('fresh', 'stale');

COMMENT ON INDEX idx_feature_values_latest_lookup IS 'Optimized index for latest value lookups in online serving';

-- ==============================================================================
-- Statistics
-- ==============================================================================

-- Analyze tables for query optimization
ANALYZE feature_groups;
ANALYZE features;
ANALYZE feature_values;

-- ==============================================================================
-- End of Migration
-- ==============================================================================
