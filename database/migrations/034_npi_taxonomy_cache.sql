-- =============================================================================
-- Migration 034: NPPES NPI taxonomy local cache
-- =============================================================================
-- Adds `npi_taxonomy` table caching CMS NPPES (National Plan and Provider
-- Enumeration System) records, ingested from the monthly bulk dump
-- (https://download.cms.gov/nppes/NPI_Files.html). The cache backs the
-- `scripts.rwd_common.lookup_npi` helper which converters and downstream
-- consumers use to enrich HCP profiles, sharpen provider-mix features,
-- and tag specialty-pharmacy / site-of-care signals.
--
-- Schema rationale:
--   * `npi`            PRIMARY KEY -- NPI is the canonical CMS identifier.
--   * `entity_type`    '1' (Individual) or '2' (Organization) per NPPES spec.
--   * `enumeration_date` issuance date; powers `years_experience` derivation.
--   * `taxonomies`     JSONB array of {code, desc, primary, license, state};
--                      query via @> / -> operators for code matching.
--   * `practice_address` JSONB {address_1, city, state, postal_code, country};
--                      structured so callers can read just zip prefix etc.
--   * `parent_organization_legal_name` TEXT -- powers `affiliation_primary`.
--   * `sole_proprietor` BOOLEAN -- powers `practice_size`/`academic_hcp` flag.
--   * `last_updated_npes` DATE -- vendor-side update timestamp from NPPES.
--   * `cached_at`      TIMESTAMPTZ DEFAULT NOW() -- local ingestion timestamp;
--                      callers use this to detect staleness (>30d → re-pull).
--
-- Indexes:
--   * PK on npi (already).
--   * GIN on taxonomies for fast taxonomy-code matching in provider-mix code.
--   * Btree on entity_type + sole_proprietor for org-vs-individual filters.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS so this migration is safe to re-run
-- on environments that pre-built the table out-of-band.
-- =============================================================================

CREATE TABLE IF NOT EXISTS npi_taxonomy (
    npi                              VARCHAR(10) PRIMARY KEY,
    entity_type                      VARCHAR(1)  NULL,
    enumeration_date                 DATE        NULL,
    last_updated_nppes               DATE        NULL,
    taxonomies                       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    practice_address                 JSONB       NULL,
    parent_organization_legal_name   TEXT        NULL,
    organization_legal_name          TEXT        NULL,
    sole_proprietor                  BOOLEAN     NULL,
    first_name                       TEXT        NULL,
    last_name                        TEXT        NULL,
    cached_at                        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source                           TEXT        NOT NULL DEFAULT 'bulk_dump'
        CHECK (source IN ('bulk_dump', 'api_fallback', 'fixture'))
);

CREATE INDEX IF NOT EXISTS idx_npi_taxonomy_taxonomies_gin
    ON npi_taxonomy USING GIN (taxonomies);

CREATE INDEX IF NOT EXISTS idx_npi_taxonomy_entity_type
    ON npi_taxonomy (entity_type, sole_proprietor);

CREATE INDEX IF NOT EXISTS idx_npi_taxonomy_cached_at
    ON npi_taxonomy (cached_at);

-- =============================================================================
-- VERIFICATION (run after migration):
--   SELECT COUNT(*) FROM npi_taxonomy;
--   \d+ npi_taxonomy
-- Expected: 0 rows initially; populated by
--   `src.tasks.nppes_tasks.refresh_npi_taxonomy_cache` (monthly Celery beat).
-- =============================================================================
