# Supabase Migration Validation Test Suite

**Date**: 2026-01-25
**Goal**: Create comprehensive tests to validate successful migration from Supabase Cloud to self-hosted

---

## Migration Requirements to Verify

| Metric | Expected Value |
|--------|----------------|
| Public tables | 84 |
| Auth tables | 20 |
| Auth users | 6 |
| agent_registry rows | 11 |
| causal_paths rows | 50 |
| agent_tier_mapping rows | 21 |

---

## File Structure

```
tests/integration/test_migration/
├── __init__.py
├── conftest.py                    # Fixtures & expected counts
├── test_phase1_connectivity.py    # DB connection & extensions
├── test_phase2_schema.py          # Table/enum/view validation
├── test_phase3_data.py            # Row counts & data integrity
├── test_phase4_auth.py            # Auth schema & users
└── test_phase5_api.py             # API endpoint validation
```

---

## Implementation Phases

### Wave 1: Setup (conftest.py)
Create fixtures for:
- `pg_connection` - Direct PostgreSQL connection via psycopg2
- `supabase_client` - Supabase Python client
- `expected_counts` - Dictionary of expected row counts
- Environment variable handling with defaults
- Skip markers for service availability

### Wave 2: Connectivity Tests (test_phase1_connectivity.py)
~5 tests, 1 min runtime

| Test | Assertion |
|------|-----------|
| `test_database_connection` | PostgreSQL connection succeeds |
| `test_database_version` | PostgreSQL >= 15.0 |
| `test_extension_uuid_ossp` | uuid-ossp installed |
| `test_extension_pgcrypto` | pgcrypto installed |
| `test_supabase_api_reachable` | Kong API responds |

### Wave 3: Schema Tests (test_phase2_schema.py)
~10 tests, 2 min runtime

| Test | Assertion |
|------|-----------|
| `test_public_table_count` | >= 84 tables in public schema |
| `test_auth_table_count` | >= 20 tables in auth schema |
| `test_core_tables_exist` | agent_registry, causal_paths, agent_tier_mapping exist |
| `test_agent_registry_columns` | All expected columns present |
| `test_causal_paths_columns` | All expected columns present |
| `test_enum_types_exist` | data_split_type, brand_type, agent_tier_type |
| `test_split_aware_views` | v_train_*, v_test_*, v_holdout_* views |
| `test_indexes_exist` | Critical indexes present |

### Wave 4: Data Tests (test_phase3_data.py)
~8 tests, 2 min runtime

| Test | Assertion |
|------|-----------|
| `test_agent_registry_count` | == 11 rows |
| `test_agent_tier_mapping_count` | == 21 rows |
| `test_causal_paths_count` | >= 50 rows |
| `test_agent_registry_has_orchestrator` | orchestrator agent exists |
| `test_agent_registry_has_all_tiers` | Tiers 1-5 represented |
| `test_tier_mapping_covers_all_tiers` | tier_0 through tier_5 |
| `test_causal_paths_have_data` | causal_chain JSONB populated |
| `test_timestamps_valid` | created_at values reasonable |

### Wave 5: Auth Tests (test_phase4_auth.py)
~5 tests, 1 min runtime

| Test | Assertion |
|------|-----------|
| `test_auth_schema_exists` | auth schema present |
| `test_auth_users_count` | == 6 users |
| `test_auth_users_have_email` | All users have email |
| `test_gotrue_health` | GoTrue API responds |
| `test_auth_admin_api` | Admin can list users |

### Wave 6: API Tests (test_phase5_api.py)
~5 tests, 2 min runtime

| Test | Assertion |
|------|-----------|
| `test_postgrest_health` | REST API responds |
| `test_query_agent_registry_via_api` | Can SELECT via PostgREST |
| `test_e2i_api_health` | E2I FastAPI healthy |
| `test_e2i_kpi_endpoint` | /api/kpis returns data |
| `test_copilotkit_endpoint` | /api/copilotkit returns actions |

---

## Droplet Execution Commands

**Important**: Tests require environment variables. The supabase-db container is at Docker IP 172.22.0.4.

```bash
# Run all phases (uses .env from droplet for keys)
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  source .env && \
  SUPABASE_DB_HOST=172.22.0.4 \
  .venv/bin/pytest tests/integration/test_migration/ -v --tb=short"

# Or with explicit environment variables:
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  SUPABASE_DB_HOST=172.22.0.4 \
  POSTGRES_PASSWORD=<pw> \
  SUPABASE_URL=http://localhost:54321 \
  SUPABASE_ANON_KEY=<anon_key> \
  SUPABASE_SERVICE_KEY=<service_key> \
  .venv/bin/pytest tests/integration/test_migration/ -v --tb=short"
```

---

## Critical Files to Reference

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Service availability pattern |
| `scripts/supabase/validate_migration.py` | SQL queries to adapt |
| `tests/integration/test_shap_analysis_repository.py` | Supabase skip pattern |
| `database/core/e2i_ml_complete_v3_schema.sql` | Expected schema |

---

## To-Do List

### Wave 1: Setup
- [x] Create `tests/integration/test_migration/__init__.py`
- [x] Create `tests/integration/test_migration/conftest.py` with fixtures
- [x] Use psycopg2 (installed on droplet) for PostgreSQL connection
- [x] Configure Docker container IP (172.22.0.4) for direct DB access

### Wave 2: Phase 1 Tests
- [x] Create `test_phase1_connectivity.py`
- [x] Run on droplet, verify all pass (6/6 PASSED)

### Wave 3: Phase 2 Tests
- [x] Create `test_phase2_schema.py`
- [x] Run on droplet, verify all pass (11/11 PASSED)
- [x] Fixed: Updated essential columns for agent_registry (agent_name not id)
- [x] Fixed: Updated essential columns for causal_paths (path_id not id)
- [x] Fixed: Updated CORE_TABLES to use `chatbot_conversations` (actual table name)

### Wave 4: Phase 3 Tests
- [x] Create `test_phase3_data.py`
- [x] Run on droplet, verify all pass (9/9 PASSED)

### Wave 5: Phase 4 Tests
- [x] Create `test_phase4_auth.py`
- [x] Run on droplet, verify all pass (7/7 PASSED)
- [x] Fixed: Accept 401 status from GoTrue (Kong requires auth for self-hosted)

### Wave 6: Phase 5 Tests
- [x] Create `test_phase5_api.py`
- [x] Run on droplet, verify all pass (7/7 PASSED)
- [x] Fixed: Accept 401 status for PostgREST and OpenAPI endpoints (Kong auth)

### Wave 7: Finalize
- [x] Run full test suite on droplet (40/40 PASSED)
- [x] Commit and push
- [x] Sync to droplet

### Wave 8: Phase 6 - pgvector Extension (Added 2026-01-25)
- [x] Create `test_phase6_pgvector.py`
- [x] Apply RAG schema migration (`database/rag/001_rag_schema.sql`)
- [x] Run on droplet, verify all pass (23/23 PASSED)

---

## Verification

After implementation, run:
```bash
# Full validation (should take ~8 minutes)
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  source .env && SUPABASE_DB_HOST=172.22.0.4 \
  .venv/bin/pytest tests/integration/test_migration/ -v --tb=short"
```

Expected output: **63 passed** ✅ (Verified 2026-01-25)
- Phase 1-5: 40 tests
- Phase 6 (pgvector): 23 tests
