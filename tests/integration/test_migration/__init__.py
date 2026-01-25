"""
Supabase Migration Validation Test Suite.

This package contains integration tests to validate successful migration
from Supabase Cloud to self-hosted infrastructure.

Tests are organized into phases:
- Phase 1: Connectivity (database connection, extensions)
- Phase 2: Schema (tables, enums, views, indexes)
- Phase 3: Data (row counts, data integrity)
- Phase 4: Auth (auth schema, users)
- Phase 5: API (PostgREST, E2I API endpoints)

Run all tests:
    pytest tests/integration/test_migration/ -v -n 1

Run individual phases:
    pytest tests/integration/test_migration/test_phase1_connectivity.py -v
"""
