# Supabase Self-Hosted Port Configuration Correction Plan

## Overview

Correct Supabase configuration throughout the codebase from cloud configuration (port 8000) to self-hosted configuration (port 54321).

### Self-Hosted Supabase Details

| Component | Value |
|-----------|-------|
| External URL | http://138.197.4.36:54321 |
| Internal URL | http://localhost:54321 |
| PostgreSQL | port 5432 |
| GoTrue Auth | v2.184.0 |

### Key Finding

The primary issue is a **port mismatch**: configuration files reference port 8000 (internal Docker port), but the self-hosted Supabase Kong API Gateway is exposed on port 54321 externally.

---

## Files Requiring Updates (14 total)

### Category 1: Configuration Templates (3 files)

#### 1. `config/supabase_self_hosted.example.env`
- **Issue**: May reference port 8000 instead of 54321
- **Change**: Update `SUPABASE_URL` to use port 54321

#### 2. `docker/env.example`
- **Issue**: Example environment may have incorrect port
- **Change**: Update Supabase URL port from 8000 to 54321

#### 3. `docker/supabase/docker-compose.override.yml`
- **Issue**: Kong API Gateway port mapping
- **Change**: Verify external port is 54321 (maps to internal 8000)

### Category 2: Scripts (3 files)

#### 4. `scripts/supabase/validate_migration.py`
- **Issue**: Hardcoded port 8000 for validation checks
- **Change**: Update to port 54321 for external access

#### 5. `scripts/supabase/setup_self_hosted.sh`
- **Issue**: Setup script may configure wrong port
- **Change**: Ensure port 54321 is used for self-hosted setup

#### 6. `docker/start.sh`
- **Issue**: Startup script port references
- **Change**: Update health checks and URLs to port 54321

### Category 3: Documentation (4 files)

#### 7. `INFRASTRUCTURE.md`
- **Issue**: Documentation may reference incorrect port
- **Change**: Update all Supabase URL references to port 54321

#### 8. `README.md`
- **Issue**: Setup instructions may have wrong port
- **Change**: Update Supabase connection instructions

#### 9. `docs/AUTHENTICATION.md`
- **Issue**: Auth endpoint documentation
- **Change**: Update GoTrue/Auth URLs to port 54321

#### 10. `scripts/data_loader.py`
- **Issue**: Data loading script Supabase connection
- **Change**: Update connection URL to port 54321

### Category 4: Source Code (2 files)

#### 11. `src/api/main.py`
- **Issue**: CORS configuration may include port 8000
- **Change**: Update CORS allowed origins to include port 54321

#### 12. `src/feature_store/client.py`
- **Issue**: Feature store Supabase client connection
- **Change**: Ensure URL uses port 54321

### Category 5: Tests (2 files)

#### 13. `tests/integration/test_cors_configuration.py`
- **Issue**: CORS tests may expect port 8000
- **Change**: Update test expectations to port 54321

#### 14. `tests/integration/test_phase5_api.py`
- **Issue**: API integration tests Supabase URLs
- **Change**: Update test URLs to port 54321

---

## Implementation Steps

### Step 1: Update Configuration Files
1. Read and update `config/supabase_self_hosted.example.env`
2. Read and update `docker/env.example`
3. Read and verify `docker/supabase/docker-compose.override.yml` port mapping

### Step 2: Update Scripts
1. Update `scripts/supabase/validate_migration.py`
2. Update `scripts/supabase/setup_self_hosted.sh`
3. Update `docker/start.sh`

### Step 3: Update Documentation
1. Update `INFRASTRUCTURE.md`
2. Update `README.md`
3. Update `docs/AUTHENTICATION.md`
4. Update `scripts/data_loader.py`

### Step 4: Update Source Code
1. Update `src/api/main.py` CORS origins
2. Update `src/feature_store/client.py`

### Step 5: Update Tests
1. Update `tests/integration/test_cors_configuration.py`
2. Update `tests/integration/test_phase5_api.py`

### Step 6: Verify Environment Variables
Ensure production `.env` file has:
```
SUPABASE_URL=http://138.197.4.36:54321
# or for internal Docker access:
SUPABASE_URL=http://localhost:54321
```

### Step 7: Test and Deploy
1. Run tests locally
2. Deploy to droplet
3. Verify Supabase connectivity
4. Test authentication flow
5. Test data operations

---

## Port Reference Summary

| Context | Old Port | New Port | Notes |
|---------|----------|----------|-------|
| External Access | 8000 | 54321 | Kong API Gateway external |
| Internal Docker | 8000 | 8000 | Internal remains unchanged |
| PostgreSQL | 5432 | 5432 | No change needed |

---

## Risk Assessment

- **Low Risk**: Documentation and example files
- **Medium Risk**: Scripts and test files
- **Higher Risk**: Source code (CORS, client connections)

All changes should be tested before production deployment.

---

## Verification

After implementation:
1. Test Supabase connectivity: `curl http://138.197.4.36:54321/rest/v1/`
2. Test auth endpoint: `curl http://138.197.4.36:54321/auth/v1/health`
3. Run integration tests
4. Verify frontend can connect to Supabase
5. Test a full authentication flow
