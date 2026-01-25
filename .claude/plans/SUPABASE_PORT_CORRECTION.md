# Supabase Self-Hosted Port Configuration Correction Plan

**Status**: ✅ COMPLETED (2026-01-25)
**Commit**: `dbf45c0`

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

## Files Updated (12 total)

### Category 1: Configuration Templates (3 files)

#### 1. `config/supabase_self_hosted.example.env` ✅
- **Change**: Updated `SUPABASE_URL` and `VITE_SUPABASE_URL` to use port 54321

#### 2. `docker/env.example` ✅
- **Change**: Updated Supabase URL port from 8000 to 54321

#### 3. `docker/supabase/docker-compose.override.yml` ✅
- **Change**: Updated Kong port mapping to `"54321:8000"` (external:internal)

### Category 2: Scripts (2 files)

#### 4. `scripts/supabase/validate_migration.py` ✅
- **Change**: Updated default API URL to port 54321

#### 5. `scripts/supabase/setup_self_hosted.sh` ✅
- **Change**: Updated KONG_HTTP_PORT, API_EXTERNAL_URL, SUPABASE_PUBLIC_URL to port 54321

#### ~~6. `docker/start.sh`~~ - Not applicable (file doesn't contain Supabase URLs)

### Category 3: Documentation (4 files)

#### 7. `INFRASTRUCTURE.md` ✅
- **Change**: Updated Supabase API port reference to 54321

#### ~~8. `README.md`~~ - Not applicable (no port 8000 references found)

#### 9. `docs/AUTHENTICATION.md` ✅
- **Change**: Updated auth endpoint URLs to port 54321

#### 10. `docs/FEATURE_STORE_QUICKSTART.md` ✅
- **Change**: Updated all Supabase URL references to port 54321

#### 11. `src/ml/data_loader.py` ✅
- **Change**: Updated all Supabase URL references in docstrings and comments

### Category 4: Source Code (2 files)

#### 12. `src/api/main.py` ✅
- **Change**: Updated CORS allowed origins to include `http://138.197.4.36:54321`

#### 13. `src/feature_store/client.py` ✅
- **Change**: Updated docstring example URL to port 54321

### Category 5: Tests (1 file)

#### 14. `tests/unit/test_api/test_cors_configuration.py` ✅
- **Change**: Updated test assertion to expect port 54321 in default origins

#### ~~15. `tests/integration/test_phase5_api.py`~~ - Not applicable (no port 8000 references)

---

## Implementation Steps - COMPLETED

### Step 1: Update Configuration Files ✅
1. ✅ Updated `config/supabase_self_hosted.example.env`
2. ✅ Updated `docker/env.example`
3. ✅ Updated `docker/supabase/docker-compose.override.yml` port mapping

### Step 2: Update Scripts ✅
1. ✅ Updated `scripts/supabase/validate_migration.py`
2. ✅ Updated `scripts/supabase/setup_self_hosted.sh`

### Step 3: Update Documentation ✅
1. ✅ Updated `INFRASTRUCTURE.md`
2. ✅ Updated `docs/AUTHENTICATION.md`
3. ✅ Updated `docs/FEATURE_STORE_QUICKSTART.md`
4. ✅ Updated `src/ml/data_loader.py`

### Step 4: Update Source Code ✅
1. ✅ Updated `src/api/main.py` CORS origins
2. ✅ Updated `src/feature_store/client.py`

### Step 5: Update Tests ✅
1. ✅ Updated `tests/unit/test_api/test_cors_configuration.py`

### Step 6: Verify Environment Variables ✅
Production `.env` confirmed:
```
SUPABASE_URL=http://localhost:54321
```
(Using localhost is correct for internal access on the droplet)

### Step 7: Test and Deploy ✅
1. ✅ Ran tests locally - 14/14 CORS tests passed
2. ✅ Deployed to droplet - `git pull` successful
3. ✅ Restarted API service - `sudo systemctl restart e2i-api`
4. ✅ Verified Supabase connectivity - HTTP 401 (auth required, expected)

---

## Port Reference Summary

| Context | Old Port | New Port | Notes |
|---------|----------|----------|-------|
| External Access | 8000 | 54321 | Kong API Gateway external |
| Internal Docker | 8000 | 8000 | Internal remains unchanged |
| PostgreSQL | 5432 | 5432 | No change needed |

---

## Verification Results ✅

| Test | Result | Notes |
|------|--------|-------|
| Local Supabase | ✅ HTTP 401 | `curl localhost:54321/rest/v1/` |
| External Supabase | ✅ HTTP 401 | `curl 138.197.4.36:54321/rest/v1/` |
| API Health | ✅ Healthy | `{"status":"healthy"}` |
| CORS Tests | ✅ 14/14 passed | All tests pass with port 54321 |
| Production .env | ✅ Verified | `SUPABASE_URL=http://localhost:54321` |

---

## Commit Details

```
Commit: dbf45c0
Author: Claude Opus 4.5 <noreply@anthropic.com>
Date: 2026-01-25

fix(config): correct Supabase port from 8000 to 54321 for self-hosted deployment

The self-hosted Supabase Kong API Gateway is exposed on external port 54321,
not port 8000 (which is the internal Docker port). This change updates all
configuration files, documentation, and source code to use the correct
external port for API access.

Files changed: 12
Insertions: 26
Deletions: 26
```
