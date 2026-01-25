# Supabase Self-Hosted Migration - Implementation Plan

**Date**: 2025-01-25
**Status**: Ready for Execution
**Target Environment**: DigitalOcean Droplet (138.197.4.36)

---

## Executive Summary

Migrate E2I Causal Analytics from Supabase Cloud to self-hosted Supabase on the existing production droplet. The infrastructure (scripts, Docker configs, validation tools) already exists and is ready for execution.

---

## Prerequisites

### Required Credentials
| Item | Description | Status |
|------|-------------|--------|
| **Supabase Project Ref** | Your cloud project reference (e.g., `abcdefghijklmnop`) | ✅ In `.env` file |
| **Supabase DB Password** | Database password from Supabase Dashboard > Settings > Database | ✅ In `.env` file |
| **Supabase Connection String** | Full PostgreSQL URL (alternative to above) | ✅ In `.env` file |

> **Note**: Credentials are stored in the project `.env` file. Will be used during execution.

### Configuration Decisions
| Question | Options | Your Choice |
|----------|---------|-------------|
| **Export Method** | CLI (recommended) or pg_dump direct | ✅ Supabase CLI |
| **Auth Usage** | Do you use Supabase Auth (GoTrue) for user authentication? | ✅ YES (verified in code) |
| **Storage Usage** | Do you use Supabase Storage buckets for files? | ❌ NO (not used) |
| **Realtime Usage** | Do you use Supabase Realtime subscriptions? | ❌ NO (not used) |
| **Downtime Window** | Can we have 1-2 hours of planned downtime? | ✅ 1-2 hours OK |
| **Rollback Strategy** | Keep cloud active for 48h as fallback? | ✅ Keep until confirmed |

### Access Verification
- [x] SSH access to droplet confirmed (just verified - working)
- [x] Docker running on droplet (14 containers healthy)
- [x] Sufficient disk space (124 GB available - ample)

---

## Verified Supabase Feature Usage (Code Analysis)

| Feature | Status | Evidence |
|---------|--------|----------|
| **Auth (GoTrue)** | ✅ HEAVILY USED | Backend: `src/api/dependencies/auth.py` (JWT verification, RBAC). Frontend: `AuthProvider.tsx` (sign in/up/out, password reset, session management) |
| **Database (PostgREST)** | ✅ HEAVILY USED | 67+ repository files in `src/repositories/`. Repository pattern with split-aware ML data loading |
| **Storage** | ❌ NOT USED | No `.storage()`, `.bucket()`, or file operations found |
| **Realtime** | ❌ NOT USED | No `.channel()`, `postgres_changes`, or subscriptions found |

### Auth Migration Implications
Since Auth is heavily used, the migration must:
1. **Preserve JWT_SECRET consistency** - All services must use the same secret
2. **Migrate user data** - Export/import `auth.users` table if users exist
3. **Update SITE_URL** - Point to droplet IP for redirect URLs
4. **Test all auth flows** - Sign in, sign up, password reset, token refresh

---

## Implementation Phases

### Phase 1: Cloud Export (Local Machine)
**Context Size**: Small (~2k tokens)
**Duration**: 15-30 min

**Tasks**:
1. Install Supabase CLI (if not present)
2. Link to cloud project
3. Run `scripts/supabase/export_cloud_db.sh`
4. Verify export files created

**Verification**:
```bash
ls -la backups/supabase_export_*/
# Should see: schema.sql, data.sql, roles.sql, manifest.json
```

**Outputs**: `backups/supabase_export_<timestamp>/`

---

### Phase 2: Self-Hosted Setup (Droplet)
**Context Size**: Small (~3k tokens)
**Duration**: 20-30 min

**Tasks**:
1. Clone Supabase repo to `/opt/supabase`
2. Generate secrets (JWT_SECRET, POSTGRES_PASSWORD)
3. Generate API keys via Supabase key generator
4. Configure `/opt/supabase/docker/.env`
5. **Configure Auth settings** (SITE_URL, redirect URLs)
6. Start Supabase services

**Commands**:
```bash
# On droplet
./scripts/supabase/setup_self_hosted.sh --generate-keys
# Copy generated secrets, then generate API keys at:
# https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys
./docker/supabase/start.sh
```

**Auth Configuration** (in `/opt/supabase/docker/.env`):
```env
SITE_URL=http://138.197.4.36
API_EXTERNAL_URL=http://138.197.4.36:8000
ADDITIONAL_REDIRECT_URLS=http://localhost:5174,http://138.197.4.36
GOTRUE_EXTERNAL_EMAIL_ENABLED=true
GOTRUE_MAILER_AUTOCONFIRM=false  # Set false for production
```

**Verification**:
```bash
docker ps | grep supabase
curl -s http://localhost:8000/rest/v1/ -H "apikey: <ANON_KEY>"
curl -s http://localhost:8000/auth/v1/health  # Auth service health
```

---

### Phase 3: Data Import (Droplet)
**Context Size**: Medium (~4k tokens)
**Duration**: 30-60 min (depends on data size)

**Tasks**:
1. Transfer export files to droplet
2. Run `scripts/supabase/import_data.sh`
3. Apply E2I-specific migrations
4. Verify table counts and data integrity

**Commands**:
```bash
# Transfer files
scp -r backups/supabase_export_* enunez@138.197.4.36:~/

# On droplet
export POSTGRES_PASSWORD="<your-password>"
./scripts/supabase/import_data.sh ~/supabase_export_<timestamp>
```

**Verification**:
```bash
python scripts/supabase/validate_migration.py --verbose
```

---

### Phase 4: Application Configuration (Droplet)
**Context Size**: Small (~2k tokens)
**Duration**: 15-20 min

**Tasks**:
1. Update backend `.env` with self-hosted URLs
2. Update frontend environment variables
3. Restart E2I API service
4. Verify API health

**Environment Changes**:
```env
# Backend .env
SUPABASE_URL=http://localhost:8000
SUPABASE_ANON_KEY=<new-anon-key>
SUPABASE_SERVICE_KEY=<new-service-key>
DATABASE_URL=postgresql://postgres:<password>@localhost:5433/postgres

# Frontend .env
VITE_SUPABASE_URL=http://138.197.4.36:8000
VITE_SUPABASE_ANON_KEY=<new-anon-key>
```

**Verification**:
```bash
sudo systemctl restart e2i-api
curl http://localhost:8000/health
```

---

### Phase 5: Validation & Testing (Droplet)
**Context Size**: Medium (~3k tokens)
**Duration**: 30-60 min

**Tasks**:
1. Run full validation script
2. Test API endpoints
3. Test frontend connectivity
4. Verify agent functionality
5. Check MLflow/Opik integration

**Test Commands**:
```bash
# Full validation
python scripts/supabase/validate_migration.py -o validation_report.json

# API test
curl -X POST http://localhost:8000/api/health/deep

# Agent test
curl -X POST http://localhost:8000/api/agents/orchestrator/health
```

**Checklist**:
- [ ] Database connection working
- [ ] All tables present
- [ ] Row counts match expected
- [ ] RLS policies intact
- [ ] API endpoints responding
- [ ] **Auth: Sign in working**
- [ ] **Auth: Sign up working**
- [ ] **Auth: Token refresh working**
- [ ] **Auth: Password reset flow**
- [ ] Frontend loading
- [ ] Agent routing working

---

### Phase 6: Cutover & Monitoring
**Context Size**: Small (~2k tokens)
**Duration**: 15-30 min

**Tasks**:
1. Update firewall rules (if needed)
2. Verify external access
3. Set up backup cron job
4. Document completion
5. Monitor for 24h

**Backup Setup**:
```bash
# Add to crontab
0 2 * * * docker exec supabase-db pg_dump -U postgres postgres | gzip > /opt/backups/supabase_$(date +\%Y\%m\%d).sql.gz
```

**Post-Migration**:
- Keep cloud instance for 48h as rollback option
- Monitor error logs
- Verify all agent functionality

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Data loss | Multiple backups, validate before cutover |
| Extended downtime | Test migration on dev first if concerned |
| Port conflicts | Using 5433/3001 to avoid conflicts |
| Auth issues | Verify JWT_SECRET consistency across services |

---

## Rollback Plan

If issues occur:
1. Revert environment variables to Supabase Cloud URLs
2. Restart E2I API: `sudo systemctl restart e2i-api`
3. Cloud data unchanged, immediate rollback possible

---

## Files to Modify

| File | Change |
|------|--------|
| `/opt/e2i_causal_analytics/.env` | Supabase URLs and keys |
| `/opt/supabase/docker/.env` | Supabase self-hosted config |
| `frontend/.env` | Frontend Supabase URLs |

---

## Existing Infrastructure (Ready to Use)

### Scripts
- `scripts/supabase/export_cloud_db.sh` - Cloud export
- `scripts/supabase/setup_self_hosted.sh` - Self-hosted setup
- `scripts/supabase/import_data.sh` - Data import
- `scripts/supabase/validate_migration.py` - Validation

### Docker Configuration
- `docker/supabase/docker-compose.override.yml` - E2I integration
- `docker/supabase/start.sh` - Start services
- `docker/supabase/stop.sh` - Stop services

### Database Migrations (~50 files)
- `database/core/` - Core schema
- `database/memory/` - Agentic memory
- `database/chat/` - Chatbot tables
- `database/ml/` - ML/MLOps tables
- `database/causal/` - Causal inference
- `database/audit/` - Audit logging
- `database/rag/` - RAG system

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Cloud Export | ✅ Complete | Data exported successfully |
| Phase 2: Self-Hosted Setup | ✅ Complete | Supabase containers running (13 services) |
| Phase 3: Data Import | ✅ Complete | 84 public + 20 auth tables, 6 users migrated |
| Phase 4: App Configuration | ✅ Complete | Backend/frontend .env updated |
| Phase 5: Validation | ✅ Complete | All endpoints working, auth functional |
| Phase 6: Cutover | ✅ Complete | Backup cron configured, monitoring active |

---

## Migration Summary (2026-01-25)

**Self-Hosted Supabase:**
- URL: `http://138.197.4.36:54321` (external) / `http://localhost:54321` (internal)
- PostgreSQL: port 5432
- GoTrue Auth: v2.184.0
- All 13 Supabase containers healthy

**Data Migrated:**
- 84 public tables + 20 auth tables
- 6 auth users
- Key tables: agent_registry (11), causal_paths (50), agent_tier_mapping (21)

**Backup:**
- Daily at 2am: `/opt/backups/backup_supabase.sh`
- Retention: 7 days
- Location: `/opt/backups/supabase_backup_*.sql.gz`

---

*Plan Version: 1.1*
*Created: 2025-01-25*
*Completed: 2026-01-25*
