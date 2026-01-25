# Supabase Cloud to Self-Hosted Migration Plan

## Executive Summary

This document provides a comprehensive plan for migrating E2I Causal Analytics from Supabase Cloud to a self-hosted Supabase instance using Docker. The migration is **fully feasible** and officially supported by Supabase.

### Key Benefits of Self-Hosting
- **Cost Control**: No per-project fees; predictable infrastructure costs
- **Data Sovereignty**: Complete control over data location and security
- **Customization**: Full access to PostgreSQL configuration and extensions
- **Integration**: Direct network access for low-latency connections with ML services

### Resources
- [Supabase Self-Hosting with Docker](https://supabase.com/docs/guides/self-hosting/docker)
- [Supabase CLI Reference](https://supabase.com/docs/reference/cli/introduction)
- [Database Backup and Restore](https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore)
- [GitHub Discussion: Cloud to Self-Host Transfer](https://github.com/orgs/supabase/discussions/22712)

---

## Prerequisites

### System Requirements (DigitalOcean Droplet)

| Resource | Minimum | Recommended (E2I) |
|----------|---------|-------------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB | 50 GB+ |
| Docker | Latest | Latest |
| Docker Compose | v2.0+ | v2.0+ |

**Current Droplet (e2i-analytics-prod)**: 4 vCPU, 8 GB RAM, 160 GB SSD - **Meets requirements**

### Software Requirements
- Docker and Docker Compose (already installed on droplet)
- PostgreSQL client tools (`psql`, `pg_dump`) for migration
- Supabase CLI (optional, for managed exports)
- Git

### Required Credentials
- Supabase Cloud project credentials (for export)
- SSH access to droplet
- DigitalOcean API token (for firewall configuration)

---

## Migration Architecture

### Current Architecture (Cloud)
```
┌─────────────────┐     ┌─────────────────────────┐
│   Frontend      │────>│   Supabase Cloud        │
│   (React)       │     │   - PostgreSQL          │
└─────────────────┘     │   - Auth (GoTrue)       │
         │              │   - Storage             │
         │              │   - Realtime            │
         ▼              └─────────────────────────┘
┌─────────────────┐              │
│   FastAPI       │──────────────┘
│   Backend       │
└─────────────────┘
```

### Target Architecture (Self-Hosted)
```
┌────────────────────────────────────────────────────────────────┐
│                    DigitalOcean Droplet                        │
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────────────────────┐  │
│  │   Frontend      │────>│   Supabase Self-Hosted          │  │
│  │   (React)       │     │   ┌────────────────────────┐    │  │
│  └─────────────────┘     │   │ Kong API Gateway       │    │  │
│           │              │   ├────────────────────────┤    │  │
│           │              │   │ GoTrue (Auth)          │    │  │
│           ▼              │   ├────────────────────────┤    │  │
│  ┌─────────────────┐     │   │ PostgREST              │    │  │
│  │   FastAPI       │────>│   ├────────────────────────┤    │  │
│  │   Backend       │     │   │ PostgreSQL 15          │    │  │
│  └─────────────────┘     │   ├────────────────────────┤    │  │
│                          │   │ Supabase Studio        │    │  │
│                          │   └────────────────────────┘    │  │
│                          └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Migration Plan

### Phase 1: Preparation (Pre-Migration)

#### Step 1.1: Backup Current Cloud Database

```bash
# Option A: Using Supabase CLI (recommended)
# Install CLI if not present
npm install -g supabase

# Link to your cloud project
supabase link --project-ref <your-project-ref>

# Export schema (excludes Supabase-managed schemas)
supabase db dump -f schema.sql

# Export data separately
supabase db dump -f data.sql --data-only

# Export roles (if custom roles exist)
supabase db dump -f roles.sql --role-only

# Preserve migration history
supabase db dump -f migration_history.sql --use-copy --data-only --schema supabase_migrations
```

```bash
# Option B: Using pg_dump directly
# Get connection string from Supabase Dashboard > Settings > Database

# Schema only
pg_dump "postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres" \
  --schema-only \
  --no-owner \
  --no-privileges \
  --no-subscriptions \
  -f schema_dump.sql

# Data only
pg_dump "postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres" \
  --data-only \
  --use-copy \
  --no-owner \
  --no-privileges \
  -f data_dump.sql
```

#### Step 1.2: Inventory Supabase Features in Use

Based on project analysis, E2I uses:
- [x] PostgreSQL database (core tables, 50+ SQL files)
- [x] Row Level Security (RLS) policies
- [ ] Supabase Auth (GoTrue) - verify usage
- [ ] Storage buckets - verify usage
- [ ] Realtime subscriptions - verify usage
- [ ] Edge Functions - not used

#### Step 1.3: Generate Required Secrets

```bash
# Generate JWT Secret (minimum 32 characters)
openssl rand -base64 32

# You'll need these values:
# - POSTGRES_PASSWORD (24+ characters)
# - JWT_SECRET (32+ characters)
# - ANON_KEY (generated from JWT_SECRET)
# - SERVICE_ROLE_KEY (generated from JWT_SECRET)

# Generate keys using: https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys
```

---

### Phase 2: Self-Hosted Setup

#### Step 2.1: Clone Supabase Docker Configuration

```bash
# On the droplet
cd /opt
git clone --depth 1 https://github.com/supabase/supabase
cd supabase/docker
cp .env.example .env
```

#### Step 2.2: Configure Environment

Edit `/opt/supabase/docker/.env`:

```env
############
# Secrets - CHANGE THESE!
############
POSTGRES_PASSWORD=your-super-secure-password-here
JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters
ANON_KEY=<generated-anon-key>
SERVICE_ROLE_KEY=<generated-service-role-key>

############
# Database - Settings for Supabase postgres
############
POSTGRES_HOST=db
POSTGRES_DB=postgres
POSTGRES_PORT=5432

############
# API Proxy - Configuration for Kong
############
KONG_HTTP_PORT=8000
KONG_HTTPS_PORT=8443

############
# API - Configuration for PostgREST
############
PGRST_DB_SCHEMAS=public,storage,graphql_public

############
# Auth - Configuration for GoTrue
############
SITE_URL=http://138.197.4.36
ADDITIONAL_REDIRECT_URLS=
JWT_EXPIRY=3600
DISABLE_SIGNUP=false
API_EXTERNAL_URL=http://138.197.4.36:8000

############
# Studio - Configuration for Supabase Studio
############
STUDIO_DEFAULT_ORGANIZATION=E2I Causal Analytics
STUDIO_DEFAULT_PROJECT=E2I Production
STUDIO_PORT=3000

############
# Storage - Configuration for Storage
############
STORAGE_BACKEND=file
```

#### Step 2.3: Configure Docker Network Integration

The self-hosted Supabase needs to integrate with E2I's existing Docker network.

Create `/opt/e2i_causal_analytics/docker/supabase/docker-compose.override.yml`:

```yaml
# This file extends the official Supabase docker-compose.yml
# to integrate with E2I's existing network

networks:
  default:
    name: e2i-backend-network
    external: true
```

#### Step 2.4: Start Self-Hosted Supabase

```bash
cd /opt/supabase/docker
docker compose up -d

# Verify all services are running
docker compose ps

# Expected services:
# - supabase-db (PostgreSQL)
# - supabase-kong (API Gateway)
# - supabase-auth (GoTrue)
# - supabase-rest (PostgREST)
# - supabase-realtime
# - supabase-storage
# - supabase-studio
```

---

### Phase 3: Data Migration

#### Step 3.1: Restore Schema

```bash
# Connect to self-hosted PostgreSQL
psql -h localhost -p 5432 -U postgres -d postgres

# Or via Docker
docker exec -i supabase-db psql -U postgres -d postgres < schema_dump.sql
```

#### Step 3.2: Restore Data

```bash
# Restore data
docker exec -i supabase-db psql -U postgres -d postgres < data_dump.sql
```

#### Step 3.3: Apply E2I-Specific Migrations

```bash
# Apply all E2I database migrations in order
cd /opt/e2i_causal_analytics

# Core schema
psql -h localhost -p 5432 -U postgres -d postgres -f database/core/e2i_ml_complete_v3_schema.sql

# Memory system
psql -h localhost -p 5432 -U postgres -d postgres -f database/memory/001_agentic_memory_schema_v1.3.sql

# And so on for other migration files...
# See scripts/migrate_to_self_hosted.sh for automation
```

#### Step 3.4: Verify Data Integrity

```bash
# Run validation queries
psql -h localhost -p 5432 -U postgres -d postgres << 'EOF'
-- Check table counts
SELECT schemaname, relname, n_live_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC
LIMIT 20;

-- Verify RLS policies exist
SELECT schemaname, tablename, policyname
FROM pg_policies
WHERE schemaname = 'public';

-- Check extensions
SELECT * FROM pg_extension;
EOF
```

---

### Phase 4: Application Configuration

#### Step 4.1: Update Backend Environment

Update `/opt/e2i_causal_analytics/.env`:

```env
# Supabase Self-Hosted Configuration
SUPABASE_URL=http://localhost:8000
SUPABASE_ANON_KEY=<your-generated-anon-key>
SUPABASE_SERVICE_KEY=<your-generated-service-role-key>

# Direct PostgreSQL connection (for migrations and admin)
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/postgres
```

#### Step 4.2: Update Frontend Environment

Update frontend environment variables:

```env
VITE_SUPABASE_URL=http://138.197.4.36:8000
VITE_SUPABASE_ANON_KEY=<your-generated-anon-key>
```

#### Step 4.3: Restart Application Services

```bash
# Restart E2I services
sudo systemctl restart e2i-api
docker compose -f /opt/e2i_causal_analytics/docker-compose.prod.yml restart
```

---

### Phase 5: Validation

#### Step 5.1: API Health Checks

```bash
# Test Supabase API Gateway
curl http://localhost:8000/rest/v1/ -H "apikey: <ANON_KEY>"

# Test Auth endpoint
curl http://localhost:8000/auth/v1/health

# Test E2I API health
curl http://localhost:8000/health
```

#### Step 5.2: Data Validation

```bash
# Run E2I validation script
python scripts/validate_self_hosted_migration.py
```

#### Step 5.3: Frontend Testing

- [ ] Verify authentication flow
- [ ] Test database queries via Supabase client
- [ ] Verify real-time subscriptions (if used)

---

### Phase 6: Cutover

#### Step 6.1: DNS/Network Configuration

```bash
# Update firewall rules (if needed)
doctl compute firewall update <firewall-id> \
  --inbound-rules "protocol:tcp,ports:8000,address:0.0.0.0/0"
```

#### Step 6.2: Update Production Configuration

1. Point `SUPABASE_URL` to self-hosted instance
2. Deploy updated frontend with new environment variables
3. Monitor logs for errors

#### Step 6.3: Decommission Cloud Instance

1. Take final backup of cloud instance
2. Pause or delete cloud project
3. Document migration completion

---

## Migration Scripts

The following scripts are provided to automate the migration:

| Script | Purpose |
|--------|---------|
| `scripts/supabase/export_cloud_db.sh` | Export from Supabase Cloud |
| `scripts/supabase/setup_self_hosted.sh` | Setup self-hosted Supabase |
| `scripts/supabase/import_data.sh` | Import data to self-hosted |
| `scripts/supabase/validate_migration.py` | Validate migration success |

---

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**: Revert environment variables to point to cloud
2. **Data Sync**: If self-hosted was used, export any new data and import to cloud
3. **DNS**: Revert any DNS/firewall changes

---

## Post-Migration Maintenance

### Backup Strategy

```bash
# Daily backup cron job
0 2 * * * docker exec supabase-db pg_dump -U postgres postgres | gzip > /backups/supabase_$(date +\%Y\%m\%d).sql.gz
```

### Updates

```bash
# Check for Supabase updates
cd /opt/supabase/docker
git pull

# Update containers
docker compose pull
docker compose down && docker compose up -d
```

### Monitoring

- Monitor PostgreSQL connections and performance
- Set up alerts for disk space
- Monitor container health via Docker

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data loss during migration | Low | High | Multiple backups, validate before cutover |
| Extended downtime | Medium | Medium | Test migration on staging first |
| Application compatibility | Low | Medium | Test all features post-migration |
| Performance degradation | Low | Medium | Monitor and tune PostgreSQL |

---

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| Phase 1: Preparation | 1-2 hours |
| Phase 2: Self-Hosted Setup | 1 hour |
| Phase 3: Data Migration | 1-2 hours (depends on data size) |
| Phase 4: Configuration | 30 minutes |
| Phase 5: Validation | 1-2 hours |
| Phase 6: Cutover | 30 minutes |
| **Total** | **5-8 hours** |

---

## Appendix

### A. Port Mapping

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| Kong (API Gateway) | 8000 | 8000 |
| Kong (HTTPS) | 8443 | 8443 |
| PostgreSQL | 5432 | 5432 (internal only) |
| Supabase Studio | 3000 | 3000 |
| GoTrue (Auth) | 9999 | Internal |
| PostgREST | 3001 | Internal |
| Realtime | 4000 | Internal |

### B. Required Extensions

These PostgreSQL extensions are required for E2I:
- `uuid-ossp`
- `pgcrypto`
- `pgjwt`
- `pg_stat_statements`
- `pgvector` (for RAG embeddings)
- `pg_trgm` (for text search)

### C. Troubleshooting

**Issue**: Connection refused to Supabase
**Solution**: Check Docker network connectivity and port bindings

**Issue**: Auth tokens not working
**Solution**: Verify JWT_SECRET matches between services

**Issue**: RLS policies blocking requests
**Solution**: Ensure correct role is set in JWT claims

---

*Document Version: 1.0*
*Created: 2025-01-24*
*Author: E2I Causal Analytics Team*
