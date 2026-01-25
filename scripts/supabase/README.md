# Supabase Migration Scripts

This directory contains scripts for migrating E2I Causal Analytics from Supabase Cloud to a self-hosted instance.

## Overview

| Script | Purpose |
|--------|---------|
| `export_cloud_db.sh` | Export schema and data from Supabase Cloud |
| `setup_self_hosted.sh` | Setup self-hosted Supabase on the droplet |
| `import_data.sh` | Import data into self-hosted instance |
| `validate_migration.py` | Validate migration success |

## Quick Start

### 1. Export from Supabase Cloud

```bash
# Using Supabase CLI (recommended)
cd /path/to/e2i_causal_analytics
supabase link --project-ref <your-project-ref>
./scripts/supabase/export_cloud_db.sh ./backups/cloud_export

# Or using direct connection
./scripts/supabase/export_cloud_db.sh --connection-string "postgresql://..." ./backups/cloud_export
```

### 2. Setup Self-Hosted Supabase

```bash
# Generate secrets and setup
./scripts/supabase/setup_self_hosted.sh --generate-keys

# Configure environment
cp docker/supabase/.env.template /opt/supabase/docker/.env
nano /opt/supabase/docker/.env  # Edit with your secrets

# Start services
./docker/supabase/start.sh
```

### 3. Import Data

```bash
export POSTGRES_PASSWORD="your-password"
./scripts/supabase/import_data.sh ./backups/cloud_export
```

### 4. Validate Migration

```bash
export POSTGRES_PASSWORD="your-password"
python scripts/supabase/validate_migration.py --verbose
```

## Full Documentation

See [docs/SUPABASE_SELF_HOSTED_MIGRATION.md](../../docs/SUPABASE_SELF_HOSTED_MIGRATION.md) for the complete migration guide.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_DB_HOST` | Self-hosted PostgreSQL host | `localhost` |
| `SUPABASE_DB_PORT` | Self-hosted PostgreSQL port | `5433` |
| `SUPABASE_DB_NAME` | Database name | `postgres` |
| `SUPABASE_DB_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | (required) |
| `SUPABASE_URL` | Supabase API URL | `http://localhost:8000` |
| `SUPABASE_ANON_KEY` | Supabase anon key | (required for API tests) |

## Troubleshooting

### Connection refused
```bash
# Check if Supabase is running
docker ps | grep supabase

# Check logs
docker logs supabase-db
docker logs supabase-kong
```

### Permission denied
```bash
# Ensure scripts are executable
chmod +x scripts/supabase/*.sh
```

### Import errors
```bash
# Check import logs
cat backups/cloud_export/schema_import.log
cat backups/cloud_export/data_import.log
```

## Support

- Main documentation: `docs/SUPABASE_SELF_HOSTED_MIGRATION.md`
- Supabase self-hosting docs: https://supabase.com/docs/guides/self-hosting
- Supabase CLI reference: https://supabase.com/docs/reference/cli
