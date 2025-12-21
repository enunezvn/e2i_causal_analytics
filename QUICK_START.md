# Quick Start Guide - E2I Causal Analytics

**Get up and running in 5 minutes!**

---

## Prerequisites

✅ Docker Desktop installed and running
✅ Supabase account (or local PostgreSQL)
✅ Claude API key

---

## Quick Setup (Copy & Paste)

### 1. Configure Environment

```bash
# .env.dev already created - just edit it with your credentials
nano .env.dev

# Fill in these 4 required values:
# - DATABASE_URL (from Supabase)
# - SUPABASE_URL (from Supabase)
# - SUPABASE_ANON_KEY (from Supabase)
# - CLAUDE_API_KEY (from Anthropic Console)
```

### 2. Run Pre-Flight Check

```bash
./scripts/preflight-check.sh
```

✅ All checks should pass

### 3. Start Services

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

⏱️ First run takes 5-10 minutes to build images
⏱️ Subsequent runs take <1 minute

---

## Access Services

Once services are running:

| Service | URL | Purpose |
|---------|-----|---------|
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Frontend** | http://localhost:3000 | React Application |
| **MLflow** | http://localhost:5000 | Experiment Tracking |
| **API Health** | http://localhost:8000/health | Health Check |

---

## Verify It's Working

```bash
# Check service status
docker compose ps

# All should show "Up" and "healthy"

# Test API
curl http://localhost:8000/health

# Expected: {"status":"ok",...}
```

---

## Common Commands

```bash
# Start (foreground - see logs)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Start (background)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Stop
docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Restart a service
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart fastapi

# Rebuild after code changes
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

---

## Hot Reload

Code changes apply instantly - no rebuild needed!

**FastAPI**: Edit `src/**/*.py` → Auto-reload
**Frontend**: Edit `frontend/src/**` → Hot-reload in browser

---

## Troubleshooting

### Services won't start

```bash
# Check what's using ports
sudo lsof -i :8000
sudo lsof -i :3000

# Clean restart
docker compose down -v
docker compose up --build
```

### Database connection fails

```bash
# Verify DATABASE_URL in .env.dev
grep DATABASE_URL .env.dev

# Test connection
docker compose exec fastapi python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
engine.connect()
print('✓ Connected!')
"
```

### Need to see detailed logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f fastapi

# Search for errors
docker compose logs | grep -i error
```

---

## Where to Get Credentials

### Supabase (Database)
1. Go to https://app.supabase.com/
2. Select your project
3. Go to Settings → Database
4. Copy **Connection String** (Pooling mode)
5. Go to Settings → API
6. Copy **URL** and **anon/public key**

### Claude API
1. Go to https://console.anthropic.com/settings/keys
2. Create an API key
3. Copy the key (starts with `sk-ant-`)

---

## Need Help?

**Full Documentation:**
- `docs/LOCAL_TESTING_GUIDE.md` - Comprehensive testing guide
- `docs/DOCKER_COMPOSE_USAGE.md` - All Docker commands
- `docs/DOCKER_DEPLOYMENT_PLAN.md` - Architecture details

**Pre-Flight Check:**
```bash
./scripts/preflight-check.sh
```

**Docker Compose Config:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml config
```

---

## Useful Aliases (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias dc='docker compose -f docker-compose.yml -f docker-compose.dev.yml'
alias dc-up='dc up'
alias dc-down='dc down'
alias dc-logs='dc logs -f'
alias dc-ps='dc ps'
```

Then just use:
```bash
dc-up      # Start
dc-logs    # View logs
dc-down    # Stop
```

---

**Ready to test?** Run `./scripts/preflight-check.sh` to verify your setup!
