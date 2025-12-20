# Local Development Testing Guide

**Project**: E2I Causal Analytics
**Created**: 2025-12-18
**Purpose**: Step-by-step guide to test Docker setup locally

---

## Prerequisites

Before starting, ensure you have:

- ✅ Docker Desktop installed (or Docker + Docker Compose on Linux)
- ✅ Git installed
- ✅ Access to Supabase project (or local PostgreSQL)
- ✅ Claude API key from Anthropic
- ✅ At least 8GB RAM available
- ✅ 10GB free disk space

---

## Step 1: Prepare Environment Configuration

### 1.1 Copy the Environment Template

The `.env.dev` file has already been created for you from the template.

### 1.2 Fill in Required Credentials

Edit `.env.dev` and replace the placeholder values:

```bash
# Open the file in your editor
nano .env.dev
# or
code .env.dev
# or
vim .env.dev
```

**Required Changes:**

#### A. Database Configuration (Supabase)

Get these from your Supabase dashboard: https://app.supabase.com/project/_/settings/database

```bash
DATABASE_URL=postgresql://postgres.[YOUR-PROJECT]:PASSWORD@aws-0-us-west-1.pooler.supabase.com:6543/postgres

SUPABASE_URL=https://[YOUR-PROJECT].supabase.co

SUPABASE_ANON_KEY=eyJhbGci... (copy from Supabase dashboard)

SUPABASE_SERVICE_ROLE_KEY=eyJhbGci... (copy from Supabase dashboard)
```

#### B. Claude API Configuration

Get your API key from: https://console.anthropic.com/settings/keys

```bash
CLAUDE_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

#### C. Generate Security Keys (Optional but Recommended)

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate JWT_SECRET_KEY
openssl rand -hex 32

# Copy the outputs and paste into .env.dev
SECRET_KEY=<paste-here>
JWT_SECRET_KEY=<paste-here>
```

### 1.3 Verify Configuration

Check that all required variables are set:

```bash
grep -E "(DATABASE_URL|CLAUDE_API_KEY|SUPABASE_URL|SUPABASE_ANON_KEY)" .env.dev
```

Make sure none contain "xxxxx" or "TODO".

---

## Step 2: Run Pre-Flight Check

We've created a script that validates your setup:

```bash
./scripts/preflight-check.sh
```

**What it checks:**
- ✅ Docker and Docker Compose installed
- ✅ Environment file exists
- ✅ Required environment variables set
- ✅ Docker daemon running
- ✅ Ports available (8000, 3000, 5000, 6379)
- ✅ Project structure complete
- ✅ Docker Compose configuration valid

**Expected Output:**

```
==========================================
E2I Causal Analytics - Pre-Flight Check
==========================================

[1/7] Checking Docker installation...
✓ Docker installed: Docker version 24.0.6
✓ Docker Compose installed: Docker Compose version v2.23.0

[2/7] Checking environment file...
✓ .env.dev file found

[3/7] Checking required environment variables...
✓ DATABASE_URL set: postgresql...
✓ CLAUDE_API_KEY set: sk-ant-api...
✓ SUPABASE_URL set: https://xx...
✓ SUPABASE_ANON_KEY set: eyJhbGciOi...

[4/7] Checking Docker daemon...
✓ Docker daemon is running

[5/7] Checking port availability...
✓ Port 8000 (FastAPI) is available
✓ Port 3000 (Frontend) is available
✓ Port 5000 (MLflow) is available
✓ Port 6379 (Redis) is available

[6/7] Checking project structure...
✓ docker-compose.yml exists
✓ docker-compose.dev.yml exists
...

[7/7] Validating Docker Compose configuration...
✓ Docker Compose configuration is valid

==========================================
Pre-Flight Check Summary
==========================================
✓ All checks passed! Ready to start.
```

**If you see errors:**
- Fix each error listed
- Re-run `./scripts/preflight-check.sh`
- Don't proceed until all errors are resolved

---

## Step 3: Start Docker Services

### 3.1 First-Time Startup (Foreground)

Start services in the foreground to see logs:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**What happens:**
1. Docker builds all images (first time only, ~5-10 minutes)
2. Services start in order (redis → fastapi → frontend → mlflow)
3. Health checks verify each service
4. Logs appear in your terminal

**Expected Log Output:**

```
[+] Running 6/6
 ✔ Network e2i-backend-network    Created
 ✔ Network e2i-frontend-network   Created
 ✔ Container e2i-redis            Started
 ✔ Container e2i-fastapi          Started
 ✔ Container e2i-mlflow           Started
 ✔ Container e2i-frontend         Started
 ✔ Container e2i-agent-worker     Started

e2i-fastapi  | ==========================================
e2i-fastapi  | E2I Causal Analytics - FastAPI Starting
e2i-fastapi  | ==========================================
e2i-fastapi  | Waiting for database connection...
e2i-fastapi  | Database is ready!
e2i-fastapi  | Starting application...
e2i-fastapi  | INFO:     Uvicorn running on http://0.0.0.0:8000

e2i-frontend | Compiled successfully!
e2i-frontend | webpack compiled with 0 warnings

e2i-mlflow   | [INFO] Starting MLflow server...
```

**Watch for:**
- ✅ All containers show "Started"
- ✅ No error messages in logs
- ✅ Services report "running" or "compiled successfully"

### 3.2 Stop Services (Foreground Mode)

Press `Ctrl+C` in the terminal to stop all services.

### 3.3 Start Services in Background

Once everything works, run in background:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

**View logs:**
```bash
# All services
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Specific service
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f fastapi
```

**Stop background services:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

---

## Step 4: Verify Services

### 4.1 Check Service Status

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

**Expected Output:**

```
NAME                IMAGE                    STATUS          PORTS
e2i-agent-worker    e2i-fastapi             Up 2 minutes
e2i-fastapi         e2i-fastapi             Up 2 minutes    0.0.0.0:8000->8000/tcp
e2i-frontend        e2i-frontend            Up 2 minutes    0.0.0.0:3000->3000/tcp
e2i-mlflow          e2i-mlflow              Up 2 minutes    0.0.0.0:5000->5000/tcp
e2i-redis           redis:7-alpine          Up 2 minutes    0.0.0.0:6379->6379/tcp
```

All should show `Up` status.

### 4.2 Test Service Endpoints

#### FastAPI Backend

```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status":"ok","version":"1.0.0"}

# API Documentation
# Open in browser: http://localhost:8000/docs
```

#### Frontend

```bash
# Home page
curl http://localhost:3000

# Or open in browser: http://localhost:3000
```

#### MLflow

```bash
# Health check
curl http://localhost:5000/health

# Or open in browser: http://localhost:5000
```

#### Redis

```bash
# Test connection
docker compose exec redis redis-cli -a devpassword ping

# Expected: PONG
```

### 4.3 Check Service Health

```bash
# View health status of all services
docker compose ps --format "table {{.Name}}\t{{.Status}}"
```

Should show "Up" and "healthy" for all services.

---

## Step 5: Test Hot Reload

Hot reload allows you to see code changes instantly without restarting Docker.

### 5.1 Test FastAPI Hot Reload

**Make a change:**

```bash
# Edit a file (e.g., add a comment)
echo "# Test change" >> src/api/main.py
```

**Watch the logs:**

```bash
docker compose logs -f fastapi
```

**Expected:**

```
e2i-fastapi | INFO:     Will watch for changes in these directories: ['/app/src']
e2i-fastapi | INFO:     Detected file change in 'main.py'
e2i-fastapi | INFO:     Reloading...
e2i-fastapi | INFO:     Application startup complete.
```

**Verify:**
```bash
curl http://localhost:8000/health
# Should still respond (service reloaded)
```

### 5.2 Test Frontend Hot Reload

**Make a change:**

Edit a React component (e.g., `frontend/src/App.tsx`):

```typescript
// Add a comment or change some text
<h1>E2I Causal Analytics - TEST</h1>
```

**Watch the browser:**
- Open http://localhost:3000
- Changes should appear automatically (within 1-2 seconds)

**Check logs:**

```bash
docker compose logs -f frontend
```

**Expected:**

```
e2i-frontend | Compiling...
e2i-frontend | Compiled successfully!
```

---

## Step 6: Test Service Integration

### 6.1 Test API → Database Connection

```bash
docker compose exec fastapi python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT 1'))
    print('✓ Database connection successful!')
"
```

### 6.2 Test API → Redis Connection

```bash
docker compose exec fastapi python -c "
import redis
import os

r = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
r.set('test_key', 'test_value')
value = r.get('test_key')
print(f'✓ Redis connection successful! Value: {value}')
"
```

### 6.3 Test API → Claude API Connection

```bash
docker compose exec fastapi python -c "
import os
import requests

api_key = os.getenv('CLAUDE_API_KEY')
headers = {
    'x-api-key': api_key,
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json'
}

# Simple test (check API key format)
if api_key and api_key.startswith('sk-ant-'):
    print('✓ Claude API key format is valid')
else:
    print('✗ Claude API key format invalid')
"
```

### 6.4 Test Frontend → API Connection

**In your browser:**

1. Open http://localhost:3000
2. Open browser DevTools (F12)
3. Go to Network tab
4. Interact with the application
5. Look for requests to http://localhost:8000

**Or use curl:**

```bash
# Test CORS (frontend calling API)
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://localhost:8000/health
```

---

## Step 7: Run Tests (Optional)

### 7.1 Backend Tests

```bash
# Run all tests
docker compose exec fastapi pytest

# Run specific test file
docker compose exec fastapi pytest tests/test_api.py

# Run with coverage
docker compose exec fastapi pytest --cov=src tests/
```

### 7.2 Frontend Tests

```bash
# Run all tests
docker compose exec frontend npm test

# Run with coverage
docker compose exec frontend npm test -- --coverage
```

---

## Step 8: Inspect Logs

### View All Logs

```bash
docker compose logs -f
```

### View Specific Service

```bash
docker compose logs -f fastapi
docker compose logs -f frontend
docker compose logs -f mlflow
docker compose logs -f redis
docker compose logs -f agent-worker
```

### View Last N Lines

```bash
docker compose logs --tail=50 fastapi
```

### Search Logs

```bash
docker compose logs fastapi | grep ERROR
docker compose logs | grep -i "warning\|error"
```

---

## Step 9: Debugging

### Access Container Shell

```bash
# FastAPI
docker compose exec fastapi bash

# Frontend
docker compose exec frontend sh

# MLflow
docker compose exec mlflow bash

# Redis
docker compose exec redis sh
```

### Check Environment Variables

```bash
docker compose exec fastapi env | grep DATABASE_URL
docker compose exec fastapi env | grep CLAUDE_API_KEY
```

### Check Python Packages

```bash
docker compose exec fastapi pip list
docker compose exec fastapi pip show mlflow
```

### Check Node Modules

```bash
docker compose exec frontend npm list
```

### Inspect Service Configuration

```bash
# View merged docker-compose configuration
docker compose -f docker-compose.yml -f docker-compose.dev.yml config

# View environment variables in running container
docker compose exec fastapi printenv
```

---

## Step 10: Clean Up and Reset

### Restart All Services

```bash
docker compose restart
```

### Restart Specific Service

```bash
docker compose restart fastapi
```

### Rebuild Images

```bash
# Rebuild all
docker compose build --no-cache

# Rebuild specific service
docker compose build --no-cache fastapi
```

### Remove All Containers and Volumes

⚠️ **WARNING**: This deletes all data!

```bash
docker compose down -v
```

### Complete Fresh Start

```bash
# Stop and remove everything
docker compose down -v

# Remove all images
docker compose -f docker-compose.yml -f docker-compose.dev.yml down --rmi all

# Rebuild from scratch
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

---

## Common Issues and Solutions

### Issue: Port Already in Use

**Error:**
```
Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use
```

**Solution:**
```bash
# Find what's using the port
sudo lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change the port in docker-compose.dev.yml
```

### Issue: Database Connection Failed

**Error:**
```
ERROR: Could not connect to database
```

**Solution:**
1. Verify DATABASE_URL in .env.dev
2. Check Supabase project is running
3. Test connection directly:
   ```bash
   docker compose exec fastapi python -c "
   from sqlalchemy import create_engine
   import os
   engine = create_engine(os.getenv('DATABASE_URL'))
   engine.connect()
   "
   ```

### Issue: Hot Reload Not Working

**Solution:**

1. Verify volume mounts in docker-compose.dev.yml
2. On Windows WSL: Ensure files are in WSL filesystem, not `/mnt/c/`
3. Restart the service:
   ```bash
   docker compose restart fastapi
   ```

### Issue: Frontend Shows Blank Page

**Solution:**

1. Check browser console (F12) for errors
2. Verify REACT_APP_API_URL in .env.dev
3. Check frontend logs:
   ```bash
   docker compose logs frontend
   ```
4. Rebuild frontend:
   ```bash
   docker compose build frontend
   docker compose up -d frontend
   ```

### Issue: Out of Disk Space

**Solution:**

```bash
# Check disk usage
docker system df

# Clean up unused resources
docker system prune -a

# Remove dangling volumes
docker volume prune
```

### Issue: Services Slow to Start

**Possible causes:**
- First-time build (images being downloaded/built)
- Large dataset initialization
- Slow internet connection

**Solution:**
- Be patient on first run (can take 10-15 minutes)
- Subsequent starts should be fast (<1 minute)

---

## Testing Checklist

Use this checklist to verify everything works:

### Pre-Flight
- [ ] Docker Desktop running
- [ ] .env.dev file created and filled
- [ ] Pre-flight check passes
- [ ] All ports available

### Service Health
- [ ] All containers show "Up" status
- [ ] FastAPI health endpoint responds
- [ ] Frontend loads in browser
- [ ] MLflow UI accessible
- [ ] Redis responds to PING

### Hot Reload
- [ ] FastAPI reloads on file change
- [ ] Frontend hot-reloads on file change
- [ ] No errors in logs after reload

### Integration
- [ ] FastAPI → Database connection works
- [ ] FastAPI → Redis connection works
- [ ] FastAPI → Claude API configured
- [ ] Frontend → API requests work
- [ ] CORS configured correctly

### Optional
- [ ] Backend tests pass
- [ ] Frontend tests pass
- [ ] Can access container shells
- [ ] Logs are readable and helpful

---

## Next Steps

Once local development is working:

1. **Document any issues** you encountered and how you fixed them
2. **Test your actual application code** (agents, API endpoints, etc.)
3. **Verify data flows** through the entire stack
4. **Consider adding more tests** for your specific use cases
5. **Proceed to production deployment** planning

---

## Useful Aliases (Optional)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias dc-dev='docker compose -f docker-compose.yml -f docker-compose.dev.yml'
alias dc-up='dc-dev up'
alias dc-down='dc-dev down'
alias dc-logs='dc-dev logs -f'
alias dc-ps='dc-dev ps'
alias dc-restart='dc-dev restart'
alias dc-build='dc-dev build'
```

Then use:
```bash
dc-up         # Start services
dc-logs       # View logs
dc-down       # Stop services
```

---

## Support Resources

- **Docker Compose Docs**: https://docs.docker.com/compose/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **Supabase Docs**: https://supabase.com/docs
- **MLflow Docs**: https://mlflow.org/docs/latest/

---

**Last Updated**: 2025-12-18
**Status**: Ready for Testing ✅
