# E2I Causal Analytics - Deployment Guide

This document provides step-by-step instructions for deploying the E2I Causal Analytics application to the production droplet.

> **For Team Members**: If you're setting up a local development environment using Docker, see [Docker Deployment for Team](#docker-deployment-for-team) at the end of this document. The sections below describe the production droplet deployment (systemd + nginx), which is managed by the project maintainer.

## ⚠️ Important: Access Methods

| Service | Method | Local URL |
|---------|--------|-----------|
| **Frontend** | SSH Tunnel | http://localhost:8080 |
| **MLflow** | SSH Tunnel | http://localhost:5000 |
| **Opik** | SSH Tunnel | http://localhost:5173 |
| **FalkorDB Browser** | SSH Tunnel | http://localhost:3030 |
| **Direct IP access** | ❌ Blocked | http://138.197.4.36/ won't work |

**Direct access to the droplet IP is blocked by network/firewall.** Always use SSH tunnel for browser access.

---

## Quick Reference

| Component | Location on Droplet |
|-----------|---------------------|
| **Project** | `/opt/e2i_causal_analytics/` |
| **Python venv** | `/opt/e2i_causal_analytics/venv/` |
| **Frontend dist** | `/var/www/html/` |
| **Nginx config** | `/etc/nginx/sites-available/e2i-analytics` |
| **API service** | `systemctl status e2i-api` |

## Droplet Information

| Property | Value |
|----------|-------|
| **IP Address** | 138.197.4.36 |
| **SSH User** | enunez |
| **SSH Key** | `~/.ssh/replit` |

---

## Frontend Deployment

### Step 1: Pull Latest Code

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && git pull origin main"
```

### Step 2: Build Frontend

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics/frontend && npm run build"
```

**Expected output**: Build completes with `built in X.XXs`

### Step 3: Deploy to Nginx

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo rm -rf /var/www/html/* && sudo cp -r /opt/e2i_causal_analytics/frontend/dist/* /var/www/html/ && sudo chown -R www-data:www-data /var/www/html"
```

### Step 4: Verify Deployment

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "curl -s http://localhost/ | head -15"
```

**Expected output**: HTML with `<title>E2I Causal Analytics</title>`

---

## Backend API Deployment

### Step 1: Pull Latest Code

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && git pull origin main"
```

### Step 2: Restart API Service

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl restart e2i-api"
```

### Step 3: Verify API Health

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sleep 5 && curl -s localhost:8000/health"
```

**Expected output**: JSON with `"status":"healthy"`

---

## Full Deployment (Frontend + Backend)

Run this single command for a complete deployment:

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  git pull origin main && \
  cd frontend && npm run build && \
  sudo rm -rf /var/www/html/* && \
  sudo cp -r dist/* /var/www/html/ && \
  sudo chown -R www-data:www-data /var/www/html && \
  sudo systemctl restart e2i-api && \
  sleep 5 && \
  echo '=== Frontend ===' && curl -s http://localhost/ | grep '<title>' && \
  echo '=== API ===' && curl -s localhost:8000/health | head -c 100"
```

---

## Accessing the Application in Browser

**SSH tunnel is required** to view the application in your browser. Direct access to the droplet IP is blocked.

### Step 1: Create SSH Tunnel

```bash
# Create tunnel (runs in background)
ssh -i ~/.ssh/replit -f -N -L 8080:localhost:80 enunez@138.197.4.36
```

**Note**: Nginx proxies `/api/*` requests to the backend, so only port 80 tunnel is needed.

### Step 2: Access Application

| Service | Local URL |
|---------|-----------|
| **Frontend** | http://localhost:8080/ |
| **API Health** | http://localhost:8080/health |
| **API Docs** | http://localhost:8080/api/docs |

### Step 3: Close Tunnel When Done

```bash
pkill -f "ssh -i.*-L 8080:localhost:80"
```

---

## Accessing MLOps & Monitoring Tools

MLflow, Opik, and FalkorDB Browser require SSH tunnels for access.

### Create SSH Tunnels

```bash
# All services in one tunnel (runs in background)
ssh -i ~/.ssh/replit -f -N \
  -L 5000:localhost:5000 \
  -L 5173:localhost:5173 \
  -L 3030:localhost:3030 \
  -L 8080:localhost:80 \
  enunez@138.197.4.36
```

### Access URLs

| Service | Local URL | Description |
|---------|-----------|-------------|
| **Frontend** | http://localhost:8080 | Main application |
| **MLflow** | http://localhost:5000 | Experiment tracking |
| **Opik** | http://localhost:5173 | Agent observability |
| **FalkorDB Browser** | http://localhost:3030 | Graph database UI |

### FalkorDB Browser Configuration

When first accessing FalkorDB Browser, it will prompt for a connection URL. Use:

```
redis://e2i_falkordb:6379
```

This connects to the FalkorDB container via Docker's internal network.

### Close All Tunnels

```bash
pkill -f "ssh -i.*-L.*enunez@138.197.4.36"
```

---

## Frontend Environment Configuration

The frontend uses relative URLs for API calls (configured in `.env.production`):

```env
# /opt/e2i_causal_analytics/frontend/.env.production
VITE_COPILOT_ENABLED=true
VITE_API_URL=/api
VITE_COPILOT_RUNTIME_URL=/api/copilotkit
```

**Important**: Using relative URLs (`/api`) allows the frontend to work through any proxy or tunnel. Do NOT hardcode the droplet IP.

---

## Nginx Configuration

The nginx config at `/etc/nginx/sites-available/e2i-analytics` handles:

| Path | Destination |
|------|-------------|
| `/` | Static files from `/var/www/html/` (SPA with fallback) |
| `/api/*` | Proxy to `localhost:8000/api/` |
| `/copilotkit/*` | Proxy to `localhost:8000/copilotkit/` |
| `/health` | Proxy to `localhost:8000/health` |
| `/assets/*` | Static files with 1-year cache |

### Restart Nginx

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo nginx -t && sudo systemctl restart nginx"
```

---

## Troubleshooting

### Issue: Frontend shows old version

**Solution**: Clear nginx cache and browser cache
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo rm -rf /var/www/html/* && sudo cp -r /opt/e2i_causal_analytics/frontend/dist/* /var/www/html/ && sudo chown -R www-data:www-data /var/www/html"
```

### Issue: CopilotKit returns 401 Unauthorized

**Cause**: CopilotKit endpoints need to be public for initialization
**Solution**: Ensure `/api/copilotkit` is in PUBLIC_PATHS in `src/api/middleware/auth_middleware.py`

### Issue: API returns 502 Bad Gateway

**Cause**: API service not running or still starting
**Solution**: Check API status and wait for startup
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl status e2i-api --no-pager"
```

### Issue: Connection reset when accessing droplet directly (http://138.197.4.36/)

**Cause**: Network/firewall blocks direct HTTP traffic to the droplet (this is expected)
**Solution**: Always use SSH tunnel for browser access (see "Accessing the Application in Browser" section)

### Issue: Git pull fails with untracked files

**Solution**: Remove conflicting files
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && git clean -fd && git pull origin main"
```

---

## Service Management

### Check Status

```bash
# API
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl status e2i-api --no-pager"

# Nginx
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl status nginx --no-pager"

# Docker containers
ssh -i ~/.ssh/replit enunez@138.197.4.36 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

### Restart Services

```bash
# API only
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl restart e2i-api"

# Nginx only
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl restart nginx"

# Both
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl restart e2i-api nginx"
```

### View Logs

```bash
# API logs
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo journalctl -u e2i-api -n 50 --no-pager"

# Nginx error logs
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo tail -50 /var/log/nginx/error.log"

# Nginx access logs
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo tail -50 /var/log/nginx/access.log"
```

---

## Important Reminders

1. **NEVER install Python dependencies on droplet** - The venv uses forked repositories
2. **Always use relative URLs in frontend** - Avoid hardcoding IPs
3. **CopilotKit endpoints must be public** - For chat widget initialization
4. **Always use SSH tunnel for browser access** - Direct IP access is blocked by network/firewall
5. **Deployment verification uses SSH commands** - Runs `curl` on the droplet itself (no tunnel needed)

---

## Docker Deployment for Team

Team members can run the full stack locally using Docker Compose. This is independent of the production droplet deployment above.

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose v2)
- 8GB+ RAM recommended for builds
- Credentials: Supabase URL/keys, Anthropic API key

### Quick Start

```bash
# Clone the repo
git clone https://github.com/enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics/docker

# Create environment file from template
cp env.example .env

# Edit .env with your credentials
# Required: SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY
# Change REDIS_PASSWORD from 'changeme' for any non-local deployment

# Build and start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f api frontend
```

### Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | http://localhost:3001 | React dashboard |
| API | http://localhost:8000 | FastAPI backend |
| MLflow | http://localhost:5000 | Experiment tracking (localhost only) |
| Opik | http://localhost:5173 | Agent observability (localhost only) |
| FalkorDB Browser | http://localhost:3030 | Graph UI (debug profile only) |

Management UIs (MLflow, Opik, etc.) are bound to `127.0.0.1` and not accessible from other machines.

### Debug Tools

To include FalkorDB Browser:

```bash
docker compose --profile debug up -d
```

### Stopping

```bash
docker compose down          # Stop containers, keep volumes
docker compose down -v       # Stop and remove volumes (data loss!)
```

### Troubleshooting

**Build fails with OOM**: Increase Docker memory limit or build services one at a time:
```bash
docker compose build api
docker compose build frontend
```

**Redis authentication errors**: Ensure all `REDIS_URL`, `CELERY_BROKER_URL`, and `CELERY_RESULT_BACKEND` use the password format: `redis://:${REDIS_PASSWORD}@redis:6379/N`

**Feast service fails**: Feast requires `docker/Dockerfile.feast`. If not needed, comment out the `feast` service in `docker-compose.yml`.

---

*Last Updated: 2026-02-05*
