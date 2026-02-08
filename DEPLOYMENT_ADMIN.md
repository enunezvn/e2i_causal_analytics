# E2I Causal Analytics - Deployment Guide

All services run via **Docker Compose with the dev overlay** (volume mounts + hot-reload). Host nginx handles SSL termination and proxies to Docker containers.

## Access Methods

| Service | Method | Local URL |
|---------|--------|-----------|
| **Frontend** | SSH Tunnel | https://localhost:8443 |
| **API Docs** | SSH Tunnel | https://localhost:8443/api/docs |
| **MLflow** | SSH Tunnel | http://localhost:5000 |
| **BentoML** | SSH Tunnel | http://localhost:3000 |
| **Opik** | SSH Tunnel | http://localhost:5173 |
| **FalkorDB Browser** | SSH Tunnel | http://localhost:3030 |
| **Supabase Studio** | SSH Tunnel | http://localhost:3001 |
| **Grafana** | SSH Tunnel | http://localhost:3200 |
| **Alertmanager** | SSH Tunnel | http://localhost:9093 |

**Direct access to the droplet IP is blocked by network/firewall.** Always use SSH tunnel for browser access.

---

## Quick Reference

| Component | Location on Droplet |
|-----------|---------------------|
| **Project (git tree)** | `/home/enunez/Projects/e2i_causal_analytics/` |
| **Compose files** | `docker/docker-compose.yml` + `docker/docker-compose.dev.yml` |
| **Nginx config** | `/etc/nginx/sites-available/e2i-analytics` |
| **Docker services** | `docker compose ps` |

## Droplet Information

| Property | Value |
|----------|-------|
| **IP Address** | 138.197.4.36 |
| **SSH User** | enunez |
| **SSH Key** | `~/.ssh/replit` |
| **Droplet Size** | 8 vCPU / 32 GB RAM |

---

## Deploying Code Changes

### Quick Deploy (most common)

Code changes to Python files and React components auto-reload via volume mounts. Workers need an explicit restart:

```bash
# On the droplet (or via SSH):
cd /home/enunez/Projects/e2i_causal_analytics
./scripts/deploy.sh
```

This runs `git pull` and restarts workers. API auto-reloads via `uvicorn --reload`, frontend via Vite HMR.

### Deploy with Image Rebuild

When `requirements.txt`, `package.json`, or Dockerfiles change:

```bash
./scripts/deploy.sh --build
```

### Remote Deploy (from local machine)

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /home/enunez/Projects/e2i_causal_analytics && ./scripts/deploy.sh"
```

### Verify Deployment

```bash
./scripts/health_check.sh
curl -sf https://eznomics.site/health
curl -sf https://eznomics.site/api/v1/health
```

---

## Starting / Stopping Services

### Start All Services

```bash
cd /home/enunez/Projects/e2i_causal_analytics
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

Or use the Makefile:

```bash
make docker-up
```

### Stop All Services

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down
```

### View Logs

```bash
# API + frontend
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f api frontend

# All workers
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f worker_light worker_medium
```

---

## SSH Tunnels

Use the provided tunnel script to access all services from your local machine:

```bash
bash scripts/ssh-tunnels/tunnels.sh
```

Or create a minimal tunnel manually:

```bash
ssh -N -L 8443:localhost:443 enunez@138.197.4.36
```

Access the app at https://localhost:8443/

### Close Tunnels

```bash
bash scripts/ssh-tunnels/tunnels.sh stop
```

---

## Architecture

All services run in Docker containers via compose:

| Service | Container | Port | Auto-reload |
|---------|-----------|------|-------------|
| API (FastAPI) | `e2i_api_dev` | 8000 | Yes (uvicorn --reload) |
| Frontend (Vite) | `e2i_frontend_dev` | 3002 | Yes (HMR) |
| Worker Light | auto-numbered (x2 replicas) | - | No (restart needed) |
| Worker Medium | `e2i_worker_medium_dev` | - | No (restart needed) |
| Scheduler | `e2i_scheduler_dev` | - | No (restart needed) |
| Redis | `e2i_redis_dev` | 6382 | N/A |
| FalkorDB | `e2i_falkordb_dev` | 6381 | N/A |
| MLflow | `e2i_mlflow_dev` | 5000 | N/A |
| Prometheus | `e2i_prometheus` | 9091 | N/A |
| Grafana | `e2i_grafana` | 3200 | N/A |

Host nginx (`/etc/nginx/sites-available/e2i-analytics`) handles SSL termination and proxies:
- `/` → Frontend Vite dev server (port 3002)
- `/api/*` → API container (port 8000)
- `/ws` → WebSocket to API
- `/mlflow/`, `/opik/`, `/falkordb/` → respective containers

---

## Nginx Configuration

The nginx config proxies all traffic to Docker containers:

| Path | Destination |
|------|-------------|
| `/` | Proxy to `localhost:3002` (Vite dev server) |
| `/api/*` | Proxy to `localhost:8000` |
| `/copilotkit/*` | Proxy to `localhost:8000` |
| `/health` | Proxy to `localhost:8000` |
| `/ws` | WebSocket proxy to `localhost:8000` |

### Reload Nginx

```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## Frontend Environment Configuration

The frontend uses relative URLs for API calls:

```env
VITE_API_URL=/api
VITE_WS_URL=wss://eznomics.site
```

These are set in `docker-compose.dev.yml`. Using relative URLs (`/api`) allows the frontend to work through the nginx proxy.

---

## Troubleshooting

### API returns 502 Bad Gateway

**Cause**: API container not running or still starting.
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs api
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
```

### Frontend shows old version

**Cause**: Vite HMR disconnected or browser cache.
**Solution**: Hard refresh (Ctrl+Shift+R) or restart the container:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart frontend
```

### Workers not picking up code changes

Workers don't auto-reload. Restart them:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart worker_light worker_medium scheduler
```

### Connection reset when accessing droplet directly

**Cause**: Network/firewall blocks direct HTTP traffic (expected).
**Solution**: Use SSH tunnel (see above).

### Git pull fails with untracked files

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git clean -fd && git pull origin main
```

---

## Rollback Plan

If something goes wrong after a deploy:

```bash
# Revert to previous commit
git log --oneline -5  # find the commit to revert to
git checkout <commit>

# Restart workers to pick up reverted code
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart worker_light worker_medium scheduler
```

API and frontend auto-reload on the git checkout.

---

## Important Reminders

1. **All services run in Docker** — no systemd, no `/opt/` copy
2. **Always use relative URLs in frontend** — avoid hardcoding IPs
3. **Always use SSH tunnel for browser access** — direct IP access is blocked
4. **`uvicorn --reload`** watches `/app/src` — Python changes apply automatically
5. **Workers need restart** — Celery doesn't auto-reload
6. **Use `--build` flag** when dependencies change (requirements.txt, package.json)

---

## Docker Deployment for Team

Team members can run the full stack locally using Docker Compose.

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose v2)
- 8 GB+ RAM recommended for builds
- Credentials: Supabase URL/keys, Anthropic API key

### Quick Start

```bash
git clone https://github.com/enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics

# Create environment file
cp .env.example .env
# Edit .env — fill in required keys

# Start everything
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Check status
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps
```

### Stopping

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down      # Keep volumes
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down -v    # Remove volumes (data loss!)
```

---

*Last Updated: 2026-02-06*
