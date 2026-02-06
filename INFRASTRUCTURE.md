# Infrastructure Reference

## Development Environment

**Development happens directly on the droplet.** All services run via Docker Compose
with dev overlays (volume mounts for hot-reload). The single git tree at
`/home/enunez/Projects/e2i_causal_analytics/` is the canonical project root.

**Service access from the droplet (localhost):**
```bash
# API health check
curl -s localhost:8000/health | python3 -m json.tool

# Check running containers
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Full health report
./scripts/droplet_report.sh

# Quick health check (CI-friendly)
./scripts/health_check.sh
```

## Quick Connect (External/Remote Access)

```bash
# SSH to droplet (only needed when connecting from a remote machine)
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Or with SSH config alias (if configured):
ssh e2i-prod
```

**Domain**: `eznomics.site` (Hostinger DNS -> DigitalOcean droplet)

**All Services (via Nginx + HTTPS):**
| Service | URL | Local (on-droplet) | Description |
|---------|-----|--------------------|-------------|
| Frontend | https://eznomics.site/ | http://localhost:3002 | React dashboard (Vite dev server) |
| API | https://eznomics.site/api/ | http://localhost:8000 | FastAPI endpoints |
| Health | https://eznomics.site/health | http://localhost:8000/health | Health check |
| Chatbot | https://eznomics.site/copilotkit | http://localhost:8000/api/copilotkit | CopilotKit endpoint |
| MLflow | https://eznomics.site/mlflow/ | http://localhost:5000 | Experiment tracking (auth required) |
| Opik | https://eznomics.site/opik/ | http://localhost:5173 | Agent observability |
| FalkorDB | https://eznomics.site/falkordb/ | http://localhost:3030 | Graph database browser |
| BentoML | — | http://localhost:3000 | Model serving |
| Grafana | — | http://localhost:3200 | Dashboards & visualization |
| Prometheus | — | http://localhost:9091 | Metrics collection |
| Alertmanager | — | http://localhost:9093 | Alert routing |
| Loki | — | http://localhost:3101 | Log aggregation |
| Supabase Studio | — | http://localhost:3001 | Database management UI |
| Supabase API | — | http://localhost:54321 | Kong API Gateway |

> **Note**: Direct port access (e.g., `:8000`, `:5000`) is blocked by the firewall for external traffic.
> From the droplet itself, services are accessible via localhost. Externally, use nginx proxy URLs or SSH tunnels.

**FalkorDB Browser Connection URL:**
```
redis://falkordb:6379
```
Use this URL in the FalkorDB Browser login form to connect to the graph database.

**Common Commands (run on droplet):**
```bash
# Start all services (base + dev overlay)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Start including Opik stack
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml up -d

# View logs (API + frontend)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f api frontend

# Restart workers (after code changes)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart worker_light worker_medium scheduler

# Deploy latest code
./scripts/deploy.sh

# Deploy with image rebuild (when requirements.txt or Dockerfiles change)
./scripts/deploy.sh --build

# Check nginx status
sudo systemctl status nginx
```

---

## DigitalOcean Droplets

### Primary Droplet - E2I Causal Analytics

**Created**: 2026-01-16 (Recreated with security hardening)

**Droplet Details:**
- **Name**: e2i-analytics-prod
- **Droplet ID**: 544907207
- **Public IPv4**: 138.197.4.36
- **Region**: NYC3 (New York)
- **Image**: Ubuntu 24.04.3 LTS x64
- **Size**: s-8vcpu-32gb-amd
  - 8 vCPUs (AMD)
  - 32 GB RAM
  - 320 GB SSD
- **Swap**: 8 GB
- **Status**: Active
- **Features**: droplet_agent, monitoring

### Security Configuration

| Feature | Status | Details |
|---------|--------|---------|
| **Non-root User** | Enabled | `enunez` with sudo |
| **Root Login** | Disabled | `PermitRootLogin no` |
| **SSH Key Auth** | ED25519 | Password auth disabled |
| **UFW Firewall** | Active | Ports 22, 80, 443 only |
| **Docker Firewall** | Active | DOCKER-USER chain blocks external access |
| **SSL/TLS** | Let's Encrypt | Auto-renewing via certbot |
| **Domain** | eznomics.site | Hostinger DNS A record |
| **Fail2ban** | Active | sshd jail enabled |

### SSH Access

```bash
# Connect as enunez (recommended)
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Or with SSH config alias (if configured)
ssh e2i-prod
```

**Note**: Root login is disabled for security. Use `enunez` user with `sudo` for admin tasks.

**Recommended: Add to ~/.ssh/config for easy access:**
```
Host e2i-prod
    HostName 138.197.4.36
    User enunez
    IdentityFile ~/.ssh/replit
```

Then simply use: `ssh e2i-prod`

### Useful doctl Commands

#### Authentication
```bash
# Authenticate doctl (token stored in .env as DIGITALOCEAN_TOKEN)
source .env && doctl auth init --access-token "$DIGITALOCEAN_TOKEN"
```

#### Droplet Management
```bash
# List all droplets
doctl compute droplet list

# Get specific droplet info
doctl compute droplet get 544907207

# Get droplet by name
doctl compute droplet get e2i-analytics-prod

# Delete droplet (CAREFUL!)
doctl compute droplet delete 544907207

# Create droplet snapshot
doctl compute droplet-action snapshot 544907207 --snapshot-name "backup-$(date +%Y%m%d)"

# Reboot droplet
doctl compute droplet-action reboot 544907207

# Power off droplet
doctl compute droplet-action power-off 544907207

# Power on droplet
doctl compute droplet-action power-on 544907207
```

#### SSH Key Management
```bash
# List SSH keys
doctl compute ssh-key list

# Current key: replit-ed25519 (ID: 53352421)
```

#### Monitoring
```bash
# Get droplet actions
doctl compute droplet-action list 544907207

# Get droplet neighbors (VMs on same hypervisor)
doctl compute droplet neighbors 544907207
```

### Environment Variables

Location: `.env` (project root, gitignored; `docker/.env` is a symlink to `../.env`)

```bash
DIGITALOCEAN_TOKEN=dop_v1_...
```

**Required env vars for Docker Compose:**
- `REDIS_PASSWORD` — Redis authentication (no defaults)
- `FALKORDB_PASSWORD` — FalkorDB authentication (no defaults)
- `GRAFANA_ADMIN_PASSWORD` — Grafana admin password
- `SUPABASE_DB_URL` — PostgreSQL connection string for Supabase
- `SUPABASE_URL`, `SUPABASE_KEY` — Supabase API access
- `ANTHROPIC_API_KEY` — Claude API key

### Creation Command Reference

```bash
# Create with cloud-init for secure setup
doctl compute droplet create e2i-analytics-prod \
    --image ubuntu-24-04-x64 \
    --size s-8vcpu-32gb-amd \
    --region nyc3 \
    --ssh-keys 53352421 \
    --user-data-file cloud-init.yaml \
    --enable-monitoring \
    --wait
```

### Available Regions

```bash
# List all regions
doctl compute region list
```

Common regions:
- `nyc1`, `nyc3` - New York
- `sfo3` - San Francisco
- `ams3` - Amsterdam
- `sgp1` - Singapore
- `lon1` - London
- `fra1` - Frankfurt
- `tor1` - Toronto
- `blr1` - Bangalore

### Available Sizes

```bash
# List all droplet sizes
doctl compute size list
```

Current size specs:
- `s-8vcpu-32gb-amd`: 8 vCPU, 32 GB RAM, 320 GB SSD (AMD) - **current**
- `s-4vcpu-16gb-amd`: 4 vCPU, 16 GB RAM, 200 GB SSD (AMD) - previous size

### Firewall & Security

**UFW Firewall Rules (3 ports only):**

| Port | Protocol | Purpose |
|------|----------|---------|
| 22 | TCP | SSH |
| 80 | TCP | HTTP (redirects to HTTPS) |
| 443 | TCP | HTTPS (all traffic via nginx) |

> All other ports (8000, 5000, 5173, 6381, 6382, 3030, etc.) are blocked.
> Services are accessible only via nginx proxy paths or SSH tunnels.

**Docker Network Isolation:**

Docker containers publish ports on `0.0.0.0` which bypasses UFW. The `DOCKER-USER`
iptables chain blocks all external access to Docker-published ports:

```bash
# Rules in /etc/iptables/rules.v4 (persisted):
# Chain DOCKER-USER:
#   RETURN - state RELATED,ESTABLISHED
#   DROP - eth0 (all new external connections)
```

```bash
# Check firewall status on droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo ufw status verbose"

# Check Docker firewall
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo iptables -L DOCKER-USER -n"

# Check fail2ban status
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo fail2ban-client status sshd"
```

### Systemd Services

Only host-level services are managed by systemd. Application services run in Docker.

| Service | Description | Status |
|---------|-------------|--------|
| `nginx` | Reverse proxy / SSL termination | Active |
| `docker` | Container runtime | Active |
| `fail2ban` | Intrusion prevention | Active |

### Nginx Reverse Proxy

- **Config file**: `/etc/nginx/sites-available/e2i-analytics`
- **SSL**: Let's Encrypt via certbot (auto-renewed)
- **Security headers**: HSTS, X-Frame-Options, X-Content-Type-Options, CSP, Permissions-Policy
- **Rate limiting**: API (10r/s), CopilotKit (30r/s), General (100r/s)
- **Malicious path blocking**: `.php`, `.env`, `.git`, `wp-*`, `phpmyadmin`
- **WebSocket support** for `/ws`, `/opik/`, `/falkordb/`
- **Logs**: `/var/log/nginx/e2i-analytics.access.log`, `/var/log/nginx/e2i-analytics.error.log`

**Nginx Proxy Routes:**
| Route | Backend | Port |
|-------|---------|------|
| `/` | Frontend Vite dev server | 3002 |
| `/api/` | API container | 8000 |
| `/health` | API container | 8000 |
| `/copilotkit` | API container | 8000 |
| `/ws` | API container (WebSocket) | 8000 |
| `/mlflow/` | MLflow container (auth required) | 5000 |
| `/opik/` | Opik frontend container | 5173 |
| `/falkordb/` | FalkorDB Browser container | 3030 |

### Docker Services

**All services run via Docker Compose** from `/home/enunez/Projects/e2i_causal_analytics/`:

```bash
# Primary compose command (base + dev overlay):
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# With Opik stack:
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml up -d
```

**E2I Core Services** (`docker/docker-compose.yml` + `docker/docker-compose.dev.yml`):

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| `e2i_api_dev` | Local build (`docker/Dockerfile`) | 8000 | FastAPI backend (uvicorn --reload) |
| `e2i_frontend_dev` | Local build (`docker/frontend/Dockerfile`) | 3002 | React Vite dev server (HMR) |
| worker_light (x2) | Local build | — | Celery light workers |
| `e2i_worker_medium_dev` | Local build | — | Celery medium worker |
| `e2i_scheduler_dev` | Local build | — | Celery beat scheduler |
| `e2i_redis_dev` | redis:7-alpine | 6382 | Working memory cache |
| `e2i_falkordb_dev` | falkordb/falkordb:v4.14.11 | 6381 | Graph database |
| `e2i_mlflow_dev` | ghcr.io/mlflow/mlflow:v3.1.0 | 5000 | Experiment tracking |
| `e2i_feast_dev` | Local build | — | Feature store server |
| `e2i_bentoml_dev` | Local build (`docker/bentoml/Dockerfile`) | 3000 | Model serving |

**Observability Stack** (in base `docker/docker-compose.yml`):

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| `e2i_prometheus` | prom/prometheus:v3.2.1 | 9091 | Metrics collection |
| `e2i_node_exporter` | prom/node-exporter:v1.9.0 | — | Host metrics |
| `e2i_postgres_exporter` | prometheuscommunity/postgres-exporter:v0.16.0 | — | Supabase DB metrics |
| `e2i_grafana` | grafana/grafana:11.5.2 | 3200 | Dashboards & visualization |
| `e2i_loki` | grafana/loki:3.4.2 | 3101 | Log aggregation |
| `e2i_promtail` | grafana/promtail:3.4.2 | — | Log shipping |
| `e2i_alertmanager` | prom/alertmanager:v0.28.1 | 9093 | Alert routing |

**Opik Stack** (`docker/docker-compose.opik.yml`):

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| opik-frontend | ghcr.io/comet-ml/opik/opik-frontend | 5173 | Opik UI |
| opik-backend | ghcr.io/comet-ml/opik/opik-backend | 8084 | Opik API |
| opik-python-backend | ghcr.io/comet-ml/opik/opik-python-backend | 8001 | Python backend |
| opik-clickhouse | clickhouse/clickhouse-server:25.3.6.56-alpine | — | Analytics DB |
| opik-mysql | mysql:8.4.2 | — | Metadata DB |
| opik-redis | redis:7.2.4-alpine3.19 | — | Cache |
| opik-minio | minio/minio | 9090 | Object storage |
| opik-zookeeper | zookeeper:3.9.4 | — | Coordination |

**Debug-only services** (behind `profiles: [debug]`):
- FalkorDB Browser (`falkordb/falkordb-browser:v1.7.1`) — port 3030
- Redis Commander (`rediscommander/redis-commander:0.8.1`)

**Other Docker projects on the droplet:**
- Supabase at `/opt/supabase/docker/` (separate compose project)

### Monitoring Stack

| Service | URL (on droplet) | URL (via SSH tunnel) | Description |
|---------|-------------------|----------------------|-------------|
| **Grafana** | http://localhost:3200 | http://localhost:3200 | Dashboards & visualization |
| **Prometheus** | http://localhost:9091 | http://localhost:9091 | Metrics collection & queries |
| **Loki** | http://localhost:3101 | http://localhost:3101 | Log aggregation |
| **Alertmanager** | http://localhost:9093 | http://localhost:9093 | Alert routing |

> Grafana, Prometheus, and Alertmanager require passwords set via environment variables (`GRAFANA_ADMIN_PASSWORD`, etc.). No default credentials.

**Available Dashboards:**
| Dashboard | Purpose |
|-----------|---------|
| E2I API Overview | API performance, request rates, error rates, latency percentiles |
| System Resources | CPU, memory, disk, network metrics |
| PostgreSQL Performance | Database connections, query performance, replication status |

**Prometheus Targets:**
- API metrics: Request count, duration, status codes
- System metrics: CPU, memory, disk I/O, network (via Node Exporter)
- PostgreSQL metrics: Connections, queries, locks, replication lag (via Postgres Exporter)
- Container metrics: Docker container resource usage

**Start/Stop Monitoring:**
```bash
# Monitoring services start with the main compose stack
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d prometheus grafana loki alertmanager promtail node-exporter postgres-exporter

# View Prometheus targets (should all be UP)
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

### Self-Hosted Supabase

**Database Infrastructure** (separate compose project at `/opt/supabase/docker/`):

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Supabase API | 54321 | http://localhost:54321 (via SSH tunnel) | Kong API Gateway |
| Supabase Studio | 3001 | http://localhost:3001 (via SSH tunnel) | Database management UI |
| PostgreSQL | 5433 | — | Direct database access |

**Connection Strings:**
```bash
# For application use (via Kong API)
SUPABASE_URL=http://localhost:54321  # via SSH tunnel or on-droplet

# For direct PostgreSQL (migrations, admin)
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5433/postgres  # via SSH tunnel
```

### Application Paths (on droplet)

| Path | Description |
|------|-------------|
| `/home/enunez/Projects/e2i_causal_analytics` | Project root (single git tree) |
| `/home/enunez/Projects/e2i_causal_analytics/.env` | Environment variables (gitignored) |
| `/home/enunez/Projects/e2i_causal_analytics/docker/.env` | Symlink to `../.env` |
| `/home/enunez/Projects/e2i_causal_analytics/docker/` | Compose files, Dockerfiles, nginx configs |
| `/home/enunez/Projects/e2i_causal_analytics/scripts/` | Operational scripts |
| `/etc/nginx/sites-available/e2i-analytics` | Host nginx config |
| `/var/log/nginx/e2i-analytics.access.log` | Nginx access logs |
| `/var/log/nginx/e2i-analytics.error.log` | Nginx error logs |
| `/opt/supabase/docker/` | Self-hosted Supabase compose project |

### Updating the Application

**Standard deploy (most common):**
```bash
cd /home/enunez/Projects/e2i_causal_analytics
./scripts/deploy.sh
```

This runs `git pull` and restarts workers. API auto-reloads via `uvicorn --reload`, frontend via Vite HMR.

**Deploy with image rebuild** (when requirements.txt, package.json, or Dockerfiles change):
```bash
./scripts/deploy.sh --build
```

**Remote deploy** (from local machine):
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /home/enunez/Projects/e2i_causal_analytics && ./scripts/deploy.sh"
```

**Verify deployment:**
```bash
./scripts/health_check.sh
curl -sf https://eznomics.site/health
```

### Cost Information

- **Current Plan**: s-8vcpu-32gb-amd
- **Estimated Cost**: ~$168/month

### SSH Keys

**Workstation Key (registered in DigitalOcean):**
- **Name**: replit-ed25519
- **ID**: 53352421
- **Fingerprint**: 72:91:c9:d1:2e:e5:09:bd:f4:68:4d:7c:d5:5c:1a:b0
- **Public Key**: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF7j9C0aZuxZ4YUXOW+IrosLczi/dTR1wBc38dgbWsyB enunez@PHUSEH-L88724`
- **Private Key Location**: `~/.ssh/replit`

**Deploy Key (GitHub Actions CD pipeline):**
- **Purpose**: Dedicated key for automated deployments via GitHub Actions
- **Location**: `~/.ssh/deploy_ed25519` (on droplet)
- **Comment**: `github-actions-deploy`
- **Added to**: `~/.ssh/authorized_keys` (entry 2 of 2)
- **GitHub Secret**: Stored as `DEPLOY_SSH_KEY` in repository secrets

**SSH Access:**
```bash
# Connect as enunez user (from workstation)
ssh -i ~/.ssh/replit enunez@138.197.4.36
```

**Status**: Both keys configured on droplet

### Corporate Proxy Configuration

When behind a corporate proxy (e.g., Novartis), direct HTTP requests to the droplet are blocked. Add the droplet IP to `no_proxy`:

```bash
# Add to ~/.bashrc for persistence
export no_proxy="$no_proxy,138.197.4.36"

# Or use --noproxy flag with curl
curl --noproxy '*' http://138.197.4.36:8000/health
```

### SSH Tunneling (Remote Access Only)

> **Note**: SSH tunnels are only needed when accessing internal services from a
> **remote machine** (e.g., a laptop). When developing directly on the droplet,
> all services are accessible via localhost — no tunnels needed.

With firewall hardening, internal services (MLflow, Opik, Grafana, etc.) are blocked
from external access. From a remote machine, use SSH tunnels for direct port access.

#### Quick Start (from remote machine)

Use the provided tunnel script:
```bash
bash scripts/ssh-tunnels/tunnels.sh
```

This forwards:
| Local Port | Remote Port | Service |
|------------|-------------|---------|
| 8443 | 443 | Frontend (HTTPS via nginx) |
| 3002 | 3002 | Frontend (direct Vite) |
| 5000 | 5000 | MLflow |
| 3000 | 3000 | BentoML |
| 5173 | 5173 | Opik |
| 3030 | 3030 | FalkorDB Browser |
| 3001 | 3001 | Supabase Studio |
| 3100 | 3100 | Grafana |
| 9093 | 9093 | Alertmanager |

**Stop tunnels:**
```bash
bash scripts/ssh-tunnels/tunnels.sh stop
```

For persistent tunnels via systemd autossh, see `scripts/ssh-tunnels/setup.sh`.

**Flags:**
- `-L [local_port]:[remote_host]:[remote_port]` - Forward local port to remote
- `-N` - Don't execute remote command (tunnel only)
- `-f` - Run in background

### Cloud-Init Configuration

The droplet was provisioned with cloud-init for automated setup:

```yaml
#cloud-config
users:
  - name: enunez
    groups: sudo, docker
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF7j9C0aZuxZ4YUXOW+IrosLczi/dTR1wBc38dgbWsyB

packages:
  - fail2ban, ufw, docker.io, docker-compose, nginx, python3-pip, git, htop

runcmd:
  - systemctl enable ssh && systemctl start ssh
  - ufw allow 22,80,443,8000/tcp && ufw --force enable
  - systemctl enable fail2ban docker
  - sed -i 's/^PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config

swap:
  filename: /swapfile
  size: 8G
```

### CI/CD Pipeline

**Backend CI** (`.github/workflows/backend-tests.yml`):
- Triggers on PRs/pushes touching `src/`, `tests/`, `pyproject.toml`
- Jobs: lint (Ruff) -> unit-tests (pytest + coverage) -> integration-tests (with Redis)
- Max 4 workers, 30s timeout per pyproject.toml

**Deploy CD** (`.github/workflows/deploy.yml`):
- Triggers on merge to `main` touching `src/`, `config/`, `docker/Dockerfile`, `docker/frontend/`, `frontend/`, `requirements.txt`, `pyproject.toml`
- Also supports manual trigger via `workflow_dispatch`
- Pipeline: test -> build-and-push (GHCR images for CI artifact) -> deploy (SSH to droplet)
- Deploy step: `git pull` + restart workers on droplet (API/frontend auto-reload via volume mounts)
- Health check with 60s timeout; fails the workflow if unhealthy

**CD Pipeline Configuration:**

| Component | Details |
|-----------|---------|
| **Deploy SSH Key** | `~/.ssh/deploy_ed25519` (dedicated key for GitHub Actions) |
| **GitHub Secrets** | `DEPLOY_SSH_KEY` (private key), `DEPLOY_HOST` (`138.197.4.36`), `DEPLOY_USER` (`enunez`) |
| **GitHub Environment** | `production` |
| **SSH Action** | `appleboy/ssh-action@v1` |

**Deploy Flow:**
1. `test` job runs backend-tests (lint + unit + integration)
2. `build-and-push` builds Docker images and pushes to GHCR (CI artifact tracking)
3. `deploy` SSHes into droplet, runs `git pull`, restarts workers
4. Health check validates deployment; workflow fails if check doesn't pass within 60s

**Trigger a deploy:**
```bash
# Manual trigger via GitHub Actions UI: Actions -> "Deploy to Production" -> "Run workflow"
# Or via gh CLI:
gh workflow run deploy.yml --repo enunezvn/e2i_causal_analytics

# Automatic: push to main touching src/, config/, docker/Dockerfile, frontend/, requirements.txt, or pyproject.toml
```

### Troubleshooting

**API not responding:**
```bash
# Check if container is running
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps api

# Check if port is listening
ss -tlnp | grep 8000

# View recent logs
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs --tail 100 api

# Restart the container
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
```

**BentoML not responding:**
```bash
# Check container
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps bentoml

# View logs
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs --tail 100 bentoml

# Restart
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart bentoml
```

**Opik not responding:**
```bash
# Check Opik containers
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml ps

# View Opik backend logs
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml logs --tail 100 opik-backend

# Restart Opik stack
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml restart opik-backend opik-frontend opik-python-backend
```

**SSH connection refused:**
```bash
# Check if droplet is running (from local machine)
doctl compute droplet get 544907207

# Reboot droplet if needed
doctl compute droplet-action reboot 544907207
```

**Nginx issues:**
```bash
# Test nginx config
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

# Check nginx logs
tail -f /var/log/nginx/e2i-analytics.error.log
```

**Check system resources:**
```bash
# Full droplet health report
./scripts/droplet_report.sh

# Memory usage
free -h

# Disk usage
df -h

# CPU/processes
htop
```

### Maintenance Scripts

**Scripts Location:** `/home/enunez/Projects/e2i_causal_analytics/scripts/`

| Script | Purpose |
|--------|---------|
| `deploy.sh` | Deploy latest code (git pull + restart workers) |
| `health_check.sh` | Quick service health check (CI-friendly) |
| `droplet_report.sh` | Comprehensive droplet health report |
| `run_tests_batched.sh` | Run full test suite in batches |
| `run_migrations.sh` | Run database migrations |
| `backup_cron.sh` | Automated backup scheduling |
| `backup_data_stores.sh` | Backup data stores |
| `setup_branch_protection.sh` | Configure GitHub branch protection |
| `maintenance/cleanup_orphans.sh` | Kill orphan processes |
| `maintenance/memory_monitor.sh` | Monitor memory usage |

**Quick Commands (after maintenance setup):**
```bash
# Check for orphan processes
e2i-orphans

# Run cleanup (dry-run first)
e2i-cleanup-dry
e2i-cleanup

# Check memory status
e2i-memcheck

# View maintenance logs
e2i-logs
```

**Log Files:**
- `/var/log/e2i/orphan_cleanup.log` - Cleanup history
- `/var/log/e2i/memory_monitor.log` - Memory alerts

### Completed Setup

- [x] Create non-root sudo user (`enunez`)
- [x] Configure SSH key authentication (ED25519)
- [x] Disable root SSH login
- [x] Set up UFW firewall rules (ports 22, 80, 443 only)
- [x] Configure Docker network isolation (DOCKER-USER chain)
- [x] Configure fail2ban for SSH protection
- [x] Configure swap (8GB)
- [x] Install Docker and Docker Compose
- [x] Install Nginx with security hardening
- [x] Configure SSL/TLS certificates (Let's Encrypt via certbot)
- [x] Configure domain/DNS (eznomics.site via Hostinger)
- [x] Set up Backend CI pipeline (GitHub Actions)
- [x] Set up Deploy CD pipeline (GitHub Actions)
- [x] Generate deploy SSH key for GitHub Actions
- [x] Configure GitHub repository secrets
- [x] Create GitHub `production` environment
- [x] Migrate all services to Docker Compose
- [x] Set up observability stack (Prometheus, Grafana, Loki, Alertmanager)
- [x] Production hardening (auth, circuit breakers, DLQ, no default passwords)
- [x] Upgrade droplet to 8 vCPU / 32 GB
- [ ] Configure automatic backups
