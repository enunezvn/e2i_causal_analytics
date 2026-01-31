# Infrastructure Reference

## Quick Connect

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Or with SSH config alias (if configured):
ssh e2i-prod
```

**Domain**: `eznomics.site` (Hostinger DNS → DigitalOcean droplet)

**All Services (via Nginx + HTTPS):**
| Service | URL | Description |
|---------|-----|-------------|
| Frontend | https://eznomics.site/ | React dashboard |
| API | https://eznomics.site/api/ | FastAPI endpoints |
| Health | https://eznomics.site/health | Health check |
| Chatbot | https://eznomics.site/copilotkit | CopilotKit endpoint |
| MLflow | https://eznomics.site/mlflow/ | Experiment tracking |
| Opik | https://eznomics.site/opik/ | Agent observability |
| FalkorDB | https://eznomics.site/falkordb/ | Graph database browser |

> **Note**: Direct port access (e.g., `:8000`, `:5000`) is blocked by firewall.
> All services are accessible only via the nginx proxy above or SSH tunnels.

**FalkorDB Browser Connection URL:**
```
redis://falkordb:6379
```
Use this URL in the FalkorDB Browser login form to connect to the graph database.

**Common Commands (run on droplet):**
```bash
# Check API status
sudo systemctl status e2i-api

# Restart API
sudo systemctl restart e2i-api

# View API logs
sudo journalctl -u e2i-api -f

# Check BentoML model serving
sudo systemctl status e2i-bentoml

# Restart BentoML (e.g. after training new model)
sudo systemctl restart e2i-bentoml

# View BentoML logs
sudo journalctl -u e2i-bentoml -f

# Deploy a specific model to BentoML
./scripts/bentoml/deploy_model.sh --model-tag tier0_abc123:v5

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
- **Size**: s-4vcpu-16gb-amd
  - 4 vCPUs (AMD)
  - 16 GB RAM
  - 200 GB SSD
- **Status**: Active
- **Features**: droplet_agent, monitoring

### Security Configuration

| Feature | Status | Details |
|---------|--------|---------|
| **Non-root User** | ✅ Enabled | `enunez` with sudo |
| **Root Login** | ✅ Disabled | `PermitRootLogin no` |
| **SSH Key Auth** | ✅ ED25519 | Password auth disabled |
| **UFW Firewall** | ✅ Active | Ports 22, 80, 443 only |
| **Docker Firewall** | ✅ Active | DOCKER-USER chain blocks external access |
| **SSL/TLS** | ✅ Let's Encrypt | Auto-renewing via certbot |
| **Domain** | ✅ eznomics.site | Hostinger DNS A record |
| **Fail2ban** | ✅ Active | sshd jail enabled |
| **Swap** | ✅ Configured | 4GB swapfile |

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

Location: `.env`

```bash
DIGITALOCEAN_TOKEN=dop_v1_...
```

### Creation Command Reference

```bash
# Create with cloud-init for secure setup
doctl compute droplet create e2i-analytics-prod \
    --image ubuntu-24-04-x64 \
    --size s-4vcpu-8gb-amd \
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
- `s-4vcpu-16gb-amd`: 4 vCPU, 16 GB RAM, 200 GB SSD (AMD) - **current**
- `s-4vcpu-8gb-amd`: 4 vCPU, 8 GB RAM, 160 GB SSD (AMD) - previous size

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

E2I application services managed by systemd:

| Service | Description | Port | Status |
|---------|-------------|------|--------|
| `e2i-api.service` | FastAPI backend (uvicorn) | 8000 | ✅ Active |
| `e2i-bentoml.service` | BentoML model serving | 3000 | ✅ Active |
| `nginx` | Reverse proxy | 80/443 | ✅ Active |
| `docker` | Container runtime | - | ✅ Active |
| `fail2ban` | Intrusion prevention | - | ✅ Active |

**Nginx Reverse Proxy Configuration:**
- Config file: `/etc/nginx/sites-available/e2i-app`
- SSL: Let's Encrypt via certbot (auto-renewed)
- Security headers: HSTS, X-Frame-Options, X-Content-Type-Options, CSP, Permissions-Policy
- Rate limiting: API (10r/s), CopilotKit (30r/s), General (100r/s)
- Malicious path blocking: `.php`, `.env`, `.git`, `wp-*`, `phpmyadmin`
- WebSocket support for `/ws`, `/opik/`, `/falkordb/`
- Logs: `/var/log/nginx/e2i-app.access.log`, `/var/log/nginx/e2i-app.error.log`

**Nginx Proxy Routes:**
| Route | Backend | Port |
|-------|---------|------|
| `/` | Frontend static files | - |
| `/api/` | e2i_api | 8000 |
| `/health` | e2i_api | 8000 |
| `/copilotkit` | e2i_api | 8000 |
| `/ws` | e2i_api (WebSocket) | 8000 |
| `/mlflow/` | mlflow_ui | 5000 |
| `/opik/` | opik_ui | 5173 |
| `/falkordb/` | falkordb_browser | 3030 |

### Docker Containers

**E2I Core Services** (from `/opt/e2i_causal_analytics/docker/docker-compose.yml`):
| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| e2i_redis | redis:7-alpine | 6382 | Working memory cache |
| e2i_falkordb | falkordb/falkordb:latest | 6381 | Graph database |
| e2i_falkordb_browser | falkordb/falkordb-browser:latest | 3030 | Graph DB explorer |
| e2i_mlflow | mlflow | 5000 | Experiment tracking |

**Opik Stack** (from `/home/enunez/opik/deployment/docker-compose/`):
| Container | Port | Purpose |
|-----------|------|---------|
| opik-frontend-1 | 5173 | Opik UI |
| opik-backend-1 | 8080, 3003 | Opik API |
| opik-python-backend-1 | 8001 | Python backend |
| opik-clickhouse-1 | 8123, 9000 | Analytics DB |
| opik-mysql-1 | 3306 | Metadata DB |
| opik-redis-1 | 6379 | Cache |
| opik-minio-1 | 9001, 9090 | Object storage |
| opik-zookeeper-1 | 2181 | Coordination |

**Start/Stop Commands:**
```bash
# Start E2I containers (Redis, FalkorDB, MLflow)
cd /opt/e2i_causal_analytics && docker compose -f docker/docker-compose.yml up -d redis falkordb falkordb-browser mlflow

# Start Opik stack
cd /home/enunez/opik/deployment/docker-compose && docker compose up -d

# Check all containers
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Stop all E2I containers
cd /opt/e2i_causal_analytics && docker compose -f docker/docker-compose.yml down

# Stop Opik stack
cd /home/enunez/opik/deployment/docker-compose && docker compose down
```

### Monitoring Stack

**Prometheus & Grafana** for comprehensive system monitoring (access via SSH tunnel):

| Service | Local URL (via SSH tunnel) | Description | Credentials |
|---------|---------------------------|-------------|-------------|
| **Grafana** | http://localhost:3100 | Dashboards & visualization | admin / admin |
| **Prometheus** | http://localhost:9091 | Metrics collection & queries | - |

```bash
# SSH tunnel for monitoring tools
ssh -i ~/.ssh/replit -L 3100:localhost:3100 -L 9091:localhost:9091 -N -f enunez@138.197.4.36
```

**Available Dashboards:**
| Dashboard | Purpose |
|-----------|---------|
| E2I API Overview | API performance, request rates, error rates, latency percentiles |
| System Resources | CPU, memory, disk, network metrics |
| PostgreSQL Performance | Database connections, query performance, replication status |

**Metrics Collected (391 total):**
- API metrics: Request count, duration, status codes
- System metrics: CPU, memory, disk I/O, network
- PostgreSQL metrics: Connections, queries, locks, replication lag

**Start/Stop Monitoring Stack:**
```bash
# Start Prometheus & Grafana
cd /opt/e2i_causal_analytics && docker compose -f docker/docker-compose.yml up -d prometheus grafana

# Check monitoring containers
docker ps --filter "name=prometheus" --filter "name=grafana"

# View Prometheus targets (should all be UP)
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

### Self-Hosted Supabase

**Database Infrastructure** (migrated from Supabase Cloud to self-hosted):

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Supabase API | 54321 | http://localhost:54321 (via SSH tunnel) | Kong API Gateway |
| Supabase Studio | 3001 | http://localhost:3001 (via SSH tunnel) | Database management UI |
| PostgreSQL | 5433 | - | Direct database access |

**Connection Strings:**
```bash
# For application use (via Kong API)
SUPABASE_URL=http://localhost:54321  # via SSH tunnel or on-droplet

# For direct PostgreSQL (migrations, admin)
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5433/postgres  # via SSH tunnel
```

**Start/Stop Supabase:**
```bash
# Start Supabase stack
cd /opt/e2i_causal_analytics && docker compose -f docker/docker-compose.yml up -d supabase-db supabase-kong supabase-studio

# Check Supabase containers
docker ps --filter "name=supabase"
```

### Application Paths (on droplet)

**Canonical Project Location:** `/opt/e2i_causal_analytics/`

| Path | Description |
|------|-------------|
| `/opt/e2i_causal_analytics` | **Canonical** application root |
| `/opt/e2i_causal_analytics/.venv` | Python virtual environment (consolidated 2026-01-26) |
| `/opt/e2i_causal_analytics/.env` | Environment variables |
| `/home/enunez/opik` | Opik installation |
| `/home/enunez/opik/deployment/docker-compose` | Opik Docker compose |
| `/etc/systemd/system/e2i-api.service` | API systemd service file |
| `/etc/systemd/system/e2i-bentoml.service` | BentoML systemd service file |
| `/opt/e2i_causal_analytics/deploy/e2i-bentoml.env` | BentoML environment config |
| `/etc/nginx/sites-available/e2i-app` | Nginx config |
| `/var/log/nginx/e2i-app.access.log` | Nginx access logs |
| `/var/log/nginx/e2i-app.error.log` | Nginx error logs |

> **Note**: Use `/opt/e2i_causal_analytics/` for all operations. The venv contains patched packages and should not be reinstalled.

### Updating the Application

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Navigate to canonical app directory
cd /opt/e2i_causal_analytics

# Pull latest changes
git pull origin main

# NOTE: Avoid installing dependencies unless necessary (venv has patches applied)
# Only use pip install if specifically required

# Restart the API service
sudo systemctl restart e2i-api

# Verify it's running
sudo systemctl status e2i-api
curl localhost:8000/health
```

**Service Management:**

```bash
# SSH to droplet first
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Check status
sudo systemctl status e2i-api
sudo systemctl status nginx

# Restart services
sudo systemctl restart e2i-api
sudo systemctl restart nginx

# View logs
sudo journalctl -u e2i-api -f
sudo journalctl -u nginx -f
```

### Cost Information

- **Current Plan**: s-4vcpu-16gb-amd
- **Estimated Cost**: $84/month

### SSH Keys

**Registered Keys in DigitalOcean:**
- **Name**: replit-ed25519
- **ID**: 53352421
- **Fingerprint**: 72:91:c9:d1:2e:e5:09:bd:f4:68:4d:7c:d5:5c:1a:b0
- **Public Key**: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF7j9C0aZuxZ4YUXOW+IrosLczi/dTR1wBc38dgbWsyB enunez@PHUSEH-L88724`
- **Private Key Location**: `~/.ssh/replit`

**SSH Access:**
```bash
# Connect as enunez user
ssh -i ~/.ssh/replit enunez@138.197.4.36
```

**Status**: ✅ SSH key successfully configured on droplet

### Corporate Proxy Configuration

When behind a corporate proxy (e.g., Novartis), direct HTTP requests to the droplet are blocked. Add the droplet IP to `no_proxy`:

```bash
# Add to ~/.bashrc for persistence
export no_proxy="$no_proxy,138.197.4.36"

# Or use --noproxy flag with curl
curl --noproxy '*' http://138.197.4.36:8000/health
```

### SSH Tunneling (Internal Services)

With firewall hardening, internal services (MLflow, Opik, Redis, etc.) are only
accessible via nginx proxy paths or SSH tunnels. Use tunnels for direct port access.

#### Quick Start

```bash
# Forward internal services for local development
ssh -i ~/.ssh/replit \
    -L 5000:localhost:5000 \
    -L 5173:localhost:5173 \
    -L 6382:localhost:6382 \
    -L 6381:localhost:6381 \
    -L 3100:localhost:3100 \
    -L 9091:localhost:9091 \
    -N -f enunez@138.197.4.36
```

**Flags:**
- `-L [local_port]:[remote_host]:[remote_port]` - Forward local port to remote
- `-N` - Don't execute remote command (tunnel only)
- `-f` - Run in background

**Stop SSH Tunnel:**

```bash
# Kill all SSH tunnels to the droplet
pkill -f "ssh.*138.197.4.36.*-L"
```

**Persistent Tunnel (Optional):**

Add to `~/.ssh/config` for easier access:

```
Host e2i-prod
    HostName 138.197.4.36
    User enunez
    IdentityFile ~/.ssh/replit
    LocalForward 8000 localhost:8000
    LocalForward 5000 localhost:5000
```

Then connect with: `ssh -N -f e2i-prod`

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
  size: 4G
```

### Notes

- ✅ Non-root user `enunez` with sudo access (security best practice)
- ✅ Root SSH login disabled
- ✅ Fail2ban active with sshd jail
- ✅ UFW firewall configured
- ✅ 4GB swap configured
- ✅ Docker and Docker Compose installed
- ✅ Nginx installed
- ✅ **Venv consolidated** (2026-01-26): Single canonical venv at `/opt/e2i_causal_analytics/.venv`
  - Contains patched packages (ag-ui-langgraph, copilotkit)
  - Avoid pip install unless strictly necessary

### CI/CD Pipeline

**Backend CI** (`.github/workflows/backend-tests.yml`):
- Triggers on PRs/pushes touching `src/`, `tests/`, `pyproject.toml`
- Jobs: lint (Ruff) → unit-tests (pytest + coverage) → integration-tests (with Redis)
- Max 4 workers, 30s timeout per pyproject.toml

**Docker CD** (`.github/workflows/deploy.yml`):
- Triggers on merge to `main` touching `src/`, `config/`, `docker/fastapi/`
- Builds Docker image → pushes to GHCR (`ghcr.io/enunezvn/e2i-api`)
- Deploys via SSH → health check → auto-rollback on failure
- Deployment script: `scripts/deploy-docker.sh`

**GHCR Images:**
```bash
# Pull latest image
docker pull ghcr.io/enunezvn/e2i-api:latest

# View available tags
docker image ls ghcr.io/enunezvn/e2i-api
```

### Completed Setup

- [x] Create non-root sudo user (`enunez`)
- [x] Configure SSH key authentication (ED25519)
- [x] Disable root SSH login
- [x] Set up UFW firewall rules (ports 22, 80, 443 only)
- [x] Configure Docker network isolation (DOCKER-USER chain)
- [x] Configure fail2ban for SSH protection
- [x] Configure swap (4GB)
- [x] Install Docker and Docker Compose
- [x] Install Nginx with security hardening
- [x] Configure SSL/TLS certificates (Let's Encrypt via certbot)
- [x] Configure domain/DNS (eznomics.site via Hostinger)
- [x] Set up Backend CI pipeline (GitHub Actions)
- [x] Set up Docker CD pipeline (GHCR + auto-deploy)

### Troubleshooting

**API not responding:**
```bash
# Check if service is running
sudo systemctl status e2i-api

# Check if port is listening
ss -tlnp | grep 8000

# View recent logs
sudo journalctl -u e2i-api -n 100 --no-pager

# Restart the service
sudo systemctl restart e2i-api
```

**BentoML model serving not responding:**
```bash
# Check if service is running
sudo systemctl status e2i-bentoml

# Check if port is listening
ss -tlnp | grep 3000

# View recent logs
sudo journalctl -u e2i-bentoml -n 100 --no-pager

# Restart the service
sudo systemctl restart e2i-bentoml

# Deploy a specific model
./scripts/bentoml/deploy_model.sh --model-tag tier0_abc123:v5
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
tail -f /var/log/nginx/e2i-api.error.log
```

**Check system resources:**
```bash
# Memory usage
free -h

# Disk usage
df -h

# CPU/processes
htop
```

### Maintenance Scripts

The droplet includes automated maintenance scripts to prevent terminal freezing and memory issues.

**Scripts Location:** `/opt/e2i_causal_analytics/scripts/maintenance/`

| Script | Purpose | Schedule |
|--------|---------|----------|
| `cleanup_orphans.sh` | Kill orphan processes (exec(eval), zombie nodes) | Every 15 min |
| `memory_monitor.sh` | Monitor memory, alert on high usage, auto-cleanup | Every 5 min |
| `setup_cron.sh` | Install cron jobs and aliases | Run once |

**Quick Commands (after setup):**
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

**Manual Orphan Check:**
```bash
ps aux | grep "exec(eval" | grep -v grep
```

**Log Files:**
- `/var/log/e2i/orphan_cleanup.log` - Cleanup history
- `/var/log/e2i/memory_monitor.log` - Memory alerts

**Session Start Check:**
On login, you'll see a warning if orphan processes are detected.

### Next Steps / TODO

- [x] Clone E2I repository to droplet
- [x] Configure E2I API systemd service
- [x] Set up Nginx reverse proxy for API
- [x] Set up monitoring/alerting (memory + orphan cleanup)
- [x] Install and configure MLflow (Docker container + nginx proxy)
- [x] Install and configure Opik (Agent observability)
- [x] Install and configure FalkorDB Browser
- [x] Configure nginx proxies for MLOps tools (/mlflow/, /opik/, /falkordb/)
- [x] Configure SSL/TLS certificates (Let's Encrypt via certbot)
- [x] Configure domain/DNS (eznomics.site)
- [x] Harden firewall (UFW + Docker isolation)
- [x] Set up CI/CD pipeline (GitHub Actions + GHCR)
- [ ] Configure automatic backups
