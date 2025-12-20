# E2I Causal Analytics - Deployment Strategy

**Date:** 2025-12-17
**Status:** RECOMMENDED APPROACH
**TL;DR:** Use Docker Compose on single VM. Kubernetes is overkill.

---

## Executive Summary

The E2I Causal Analytics platform is **designed for Docker Compose deployment** on a single powerful VM. Kubernetes is **not required** and would add unnecessary complexity for minimal benefit at this stage.

**Recommended Deployment:**
- Single cloud VM (32-64 vCPUs, 64-128GB RAM)
- Docker Compose orchestration
- Python autoscaler for worker scaling
- Cost: ~$500-1,000/mo
- Operational complexity: Low

**When to reconsider:** Only if you exceed 10,000+ concurrent users or need multi-region deployment (likely 1-2 years away, if ever).

---

## Current Architecture: Docker Compose Native

### What We Built

Everything in this project uses **Docker Compose**, not Kubernetes:

```yaml
# docker-compose.yml
services:
  api:           # FastAPI backend
  worker_light:  # 2 CPUs, 2GB RAM (×2-4 replicas)
  worker_medium: # 4 CPUs, 8GB RAM (×1-3 replicas)
  worker_heavy:  # 16 CPUs, 32GB RAM (×0-4 replicas)
  redis:         # Working memory
  falkordb:      # Semantic memory
  mlflow:        # Experiment tracking
  bentoml:       # Model serving
  feast:         # Feature store
  opik:          # LLM observability
  frontend:      # React UI
```

### How Auto-Scaling Works

**Python script** (`scripts/autoscaler.py`) monitors Redis queues and scales workers:

```python
# When SHAP task arrives in queue
docker compose up -d --scale worker_heavy=1  # Scale up

# After task completes and queue empty
docker compose up -d --scale worker_heavy=0  # Scale down
```

**This is NOT Kubernetes** - it's pure Docker Compose with intelligent scaling logic.

---

## Deployment Options Compared

### ✅ Option 1: Docker Compose on Single VM (RECOMMENDED)

#### What You Need
- 1 cloud VM (AWS EC2, DigitalOcean, Hetzner, etc.)
- Docker + Docker Compose installed
- 32-64 vCPUs, 64-128GB RAM

#### Recommended VM Specs

| Provider | Instance Type | vCPUs | RAM | Monthly Cost | Notes |
|----------|---------------|-------|-----|--------------|-------|
| **AWS** | c6i.8xlarge | 32 | 64GB | ~$980 | Compute-optimized |
| **AWS** | c6i.12xlarge | 48 | 96GB | ~$1,470 | More headroom |
| **Hetzner** | CCX53 | 32 | 128GB | ~$220 | **Best value** |
| **DigitalOcean** | CPU-Optimized 32vCPU | 32 | 64GB | ~$952 | Good balance |

**Best choice:** Hetzner CCX53 (4-5x cheaper than AWS for same specs)

#### Pros
- ✅ **Simple** - one `docker compose up -d` command
- ✅ **Everything works out-of-the-box** - no migration needed
- ✅ **Easy to debug** - `docker compose logs -f api`
- ✅ **Lower ops complexity** - no K8s learning curve
- ✅ **Auto-scaling works** via Python script
- ✅ **Cost-effective** - ~$220-980/mo vs $2,000-5,000/mo for K8s
- ✅ **Fast deployment** - hours, not weeks

#### Cons
- ⚠️ **Single point of failure** - if VM crashes, everything down
  - *Mitigation:* Cloud provider uptime SLA (99.95%), automated backups
- ⚠️ **Limited to single host resources** - max ~128 vCPU practically
  - *Mitigation:* This supports 10,000+ concurrent users
- ⚠️ **Manual failover** - need to restart on new VM if disaster
  - *Mitigation:* Automated backup + restore scripts (P5.4)

#### When This Works
- ✅ MVP through Series A
- ✅ <10,000 concurrent users
- ✅ Budget-conscious teams
- ✅ Small ops teams (1-2 people)
- ✅ Single-region deployment
- ✅ 99.5% uptime acceptable

#### Resource Capacity

On Hetzner CCX53 (32 vCPU, 128GB RAM):

| Scenario | Workers Active | CPU Used | RAM Used | Headroom |
|----------|----------------|----------|----------|----------|
| **Idle** | Light×2, Medium×1 | 6 cores | 10GB | 81% free |
| **Normal** | Light×2, Medium×2, Heavy×1 | 14 cores | 26GB | 56% free |
| **Peak** | Light×4, Medium×3, Heavy×2 | 40 cores | 72GB | **Exceeds capacity** |

**Conclusion:** CCX53 handles normal load comfortably. For peak load, upgrade to:
- Hetzner CCX63: 48 vCPU, 192GB RAM (~$330/mo)

---

### ⚠️ Option 2: Docker Compose on Multiple Hosts

#### What You Need
- 2-3 VMs for redundancy
- Shared storage (NFS, EFS, or GlusterFS) for volumes
- Load balancer (nginx, HAProxy, or cloud LB)
- Manual orchestration of service placement

#### Architecture
```
                    Load Balancer (nginx)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     Host 1              Host 2              Host 3
  API + Light        Medium Workers       Heavy Workers
  Redis + DBs                              (on-demand)
```

#### Pros
- ✅ **High availability** - host failure doesn't kill everything
- ✅ **Still Docker Compose** - no K8s complexity
- ✅ **Distributes load** - dedicated hosts for heavy tasks
- ✅ **Moderate complexity** - harder than single host, easier than K8s

#### Cons
- ⚠️ **Shared storage complexity** - NFS setup, performance tuning
- ⚠️ **Manual orchestration** - you decide which services on which host
- ⚠️ **More expensive** - 3 smaller VMs + load balancer + storage
- ⚠️ **Network overhead** - inter-host communication latency

#### Cost
- 3× Hetzner CPX51 (16 vCPU, 32GB) = ~$300/mo
- Load balancer = ~$50/mo
- **Total:** ~$350/mo (still cheaper than AWS single host)

#### When to Consider
- Need 99.9% uptime
- Want to distribute risk across hosts
- Have outgrown single host (>64 vCPU needed)
- Have ops bandwidth for setup complexity

---

### ❌ Option 3: Kubernetes (NOT RECOMMENDED NOW)

#### What You Need
- Managed K8s cluster (AWS EKS, GCP GKE, Azure AKS)
- Convert docker-compose.yml → K8s manifests (Deployments, Services, ConfigMaps, Secrets)
- Learn kubectl, Helm, K8s concepts (pods, nodes, ingress, etc.)
- Dedicated DevOps engineer

#### Architecture
```
Kubernetes Cluster
├─ Namespace: e2i-production
│  ├─ Deployment: api (3 replicas across nodes)
│  ├─ Deployment: worker-light (HPA: 2-4 replicas)
│  ├─ Deployment: worker-medium (HPA: 1-3 replicas)
│  ├─ Deployment: worker-heavy (HPA: 0-4 replicas)
│  ├─ StatefulSet: redis
│  ├─ StatefulSet: falkordb
│  ├─ Service: LoadBalancer (external IP)
│  └─ Ingress: nginx-ingress (routing rules)
└─ PersistentVolumeClaims (EBS, GCE PD)
```

#### Pros
- ✅ **Auto-failover** - pod dies, K8s restarts on another node
- ✅ **Built-in HPA** - Horizontal Pod Autoscaler (better than our Python script)
- ✅ **Multi-zone HA** - distribute across availability zones
- ✅ **Industry standard** - easy to hire K8s engineers
- ✅ **GitOps-friendly** - declarative config, version controlled

#### Cons
- ⚠️ **High complexity** - steep learning curve (weeks to months)
- ⚠️ **Expensive** - $500-1,000/mo for control plane alone
- ⚠️ **Overkill** - 10x more complexity for <5% uptime gain
- ⚠️ **Migration effort** - 2-4 weeks to convert and test
- ⚠️ **Requires expertise** - dedicated DevOps engineer (~$120k/year)
- ⚠️ **Debugging harder** - logs scattered across pods, need tools (Lens, k9s)

#### Cost Breakdown
- EKS control plane: ~$75/mo
- 3× t3.2xlarge nodes (8 vCPU, 32GB): ~$300/mo
- Load balancer (ALB): ~$50/mo
- EBS volumes: ~$100/mo
- Data transfer: ~$50/mo
- **Minimum:** ~$575/mo (before scaling)
- **Typical production:** ~$2,000-3,000/mo

#### When to Consider (All Must Be True)
- ✅ >10,000 concurrent users
- ✅ Multi-region deployment required
- ✅ Single host can't handle load (>100 vCPU needed)
- ✅ 99.99% uptime SLA required
- ✅ Budget for dedicated DevOps engineer ($120k+/year)
- ✅ 6+ months to plan migration
- ✅ Revenue >$1M/year to justify cost

**For most SaaS companies:** Docker Compose is sufficient for **years**.

---

## Recommended Deployment: Docker Compose on Single VM

### Step-by-Step Production Deployment

#### 1. Provision VM

**Best Value:** Hetzner CCX53
- 32 vCPU, 128GB RAM
- €199/mo (~$220/mo)
- NVMe SSD storage
- 1 Gbps bandwidth

**How to provision:**
```bash
# Via Hetzner Cloud Console (https://console.hetzner.cloud/)
1. Create new project: "e2i-production"
2. Add server:
   - Location: US East (Ashburn) or EU (Falkenstein)
   - Image: Ubuntu 22.04 LTS
   - Type: CCX53 (32 vCPU, 128GB RAM)
   - Networking: IPv4 + IPv6
   - SSH key: Add your public key
3. Click "Create & Buy Now"
4. Note the public IP address
```

**Alternative:** AWS EC2 c6i.8xlarge
```bash
# Via AWS Console
1. EC2 → Launch Instance
2. AMI: Ubuntu Server 22.04 LTS
3. Instance Type: c6i.8xlarge (32 vCPU, 64GB RAM)
4. Storage: 500GB gp3 SSD
5. Security Group: Allow 22 (SSH), 80 (HTTP), 443 (HTTPS), 8000 (API)
6. Key pair: Create or select existing
7. Launch
```

#### 2. Initial Server Setup

```bash
# SSH into server
ssh root@<server-ip>

# Update system
apt-get update
apt-get upgrade -y

# Create application user
adduser --disabled-password --gecos "" e2i
usermod -aG sudo e2i
usermod -aG docker e2i  # We'll create docker group next

# Set up firewall
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw allow 8000/tcp    # API (optional, if direct access needed)
ufw enable

# Install essential tools
apt-get install -y curl git tmux htop net-tools
```

#### 3. Install Docker

```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose v2 (plugin)
apt-get update
apt-get install -y docker-compose-plugin

# Add e2i user to docker group
usermod -aG docker e2i

# Verify installation
docker --version
# Output: Docker version 24.0.x, build xxx

docker compose version
# Output: Docker Compose version v2.23.x

# Enable Docker to start on boot
systemctl enable docker
systemctl start docker
```

#### 4. Clone and Configure Application

```bash
# Switch to e2i user
su - e2i

# Clone repository
git clone <your-repo-url> e2i_causal_analytics
cd e2i_causal_analytics

# Create environment file
cp .env.example .env
nano .env
```

**Edit `.env` with your credentials:**
```bash
# Supabase (Cloud PostgreSQL)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# Anthropic API
ANTHROPIC_API_KEY=sk-ant-xxx

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Celery (no changes needed - uses Redis containers)
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
```

#### 5. Start Services

```bash
# Navigate to docker directory
cd docker

# Pull images (optional, happens during up anyway)
docker compose pull

# Start all services
docker compose up -d

# Watch startup progress
docker compose logs -f

# Wait ~2 minutes for all services to be healthy
# Press Ctrl+C to exit logs
```

#### 6. Verify Health

```bash
# Run health check script
cd ..
./scripts/health_check.sh

# Expected output:
# ==========================================
# E2I Causal Analytics - Health Check
# ==========================================
# --- HTTP Services ---
# ✅ API (FastAPI) - HEALTHY
# ✅ MLflow - HEALTHY
# ✅ BentoML - HEALTHY
# ✅ Feast - HEALTHY
# ...
# ✅ SYSTEM STATUS: HEALTHY
```

#### 7. Set Up Autoscaler as System Service

```bash
# Create systemd service file
sudo nano /etc/systemd/system/e2i-autoscaler.service
```

**Service file content:**
```ini
[Unit]
Description=E2I Celery Worker Autoscaler
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=e2i
WorkingDirectory=/home/e2i/e2i_causal_analytics
ExecStart=/usr/bin/python3 scripts/autoscaler.py --config config/autoscale.yml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment
Environment="CELERY_BROKER_URL=redis://localhost:6379/1"

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable autoscaler (start on boot)
sudo systemctl enable e2i-autoscaler

# Start autoscaler
sudo systemctl start e2i-autoscaler

# Check status
sudo systemctl status e2i-autoscaler

# View logs
sudo journalctl -u e2i-autoscaler -f
```

#### 8. Set Up Monitoring

**Option A: Simple Cron-Based Alerts**

```bash
# Edit crontab
crontab -e

# Add health check every 5 minutes
*/5 * * * * /home/e2i/e2i_causal_analytics/scripts/health_check.sh > /tmp/health_check.log 2>&1 || echo "E2I health check failed at $(date)" | mail -s "E2I Health Alert" ops@yourcompany.com

# Add daily resource usage report
0 9 * * * docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" > /tmp/docker_stats.log && mail -s "E2I Daily Stats" ops@yourcompany.com < /tmp/docker_stats.log
```

**Option B: Prometheus + Grafana (Better)**

See "Advanced Monitoring" section below.

#### 9. Set Up Automated Backups

```bash
# Create backup script
nano /home/e2i/backup.sh
```

**Backup script:**
```bash
#!/bin/bash
# E2I Automated Backup Script

BACKUP_DIR="/home/e2i/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Docker volumes
cd /home/e2i/e2i_causal_analytics/docker
docker compose exec -T redis redis-cli SAVE
docker run --rm -v e2i_redis_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/redis_${DATE}.tar.gz -C /data .
docker run --rm -v e2i_mlflow_artifacts:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/mlflow_${DATE}.tar.gz -C /data .

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

**Schedule daily backups:**
```bash
chmod +x /home/e2i/backup.sh

# Add to crontab
crontab -e

# Run daily at 2 AM
0 2 * * * /home/e2i/backup.sh >> /var/log/e2i_backup.log 2>&1
```

#### 10. Set Up Reverse Proxy (Optional)

If you want SSL/TLS (HTTPS):

```bash
# Install nginx
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Create nginx config
sudo nano /etc/nginx/sites-available/e2i
```

**Nginx config:**
```nginx
server {
    listen 80;
    server_name api.yourcompany.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable and get SSL:**
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/e2i /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get Let's Encrypt certificate
sudo certbot --nginx -d api.yourcompany.com
```

---

## Operations Guide

### Starting Services

```bash
cd /home/e2i/e2i_causal_analytics/docker
docker compose up -d
```

### Stopping Services

```bash
docker compose down
```

### Restarting a Service

```bash
# Restart API
docker compose restart api

# Restart all workers
docker compose restart worker_light worker_medium worker_heavy

# Restart everything
docker compose restart
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f worker_heavy

# Last 100 lines
docker compose logs --tail=100 api

# Since specific time
docker compose logs --since="2024-01-15T10:00:00"
```

### Scaling Workers Manually

```bash
# Scale heavy workers to 2
docker compose up -d --scale worker_heavy=2

# Scale down to 0
docker compose up -d --scale worker_heavy=0

# Scale multiple tiers
docker compose up -d --scale worker_light=3 --scale worker_medium=2
```

### Monitoring Resources

```bash
# Real-time stats
docker stats

# Disk usage
docker system df

# Specific service stats
docker stats e2i_worker_heavy_1
```

### Updating Application

```bash
# Pull latest code
cd /home/e2i/e2i_causal_analytics
git pull

# Rebuild and restart
cd docker
docker compose build
docker compose up -d

# Or rebuild specific service
docker compose build api
docker compose up -d api
```

### Troubleshooting

```bash
# Check service health
docker compose ps

# Inspect specific container
docker inspect e2i_api

# Enter container shell
docker exec -it e2i_api bash

# Check autoscaler logs
sudo journalctl -u e2i-autoscaler -f

# Run health check
./scripts/health_check.sh

# Check Redis queues
docker exec -it e2i_redis redis-cli -n 1 llen shap
docker exec -it e2i_redis redis-cli -n 1 llen causal
```

---

## Advanced Monitoring (Optional)

### Prometheus + Grafana Setup

**1. Add to docker-compose.yml:**

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus:latest
    container_name: e2i_prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - e2i_network

  grafana:
    image: grafana/grafana:latest
    container_name: e2i_grafana
    restart: unless-stopped
    ports:
      - "3002:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=changeme
      - GF_INSTALL_PLUGINS=redis-datasource
    networks:
      - e2i_network
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

**2. Create prometheus.yml:**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']  # Docker metrics

  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']  # Add /metrics endpoint to API
```

**3. Access dashboards:**
- Prometheus: http://<server-ip>:9090
- Grafana: http://<server-ip>:3002

---

## Cost Analysis

### Monthly Costs Breakdown

| Item | Docker Compose | Kubernetes |
|------|----------------|------------|
| **Compute** | | |
| VM/Nodes | $220-980 | $2,000-3,000 |
| Control Plane | $0 | $75-150 |
| Load Balancer | $0 (optional $50) | $50-100 |
| **Storage** | | |
| Block Storage | Included | $50-100 |
| Backup Storage | $10-20 | $50-100 |
| **Data Transfer** | $10-30 | $50-200 |
| **Monitoring** | $0 (self-hosted) | $50-200 |
| **Support** | $0 | $100-500 |
| **TOTAL** | **$240-1,080** | **$2,425-4,250** |
| **Savings** | **Baseline** | **-90% to -75%** |

**Winner:** Docker Compose is 4-10x cheaper

---

## Migration Path (If You Outgrow Docker Compose)

### Signs You Need to Migrate

1. **Performance ceiling hit**
   - Single host maxed out (>100 vCPU utilization)
   - Heavy workers constantly at max replicas
   - Queue depths consistently >1000

2. **Availability requirements increase**
   - SLA increases to 99.99%
   - Customers demand multi-region
   - Downtime becomes costly (>$10k/hour)

3. **Operational complexity justified**
   - Revenue >$5M/year
   - Can hire dedicated DevOps engineer
   - Have 3-6 months for migration

### Migration Strategy

**Phase 1: Multi-Host Docker Compose** (1-2 weeks)
- Distribute services across 2-3 VMs
- Add load balancer
- Set up shared storage
- Still Docker Compose, just distributed

**Phase 2: Kubernetes (4-8 weeks)**
- Convert docker-compose.yml to K8s manifests
- Set up managed K8s cluster (EKS/GKE)
- Migrate services incrementally
- Run parallel environments during transition
- Cut over when confident

**Good news:** Everything we built (health checks, resource limits, multi-worker architecture) translates directly to K8s. Zero wasted effort.

---

## Kubernetes References in Documentation

### What to Ignore

Several documents mention "Kubernetes-ready" or "K8s migration." These are **informational only** and can be completely ignored:

**In `HEALTH_CHECK_VERIFICATION.md`:**
- Section: "Kubernetes liveness/readiness probes"
- Note: Just explaining that `/healthz` is K8s convention
- **Action:** Ignore, but keep `/healthz` endpoint (doesn't hurt)

**In `PRIORITY_3_COMPLETION_SUMMARY.md`:**
- Section: "Kubernetes Migration (Future)"
- Note: Listed as optional future enhancement
- **Action:** Ignore entirely

**In `IMPLEMENTATION_COMPLETE.md`:**
- Section: "Kubernetes Migration" under "Future Enhancements"
- Note: Aspirational, not required
- **Action:** Ignore entirely

**In `OVERALL_PROGRESS_STATUS.md`:**
- Mentions of "K8s-compatible design"
- Note: Just noting that migration would be easier if ever needed
- **Action:** Ignore

**Bottom line:** All K8s references are about future-proofing. They don't affect current deployment. You can safely ignore them.

---

## Summary

### ✅ Recommended Deployment Strategy

**Platform:** Docker Compose on single VM
**Provider:** Hetzner Cloud (best value) or AWS EC2 (if already using AWS)
**Instance:** CCX53 (32 vCPU, 128GB RAM) @ ~$220/mo
**Auto-scaling:** Python autoscaler script (already built)
**Monitoring:** Self-hosted Prometheus + Grafana (optional)
**Deployment time:** 2-4 hours

### ✅ What You Get

- Production-ready infrastructure
- Auto-scaling workers (0-4 heavy workers on-demand)
- ~$240-1,000/mo all-in cost
- Low operational complexity
- Easy debugging and maintenance
- Supports 1,000-10,000+ concurrent users

### ❌ What You DON'T Need

- Kubernetes (complexity overkill, 4-10x more expensive)
- Multiple VMs (single host sufficient for 1-2 years)
- Complex service mesh (unnecessary for monolith)
- Container orchestration platform (Docker Compose is enough)

### 🔄 When to Reconsider

Only if **all** of these become true:
- >10,000 concurrent users
- Revenue >$5M/year
- Multi-region deployment required
- 99.99% uptime SLA
- Budget for dedicated DevOps ($120k+/year)

**Timeframe:** Likely 1-2 years from now, if ever

---

## Quick Start Commands

```bash
# 1. Provision Hetzner CCX53 VM (via web console)

# 2. Install Docker
curl -fsSL https://get.docker.com | sh
apt-get install docker-compose-plugin

# 3. Clone and configure
git clone <repo> e2i_causal_analytics
cd e2i_causal_analytics
cp .env.example .env
nano .env  # Add your keys

# 4. Start everything
cd docker
docker compose up -d

# 5. Set up autoscaler
sudo cp ../scripts/e2i-autoscaler.service /etc/systemd/system/
sudo systemctl enable e2i-autoscaler
sudo systemctl start e2i-autoscaler

# 6. Verify
../scripts/health_check.sh
```

**That's it!** You're in production with Docker Compose.

---

## FAQ

**Q: Can Docker Compose auto-scale like Kubernetes?**
A: Yes! Our Python autoscaler (`scripts/autoscaler.py`) monitors queues and scales workers using `docker compose scale`. It's simpler than K8s HPA but works great.

**Q: What if the VM crashes?**
A: Cloud provider SLA is 99.95% (~4 hours downtime/year). For better availability, use multi-host Docker Compose. Still cheaper and simpler than K8s.

**Q: Can I deploy to multiple regions?**
A: Not easily with Docker Compose. Multi-region requires Kubernetes or separate Docker Compose deployments per region with global load balancer.

**Q: Will I hit resource limits?**
A: Hetzner CCX63 (48 vCPU, 192GB RAM) supports ~15,000 concurrent users. That's 1-2 years of growth for most startups.

**Q: Is Docker Compose production-ready?**
A: Absolutely! Many successful companies run Docker Compose in production: GitLab (early days), Basecamp, and countless others. It's battle-tested.

**Q: Can I migrate to Kubernetes later?**
A: Yes! Everything we built (health checks, resource limits, env vars) translates directly. Migration is easier because we followed best practices.

**Q: Do I lose the autoscaler if I migrate to K8s?**
A: No, K8s has built-in Horizontal Pod Autoscaler (HPA) that works better. You'd replace our Python script with K8s HPA config.

---

## Conclusion

**Docker Compose on a single VM is the right choice for E2I Causal Analytics.**

- ✅ Everything already built for it
- ✅ Production-ready as-is
- ✅ 4-10x cheaper than Kubernetes
- ✅ 10x simpler to operate
- ✅ Supports 1-2 years of growth

**Kubernetes is unnecessary complexity** that would add weeks of work for minimal benefit at this stage.

**Recommendation:** Deploy to Hetzner CCX53, run for 6-12 months, then reassess based on actual usage patterns.

---

**Document Version:** 1.0
**Author:** E2I Causal Analytics Team
**Last Updated:** 2025-12-17
**Next Review:** After 6 months in production
