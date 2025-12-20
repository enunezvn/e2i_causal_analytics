# Docker Deployment Plan - E2I Causal Analytics

**Project**: E2I Causal Analytics
**Deployment Target**: Digital Ocean Droplet
**Strategy**: Git-Based Auto-Deploy with Docker Compose
**Created**: 2025-12-18
**Status**: Implementation Phase

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment Phases](#deployment-phases)
4. [Service Stack](#service-stack)
5. [Environment Strategy](#environment-strategy)
6. [Security](#security)
7. [Monitoring](#monitoring)
8. [Cost Estimates](#cost-estimates)
9. [Migration Path](#migration-path)

---

## Overview

### Objectives

- Deploy E2I Causal Analytics platform to Digital Ocean
- Maintain development/production parity
- Enable automated deployments via Git push
- Ensure zero-downtime updates
- Implement production-grade monitoring and backups

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Database** | Cloud Supabase (initially) | Faster iteration, managed backups, focus on ML features |
| **Deployment Method** | Git-based auto-deploy | Balance of automation and simplicity |
| **Orchestration** | Docker Compose | Sufficient for single-server deployment, lower complexity than K8s |
| **Reverse Proxy** | Nginx | Industry standard, SSL termination, static file serving |
| **Artifact Storage** | Digital Ocean Spaces | S3-compatible, cost-effective for MLflow artifacts |

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Digital Ocean Droplet                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Docker Compose                      │  │
│  │                                                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │  Nginx   │  │ FastAPI  │  │ MLflow   │            │  │
│  │  │  :80/443 │→ │  :8000   │  │  :5000   │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘            │  │
│  │                      ↓             ↓                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │ Frontend │  │  Redis   │  │ Worker   │            │  │
│  │  │  :3000   │  │  :6379   │  │ Agents   │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘            │  │
│  │                                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
└────────────────────────────┼─────────────────────────────────┘
                             ↓
                    ┌────────────────┐
                    │ Cloud Supabase │
                    │  (PostgreSQL)  │
                    └────────────────┘
                             ↓
                    ┌────────────────┐
                    │  DO Spaces     │
                    │ (MLflow Store) │
                    └────────────────┘
```

### Local Development Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Developer Machine                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Docker Compose (Dev Mode)                    │  │
│  │                                                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │ FastAPI  │  │ Frontend │  │ MLflow   │            │  │
│  │  │ (reload) │  │ (hot-reload) │ (local) │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘            │  │
│  │       ↓             ↓             ↓                   │  │
│  │  [Volume Mounts to ./src, ./frontend]                 │  │
│  │                                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
└────────────────────────────┼─────────────────────────────────┘
                             ↓
                    ┌────────────────┐
                    │Supabase (Dev)  │
                    │   OR Local PG  │
                    └────────────────┘
```

---

## Deployment Phases

### Phase 1: Local Docker Compose Setup (Week 1)

**Goal**: Get Docker Compose running locally with dev environment parity

**Tasks**:
1. ✅ Create base `docker-compose.yml`
2. ✅ Create `docker-compose.dev.yml` with local overrides
3. ✅ Create `docker-compose.prod.yml` with production overrides
4. ✅ Create `.env.example` template
5. ✅ Create Dockerfiles for each service
6. ✅ Test local development workflow
7. ✅ Document setup process

**Success Criteria**:
- `docker compose up` starts all services locally
- Hot reload works for FastAPI and frontend
- Can connect to Supabase dev instance
- MLflow UI accessible at localhost:5000

---

### Phase 2: Digital Ocean Droplet Setup (Week 1-2)

**Goal**: Provision and secure DO droplet

**Tasks**:
1. Provision DO droplet (8 GB / 4 vCPUs, Ubuntu 22.04)
2. Configure SSH keys (disable password auth)
3. Set up firewall (UFW)
4. Install Docker and Docker Compose
5. Set up DO Spaces for backups/artifacts
6. Clone repository
7. Create `.env.prod` with production secrets
8. Manual deployment test

**Success Criteria**:
- Can SSH into droplet with key-based auth
- Docker Compose runs successfully
- Application accessible via droplet IP
- Firewall configured correctly

---

### Phase 3: Automation & CI/CD (Week 2-3)

**Goal**: Automated deployments via Git push

**Tasks**:
1. Create deployment script (`scripts/deploy.sh`)
2. Set up GitHub webhook
3. Configure webhook listener on droplet
4. Implement health checks
5. Add rollback mechanism
6. Test full deploy cycle
7. Document deployment process

**Success Criteria**:
- Push to `main` branch triggers auto-deploy
- Health checks verify successful deployment
- Can rollback failed deployments
- Deployment logs accessible

---

### Phase 4: Production Hardening (Week 3-4)

**Goal**: Production-ready infrastructure

**Tasks**:
1. Set up Nginx reverse proxy with SSL (Let's Encrypt)
2. Implement zero-downtime deployments
3. Configure automated backups
4. Set up monitoring and alerting
5. Implement log aggregation
6. Create disaster recovery plan
7. Load testing
8. Security audit

**Success Criteria**:
- HTTPS enabled with valid certificate
- Deployments have zero downtime
- Automated daily backups
- Monitoring dashboards operational
- Incident response procedures documented

---

## Service Stack

### Services in Docker Compose

| Service | Image/Build | Port | Volume Mounts | Dependencies |
|---------|-------------|------|---------------|--------------|
| **nginx** | nginx:alpine | 80, 443 | ./docker/nginx/nginx.conf | fastapi, frontend |
| **fastapi** | Custom (./docker/fastapi/Dockerfile) | 8000 | ./src (dev only) | postgres, redis |
| **frontend** | Custom (./docker/frontend/Dockerfile) | 3000 | ./frontend (dev only) | fastapi |
| **mlflow** | Custom (./docker/mlflow/Dockerfile) | 5000 | mlflow-data | postgres |
| **redis** | redis:7-alpine | 6379 | redis-data | - |
| **agent-worker** | Custom (same as fastapi) | - | ./src (dev only) | postgres, redis, fastapi |

**Note**: PostgreSQL is external (Supabase) initially. Can be added to stack later if self-hosting.

---

## Environment Strategy

### Three-Tier Environment Configuration

```
docker-compose.yml              # Base configuration (shared)
├── docker-compose.dev.yml      # Local development overrides
└── docker-compose.prod.yml     # Production overrides
```

### Environment Variables

**.env.example** (Template, checked into Git):
```bash
# Application
APP_ENV=development
APP_NAME=E2I Causal Analytics
APP_VERSION=1.0.0

# Database (Supabase)
DATABASE_URL=postgresql://user:password@host:5432/dbname
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=xxxxx

# Claude API
CLAUDE_API_KEY=sk-ant-xxxxx

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_ROOT=s3://your-bucket/artifacts

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=generate-with-openssl-rand-hex-32
JWT_SECRET_KEY=generate-with-openssl-rand-hex-32

# DO Spaces (Production only)
DO_SPACES_KEY=xxxxx
DO_SPACES_SECRET=xxxxx
DO_SPACES_ENDPOINT=nyc3.digitaloceanspaces.com
DO_SPACES_BUCKET=e2i-mlflow-artifacts
```

**.env.dev** (Local, gitignored):
- Development database credentials
- Local service URLs
- Debug flags enabled
- Permissive CORS

**.env.prod** (On DO droplet only, gitignored):
- Production database credentials
- Production service URLs
- Debug flags disabled
- Strict CORS
- Monitoring endpoints

---

## Docker Compose Configuration Details

### Base Configuration (docker-compose.yml)

**Shared across all environments:**
- Service definitions
- Network configuration
- Base volume definitions
- Health checks
- Restart policies

### Development Overrides (docker-compose.dev.yml)

**Local-specific settings:**
- Volume mounts for hot reload (`./src:/app/src`)
- Exposed ports for direct access
- Development command overrides (e.g., `--reload`)
- No resource limits
- Verbose logging

### Production Overrides (docker-compose.prod.yml)

**Production-specific settings:**
- No volume mounts (code baked into images)
- Limited port exposure (only nginx)
- Production commands
- Resource limits (CPU/memory)
- Structured logging
- Health check intervals

---

## Security

### Network Security

**Firewall Rules (UFW)**:
```
22    (SSH)       - Allow from specific IPs only
80    (HTTP)      - Allow (redirects to HTTPS)
443   (HTTPS)     - Allow
All other ports   - Deny
```

**Docker Networks**:
- `frontend-network`: nginx ↔ fastapi, frontend
- `backend-network`: fastapi ↔ postgres, redis, mlflow
- Isolation prevents direct external access to backend services

### Application Security

**Must-Haves**:
- ✅ SSH key-based authentication only
- ✅ Secrets in `.env.prod` (never in Git)
- ✅ SSL/TLS via Let's Encrypt
- ✅ CORS configured for production domain only
- ✅ Rate limiting on API endpoints
- ✅ Input validation (Pydantic)
- ✅ SQL injection prevention (parameterized queries)

**Nice-to-Haves**:
- Fail2ban for brute force protection
- Docker secrets instead of .env files
- WAF (Web Application Firewall)
- Intrusion detection system

### Secret Management

**Development**:
- `.env.dev` on local machine (gitignored)
- Developers share template via `.env.example`

**Production**:
- `.env.prod` created during DO droplet setup
- Stored only on droplet, never in Git
- Backed up to secure location (encrypted)
- Rotated regularly (quarterly minimum)

**Future**: Migrate to proper secret management (Vault, AWS Secrets Manager, DO App Platform secrets)

---

## Monitoring

### Infrastructure Monitoring

**Digital Ocean Built-In**:
- CPU usage
- Memory usage
- Disk I/O
- Network bandwidth
- Alerts via email/Slack

**Docker Monitoring**:
```bash
docker stats              # Real-time resource usage
docker logs <container>   # Container logs
docker inspect <container> # Detailed info
```

### Application Monitoring

**Logs**:
- Structured JSON logging (production)
- Log aggregation to files
- Rotation policy (keep 30 days)
- Future: Ship to log management service (Logtail, Papertrail)

**Metrics** (Future):
- Prometheus for metrics collection
- Grafana for visualization
- Custom ML metrics (inference latency, drift scores)

**Health Checks**:
- Endpoint: `GET /health`
- Returns: Service status, dependencies, version
- Checked by: Nginx, deploy script, monitoring

---

## Backup Strategy

### What to Back Up

| Data | Location | Frequency | Retention | Method |
|------|----------|-----------|-----------|--------|
| **Database** | Supabase | Daily (automated by Supabase) | 30 days | Supabase backups |
| **MLflow Artifacts** | DO Spaces | Continuous | 90 days | Versioned in Spaces |
| **MLflow Metadata** | PostgreSQL (Supabase) | Daily | 30 days | Included in DB backup |
| **Application Config** | Git | On commit | Indefinite | Git history |
| **Environment Secrets** | Manual | Weekly | Encrypted offsite | Manual encrypted backup |
| **Docker Volumes** | DO droplet | Daily | 7 days | `scripts/backup.sh` |

### Backup Script

**Location**: `scripts/backup.sh`

**Responsibilities**:
- Export Docker volumes
- Upload to DO Spaces
- Verify backup integrity
- Prune old backups
- Send notifications

**Schedule**: Daily at 2 AM UTC (cron job)

---

## Deployment Workflow

### Development Cycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Developer makes changes locally                      │
│    - Edit code in ./src or ./frontend                   │
│    - Hot reload applies changes automatically           │
│    - Test locally with Docker Compose                   │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Local testing                                         │
│    - Run unit tests: pytest tests/                      │
│    - Run integration tests                              │
│    - Check linting: make lint                           │
│    - Verify health checks pass                          │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Commit and push                                       │
│    - git add .                                          │
│    - git commit -m "Descriptive message"                │
│    - git push origin main                               │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 4. GitHub webhook triggers deploy                       │
│    - Webhook hits DO droplet endpoint                   │
│    - Deploy script runs                                 │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Automated deployment on DO                           │
│    - Pull latest code from Git                          │
│    - Build Docker images                                │
│    - Run database migrations (if any)                   │
│    - Rolling restart (zero downtime)                    │
│    - Health checks                                      │
│    - Notification (Slack/email)                         │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Verify deployment                                     │
│    - Check application health                           │
│    - Monitor error logs                                 │
│    - Smoke tests (optional)                             │
│    - Rollback if health checks fail                     │
└─────────────────────────────────────────────────────────┘
```

### Rollback Procedure

**Automatic Rollback** (if health checks fail):
```bash
git revert HEAD
git push origin main
# Webhook triggers deploy with reverted code
```

**Manual Rollback** (emergency):
```bash
# SSH into droplet
ssh do-droplet

# Checkout previous commit
cd /opt/e2i_causal_analytics
git log --oneline -n 5
git checkout <previous-commit-hash>

# Restart services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Verify
./scripts/health-check.sh
```

---

## Cost Estimates

### Monthly Operational Costs

| Service | Tier | Cost | Notes |
|---------|------|------|-------|
| **DO Droplet** | 8 GB / 4 vCPUs | $48 | Can scale up/down |
| **DO Spaces** | 250 GB storage | $5 + $0.01/GB | MLflow artifacts, backups |
| **Bandwidth** | 1 TB included | $0 | Included with droplet |
| **Supabase** | Pro Plan | $25-500 | Depends on usage |
| **Domain + SSL** | - | $1/mo | Let's Encrypt is free |
| **Monitoring** | DO free tier | $0 | Built-in monitoring |

**Total**: $79-554/month (depending on Supabase usage)

### Cost Optimization Strategies

**Immediate**:
- Start with 4 GB droplet ($24/mo) for testing
- Use Supabase free tier if data < 500 MB
- Optimize Docker images (smaller = faster deploys)

**Future** (when costs justify):
- Migrate from Supabase to self-hosted PostgreSQL
- Add PostgreSQL to Docker Compose stack
- Use DO managed PostgreSQL if self-hosting is complex

**Break-Even Analysis**:
- Cloud Supabase: ~$25-500/mo (depending on usage)
- Self-hosted PostgreSQL: $0 (uses droplet resources)
- Self-hosting makes sense when Supabase > $100/mo

---

## Migration Path

### Current State → Target State

**Current**:
- No containerization
- Manual deployment
- Development environment not standardized

**Target (Phase 1-2)**:
- Fully containerized
- Local development via Docker Compose
- Manual deploy to DO droplet
- Cloud Supabase for database

**Target (Phase 3)**:
- Automated Git-based deployments
- Zero-downtime updates
- Monitoring and alerting

**Future** (Phase 4+):
- Self-hosted PostgreSQL (if costs justify)
- Multi-environment setup (staging, production)
- Horizontal scaling (multiple droplets + load balancer)
- Kubernetes migration (if needed at scale)

---

## File Structure

```
e2i_causal_analytics/
├── docker-compose.yml              # Base configuration
├── docker-compose.dev.yml          # Dev overrides
├── docker-compose.prod.yml         # Prod overrides
├── .env.example                    # Template (in Git)
├── .env.dev                        # Local secrets (gitignored)
├── .env.prod                       # Production secrets (gitignored)
├── .dockerignore                   # Exclude from build context
├── docker/
│   ├── fastapi/
│   │   ├── Dockerfile              # FastAPI production image
│   │   └── entrypoint.sh           # Startup script
│   ├── frontend/
│   │   ├── Dockerfile              # React production image
│   │   └── nginx.conf              # Frontend nginx config
│   ├── mlflow/
│   │   └── Dockerfile              # MLflow server
│   └── nginx/
│       ├── nginx.conf              # Main nginx config
│       └── ssl/                    # SSL certificates
├── scripts/
│   ├── deploy.sh                   # Deployment script
│   ├── backup.sh                   # Backup script
│   ├── health-check.sh             # Post-deploy health check
│   ├── setup-droplet.sh            # Initial droplet setup
│   └── rollback.sh                 # Emergency rollback
├── docs/
│   ├── DOCKER_DEPLOYMENT_PLAN.md   # This file
│   ├── DO_DROPLET_SETUP.md         # Droplet setup guide
│   └── DEPLOYMENT_RUNBOOK.md       # Operational procedures
└── .github/
    └── workflows/
        └── deploy.yml              # Future: GitHub Actions CI/CD
```

---

## Next Steps

### Immediate (This Week)

1. ✅ **Create Docker Compose files** (base, dev, prod)
2. ✅ **Create Dockerfiles** for all services
3. ✅ **Test local development workflow**
4. ✅ **Document setup process**

### Short-Term (Week 2)

1. **Provision DO droplet**
2. **Security hardening** (SSH, firewall, fail2ban)
3. **Manual deployment test**
4. **Set up DO Spaces** for backups

### Medium-Term (Week 3-4)

1. **Implement auto-deploy** (webhook + script)
2. **Set up Nginx + SSL**
3. **Configure monitoring**
4. **Production smoke tests**

---

## Success Metrics

### Development Experience

- ✅ Developer can start entire stack with one command
- ✅ Hot reload works for code changes
- ✅ Local environment matches production
- ✅ Setup documentation is clear

### Deployment

- ✅ Deployments complete in < 5 minutes
- ✅ Zero downtime during updates
- ✅ Failed deployments auto-rollback
- ✅ Deployment success rate > 95%

### Operations

- ✅ System uptime > 99.5%
- ✅ Incident response time < 30 minutes
- ✅ Backup verification passes weekly
- ✅ Security audit findings addressed

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Data loss** | Critical | Low | Automated backups, tested recovery |
| **Security breach** | Critical | Medium | SSH keys only, firewall, fail2ban, regular updates |
| **Deploy failure** | High | Medium | Health checks, auto-rollback, staging tests |
| **Downtime** | High | Low | Zero-downtime deploys, monitoring, redundancy |
| **Cost overrun** | Medium | Medium | Budget alerts, usage monitoring, optimization |
| **Knowledge loss** | Medium | Medium | Comprehensive documentation, runbooks |

---

## Support & Maintenance

### Daily
- Monitor error logs
- Check system health metrics
- Review backup success

### Weekly
- Review security logs
- Update dependencies (if patches available)
- Verify backup restoration

### Monthly
- Security updates (OS, Docker, packages)
- Cost review and optimization
- Performance tuning

### Quarterly
- Disaster recovery drill
- Security audit
- Architecture review
- Rotate secrets

---

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Digital Ocean Droplet Guide](https://docs.digitalocean.com/products/droplets/)
- [Nginx Reverse Proxy Setup](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Let's Encrypt](https://letsencrypt.org/)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployment.html)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-18
**Next Review**: After Phase 1 completion
