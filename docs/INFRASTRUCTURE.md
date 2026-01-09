# Infrastructure Reference

## DigitalOcean Droplets

### Primary Droplet - E2I Causal Analytics

**Created**: 2025-12-18

**Droplet Details:**
- **Name**: ubuntu-s-2vcpu-4gb-120gb-intel-nyc3-01
- **Droplet ID**: 538064298
- **Public IPv4**: 159.89.180.27
- **Private IPv4**: 10.108.0.2
- **Region**: NYC3 (New York)
- **Image**: Ubuntu 24.04 LTS x64
- **Size**: s-2vcpu-4gb-120gb-intel
  - 2 vCPUs
  - 4GB RAM
  - 120GB SSD
- **VPC UUID**: acd58f3d-4e52-4e14-bce4-e5e002521914
- **Status**: Active
- **Features**: droplet_agent, private_networking

### SSH Access

```bash
ssh root@159.89.180.27
```

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
doctl compute droplet get 538064298

# Get droplet by name
doctl compute droplet get ubuntu-s-2vcpu-4gb-120gb-intel-nyc3-01

# Delete droplet (CAREFUL!)
doctl compute droplet delete 538064298

# Create droplet snapshot
doctl compute droplet-action snapshot 538064298 --snapshot-name "backup-$(date +%Y%m%d)"

# Reboot droplet
doctl compute droplet-action reboot 538064298

# Power off droplet
doctl compute droplet-action power-off 538064298

# Power on droplet
doctl compute droplet-action power-on 538064298
```

#### SSH Key Management
```bash
# List SSH keys
doctl compute ssh-key list

# Add SSH key to account
doctl compute ssh-key create "my-key-name" --public-key "$(cat ~/.ssh/id_rsa.pub)"

# Create droplet with SSH key
doctl compute droplet create my-droplet \
    --image ubuntu-24-04-x64 \
    --size s-2vcpu-4gb-120gb-intel \
    --region nyc3 \
    --ssh-keys <key-id>
```

#### Monitoring
```bash
# Get droplet actions
doctl compute droplet-action list 538064298

# Get droplet neighbors (VMs on same hypervisor)
doctl compute droplet neighbors 538064298
```

### Environment Variables

Location: `.env`

```bash
DIGITALOCEAN_TOKEN=REDACTED_DO_TOKEN
```

### Creation Command Reference

```bash
doctl compute droplet create ubuntu-s-2vcpu-4gb-120gb-intel-nyc3-01 \
    --image ubuntu-24-04-x64 \
    --size s-2vcpu-4gb-120gb-intel \
    --region nyc3 \
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
- `s-2vcpu-4gb-120gb-intel`: 2 vCPU, 4GB RAM, 120GB SSD (Intel)

### Firewall & Security

```bash
# List firewalls
doctl compute firewall list

# Create firewall
doctl compute firewall create \
    --name "web-firewall" \
    --inbound-rules "protocol:tcp,ports:22,sources:addresses:0.0.0.0/0,sources:addresses:::/0 protocol:tcp,ports:80,sources:addresses:0.0.0.0/0,sources:addresses:::/0 protocol:tcp,ports:443,sources:addresses:0.0.0.0/0,sources:addresses:::/0" \
    --droplet-ids 538064298
```

### Cost Information

- **Current Plan**: s-2vcpu-4gb-120gb-intel
- **Estimated Cost**: ~$24/month (verify current pricing on DigitalOcean)

### SSH Keys

**Registered Keys in DigitalOcean:**
- **Name**: replit-key
- **ID**: 52751421
- **Fingerprint**: 72:91:c9:d1:2e:e5:09:bd:f4:68:4d:7c:d5:5c:1a:b0
- **Public Key**: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF7j9C0aZuxZ4YUXOW+IrosLczi/dTR1wBc38dgbWsyB enunez@PHUSEH-L88724`
- **Private Key Location**: `~/.ssh/replit`

**SSH Access:**
```bash
# Passwordless SSH with replit key
ssh -i ~/.ssh/replit root@159.89.180.27

# Or simply (if using default SSH config)
ssh root@159.89.180.27
```

**Status**: ✅ SSH key successfully configured on droplet

### Notes

- Droplet is running in a VPC for private networking
- Droplet agent is enabled for enhanced monitoring
- Default user is `root` - consider creating a sudo user for security
- SSH key `replit-key` (ID: 52751421) registered in DigitalOcean account
- ✅ SSH key configured for passwordless authentication

---

## Deployed Services

### Backend API Service (e2i-api)

**Service File**: `/etc/systemd/system/e2i-api.service`

```ini
[Unit]
Description=E2I Causal Analytics API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/e2i_causal_analytics
EnvironmentFile=/root/e2i_causal_analytics/.env
ExecStart=/root/e2i_causal_analytics/.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

**Management Commands**:
```bash
# Check status
systemctl status e2i-api

# Restart service
systemctl restart e2i-api

# View logs
journalctl -u e2i-api -f

# View recent logs
journalctl -u e2i-api --since "10 minutes ago"
```

**Port**: 8001

### Frontend Service (e2i-frontend)

**Service File**: `/etc/systemd/system/e2i-frontend.service`

```ini
[Unit]
Description=E2I Frontend Preview Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/e2i_causal_analytics/frontend
ExecStart=/usr/bin/npm run preview -- --host 0.0.0.0 --port 5173
Restart=always

[Install]
WantedBy=multi-user.target
```

**Management Commands**:
```bash
# Check status
systemctl status e2i-frontend

# Restart service
systemctl restart e2i-frontend

# View logs
journalctl -u e2i-frontend -f
```

**Port**: 5173

---

## Knowledge Graph (FalkorDB)

### Container Configuration

**Container Name**: `e2i_falkordb`
**Image**: `falkordb/falkordb:latest`
**Port**: 6379 (Redis protocol)

**Docker Run Command**:
```bash
docker run -d \
  --name e2i_falkordb \
  -p 6379:6379 \
  -v falkordb_data:/data \
  --restart unless-stopped \
  falkordb/falkordb:latest
```

**Management Commands**:
```bash
# Check container status
docker ps | grep falkordb

# View logs
docker logs e2i_falkordb

# Restart container
docker restart e2i_falkordb

# Access Cypher shell
docker exec -it e2i_falkordb redis-cli
# Then: GRAPH.QUERY e2i_semantic "MATCH (n) RETURN count(n)"
```

### Graph Configuration

**Graph Name**: `e2i_semantic`
**Configuration File**: `config/005_memory_config.yaml`

```yaml
semantic_memory:
  graph:
    name: e2i_semantic
    host: localhost
    port: 6379
```

### Current Graph Data (as of 2026-01-08)

**Node Types** (27 total):
- **Patient** (3): Anonymous patient entities for journey analysis
- **HCP** (5): Healthcare providers with specialties and tiers
- **Brand** (3): Remibrutinib, Fabhalta, Kisqali
- **Region** (4): Northeast, Southeast, Midwest, West
- **KPI** (6): TRx, NRx, NBRx, Market Share, NPS, Conversion Rate
- **CausalPath** (3): HCP→NRx, NRx→TRx, Multi-hop chains
- **Agent** (3): Drift Monitor, Health Score, Experiment Designer

**Relationship Types** (15 total):
- `PRESCRIBES`: HCP to Brand (with monthly_volume property)
- `TREATED_BY`: Patient to HCP
- `CAUSES`: Causal path connections (with effect_size)
- `AFFECTS`: Brand to KPI impacts
- `IMPACTS`: Intervention effects
- `MONITORS`: Agent observation relationships
- `ANALYZES`: Agent analysis targets
- `INFLUENCES`: HCP peer influence (with influence_score)
- `TRACKS`: Agent KPI tracking

### Cypher Query Examples

```cypher
# Count all nodes
MATCH (n) RETURN labels(n) as type, count(*) as count

# Get all HCPs and their prescriptions
MATCH (h:HCP)-[p:PRESCRIBES]->(b:Brand)
RETURN h.name, b.name, p.monthly_volume

# Trace causal chain
MATCH path = (start:CausalPath)-[:CAUSES*]->(end:CausalPath)
RETURN path

# Find HCP influence network
MATCH (h1:HCP)-[i:INFLUENCES]->(h2:HCP)
RETURN h1.name, h2.name, i.influence_score
```

---

## API Endpoints Reference

### Public Endpoints (No Auth Required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | API health status |
| GET | `/healthz` | Kubernetes health probe |
| GET | `/ready` | Readiness check |
| GET | `/api/docs` | Swagger documentation |
| GET | `/api/redoc` | ReDoc documentation |
| GET | `/api/kpis` | List all KPIs |
| GET | `/api/copilotkit/status` | Chatbot status |
| POST | `/api/copilotkit` | Chatbot runtime |

### Protected Endpoints (JWT Required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/nodes` | Get all graph nodes |
| GET | `/api/graph/relationships` | Get all relationships |
| GET | `/api/graph/stats` | Graph statistics |
| GET | `/api/graph/causal-chains` | Causal path analysis |
| GET | `/api/digital-twin/health` | Digital twin status |
| GET | `/api/agents/status` | Agent orchestration status |
| GET | `/api/monitoring/alerts` | System alerts |

### Testing Endpoints

```bash
# Test health (public)
curl http://159.89.180.27:8001/health

# Test graph stats (requires auth)
curl -H "Authorization: Bearer <token>" http://159.89.180.27:8001/api/graph/stats

# Test locally via SSH tunnel
ssh -L 8001:localhost:8001 root@159.89.180.27
curl http://localhost:8001/api/graph/stats
```

---

## Next Steps / TODO

- [x] Add SSH key for passwordless authentication (Completed: 2025-12-18)
- [x] Install required software for E2I Causal Analytics (Completed: 2026-01-08)
- [x] Deploy API service with systemd (Completed: 2026-01-08)
- [x] Deploy Frontend service with systemd (Completed: 2026-01-08)
- [x] Configure FalkorDB Knowledge Graph (Completed: 2026-01-08)
- [x] Seed semantic graph with domain data (Completed: 2026-01-08)
- [ ] Create non-root sudo user
- [ ] Set up firewall rules
- [ ] Configure automatic backups (if needed)
- [ ] Set up monitoring/alerting
- [ ] Configure domain/DNS (if applicable)
- [ ] Set up SSL/TLS certificates
