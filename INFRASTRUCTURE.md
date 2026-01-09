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
- **Size**: s-4vcpu-8gb-120gb-intel (resized from s-2vcpu-4gb-120gb-intel)
  - 4 vCPUs
  - 8 GB RAM
  - 120 GB SSD
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
DIGITALOCEAN_TOKEN=dop_v1_56687eccc31d38426562949cd41e61f5a381784922c456070bb8d9e8e493fea7
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
- `s-4vcpu-8gb-120gb-intel`: 4 vCPU, 8 GB RAM, 120 GB SSD (Intel) - current
- `s-2vcpu-4gb-120gb-intel`: 2 vCPU, 4 GB RAM, 120 GB SSD (Intel) - original

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

### Systemd Services

E2I application services managed by systemd:

| Service | Description | Port | Status |
|---------|-------------|------|--------|
| `e2i-api.service` | FastAPI backend (uvicorn) | 8001 | Enabled |
| `e2i-frontend.service` | React frontend (vite preview) | 5174 | Enabled |

**Service Management:**

```bash
# Check status
systemctl status e2i-api
systemctl status e2i-frontend

# Restart services
systemctl restart e2i-api
systemctl restart e2i-frontend

# View logs
journalctl -u e2i-api -f
journalctl -u e2i-frontend -f

# Stop/Start
systemctl stop e2i-frontend
systemctl start e2i-frontend
```

**Service Files:**
- `/etc/systemd/system/e2i-api.service`
- `/etc/systemd/system/e2i-frontend.service`

**Health Check:**

```bash
curl -s http://localhost:8001/health | jq .status  # API
curl -s -o /dev/null -w "%{http_code}" http://localhost:5174/  # Frontend
```

### Cost Information

- **Current Plan**: s-4vcpu-8gb-120gb-intel
- **Estimated Cost**: ~$48/month (verify current pricing on DigitalOcean)
- **Previous Plan**: s-2vcpu-4gb-120gb-intel (~$24/month)

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

### SSH Tunneling (Corporate Proxy Bypass)

When behind a corporate proxy (e.g., Novartis Netskope), direct access to web UIs on non-standard ports is blocked. Use SSH tunneling to access services via localhost.

#### Quick Start (Recommended)

Use the connection manager script for automated health checks and tunnel setup:

```bash
# Full check + start tunnel
./scripts/droplet-connect.sh

# Just check health (no tunnel)
./scripts/droplet-connect.sh --check-only

# Skip checks, just start tunnel
./scripts/droplet-connect.sh --tunnel-only

# Stop existing tunnels
./scripts/droplet-connect.sh --kill-tunnel
```

#### Manual Setup

**Available Services:**

| Service | Remote Port | Local URL (via tunnel) |
|---------|-------------|------------------------|
| E2I Frontend | 5174 | http://localhost:5174 |
| E2I API | 8001 | http://localhost:8001 |
| MLflow | 5000 | http://localhost:5000 |
| Opik UI | 5173 | http://localhost:5173 |
| Opik Backend | 8080 | http://localhost:8080 |

**Start SSH Tunnel:**

```bash
# Forward E2I services (frontend + API)
ssh -i ~/.ssh/replit -L 5174:localhost:5174 -L 8001:localhost:8001 -N -f root@159.89.180.27

# Forward all services
ssh -i ~/.ssh/replit \
    -L 5174:localhost:5174 \
    -L 8001:localhost:8001 \
    -L 5000:localhost:5000 \
    -L 5173:localhost:5173 \
    -L 8080:localhost:8080 \
    -N -f root@159.89.180.27
```

**Flags:**
- `-L [local_port]:[remote_host]:[remote_port]` - Forward local port to remote
- `-N` - Don't execute remote command (tunnel only)
- `-f` - Run in background

**Verify Tunnel:**

```bash
# Check if tunnel is working
curl -s -o /dev/null -w "MLflow: HTTP %{http_code}\n" http://localhost:5000/
curl -s -o /dev/null -w "Opik: HTTP %{http_code}\n" http://localhost:5173/
```

**Stop SSH Tunnel:**

```bash
# Kill all SSH tunnels to the droplet
pkill -f "ssh.*-L 5000:localhost:5000"

# Or find and kill specific tunnel
ps aux | grep "ssh.*-L" | grep -v grep
kill <PID>
```

**Persistent Tunnel (Optional):**

Add to `~/.ssh/config` for easier access:

```
Host e2i-tunnel
    HostName 159.89.180.27
    User root
    IdentityFile ~/.ssh/replit
    LocalForward 5174 localhost:5174
    LocalForward 8001 localhost:8001
    LocalForward 5000 localhost:5000
    LocalForward 5173 localhost:5173
    LocalForward 8080 localhost:8080
```

Then connect with: `ssh -N -f e2i-tunnel`

### Notes

- Droplet is running in a VPC for private networking
- Droplet agent is enabled for enhanced monitoring
- Default user is `root` - consider creating a sudo user for security
- SSH key `replit-key` (ID: 52751421) registered in DigitalOcean account
- ✅ SSH key configured for passwordless authentication

### Next Steps / TODO

- [x] Add SSH key for passwordless authentication (Completed: 2025-12-18)
- [ ] Create non-root sudo user
- [ ] Set up firewall rules
- [ ] Configure automatic backups (if needed)
- [ ] Install required software for E2I Causal Analytics
- [ ] Set up monitoring/alerting
- [ ] Configure domain/DNS (if applicable)
