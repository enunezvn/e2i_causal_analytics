# Infrastructure Reference

## Quick Connect

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Or if you have ~/.ssh/config configured:
ssh e2i-prod
```

**API Endpoints:**
| Endpoint | URL |
|----------|-----|
| Health Check | http://138.197.4.36/health |
| API (via nginx) | http://138.197.4.36/ |
| API (direct) | http://138.197.4.36:8000/ |

**Common Commands (run on droplet):**
```bash
# Check API status
sudo systemctl status e2i-api

# Restart API
sudo systemctl restart e2i-api

# View API logs
sudo journalctl -u e2i-api -f

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
| **UFW Firewall** | ✅ Active | Ports 22, 80, 443, 8000 |
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

**UFW Firewall Rules (Configured via cloud-init):**

| Port | Protocol | Purpose |
|------|----------|---------|
| 22 | TCP | SSH |
| 80 | TCP | HTTP |
| 443 | TCP | HTTPS |
| 8000 | TCP | API |

```bash
# Check firewall status on droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo ufw status verbose"

# Check fail2ban status
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo fail2ban-client status sshd"
```

### Systemd Services

E2I application services managed by systemd:

| Service | Description | Port | Status |
|---------|-------------|------|--------|
| `e2i-api.service` | FastAPI backend (uvicorn) | 8000 | ✅ Active |
| `nginx` | Reverse proxy | 80/443 | ✅ Active |
| `docker` | Container runtime | - | ✅ Active |
| `fail2ban` | Intrusion prevention | - | ✅ Active |

**Nginx Reverse Proxy Configuration:**
- Config file: `/etc/nginx/sites-available/e2i-api`
- Proxies port 80 → 8000 (API)
- WebSocket support at `/ws`
- Security headers enabled
- Logs: `/var/log/nginx/e2i-api.access.log`, `/var/log/nginx/e2i-api.error.log`

### Application Paths (on droplet)

| Path | Description |
|------|-------------|
| `/home/enunez/Projects/e2i_causal_analytics` | Application root |
| `/home/enunez/Projects/e2i_causal_analytics/venv` | Python virtual environment |
| `/home/enunez/Projects/e2i_causal_analytics/.env` | Environment variables |
| `/etc/systemd/system/e2i-api.service` | API systemd service file |
| `/etc/nginx/sites-available/e2i-api` | Nginx config |
| `/var/log/nginx/e2i-api.access.log` | Nginx access logs |
| `/var/log/nginx/e2i-api.error.log` | Nginx error logs |

### Updating the Application

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Navigate to app directory
cd ~/Projects/e2i_causal_analytics

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Install any new dependencies
pip install -r requirements.txt

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

### SSH Tunneling (Alternative Proxy Bypass)

When behind a corporate proxy (e.g., Novartis Netskope), direct access to web UIs on non-standard ports is blocked. Use SSH tunneling to access services via localhost.

#### Quick Start

```bash
# Forward E2I services (API)
ssh -i ~/.ssh/replit -L 8000:localhost:8000 -N -f enunez@138.197.4.36

# Forward all services (when configured)
ssh -i ~/.ssh/replit \
    -L 8000:localhost:8000 \
    -L 5000:localhost:5000 \
    -L 5173:localhost:5173 \
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

### Completed Setup

- [x] Create non-root sudo user (`enunez`)
- [x] Configure SSH key authentication (ED25519)
- [x] Disable root SSH login
- [x] Set up UFW firewall rules
- [x] Configure fail2ban for SSH protection
- [x] Configure swap (4GB)
- [x] Install Docker and Docker Compose
- [x] Install Nginx

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

**Scripts Location:** `~/Projects/e2i_causal_analytics/scripts/maintenance/`

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
- [ ] Configure SSL/TLS certificates (Let's Encrypt)
- [ ] Install and configure MLflow
- [ ] Configure automatic backups
- [ ] Configure domain/DNS (if applicable)
