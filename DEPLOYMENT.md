# E2I Causal Analytics - Deployment Guide

This document provides step-by-step instructions for deploying the E2I Causal Analytics application to the production droplet.

## ⚠️ Important: Access Methods

| Access Type | Method | URL |
|-------------|--------|-----|
| **Browser (local machine)** | SSH Tunnel required | http://localhost:8080/ |
| **Deployment verification** | SSH commands | Runs `curl` on droplet |
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

*Last Updated: 2026-01-21*
