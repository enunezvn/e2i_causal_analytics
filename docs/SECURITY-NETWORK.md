# E2I Causal Analytics - Network Security Architecture

**Version**: 1.0.0
**Date**: 2026-01-24
**Author**: E2I Causal Analytics Team

---

## Overview

This document describes the Docker network security architecture implemented for production deployment. The design follows the principle of **least privilege** - each service only has access to networks it needs for its function.

---

## Network Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               EXTERNAL TRAFFIC                                   │
│                                      │                                           │
│                               ┌──────▼──────┐                                   │
│                               │    nginx    │ ◄── public-network (port 80/443)  │
│                               │  (proxy)    │                                   │
│                               └──────┬──────┘                                   │
│                                      │                                           │
│  ┌───────────────────────────────────┼───────────────────────────────────┐      │
│  │                          app-network                                   │      │
│  │                                   │                                    │      │
│  │    ┌──────────────┐        ┌──────▼──────┐        ┌──────────────┐   │      │
│  │    │   frontend   │        │     api     │        │   workers    │   │      │
│  │    │   (React)    │        │  (FastAPI)  │        │   (Celery)   │   │      │
│  │    └──────────────┘        └──────┬──────┘        └──────┬───────┘   │      │
│  └───────────────────────────────────┼───────────────────────┼───────────┘      │
│                                      │                       │                   │
│     ┌────────────────────────────────┴───────────────────────┘                  │
│     │                                                                            │
│  ┌──┴─────────────────────────────┐    ┌──────────────────────────────────┐    │
│  │        data-network            │    │      monitoring-network          │    │
│  │        (internal: true)        │    │                                  │    │
│  │                                │    │                                  │    │
│  │  ┌────────┐    ┌──────────┐   │    │  ┌────────┐    ┌──────────┐     │    │
│  │  │ redis  │    │ falkordb │   │    │  │ mlflow │    │   opik   │     │    │
│  │  │(cache) │    │ (graph)  │   │    │  │(expmt) │    │  (obs)   │     │    │
│  │  └────────┘    └──────────┘   │    │  └────────┘    └──────────┘     │    │
│  └────────────────────────────────┘    └──────────────────────────────────┘    │
│           │                                                                      │
│           │                                                                      │
│     ┌─────┴───────────────────────────────────────────────────────────────┐    │
│     │                     celery-network (internal: true)                  │    │
│     │                                                                      │    │
│     │  ┌──────────────┐    ┌───────────────┐    ┌───────────────┐        │    │
│     │  │worker-light  │    │ worker-heavy  │    │   scheduler   │        │    │
│     │  │  (fast)      │    │ (compute)     │    │   (beat)      │        │    │
│     │  └──────────────┘    └───────────────┘    └───────────────┘        │    │
│     └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Networks

### 1. public-network

**Purpose**: External-facing network for ingress traffic.

| Property | Value |
|----------|-------|
| Type | Bridge |
| Internal | No |
| ICC | Disabled |
| Services | nginx only |

**Security Controls**:
- Only nginx is connected
- Inter-container communication disabled
- All other services unreachable from this network

### 2. app-network

**Purpose**: Application tier communication.

| Property | Value |
|----------|-------|
| Type | Bridge |
| Internal | No |
| Services | nginx, api, frontend, mlflow*, opik* |

**Security Controls**:
- Frontend only talks to nginx (reverse proxy)
- API receives requests from nginx
- MLflow/Opik accessible for UI proxying

### 3. data-network

**Purpose**: Data store access (Redis, FalkorDB).

| Property | Value |
|----------|-------|
| Type | Bridge |
| Internal | **Yes** (no external access) |
| Services | redis, falkordb, api, workers |

**Security Controls**:
- `internal: true` - No routes to external networks
- Only API and workers can access data stores
- Frontend cannot access data stores directly

### 4. monitoring-network

**Purpose**: MLOps service communication.

| Property | Value |
|----------|-------|
| Type | Bridge |
| Internal | No (for UI access via nginx) |
| Services | mlflow, opik, api, workers |

**Security Controls**:
- MLflow and Opik UIs proxied through nginx
- API and workers can log experiments/traces
- No direct external port exposure

### 5. celery-network

**Purpose**: Celery broker/result backend communication.

| Property | Value |
|----------|-------|
| Type | Bridge |
| Internal | **Yes** (no external access) |
| Services | redis, workers, scheduler |

**Security Controls**:
- `internal: true` - No routes to external networks
- Only Celery components can access broker
- Separate from data-network for isolation

---

## Service Network Access Matrix

| Service | public | app | data | monitoring | celery |
|---------|--------|-----|------|------------|--------|
| nginx | ✅ | ✅ | ❌ | ❌ | ❌ |
| frontend | ❌ | ✅ | ❌ | ❌ | ❌ |
| api | ❌ | ✅ | ✅ | ✅ | ✅ |
| redis | ❌ | ❌ | ✅ | ❌ | ✅ |
| falkordb | ❌ | ❌ | ✅ | ❌ | ❌ |
| mlflow | ❌ | ✅ | ❌ | ✅ | ❌ |
| opik | ❌ | ✅ | ❌ | ✅ | ❌ |
| worker-light | ❌ | ❌ | ✅ | ✅ | ✅ |
| worker-heavy | ❌ | ❌ | ✅ | ✅ | ✅ |
| scheduler | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Security Features

### 1. Container Hardening

All containers include:
- `security_opt: no-new-privileges:true` - Prevents privilege escalation
- `read_only: true` (where possible) - Immutable root filesystem
- Resource limits (CPU, memory) - Prevents resource exhaustion

### 2. No Direct Port Exposure

Services expose ports only via nginx proxy:
- Redis: No external port (was 6382)
- FalkorDB: No external port (was 6381)
- API: No external port (was 8000)
- Frontend: No external port (was 3001)
- MLflow: Proxied via /mlflow/
- Opik: Proxied via /opik/

### 3. Rate Limiting

Nginx implements rate limiting per endpoint:

| Zone | Rate | Endpoints |
|------|------|-----------|
| `api_limit` | 10 req/s | /api/* |
| `auth_limit` | 5 req/min | /api/auth/*, /api/login/* |
| `copilot_limit` | 30 req/s | /api/copilotkit/* |
| `general_limit` | 100 req/s | /* (frontend) |

### 4. Security Headers

Response headers added by nginx:
- `Strict-Transport-Security` - HSTS with preload
- `X-Frame-Options: SAMEORIGIN` - Clickjacking protection
- `X-Content-Type-Options: nosniff` - MIME sniffing protection
- `X-XSS-Protection: 1; mode=block` - XSS filter
- `Content-Security-Policy` - CSP policy
- `Permissions-Policy` - Feature restrictions

### 5. Request Correlation

All requests include:
- `X-Request-ID` header for end-to-end tracing
- Logged in nginx access logs
- Passed to API for correlation

---

## Deployment

### Using the Secure Configuration

```bash
# Deploy with secure network configuration
docker compose -f docker/docker-compose.secure.yml up -d

# Include debug tools (localhost only)
docker compose -f docker/docker-compose.secure.yml --profile debug up -d
```

### Generating SSL Certificates

For production, use Let's Encrypt:

```bash
# Install certbot
apt install certbot

# Generate certificate
certbot certonly --webroot -w /var/www/certbot -d your-domain.com

# Copy to nginx ssl directory
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem docker/nginx/ssl/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem docker/nginx/ssl/key.pem
```

For development, generate self-signed:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem \
  -subj "/CN=localhost"
```

### Environment Variables

Required in `.env`:

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
ANTHROPIC_API_KEY=sk-ant-...
REDIS_PASSWORD=strong-password-here

# Optional
CORS_ORIGINS=https://your-domain.com
```

---

## Monitoring

### Checking Network Isolation

```bash
# List networks
docker network ls | grep e2i

# Inspect network
docker network inspect e2i-data

# Verify service connectivity
docker exec e2i-api ping -c 1 redis     # Should succeed
docker exec e2i-frontend ping -c 1 redis # Should fail (not on data-network)
```

### Viewing Security Logs

```bash
# Nginx security log (blocked requests)
docker exec e2i-nginx tail -f /var/log/nginx/security.log

# Access log with request IDs
docker exec e2i-nginx tail -f /var/log/nginx/access.log
```

---

## Troubleshooting

### Service Cannot Reach Database

1. Verify service is on correct network:
   ```bash
   docker inspect <container> --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'
   ```

2. Check network connectivity:
   ```bash
   docker exec <container> nc -zv redis 6379
   ```

### Rate Limit Errors (429)

1. Check nginx rate limit logs:
   ```bash
   grep "limiting" /var/log/nginx/error.log
   ```

2. Adjust limits in `nginx.secure.conf` if needed

### SSL Certificate Issues

1. Verify certificates exist:
   ```bash
   ls -la docker/nginx/ssl/
   ```

2. Check certificate validity:
   ```bash
   openssl x509 -in docker/nginx/ssl/cert.pem -noout -dates
   ```

---

## References

- [Docker Network Security](https://docs.docker.com/network/network-tutorial-standalone/)
- [Nginx Security Configuration](https://www.nginx.com/blog/http-security-headers-deployment/)
- [OWASP Security Headers](https://owasp.org/www-project-secure-headers/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
