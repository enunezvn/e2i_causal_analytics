# Docker Compose Usage Guide

**Project**: E2I Causal Analytics
**Created**: 2025-12-18

---

## Quick Start

### Local Development

```bash
# 1. Copy the environment template
cp .env.template .env.dev

# 2. Edit .env.dev with your actual credentials
# - Set DATABASE_URL to your Supabase connection string
# - Set CLAUDE_API_KEY to your Claude API key
# - Set SUPABASE_URL and SUPABASE_ANON_KEY

# 3. Start all services in development mode
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or run in background
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 4. View logs
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# 5. Stop all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

### Production Deployment (Digital Ocean)

```bash
# 1. Copy the environment template
cp .env.template .env.prod

# 2. Edit .env.prod with production credentials
# - Generate secure SECRET_KEY and JWT_SECRET_KEY
# - Set production DATABASE_URL
# - Configure DO Spaces credentials
# - Set strict CORS_ORIGINS to your domain

# 3. Build and start production services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 4. View logs
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

# 5. Check health
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

---

## Service Access

### Development Mode

| Service | URL | Purpose |
|---------|-----|---------|
| **FastAPI** | http://localhost:8000 | Backend API |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Frontend** | http://localhost:3000 | React app |
| **MLflow** | http://localhost:5000 | Experiment tracking UI |
| **Redis** | localhost:6379 | Cache (password: devpassword) |

### Production Mode

| Service | URL | Purpose |
|---------|-----|---------|
| **Application** | https://yourdomain.com | All traffic via Nginx |
| **API** | https://yourdomain.com/api | Proxied to FastAPI |
| **MLflow** | https://yourdomain.com/mlflow | Proxied to MLflow (optional) |

---

## Common Commands

### Development

```bash
# Start services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Rebuild a specific service
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build fastapi

# View logs for specific service
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f fastapi

# Execute command in running container
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi python

# Run tests
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi pytest

# Stop and remove all containers
docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Stop and remove all containers + volumes (CAUTION: deletes data)
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
```

### Production

```bash
# Start with rebuild
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Rolling restart (zero downtime)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --build fastapi

# Scale agent workers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale agent-worker=4

# View resource usage
docker stats

# Check health of all services
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

---

## Environment Configuration

### Required Environment Variables

**Minimum for Development**:
```bash
DATABASE_URL=postgresql://...
CLAUDE_API_KEY=sk-ant-...
SUPABASE_URL=https://...
SUPABASE_ANON_KEY=eyJ...
```

**Additional for Production**:
```bash
APP_ENV=production
DEBUG=false
SECRET_KEY=<generate-with-openssl-rand-hex-32>
JWT_SECRET_KEY=<generate-with-openssl-rand-hex-32>
CORS_ORIGINS=https://yourdomain.com
DO_SPACES_KEY=...
DO_SPACES_SECRET=...
REDIS_PASSWORD=<secure-password>
```

### Generating Secure Keys

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate JWT_SECRET_KEY
openssl rand -hex 32

# Generate Redis password
openssl rand -base64 32
```

---

## Service Architecture

### Network Topology

```
┌─────────────────────────────────────────┐
│         frontend-network                │
│  ┌────────┐  ┌─────────┐  ┌──────────┐ │
│  │ nginx  │─→│ fastapi │  │ frontend │ │
│  └────────┘  └─────────┘  └──────────┘ │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         backend-network                 │
│  ┌─────────┐  ┌───────┐  ┌────────────┐│
│  │ fastapi │─→│ redis │  │   mlflow   ││
│  └─────────┘  └───────┘  └────────────┘│
│  ┌─────────────────┐                    │
│  │  agent-worker   │                    │
│  └─────────────────┘                    │
└─────────────────────────────────────────┘
         ↓
┌─────────────────┐
│ Cloud Supabase  │
│   (External)    │
└─────────────────┘
```

### Service Dependencies

```
nginx → fastapi, frontend
fastapi → redis, (supabase)
frontend → fastapi
mlflow → (supabase)
agent-worker → fastapi, redis
```

---

## Development Workflow

### Typical Development Session

```bash
# 1. Start services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# 2. Make code changes in ./src or ./frontend
# Changes are automatically reloaded (hot reload enabled)

# 3. Run tests
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi pytest tests/

# 4. Check logs if needed
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f fastapi

# 5. When done, stop services
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

### Debugging

```bash
# Access FastAPI container shell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi bash

# Check Python environment
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi python --version
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec fastapi pip list

# Test Redis connection
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec redis redis-cli -a devpassword ping

# View service configuration
docker compose -f docker-compose.yml -f docker-compose.dev.yml config
```

---

## Production Deployment

### Initial Deployment

```bash
# On Digital Ocean droplet:

# 1. Clone repository
git clone <your-repo-url>
cd e2i_causal_analytics

# 2. Create production environment file
cp .env.template .env.prod
nano .env.prod  # Fill in production values

# 3. Build and start services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 4. Verify all services are healthy
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

# 5. Check logs
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

# 6. Test health endpoints
curl http://localhost/health
```

### Updates and Deployments

```bash
# Pull latest code
git pull origin main

# Rebuild and restart (zero downtime)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Or rebuild specific service only
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --build fastapi
```

---

## Troubleshooting

### Services Won't Start

```bash
# Check logs for errors
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs

# Check if ports are already in use
sudo lsof -i :8000  # FastAPI
sudo lsof -i :3000  # Frontend
sudo lsof -i :5000  # MLflow

# Remove all containers and volumes, then restart
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Database Connection Issues

```bash
# Verify DATABASE_URL is correct in .env file
cat .env.dev | grep DATABASE_URL

# Test connection from FastAPI container
docker compose exec fastapi python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
connection = engine.connect()
print('Connection successful!')
connection.close()
"
```

### Hot Reload Not Working

```bash
# Ensure volumes are mounted correctly
docker compose -f docker-compose.yml -f docker-compose.dev.yml config | grep volumes -A 5

# Restart services
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart fastapi frontend
```

### Out of Disk Space

```bash
# Check disk usage
docker system df

# Remove unused images, containers, volumes
docker system prune -a --volumes

# Or selectively remove old images
docker image prune -a
```

---

## Performance Optimization

### Development

- Use Docker BuildKit for faster builds:
  ```bash
  export DOCKER_BUILDKIT=1
  export COMPOSE_DOCKER_CLI_BUILD=1
  ```

- Optimize volume mounts (delegated mode on macOS):
  - Already configured in docker-compose.dev.yml

### Production

- Resource limits are configured in docker-compose.prod.yml
- Adjust based on your droplet size
- Monitor with `docker stats`

---

## Security Checklist

**Development**:
- ✅ Use .env.dev (gitignored)
- ✅ Don't commit secrets

**Production**:
- ✅ Use strong passwords (32+ characters)
- ✅ Set APP_ENV=production
- ✅ Set DEBUG=false
- ✅ Configure strict CORS_ORIGINS
- ✅ Use HTTPS only (via Nginx)
- ✅ Don't expose internal ports (only 80/443)
- ✅ Rotate secrets quarterly

---

## Additional Resources

- Docker Compose CLI Reference: https://docs.docker.com/compose/reference/
- Docker Compose File Reference: https://docs.docker.com/compose/compose-file/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/

---

## Aliases (Optional)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Development
alias dc-dev-up='docker compose -f docker-compose.yml -f docker-compose.dev.yml up'
alias dc-dev-down='docker compose -f docker-compose.yml -f docker-compose.dev.yml down'
alias dc-dev-logs='docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f'
alias dc-dev-ps='docker compose -f docker-compose.yml -f docker-compose.dev.yml ps'

# Production
alias dc-prod-up='docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d'
alias dc-prod-down='docker compose -f docker-compose.yml -f docker-compose.prod.yml down'
alias dc-prod-logs='docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f'
alias dc-prod-ps='docker compose -f docker-compose.yml -f docker-compose.prod.yml ps'
```

---

**Last Updated**: 2025-12-18
