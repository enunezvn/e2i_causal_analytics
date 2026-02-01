# Production Hardening Plan — E2I Causal Analytics

**Created**: 2026-01-30
**Domain**: eznomics.site (Hostinger)
**Droplet**: e2i-analytics-prod (138.197.4.36)
**Registry**: GitHub Container Registry (ghcr.io)
**Overall Score**: 87/100 → Target 95/100

---

## Phase 1: DNS + Let's Encrypt SSL (P0)

**Objective**: Point `eznomics.site` at the droplet and enable trusted HTTPS via Let's Encrypt.

**Files to modify**:
- Droplet: `/etc/nginx/sites-available/e2i-app` — add SSL server block

**Steps**:

1. **Configure DNS in Hostinger**:
   - Add an **A record**: `@` → `138.197.4.36` (TTL 300)
   - Add an **A record**: `www` → `138.197.4.36` (TTL 300)
   - Wait for propagation (~5-15 min), verify with `dig eznomics.site`

2. **Install certbot on droplet**:
   ```bash
   sudo apt update && sudo apt install -y certbot python3-certbot-nginx
   ```

3. **Update nginx server_name** in `/etc/nginx/sites-available/e2i-app`:
   - Change `server_name _;` to `server_name eznomics.site www.eznomics.site;`

4. **Obtain Let's Encrypt certificate**:
   ```bash
   sudo certbot --nginx -d eznomics.site -d www.eznomics.site
   ```
   Certbot will auto-modify the nginx config to add SSL directives and HTTP→HTTPS redirect.

5. **Harden the nginx config** (merge from `docker/nginx/nginx.secure.conf`):
   - Add `server_tokens off;`
   - Add security headers: `X-Frame-Options`, `X-Content-Type-Options`, `HSTS`, `CSP`, `Permissions-Policy`
   - Add rate limiting zones: `api_limit 10r/s`, `general_limit 100r/s`
   - Add malicious path blocking (`.php`, `.env`, `.git`, `wp-*`)
   - Keep `/health` accessible on port 80 (for load balancer probes)

6. **Verify auto-renewal**: `sudo certbot renew --dry-run`

**Verification (on droplet)**:
```bash
curl -I https://eznomics.site/health         # 200 + security headers
curl -I http://eznomics.site/                 # 301 → https
curl -I https://eznomics.site/ 2>&1 | grep Strict  # HSTS present
```

**Done when**: `https://eznomics.site/health` returns 200 with valid cert and security headers.

---

## Phase 2: Firewall Hardening (P0)

**Objective**: Close all internal service ports. Services accessible only via nginx proxy or SSH tunnels.

**Steps**:

1. **Verify nginx proxy routes work** before closing ports:
   ```bash
   curl -sk https://eznomics.site/mlflow/     # proxies to :5000
   curl -sk https://eznomics.site/opik/        # proxies to :5173
   curl -sk https://eznomics.site/falkordb/    # proxies to :3030
   curl -sk https://eznomics.site/api/health   # proxies to :8000
   ```

2. **Remove UFW rules** for internal-only ports (delete in reverse numbered order):
   | Port | Service | Action |
   |------|---------|--------|
   | 54321 | Supabase Kong | **DELETE** |
   | 6382 | Redis | **DELETE** |
   | 6381 | FalkorDB | **DELETE** |
   | 5174 | Opik Backend | **DELETE** |
   | 5173 | Opik UI | **DELETE** |
   | 5000 | MLflow | **DELETE** |
   | 3030 | FalkorDB Browser | **DELETE** |
   | 3000 | BentoML | **DELETE** |
   | 8000 | API (direct) | **DELETE** — nginx proxies `/api/` |

3. **Final UFW state** (3 ports only):
   | Port | Purpose |
   |------|---------|
   | 22 | SSH |
   | 80 | HTTP (redirects to HTTPS) |
   | 443 | HTTPS (all traffic) |

4. **Reload**: `sudo ufw reload`

**Verification**:
```bash
# Should work (via nginx proxy):
curl -sk https://eznomics.site/api/health
curl -sk https://eznomics.site/mlflow/

# Should timeout (ports closed):
curl -m 3 http://138.197.4.36:6382   # Redis
curl -m 3 http://138.197.4.36:5000   # MLflow direct

# SSH tunnel still works:
ssh -L 6382:localhost:6382 -i ~/.ssh/replit enunez@138.197.4.36
```

**Done when**: Only ports 22, 80, 443 open. All services reachable via `https://eznomics.site/*` proxy paths.

---

## Phase 3: Fix Health Check + Register cohort_constructor (P1)

**Objective**: Fix the Supabase readiness probe and register the missing agent in config.

**Files to modify**:
- `src/api/dependencies/supabase_client.py` — fix `supabase_health_check()` at line 122
- `config/agent_config.yaml` — add `cohort_constructor` to `agents:` and `priority_order:`

**Steps**:

### 3a. Fix Supabase Health Check
1. Read `src/api/dependencies/supabase_client.py` (lines 99-147)
2. Replace the `business_metrics` table query with a simpler connectivity test:
   - Use PostgREST endpoint ping or `SELECT 1` via RPC
   - Remove the "table check failed" fallback path
3. Add unit test in `tests/unit/test_api/` to verify graceful handling

### 3b. Register cohort_constructor
1. Add `cohort_constructor` entry to `config/agent_config.yaml` in the `agents:` section (Tier 0, after `observability_connector` ~line 467)
2. Add `cohort_constructor` to the `priority_order:` list (~line 972)
3. Verify the agent is already in `src/agents/factory.py` AGENT_REGISTRY_CONFIG (no Python changes needed)

**Verification (on droplet after deploy)**:
```bash
curl -s http://localhost:8000/ready | python3 -m json.tool
# supabase should show "healthy" without "table check failed" note

grep cohort_constructor config/agent_config.yaml | head -3
# should appear in agents section and priority_order
```

**Done when**: `/ready` reports clean Supabase health; `cohort_constructor` in config.

---

## Phase 4: Backend CI Workflow (P1)

**Objective**: Create a GitHub Actions workflow that runs pytest on PRs touching backend code.

**File to create**:
- `.github/workflows/backend-tests.yml`

**Steps**:

1. Create workflow with these jobs:

   | Job | Purpose | Command |
   |-----|---------|---------|
   | **lint** | Ruff linting | `ruff check src/ tests/` |
   | **unit-tests** | Unit tests + coverage | `pytest tests/unit/ --cov -n 4 --dist=loadscope --timeout=30` |
   | **integration-tests** | Integration (with Redis) | `pytest tests/integration/ -n 4 --timeout=60` |
   | **ci-success** | Gate check | Depends on unit-tests |

2. Triggers: `push`/`pull_request` to `main`/`develop` on paths `src/**`, `tests/**`, `pyproject.toml`
3. Python 3.11, pip cache, `requirements.txt` install
4. Environment: `E2I_TESTING_MODE=true`, mock service URLs
5. Integration tests use `services:` block for Redis container
6. Coverage upload to Codecov
7. **Constraint**: Max 4 workers (`-n 4`), 30s timeout — per pyproject.toml

**Reference files**:
- `.github/workflows/frontend-tests.yml` — job structure template
- `.github/workflows/security.yml` — Python setup pattern
- `pyproject.toml` lines 155-248 — pytest + coverage config

**Verification**: Open a test PR touching `src/` — workflow triggers and passes.

**Done when**: Backend CI runs on PRs with lint + unit tests + coverage reporting.

---

## Phase 5: Docker CD Pipeline with GHCR (P2)

**Objective**: Build Docker images in CI, push to GHCR, pull and deploy on the droplet on merge to main.

**Files to create**:
- `.github/workflows/deploy.yml` — CD workflow
- `scripts/deploy-docker.sh` — deployment script for droplet

**Steps**:

### 5a. GHCR Setup
1. Authenticate GHCR using GitHub token (already available in Actions as `GITHUB_TOKEN`)
2. Image naming: `ghcr.io/<owner>/e2i-api:latest` and `ghcr.io/<owner>/e2i-api:<sha>`
3. No extra accounts needed — GHCR is included with GitHub

### 5b. CI Build Job
1. After backend-tests pass, build the API Docker image using `docker/fastapi/Dockerfile`
2. Tag with `latest` and commit SHA
3. Push to `ghcr.io`
4. Use `docker/buildx-action` for caching (layer cache speeds up builds)

### 5c. Deploy Job
1. SSH into droplet using `appleboy/ssh-action@v1`
2. Run `scripts/deploy-docker.sh` which:
   - `docker login ghcr.io` with GitHub token
   - `docker pull ghcr.io/<owner>/e2i-api:latest`
   - Restart the API service (or swap containers with zero-downtime)
   - Health check: `curl -sf https://eznomics.site/health`
3. GitHub Secrets needed: `DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`

### 5d. Rollback
- Keep previous image tagged as `previous`
- If health check fails, `docker tag previous latest && restart`

**Verification**: Merge a PR to main → image builds → pushes to GHCR → deploys to droplet → health check passes.

**Done when**: Automated Docker-based deployment works end-to-end on merge to main.

---

## Phase 6: Test Coverage to 70% (P2)

**Objective**: Identify lowest-coverage modules and write targeted tests to reach 70%.

**Approach**: Biggest gaps first — maximize covered lines per test written.

**Steps** (split across 2-3 sub-sessions):

### 6a. Coverage Analysis
1. Run full coverage on droplet:
   ```bash
   pytest tests/unit/ --cov=src --cov-report=term-missing -n 4 --dist=loadscope 2>&1 | tail -100
   ```
2. Identify top 10-15 modules below 50% with largest line counts
3. Create prioritized list of modules to target

### 6b. Write Tests (batch 1 — highest impact modules)
- Target ~5 modules per sub-session
- Focus on public API methods, mock external deps
- Follow patterns from existing `tests/unit/test_agents/` fixtures
- Run each batch on droplet to verify

### 6c. Write Tests (batch 2 — remaining gaps)
- Continue until overall coverage crosses 70%
- Update `pyproject.toml` `fail_under = 70`

**Verification**:
```bash
pytest tests/unit/ --cov=src --cov-report=term -n 4 | grep TOTAL
# TOTAL ... 70%+
```

**Done when**: `pytest --cov` reports >= 70%. `fail_under` updated in pyproject.toml.

---

## Phase 7: Documentation Update (Wrap-up)

**Objective**: Update all documentation to reflect hardening changes.

**Files to modify**:
- `INFRASTRUCTURE.md` — SSL, UFW, domain, CD pipeline, GHCR
- `CLAUDE.md` — update URLs to `https://eznomics.site`, version bump, CD commands
- `README.md` — update access URLs, add CI/CD badges
- `docker/nginx/ssl/README.md` — update cert instructions (Let's Encrypt replaces self-signed)

**Steps**:
1. Update all `http://138.197.4.36` references to `https://eznomics.site`
2. Document UFW rules (ports 22, 80, 443 only)
3. Document SSH tunnel instructions for internal services
4. Add GHCR image references and CD workflow description
5. Add CI/CD badges to README
6. Mark SSL/TLS as complete in INFRASTRUCTURE.md checklist

**Done when**: All docs reflect current production state.

---

## Phase Summary

| # | Phase | Priority | Status | Notes |
|---|-------|----------|--------|-------|
| 1 | DNS + Let's Encrypt SSL | P0 | ✅ Done | Droplet: nginx + certbot |
| 2 | Firewall Hardening | P0 | ✅ Done | Droplet: UFW 22/80/443 only |
| 3 | Health Check + cohort_constructor | P1 | ✅ Done | Commit 26855c9 |
| 4 | Backend CI Workflow | P1 | ✅ Done | Commit 26855c9 |
| 5 | Docker CD Pipeline (GHCR) | P2 | ✅ Done | Commit 26855c9 |
| 6 | Test Coverage to 70% | P2 | ✅ Done | 70.59% — batch 1 (aa5cf9d), batch 2 (8e99ecf), batch 3 (memory_hooks + fixes) |
| 7 | Documentation Update | — | ✅ Done | Commit 26855c9 |

## Prerequisites Checklist

- [x] DNS: Point `eznomics.site` A record → `138.197.4.36` in Hostinger DNS panel
- [x] DNS: Point `www.eznomics.site` A record → `138.197.4.36`
- [ ] GitHub: Add repository secrets (`DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`)
- [ ] GitHub: Ensure GHCR is enabled for the repository (Settings → Packages)
