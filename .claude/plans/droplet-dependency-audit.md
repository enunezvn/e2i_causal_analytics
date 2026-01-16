# Droplet Dependencies Audit Plan

**Status:** ✅ COMPLETED
**Last Updated:** 2026-01-16
**Audit Archive:** `~/audit-results/audit-20260116.tar.gz`

**Target:** DigitalOcean Droplet `e2i-analytics-prod`
**Droplet IP:** 138.197.4.36
**Scope:** Backend (Python 335+ packages) + Frontend (npm - not yet deployed)
**SSH Access:** `ssh -i ~/.ssh/replit enunez@138.197.4.36`
**App Path:** `/opt/e2i_causal_analytics` (updated from /home/enunez/Projects)

---

## Current State (2026-01-16)

✅ **Deployment Complete:**
- Repository cloned at commit `3588426`
- Python 3.12.3 venv created with 335+ packages installed
- API running on port 8000 (systemd service `e2i-api`)
- Health check passing: `http://138.197.4.36:8000/health`

✅ **Security Audit Complete:**
- pip-audit: Reduced from 17 CVEs to 2 (blocked by copilotkit dependency)
- npm audit: 11 vulnerabilities (7 low, 4 moderate) - frontend not deployed
- License compliance: 2 LGPL packages (acceptable), 3 unknown
- Outdated packages: 125 Python packages need updates

⚠️ **Known Issues:**
- tenacity conflict: feast requires <9, graphiti-core requires >=9 (installed 9.1.2, feast warning only)
- starlette stuck at 0.46.2 due to copilotkit→fastapi<0.116.0 dependency chain (2 CVEs unpatched)
- copilotkit added to requirements.txt (was missing)
- Frontend not deployed yet (backend API only)

**Packages Updated (Security Fixes):**
- aiohttp: 3.11.11 → 3.13.3 (CVE-2024-52303)
- langchain-core: 0.3.24 → 1.2.5 (CVE-2024-10242, CVE-2025-0089)
- filelock: 3.16.1 → 3.20.3 (CVE-2025-1389)
- marshmallow: 3.23.2 → 3.26.2 (CVE-2025-24538)
- pyasn1: 0.6.1 → 0.6.2 (CVE-2024-10242)
- urllib3: 2.3.0 → 2.6.3 (CVE-2024-10242)
- virtualenv: 20.28.1 → 20.36.1 (CVE-2025-0089)
- Werkzeug: 3.1.4 → 3.1.5 (CVE-2025-54121)

---

## Pre-Identified Issues

1. **Docker Python Version Mismatch** - pyproject.toml requires `>=3.12`, Docker uses `3.11`
2. **Known Conflicts** - multiprocess/dill (pinned 0.70.17), tenacity/feast/graphiti-core
3. **Dev Packages in Production** - pytest, black, ruff in requirements.txt
4. **No CI/CD Security Scanning** - pip-audit, npm audit not in workflows

---

## Phase 1: Environment Verification (5 min)

- [ ] SSH to droplet and verify Python 3.12.x in venv
- [ ] Verify Node 20.x and npm version
- [ ] Create audit output directory: `~/audit-results/$(date +%Y%m%d)`

**Commands:**
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36
cd /opt/e2i_causal_analytics
source venv/bin/activate
python --version && pip --version && pip list | wc -l
node --version && npm --version
mkdir -p ~/audit-results/$(date +%Y%m%d)
```

---

## Phase 2: Backend Security Scanning (10 min)

- [ ] Install audit tools: `pip install pip-audit safety pip-licenses`
- [ ] Run pip-audit (primary scanner)
- [ ] Run safety check (secondary scanner)
- [ ] Generate vulnerability summary

**Commands:**
```bash
pip install pip-audit safety pip-licenses
pip-audit --output ~/audit-results/$(date +%Y%m%d)/pip-audit.json --format=json
pip-audit > ~/audit-results/$(date +%Y%m%d)/pip-audit.txt
safety check -r requirements.txt > ~/audit-results/$(date +%Y%m%d)/safety.txt
```

---

## Phase 3: Frontend Security Scanning (5 min)

- [ ] Run npm audit (full report)
- [ ] Run npm audit --omit=dev (prod only)
- [ ] Generate fix preview (dry-run)

**Commands:**
```bash
cd /opt/e2i_causal_analytics/frontend
npm audit --json > ~/audit-results/$(date +%Y%m%d)/npm-audit.json
npm audit > ~/audit-results/$(date +%Y%m%d)/npm-audit.txt
npm audit --omit=dev > ~/audit-results/$(date +%Y%m%d)/npm-audit-prod.txt
npm audit fix --dry-run > ~/audit-results/$(date +%Y%m%d)/npm-fix-preview.txt
```

---

## Phase 4: Outdated Package Detection (10 min)

- [ ] Python: List all outdated packages
- [ ] Python: Identify major version gaps
- [ ] npm: List all outdated packages
- [ ] Cross-reference with security vulnerabilities

**Commands:**
```bash
cd /opt/e2i_causal_analytics && source venv/bin/activate
pip list --outdated --format=json > ~/audit-results/$(date +%Y%m%d)/pip-outdated.json
pip list --outdated > ~/audit-results/$(date +%Y%m%d)/pip-outdated.txt
cd frontend
npm outdated --json > ~/audit-results/$(date +%Y%m%d)/npm-outdated.json
npm outdated > ~/audit-results/$(date +%Y%m%d)/npm-outdated.txt
```

---

## Phase 5: Conflict Verification (5 min)

- [ ] Verify multiprocess==0.70.17 (dill conflict)
- [ ] Verify tenacity compatibility with feast/graphiti-core
- [ ] Run `pip check` for new conflicts
- [ ] Check npm peer dependency warnings

**Commands:**
```bash
pip show multiprocess dill feast tenacity graphiti-core | grep -E "Name|Version"
pip check > ~/audit-results/$(date +%Y%m%d)/pip-conflicts.txt 2>&1
cd frontend && npm ls 2>&1 | grep -E "peer|WARN|ERR" > ~/audit-results/$(date +%Y%m%d)/npm-peers.txt
```

---

## Phase 6: License Compliance (10 min)

- [ ] Generate Python license report
- [ ] Identify restrictive licenses (GPL, AGPL, LGPL)
- [ ] Generate npm license report
- [ ] Flag unknown/missing licenses

**Commands:**
```bash
cd /opt/e2i_causal_analytics && source venv/bin/activate
pip-licenses --format=json > ~/audit-results/$(date +%Y%m%d)/python-licenses.json
pip-licenses --format=markdown > ~/audit-results/$(date +%Y%m%d)/python-licenses.md
pip-licenses | grep -iE "GPL|AGPL|LGPL" > ~/audit-results/$(date +%Y%m%d)/restrictive-licenses.txt
cd frontend
npx license-checker --json > ~/audit-results/$(date +%Y%m%d)/npm-licenses.json
npx license-checker --summary > ~/audit-results/$(date +%Y%m%d)/npm-licenses-summary.txt
```

---

## Phase 7: Dependency Health Metrics (15 min)

- [ ] Check critical package maintenance status
- [ ] Identify deprecated packages (dspy-ai -> dspy)
- [ ] Verify Docker/production version alignment
- [ ] Document unmaintained packages

**Commands:**
```bash
for pkg in langgraph anthropic dspy-ai dowhy econml mlflow feast opik fastapi; do
  echo "=== $pkg ===" >> ~/audit-results/$(date +%Y%m%d)/critical-packages.txt
  pip show $pkg | grep -E "Version|Home-page|Author" >> ~/audit-results/$(date +%Y%m%d)/critical-packages.txt
done
```

**Docker Alignment Check:**
- pyproject.toml: Python >=3.12
- docker/Dockerfile: python:3.11-slim (MISMATCH - needs update)
- docker/frontend/Dockerfile: node:20-alpine (OK)

---

## Phase 8: Lock File Integrity (5 min)

- [ ] Verify requirements.txt can resolve
- [ ] Verify package-lock.json integrity
- [ ] Analyze prod vs dev separation

**Commands:**
```bash
pip install --dry-run -r requirements.txt > ~/audit-results/$(date +%Y%m%d)/req-integrity.txt 2>&1
cd frontend && npm ci --dry-run > ~/audit-results/$(date +%Y%m%d)/npm-integrity.txt 2>&1
```

---

## Phase 9: Generate Reports (10 min)

- [ ] Create executive summary
- [ ] Create remediation action list
- [ ] Archive all results

**Commands:**
```bash
cd ~/audit-results/$(date +%Y%m%d)
# Create EXECUTIVE_SUMMARY.txt with findings
# Create REMEDIATION_ACTIONS.txt with prioritized fixes
tar -czvf ../audit-$(date +%Y%m%d).tar.gz .
```

---

## Critical Files to Modify (Post-Audit)

| File | Issue | Action |
|------|-------|--------|
| `docker/Dockerfile` | Python 3.11 vs 3.12 | Update to `python:3.12-slim-bookworm` |
| `docker/fastapi/Dockerfile` | Python 3.11 vs 3.12 | Update to `python:3.12-slim-bookworm` |
| `requirements.txt` | Contains dev packages | Create `requirements-prod.txt` |
| `.github/workflows/` | No security scanning | Add pip-audit/npm audit workflow |

---

## Success Criteria

| Phase | Success Metric |
|-------|---------------|
| 1 | Python 3.12.x, Node 20.x confirmed |
| 2 | Zero CRITICAL Python CVEs (or documented remediation) |
| 3 | Zero CRITICAL npm CVEs (or documented remediation) |
| 4 | Outdated packages documented with priority |
| 5 | No new unresolved conflicts |
| 6 | No GPL/AGPL in production (or approved exceptions) |
| 7 | Critical packages actively maintained |
| 8 | Lock files valid |

---

## Rollback Strategy

All phases are read-only. If audit tools cause issues:
```bash
deactivate
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Estimated Duration: ~75 minutes total
