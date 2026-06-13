# Runbook: Frontend Serving Flip (static /var/www/html -> e2i_frontend container)

**Owner:** deploy orchestrator (manual, one-time at deploy)
**Branch that prepared this:** `fix/fe-serving-auth-build`
**Canonical config:** `docker/nginx/host-nginx.conf` (tracked) -> `/etc/nginx/sites-available/e2i-analytics` (live)

## Background

Until this flip, `https://eznomics.site` was served by HOST nginx from a static
`/var/www/html` snapshot built locally on 2026-02-09. That bundle is stale,
dev-flavored, and bakes `VITE_SUPABASE_URL=https://localhost:8443` (login fails
from any remote browser). The CI-built `e2i_frontend` container (nginx
production stage, compose `3002:80`, healthy) received no public traffic.

The tracked `docker/nginx/host-nginx.conf` now:

1. proxies `/` and `/assets/` to `127.0.0.1:3002` (the container),
2. includes the four Supabase proxy blocks (`/auth/`, `/rest/`, `/realtime/`,
   `/storage/` -> Kong `54321`) that were previously live-only drift,
3. strips the container's own security headers via `proxy_hide_header` so the
   host-level CSP/security headers stay authoritative (duplicate CSP headers
   are enforced as their intersection by browsers and would break the app),
4. returns an explicit 404 for `/mockServiceWorker.js` so browsers that
   registered the stale pre-flip mock service worker unregister it on their
   next update check (the SPA fallback would otherwise answer with 200/HTML
   and keep the registration alive forever).

## Pre-flight checks

```bash
# 1. Frontend container healthy and serving a CURRENT, PROD-flavored bundle
docker ps --filter name=e2i_frontend --format '{{.Status}}'   # expect: Up ... (healthy)
curl -fsS http://127.0.0.1:3002/health                        # expect: OK
# Image must include the fix/fe-serving-auth-build Dockerfile fix (prod bundle):
docker exec e2i_frontend sh -c 'ls /usr/share/nginx/html/mockServiceWorker.js' 2>&1
#   expect: No such file or directory
docker exec e2i_frontend sh -c 'grep -l "localhost:8443" /usr/share/nginx/html/assets/*.js' 2>&1
#   expect: no matches

# 2. Supabase Kong answering locally
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:54321/auth/v1/settings  # expect 401 (no apikey) - proves routing

# 3. Tracked config vs live: review the diff you are about to apply
sudo diff /etc/nginx/sites-enabled/e2i-analytics docker/nginx/host-nginx.conf
```

## Flip

```bash
# 1. Back up live config
sudo cp /etc/nginx/sites-enabled/e2i-analytics \
        /etc/nginx/sites-available/e2i-analytics.bak-$(date +%Y%m%d-%H%M%S)

# 2. Install tracked config (sites-enabled is expected to be a symlink to
#    sites-available; if it is a regular file, replace it the same way)
sudo cp docker/nginx/host-nginx.conf /etc/nginx/sites-available/e2i-analytics

# 3. Validate and reload (reload, NOT restart - keeps existing connections)
sudo nginx -t && sudo systemctl reload nginx
```

## Verify

```bash
# Public origin now serves the container bundle (asset hashes must match the
# container, NOT the old /var/www/html snapshot index-Cj8IDC6X.js)
curl -s https://eznomics.site/ | grep -o 'assets/index-[^"]*\.js'
docker exec e2i_frontend sh -c 'ls /usr/share/nginx/html/assets/index-*.js'

# No localhost:8443 in the served bundle
BUNDLE=$(curl -s https://eznomics.site/ | grep -o 'assets/index-[^"]*\.js' | head -1)
curl -s "https://eznomics.site/$BUNDLE" | grep -c 'localhost:8443'   # expect 0

# Supabase auth via public origin (anon key from frontend/.env.production)
curl -s -H "apikey: $ANON_KEY" https://eznomics.site/auth/v1/settings | head -c 200
# expect JSON settings, not 401/404

# Real login round-trip (use a known account)
curl -s -X POST "https://eznomics.site/auth/v1/token?grant_type=password" \
  -H "apikey: $ANON_KEY" -H "Content-Type: application/json" \
  -d '{"email":"<user>","password":"<pass>"}' | head -c 200
# expect access_token JSON

# Backend paths unaffected
curl -s -o /dev/null -w '%{http_code}\n' https://eznomics.site/health    # 200
curl -s -o /dev/null -w '%{http_code}\n' https://eznomics.site/api/docs  # 200
```

## Rollback

```bash
sudo cp /etc/nginx/sites-available/e2i-analytics.bak-<TS> /etc/nginx/sites-available/e2i-analytics
sudo nginx -t && sudo systemctl reload nginx
```

## Retire /var/www/html (AFTER verified flip; archive, do not delete)

`/var/www/html` itself must continue to exist: the HTTP:80 server block still
uses it as the ACME webroot for certbot renewals
(`location /.well-known/acme-challenge/`).

```bash
# Archive the stale app snapshot
sudo tar -czf /root/var-www-html-frontend-snapshot-$(date +%Y%m%d).tar.gz \
    -C /var/www/html assets index.html mockServiceWorker.js vite.svg

# Remove only the app files (keep the directory and any .well-known content)
sudo rm -rf /var/www/html/assets /var/www/html/index.html \
            /var/www/html/mockServiceWorker.js /var/www/html/vite.svg

# Confirm certbot renewal still works
sudo certbot renew --dry-run
```

## Ongoing operation

- CI (`deploy.yml` `build-and-push-frontend` + `deploy`) builds and pulls the
  frontend image on every main push; the container restart picks up new
  bundles automatically. No web-root copying is involved anymore.
- Any future host nginx edits MUST be made in `docker/nginx/host-nginx.conf`
  first and copied to the box (this file is how the 2026-02-09 drift happened).
