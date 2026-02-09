# E2I Project Memory

## Hard-Won Lessons (see [troubleshooting.md](troubleshooting.md) for details)
- **Docker `-e` vars override `.env` files in Vite builds** — always check `docker exec env | grep VAR` first
- **`sed -i` breaks Docker bind mounts** (new inode) — use `cat | sed > tmp && cat tmp > file` instead
- **To change Docker `-e` vars, you MUST recreate the container** — no shortcut
- **HTTPS→HTTP = silent "Failed to fetch"** — proxy backend through same nginx HTTPS server
- **Use `window.location.origin` as runtime Supabase URL default** — works for any access domain
- **Never `clearAuth()` on backend API 401** — Supabase session != backend auth; only `onAuthStateChange` should manage session
- **nginx serves `/var/www/html` (static build), NOT the Vite dev container** — must `vite build` + `docker cp` to deploy changes
- **Vite HMR unreliable with Docker bind mounts on WSL2** — `docker restart` is more reliable than relying on inotify
- **MCP `list_console_messages` may return empty** — use `initScript` monkey-patch workaround (see troubleshooting.md)

## Frontend Auth (SOLVED 2026-02-08)
- Login: `admin@e2i.local` / `E2iAdmin2026Secure` (created via Supabase admin API)
- Supabase proxied through nginx at `/auth/`, `/rest/`, `/realtime/`, `/storage/`
- `VITE_SUPABASE_URL` is empty in production build; runtime defaults to `window.location.origin`
- Nginx proxy config: `/etc/nginx/sites-enabled/e2i-analytics` (outside git, server-level)

## Services (verified 2026-02-08)
1. **Frontend** - https://localhost:8443 (HTTPS) or http://localhost:3002 (HTTP)
2. **Supabase Studio** - http://localhost:3001
3. **FalkorDB Browser** - http://localhost:3030 (password: `changeme`)
4. **MLflow** - http://localhost:5000 (credentials: `admin`/`admin`)
5. **Opik UI** - http://localhost:5173

## Setup Done
- Updated MCP config at `/home/enunez/Projects/.claude.json` to use `--browserUrl http://127.0.0.1:9222 --acceptInsecureCerts`
- Chromium launched at `DISPLAY=:0 chromium-browser --remote-debugging-port=9222 --no-first-run --no-default-browser-check --ignore-certificate-errors`
- SSH tunnels script: `/mnt/c/Programming_Projects/e2i/ssh-tunnels/tunnels.sh` connects to `enunez@138.197.4.36`
- Created `TUNNEL_URLS.md` with all 20 tunnel URLs

## Key Files
- `.env` - credentials for all services (Supabase, Redis, FalkorDB, Grafana, MLflow, Flower, Opik)
- `ssh-tunnels/tunnels.sh` - quick tunnel launcher (20 port forwards)
- `ssh-tunnels/setup.sh` - persistent systemd autossh setup
- `ssh-tunnels/ssh-config-snippet` - SSH config host entries

## Chrome DevTools MCP
- Chromium is at `/usr/bin/chromium-browser` (not `/opt/google/chrome/chrome`)
- No sudo access, so can't symlink. Instead configured `--browserUrl` to connect to running instance.
- Must have Chromium running with `--remote-debugging-port=9222` before MCP tools work.

## Credentials from .env
- MLflow: admin / admin
- Grafana: admin / admin
- Flower: admin / FlowerAdminPass2026
- FalkorDB: password=changeme
- Opik MinIO: opik-minio-admin / ij6kwztIHmzAg1KuQjZNhCgk
- Redis: password=changeme
