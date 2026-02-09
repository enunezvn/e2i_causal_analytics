# Troubleshooting Notes

## Docker + Vite Environment Variable Precedence (2026-02-08)

**Problem:** Vite build kept producing old `VITE_SUPABASE_URL` value despite `.env` and `.env.production` files being updated inside the container.

**Root Cause Chain:**
1. `sed -i` on a bind-mounted file creates a new inode (temp file + rename). The container's bind mount still points to the OLD inode, so the container sees stale data even though the host file looks correct.
2. Docker `-e` env vars (set at `docker run` time) are baked into the container and CANNOT be changed without recreating the container.
3. Vite's `import.meta.env` includes `VITE_*` vars from BOTH `.env` files AND `process.env`. Docker `-e` vars land in `process.env` and **take priority** over `.env` file values.

**Debugging Steps That Revealed It:**
- `.env` files inside container showed correct values → misleading!
- `docker exec env | grep SUPABASE` showed the OLD value from Docker `-e` → this was the real source
- Build hash never changed despite "clearing cache" → env was identical from Vite's perspective

**Solution:**
1. Recreate the container (`docker stop && docker rm && docker run`) with corrected `-e VITE_SUPABASE_URL=...`
2. For production builds, override env at build time: `docker exec -e VITE_SUPABASE_URL=... container npx vite build`

**Lessons:**
- Always check `docker exec env | grep VAR` first — Docker `-e` vars override everything
- `sed -i` breaks bind mounts (use `cat old | sed > tmp && cat tmp > old` to write in-place without new inode)
- To change a Docker `-e` var, you MUST recreate the container — no shortcut

## Mixed Content / HTTPS→HTTP Browser Blocking (2026-02-08)

**Problem:** Frontend served over HTTPS (`https://localhost:8443`) tried to call Supabase API at `http://localhost:54321` (HTTP). Browser silently blocks this as mixed content — shows "Failed to fetch" with no helpful error.

**Solution:** Proxy the backend API through the same nginx HTTPS server:
- Add `location /auth/ { proxy_pass http://127.0.0.1:54321/auth/; ... }` etc.
- Set the frontend's API URL to same-origin (empty or `window.location.origin`)

**Runtime Origin Detection Pattern:**
Instead of hardcoding the Supabase URL at build time (which breaks when accessing via different domains like `localhost:8443` vs `eznomics.site`), use runtime detection:
```ts
supabaseUrl: getEnvVar('SUPABASE_URL', typeof window !== 'undefined' ? window.location.origin : '')
```
This way the app works regardless of which domain/port the user accesses it from, as long as nginx proxies the Supabase endpoints.

## Nginx Proxy for Supabase Self-Hosted

The Supabase Kong gateway at port 54321 exposes these paths that need proxying:
- `/auth/` — GoTrue auth service
- `/rest/` — PostgREST API
- `/realtime/` — WebSocket realtime (needs `Upgrade`/`Connection` headers)
- `/storage/` — File storage (needs `client_max_body_size`)

Config lives at: `/etc/nginx/sites-enabled/e2i-analytics` (outside git repo, server-level config)

## Axios 401 Interceptor Destroying Supabase Sessions (2026-02-08)

**Problem:** Navigating to pages that make backend API calls (Agent Orchestration, KPI Dictionary, Analytics, Experiments, Monitoring) redirected to `/login` even though the user was authenticated.

**Root Cause Chain:**
1. Backend API returns 401 (it doesn't validate Supabase JWTs — separate issue)
2. Axios error interceptor in `api-client.ts` had `clearAuth()` on ANY 401 response
3. This wiped the Zustand auth store (`session = null, user = null`)
4. `ProtectedRoute` checks `isAuthenticated` which derives from `Boolean(session?.access_token && user)`
5. With both nulled → redirect to `/login`

**Key Insight:** Supabase session (`e2i-auth-token` in localStorage) was valid the whole time. The Zustand store (`e2i-auth-store`) was being nuked by the interceptor, not by Supabase.

**Fix:** Remove `clearAuth()` from the 401 interceptor. Replace with `console.warn()`. Supabase session lifecycle should ONLY be managed by `AuthProvider`'s `onAuthStateChange` listener — never by API response interceptors.

**Debugging Steps That Revealed It:**
- Injected `initScript` via Chrome DevTools MCP to monkey-patch `console.*` and capture logs into `window.__consoleLogs` (the MCP `list_console_messages` tool returned empty for this app)
- Traced auth store writes by monkey-patching `localStorage.setItem` — saw `hasSession: true` briefly then `hasSession: false`
- Console trace showed: `SIGNED_IN` → `INITIAL_SESSION` → `API 401` → session cleared

**Deployment Gotcha — Nginx Serves Static Build, Not Vite Dev:**
- `https://localhost:8443` is served by nginx from `/var/www/html` (production build), NOT from the Vite dev container
- The Docker container with `npm run dev` and bind mounts is only reachable on port 5173 directly
- Editing source files via bind mount does NOT affect what nginx serves — must rebuild and copy:
  ```bash
  docker exec e2i_frontend_dev npx vite build
  sudo docker cp e2i_frontend_dev:/app/dist/. /var/www/html/
  ```
- Verify the correct JS chunk is referenced: `grep 'index-' /var/www/html/index.html`
- Verify fix is in the build: `grep -o '.{50}401 Unauthorized.{80}' /var/www/html/assets/index-*.js`

**Vite HMR + Docker Bind Mounts:**
- Vite's file watcher relies on inotify, which doesn't reliably propagate through Docker bind mounts (especially on WSL2)
- `touch`ing the file doesn't always trigger HMR either
- `docker restart` forces Vite to reload all modules from scratch — more reliable than relying on HMR for bind-mounted files

**Auth Architecture (for reference):**
- `e2i-auth-token` (localStorage) — Supabase's own session storage (access_token, refresh_token, user)
- `e2i-auth-store` (localStorage) — Zustand persist store (session, user, isAuthenticated, isInitialized)
- `AuthProvider.tsx` — calls `supabase.auth.getSession()` on mount, listens to `onAuthStateChange`
- `useAuth` hook — derives `isAuthenticated = Boolean(session?.access_token && user)` from Zustand store
- `ProtectedRoute` — checks `isAuthenticated` and `isInitialized`, redirects to `/login` if false

**Chrome DevTools MCP Console Capture Workaround:**
The `list_console_messages` MCP tool may return empty for SPAs. Use `initScript` on `navigate_page` to monkey-patch console methods:
```js
window.__consoleLogs = [];
['log','warn','error','debug'].forEach(level => {
  const orig = console[level];
  console[level] = function() {
    window.__consoleLogs.push({ level, msg: Array.from(arguments).map(a =>
      typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '), t: Date.now() });
    orig.apply(console, arguments);
  };
});
```
Then read with `evaluate_script`: `() => window.__consoleLogs`
