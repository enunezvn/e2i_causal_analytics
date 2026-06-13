# Frontend env hygiene, CSP reconciliation, and edge-function proxy (#23 / #24)

## #23 — frontend/.env must be VITE-only

**Finding (2026-06-13):** the droplet's `frontend/.env` carried backend-only secrets
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_ACCESS_TOKEN`,
`DATABASE_URL`, `REDIS_URL`, `FALKORDB_HOST/PORT`, `DIGITALOCEAN_TOKEN`, `LLM_*`).

**Severity:** local hygiene footgun — **not** a disclosed-secret incident.
- `frontend/.env` is gitignored and has **0 commits in all branch history** → never pushed.
- Vite only bundles `VITE_*` vars → none of these non-`VITE_` keys were ever baked into the client bundle.
- Code audit: **none** of those keys are referenced by frontend code (the one `ANTHROPIC_API_KEY`
  hit is a docstring in the generated OpenAPI types; no direct `api.anthropic.com`/`api.openai.com`
  calls exist in `frontend/src`).

**Conclusion: no key rotation required.** Fix is hygiene only:
1. Strip the live droplet `frontend/.env` to the `VITE_*` contract (see `frontend/.env.example`).
2. The tracked `frontend/.env.example` documents the contract and the "no backend secrets here" rule.

**Live remediation (batch phase):** on the droplet, edit `frontend/.env` to keep only the
`VITE_*` contract (see `frontend/.env.example` for the authoritative list):
`VITE_API_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `VITE_COPILOT_ENABLED`,
`VITE_APP_VERSION`, `VITE_DEFAULT_MODEL_ID`, `VITE_DEBUG` (false), `VITE_MSW_ENABLED` (false).
Back up first. The running container reads its env from the compose/build, not this file, so
this is defense-in-depth + dev-box hygiene.

## #24 — CSP reconciliation + edge-function proxy

**Container CSP was stricter than host CSP.** `docker/frontend/nginx.conf` set
`connect-src 'self' wss:` while the host `docker/nginx/host-nginx.conf` set `connect-src 'self' wss: https:`.
On the public origin the host CSP took precedence (worked); on **direct `:3002`** access the
container's stricter CSP blocked XHR/fetch to https endpoints. Reconciled: container CSP
`connect-src` now includes `https:` so both paths behave identically. **Takes effect at the next
frontend image rebuild/deploy** (CSP is baked into the container).

**`/functions/v1` was not proxied.** Added a `location /functions/v1/` block to the host nginx,
proxying to Kong (`127.0.0.1:54321`) for parity with `/auth/`, `/rest/`, `/realtime/`.
The frontend makes **zero** `supabase.functions.invoke(...)` calls today, so this is
future-proofing — it prevents a silent 404 the first time an edge function ships. **Apply live**
by copying the tracked config to `/etc/nginx/sites-enabled/e2i-analytics`, `nginx -t`, reload
(see `frontend-serving-flip.md` for the live-copy procedure).
