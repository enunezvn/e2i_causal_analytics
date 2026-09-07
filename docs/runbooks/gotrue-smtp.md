# GoTrue SMTP — activating the password-recovery email leg (#22)

## Problem

The public Supabase stack has `GOTRUE_MAILER_AUTOCONFIRM=true` (signup needs no email) but **no
SMTP configured**, so `POST /auth/v1/recover` (password reset) fails with 500 — the
`/forgot-password` → email → `/reset-password` flow is dead end-to-end. Signup is locked and the
reviewer account has known creds, so this is **low-urgency**, but the leg should work before any
non-admin user relies on self-service recovery.

## What's already in place (PR #918)

The tracked `docker/supabase/docker-compose.override.yml` `auth:` service deliberately defines
**no** `GOTRUE_SMTP_*` keys at all — its comment block says so explicitly. It is the **base**
`docker-compose.yml` of the Supabase stack that maps the `.env` vars
`SMTP_HOST/PORT/USER/PASS/ADMIN_EMAIL/SENDER_NAME`, `SITE_URL`, `ADDITIONAL_REDIRECT_URLS` and
`MAILER_URLPATHS_RECOVERY` onto the auth service's `GOTRUE_*` names. Redefining them in the
override would merge empties on top of that mapping and clobber it. The override carries only
`GOTRUE_EXTERNAL_EMAIL_ENABLED`, `GOTRUE_MAILER_AUTOCONFIRM` and `GOTRUE_DISABLE_SIGNUP`.
**No credentials live in git.** The DIAGNOSED root cause (2026-06-13): the live `.env` `SMTP_*`
values are Supabase's template **placeholders** — `SMTP_HOST=supabase-mail` (the Inbucket dev
catcher), `SMTP_ADMIN_EMAIL=admin@example.com`, `SMTP_SENDER_NAME=fake_sender` — and the
`supabase-mail` container is **not running**, so GoTrue can't connect → `POST /auth/v1/recover`
returns 500. `SITE_URL=http://138.197.4.36` (the droplet IP), so even with working SMTP the reset
link would point at the wrong origin instead of `https://eznomics.site`.

> NOTE: configure SMTP via the **`SMTP_*`** env vars (the base `docker-compose.yml` already maps
> them to the auth service's `GOTRUE_SMTP_*`). Do NOT add `GOTRUE_SMTP_*` to the override — that
> would clobber the base mapping with empties.

## Activation (requires an SMTP provider + credentials — a USER decision)

Pick a provider and obtain SMTP creds. Options, cheapest-faithful first:

| Provider | Notes |
|---|---|
| **SendGrid** | Free tier 100 emails/day; `smtp.sendgrid.net:587`, user=`apikey`, pass=`<API key>`. Simplest for transactional. |
| **AWS SES** | Cheap at scale; needs domain verification + move out of sandbox. `email-smtp.<region>.amazonaws.com:587`. |
| **Mailgun / Postmark** | Similar transactional model; Postmark has good deliverability. |
| **Gmail SMTP** | `smtp.gmail.com:587` + an App Password. Fine for a single low-volume reviewer flow; not for production scale. |

Then, on the droplet, REPLACE the placeholders in `/opt/supabase/docker/.env` (NOT the repo):

```
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASS=<provider secret>
SMTP_ADMIN_EMAIL=no-reply@eznomics.site   # verified sender
SMTP_SENDER_NAME=E2I Causal Analytics
# Fix the reset-link origin (live default is the bare droplet IP):
SITE_URL=https://eznomics.site
ADDITIONAL_REDIRECT_URLS=https://eznomics.site/reset-password
```

> **Do NOT set `MAILER_URLPATHS_RECOVERY=/reset-password`** (this runbook previously said to).
> GoTrue's default for that key is its own verification endpoint, **`/auth/v1/verify`** — which
> is what the live container has (`docker exec supabase-auth printenv
> GOTRUE_MAILER_URLPATHS_RECOVERY` → `/auth/v1/verify`, 2026-09-07). The emailed link is meant to
> hit GoTrue first so it can consume the recovery token; GoTrue then redirects the browser to the
> SPA route. The SPA route is reached via **`redirectTo`**, which the frontend already sends:
> `supabase.auth.resetPasswordForEmail(email, { redirectTo: \`${window.location.origin}/reset-password\` })`
> (`frontend/src/providers/AuthProvider.tsx:363-364`). Pointing `MAILER_URLPATHS_RECOVERY` at the
> SPA route instead would hand the SPA a token GoTrue never verified.
>
> So the recommendation is: leave `MAILER_URLPATHS_RECOVERY` at its default, and set `SITE_URL`
> + `ADDITIONAL_REDIRECT_URLS` so the `redirectTo` target is allowed.
>
> **UNCERTAIN end-to-end (2026-09-07):** this has never been walked through on this box — there
> is no working SMTP path here (see the DO block below), so no recovery email has ever been
> delivered. Treat the flow as reasoned-from-config, not certified.

Apply + verify:

```bash
cd /opt/supabase/docker && docker compose up -d auth      # recreate gotrue with new env
# Faithful test: trigger a real recovery and confirm a 200 + delivered email:
curl -s -X POST https://eznomics.site/auth/v1/recover \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_MAILBOX\"}" -o /dev/null -w '%{http_code}\n'   # expect 200, NOT 500
# Set TEST_MAILBOX in your shell to a mailbox you can read. Do not hard-code a
# personal address in this runbook.
```

Then walk the UI flow: `/forgot-password` → receive email → click link → `/reset-password` →
log in with the new password. The reset link must land on the public origin (covered by
`SITE_URL` + `ADDITIONAL_REDIRECT_URLS` above).

## ⚠️ CRITICAL: DigitalOcean blocks outbound SMTP on this droplet (verified 2026-06-13)

Gmail SMTP creds were added to `/opt/supabase/docker/.env` and the auth container loaded them
correctly (`GOTRUE_SMTP_HOST=smtp.gmail.com`, port 465, an app-password user, and
`SITE_URL=https://eznomics.site`) — **that state is gone as of 2026-09-07, see Status**.
But a live `POST /auth/v1/recover` returned **504 (10s timeout, `context deadline exceeded`)** — the
SMTP send hung. Cheap-disproof from the droplet AND inside the auth container:

```
port 465: BLOCKED/timeout   port 587: BLOCKED/timeout   port 25: BLOCKED/timeout
smtp.sendgrid.net:2525: OPEN   smtp.mailgun.org:2525: OPEN   api.sendgrid.com:443: OPEN
```

**DigitalOcean blocks outbound 25/465/587 by default (anti-spam).** So **Gmail SMTP cannot work
here** (Gmail offers only 465/587, no 2525). The creds were correct; the network path is blocked.

> That port table is a **2026-06-13 measurement** and has not been re-taken since. DO lifts the
> block per-account after review, so it can change without anything in this repo changing.
> Re-probe before relying on it (read-only, seconds):
>
> ```bash
> for hp in smtp.gmail.com:465 smtp.gmail.com:587 smtp.sendgrid.net:2525 api.sendgrid.com:443; do
>   timeout 5 bash -c "</dev/tcp/${hp%:*}/${hp##*:}" 2>/dev/null \
>     && echo "$hp OPEN" || echo "$hp BLOCKED/timeout"
> done
> ```
>
> Run it **inside the auth container too** — that is the path GoTrue actually takes:
> `docker exec supabase-auth sh -c '...'`.

### Working paths (pick one)
1. **Provider on port 2525** (RECOMMENDED — 2525 is open, no DO ticket needed). Use SendGrid /
   Mailgun / Postmark free tier:
   ```
   SMTP_HOST=smtp.sendgrid.net   SMTP_PORT=2525   SMTP_USER=apikey   SMTP_PASS=<sendgrid API key>
   SMTP_ADMIN_EMAIL=<verified sender>   SMTP_SENDER_NAME=E2I Causal Analytics
   ```
   then `cd /opt/supabase/docker && docker compose up -d auth` and re-run the recover test.
2. **Ask DigitalOcean to unblock SMTP** (support ticket; they lift 25/465/587 after account
   review). Then the current Gmail-on-587 config works (switch `SMTP_PORT=465`→`587`).
3. **GoTrue Send-Email Hook over HTTPS** (port 443 is open) — call a provider's REST API from a
   hook instead of SMTP. Most setup; only if 1 & 2 are unavailable.

## Status — re-checked 2026-09-07

**The live auth container is back on the template placeholders. The Gmail credentials this
runbook's §"CRITICAL" section describes are NOT loaded any more.** Whatever put them there
(a hand edit to `/opt/supabase/docker/.env`) has since been reverted or lost — most likely a
stack recreate from the template `.env`.

Measured, read-only, on 2026-09-07:

```bash
docker exec supabase-auth printenv | grep -E '^GOTRUE_(SMTP_HOST|SITE_URL)='
# GOTRUE_SMTP_HOST=supabase-mail
# GOTRUE_SITE_URL=http://138.197.4.36
```

| Key | Live value (2026-09-07) | Meaning |
| --- | --- | --- |
| `GOTRUE_SMTP_HOST` | `supabase-mail` | the Inbucket dev catcher — **placeholder**, and that container is not running |
| `GOTRUE_SMTP_PORT` | `2500` | Inbucket's port — **placeholder** |
| `GOTRUE_SMTP_USER` | placeholder (not an email address) | **placeholder** |
| `GOTRUE_SMTP_SENDER_NAME` | `fake_sender` | **placeholder** |
| `GOTRUE_SITE_URL` | `http://138.197.4.36` | bare droplet IP, not the public origin |
| `GOTRUE_MAILER_URLPATHS_RECOVERY` | `/auth/v1/verify` | GoTrue's own default — correct, leave it |
| `GOTRUE_MAILER_AUTOCONFIRM` | `true` | dev-era setting; revisit when SMTP lands |
| `GOTRUE_DISABLE_SIGNUP` | `true` | signup admin-locked (see `reviewer-provisioning.md`) |

So `POST /auth/v1/recover` is expected to fail here, and the state is the **pre-#918 baseline**,
not the Gmail configuration. Two separate blockers now stand between this box and a working
recovery email:

1. **No real SMTP credentials are configured** (this section) — fix by setting real values in
   `/opt/supabase/docker/.env` and `docker compose up -d auth`.
2. **DigitalOcean blocks outbound 25/465/587** (measured 2026-06-13, re-probe above) — so
   whatever provider is chosen must offer **port 2525**, or DO must lift the block.

Still low-urgency: signup is locked and reviewer accounts are admin-provisioned with known
credentials, so self-service recovery is rarely exercised. When it is picked up, path 1 in
"Working paths" (a 2525 provider) is the recommendation — it needs no infra ticket.
