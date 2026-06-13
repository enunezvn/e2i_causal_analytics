# GoTrue SMTP — activating the password-recovery email leg (#22)

## Problem

The public Supabase stack has `GOTRUE_MAILER_AUTOCONFIRM=true` (signup needs no email) but **no
SMTP configured**, so `POST /auth/v1/recover` (password reset) fails with 500 — the
`/forgot-password` → email → `/reset-password` flow is dead end-to-end. Signup is locked and the
reviewer account has known creds, so this is **low-urgency**, but the leg should work before any
non-admin user relies on self-service recovery.

## What's already in place (PR #918)

The tracked `docker/supabase/docker-compose.override.yml` `auth:` service now interpolates the
the `.env` vars `SMTP_*`, `SITE_URL`, `ADDITIONAL_REDIRECT_URLS`, `MAILER_URLPATHS_RECOVERY`.
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
MAILER_URLPATHS_RECOVERY=/reset-password
ADDITIONAL_REDIRECT_URLS=https://eznomics.site/reset-password
```

Apply + verify:

```bash
cd /opt/supabase/docker && docker compose up -d auth      # recreate gotrue with new env
# Faithful test: trigger a real recovery and confirm a 200 + delivered email:
curl -s -X POST https://eznomics.site/auth/v1/recover \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d '{"email":"etn3724@gmail.com"}' -o /dev/null -w '%{http_code}\n'   # expect 200, NOT 500
```

Then walk the UI flow: `/forgot-password` → receive email → click link → `/reset-password` →
log in with the new password. The reset link must land on the public origin (covered by
`SITE_URL` + `ADDITIONAL_REDIRECT_URLS` above).

## ⚠️ CRITICAL: DigitalOcean blocks outbound SMTP on this droplet (verified 2026-06-13)

Gmail SMTP creds were added to `/opt/supabase/docker/.env` and the auth container loaded them
correctly (`GOTRUE_SMTP_HOST=smtp.gmail.com:465`, user `etn3724@gmail.com`, `SITE_URL=https://eznomics.site`).
But a live `POST /auth/v1/recover` returned **504 (10s timeout, `context deadline exceeded`)** — the
SMTP send hung. Cheap-disproof from the droplet AND inside the auth container:

```
port 465: BLOCKED/timeout   port 587: BLOCKED/timeout   port 25: BLOCKED/timeout
smtp.sendgrid.net:2525: OPEN   smtp.mailgun.org:2525: OPEN   api.sendgrid.com:443: OPEN
```

**DigitalOcean blocks outbound 25/465/587 by default (anti-spam).** So **Gmail SMTP cannot work
here** (Gmail offers only 465/587, no 2525). The creds are correct; the network path is blocked.

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

## Status

Config is CORRECT and the auth container loads it; **the live blocker is DO's outbound-SMTP
firewall, not creds**. Gmail is infeasible on this droplet. Resolution = a 2525 provider (path 1,
no infra change) or a DO SMTP-unblock ticket (path 2). Low-urgency: signup is locked and the
reviewer account has known creds, so self-service recovery is rarely exercised.
