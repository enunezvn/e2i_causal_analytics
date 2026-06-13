# GoTrue SMTP — activating the password-recovery email leg (#22)

## Problem

The public Supabase stack has `GOTRUE_MAILER_AUTOCONFIRM=true` (signup needs no email) but **no
SMTP configured**, so `POST /auth/v1/recover` (password reset) fails with 500 — the
`/forgot-password` → email → `/reset-password` flow is dead end-to-end. Signup is locked and the
reviewer account has known creds, so this is **low-urgency**, but the leg should work before any
non-admin user relies on self-service recovery.

## What's already in place (PR #918)

The tracked `docker/supabase/docker-compose.override.yml` `auth:` service now interpolates the
SMTP vars from the live `/opt/supabase/docker/.env`. **No credentials live in git.** Until
`GOTRUE_SMTP_HOST` is set, the block is inert (defaults are empty) and behavior is unchanged.

## Activation (requires an SMTP provider + credentials — a USER decision)

Pick a provider and obtain SMTP creds. Options, cheapest-faithful first:

| Provider | Notes |
|---|---|
| **SendGrid** | Free tier 100 emails/day; `smtp.sendgrid.net:587`, user=`apikey`, pass=`<API key>`. Simplest for transactional. |
| **AWS SES** | Cheap at scale; needs domain verification + move out of sandbox. `email-smtp.<region>.amazonaws.com:587`. |
| **Mailgun / Postmark** | Similar transactional model; Postmark has good deliverability. |
| **Gmail SMTP** | `smtp.gmail.com:587` + an App Password. Fine for a single low-volume reviewer flow; not for production scale. |

Then, on the droplet, add to `/opt/supabase/docker/.env` (NOT the repo):

```
GOTRUE_SMTP_HOST=smtp.sendgrid.net
GOTRUE_SMTP_PORT=587
GOTRUE_SMTP_USER=apikey
GOTRUE_SMTP_PASS=<provider secret>
GOTRUE_SMTP_ADMIN_EMAIL=no-reply@eznomics.site   # verified sender
GOTRUE_SMTP_SENDER_NAME=E2I Causal Analytics
# Optional overrides (defaults already point at the public origin):
# GOTRUE_SITE_URL=https://eznomics.site
# GOTRUE_MAILER_URLPATHS_RECOVERY=/reset-password
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
`GOTRUE_SITE_URL` / `GOTRUE_URI_ALLOW_LIST`).

## Status

Config-ready (PR #918). **Blocked on the SMTP provider + credentials decision** — these are
external and the user's to provide. No code/repo change can complete it without creds.
