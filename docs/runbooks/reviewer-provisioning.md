# Reviewer Account Provisioning (signup is admin-locked)

Public self-registration on the eznomics.site Supabase (GoTrue) is **disabled**
(`GOTRUE_DISABLE_SIGNUP: "true"` — locked 2026-06-12, decision record: task #21
of the frontend functional-review-readiness remediation). Strangers hitting
`/auth/v1/signup` receive `422 signup_disabled`. Existing accounts and logins
are completely unaffected — this gate touches account *creation* only.

Accounts are provisioned through the GoTrue **admin API** using the service-role
key. Admin-created accounts are confirmed directly (`email_confirm: true`), so
the missing-SMTP gap (GoTrue has no mailer configured) does not block onboarding.

## Where the config lives

- **Live**: `/opt/supabase/docker/docker-compose.override.yml` line ~81
  (the supabase compose project runs from `/opt/supabase/docker`, NOT the repo).
- **Repo mirror**: `docker/supabase/docker-compose.override.yml` — keep them in
  sync; the nginx split-brain taught us what drift costs.
- Apply a change with: `cd /opt/supabase/docker && docker compose up -d auth`
  then verify via the settings probe below.

## Provision a reviewer

```bash
ANON=$(docker exec e2i_api printenv SUPABASE_ANON_KEY)
SVC=$(docker exec e2i_api printenv SUPABASE_SERVICE_ROLE_KEY)
EMAIL="reviewer@example.com"
PW=$(openssl rand -base64 15 | tr -d '/+=' | head -c 16)

# Create (409/422 email_exists → use the password-reset flow below instead)
curl -s -X POST "https://eznomics.site/auth/v1/admin/users" \
  -H "apikey: $SVC" -H "Authorization: Bearer $SVC" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PW\",\"email_confirm\":true}"

# Prove the login before handing out credentials
curl -s -X POST "https://eznomics.site/auth/v1/token?grant_type=password" \
  -H "apikey: $ANON" -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PW\"}" | grep -c access_token
```

## Reset an existing account's password

```bash
# Find the user id
curl -s "https://eznomics.site/auth/v1/admin/users?per_page=100" \
  -H "apikey: $SVC" -H "Authorization: Bearer $SVC" | jq -r \
  '.users[] | select(.email=="reviewer@example.com") | .id'

# Set the new password (replaces the old one)
curl -s -X PUT "https://eznomics.site/auth/v1/admin/users/<id>" \
  -H "apikey: $SVC" -H "Authorization: Bearer $SVC" \
  -H "Content-Type: application/json" \
  -d "{\"password\":\"$PW\",\"email_confirm\":true}"
```

Note: `UID` is a readonly bash builtin — don't use it as the variable name.

## Deprovision

```bash
curl -s -X DELETE "https://eznomics.site/auth/v1/admin/users/<id>" \
  -H "apikey: $SVC" -H "Authorization: Bearer $SVC"
```

## Verify / flip the lock

```bash
# Current state (expects "disable_signup": true)
curl -s "https://eznomics.site/auth/v1/settings" -H "apikey: $ANON" | jq .disable_signup

# To re-open signup: set GOTRUE_DISABLE_SIGNUP: "false" in BOTH the live file
# and the repo mirror, then: cd /opt/supabase/docker && docker compose up -d auth
```

## Related follow-ups

- SMTP is unconfigured (`/auth/v1/recover` → 500): the in-app password-recovery
  *email* leg stays dead until ops adds a mailer. Admin resets (above) are the
  workaround. When SMTP lands, revisit `GOTRUE_MAILER_AUTOCONFIRM` (currently
  `"true"`, a dev-era setting) and consider verified-email signup or invite-only.
