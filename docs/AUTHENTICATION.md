# E2I API Authentication Guide

**Version**: 4.2.0
**Auth Provider**: Supabase Auth

---

## Overview

The E2I API uses JWT (JSON Web Token) authentication via Supabase Auth.
Protected endpoints require a valid JWT token in the Authorization header.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   React     │────▶│   Supabase   │────▶│   E2I API   │
│  Frontend   │     │     Auth     │     │  (FastAPI)  │
└─────────────┘     └──────────────┘     └─────────────┘
      │                    │                    │
      │  1. Login          │                    │
      │───────────────────▶│                    │
      │                    │                    │
      │  2. JWT Token      │                    │
      │◀───────────────────│                    │
      │                    │                    │
      │  3. API Request + Bearer Token          │
      │────────────────────────────────────────▶│
      │                    │                    │
      │                    │  4. Verify Token   │
      │                    │◀───────────────────│
      │                    │                    │
      │  5. Response       │                    │
      │◀────────────────────────────────────────│
```

---

## Setup Instructions

### Step 1: Create Supabase Users

1. Go to your [Supabase Dashboard](https://app.supabase.com)
2. Select your project
3. Navigate to **Authentication** → **Users**
4. Click **Add User** → **Create New User**
5. Enter email and password
6. Click **Create User**

**For admin users**, after creating the user:
1. Go to **Table Editor** → **auth.users**
2. Find the user row
3. Edit `raw_app_meta_data` to add: `{"role": "admin"}`

### Step 2: Configure Environment Variables

Add to your `.env` file on the droplet:

```bash
# Required for JWT validation
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Optional: For admin operations
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Optional: Restrict CORS origins (comma-separated)
ALLOWED_ORIGINS=https://your-frontend.com,http://localhost:3000
```

### Step 3: Restart the API

```bash
# On droplet
cd /root/Projects/e2i_causal_analytics
pkill -f uvicorn
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 &
```

---

## Public Endpoints (No Auth Required)

These endpoints are accessible without authentication:

| Method | Path | Description |
|--------|------|-------------|
| * | `/` | Root info |
| * | `/health` | Health check |
| * | `/healthz` | K8s health |
| * | `/ready` | Readiness check |
| * | `/api/docs` | Swagger UI |
| * | `/api/redoc` | ReDoc |
| * | `/api/openapi.json` | OpenAPI spec |
| GET | `/api/kpis` | List KPIs |
| GET | `/api/kpis/workstreams` | List workstreams |
| GET | `/api/kpis/health` | KPI health |
| GET | `/causal/estimators` | List estimators |
| GET | `/causal/health` | Causal health |
| GET | `/graph/health` | Graph health |
| GET | `/api/copilotkit/status` | CopilotKit status |

---

## Protected Endpoints (Auth Required)

All other endpoints require a valid JWT token:

- POST/PUT/DELETE on any endpoint
- `/memory/*` (except health)
- `/api/v1/rag/*` (except health)
- `/cognitive/*`
- `/experiments/*`
- `/digital-twin/*`
- `/monitoring/*` (write operations)

---

## Using Authentication

### From React Frontend (Supabase JS)

```typescript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

// Login
const { data, error } = await supabase.auth.signInWithPassword({
  email: 'user@example.com',
  password: 'password123'
})

// Get session token
const { data: { session } } = await supabase.auth.getSession()
const token = session?.access_token

// Call protected API endpoint
const response = await fetch('http://api.example.com/memory/search', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ query: 'market share' })
})
```

### From curl/CLI

```bash
# 1. Get a token (via Supabase REST API)
TOKEN=$(curl -s -X POST \
  'https://your-project.supabase.co/auth/v1/token?grant_type=password' \
  -H 'apikey: YOUR_ANON_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"password123"}' \
  | jq -r '.access_token')

# 2. Use the token
curl -X POST http://159.89.180.27:8001/memory/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "market share"}'
```

---

## Error Responses

### 401 Unauthorized - Missing Token
```json
{
  "error": "authentication_required",
  "message": "Missing Authorization header",
  "hint": "Include 'Authorization: Bearer <token>' header"
}
```

### 401 Unauthorized - Invalid Token
```json
{
  "error": "invalid_token",
  "message": "Invalid or expired token",
  "hint": "Login again to get a fresh token"
}
```

### 403 Forbidden - Admin Required
```json
{
  "error": "authentication_error",
  "message": "Admin privileges required"
}
```

---

## Testing Authentication

### Check if auth is enabled
```bash
curl http://159.89.180.27:8001/
# Response includes: "auth_enabled": true/false
```

### Test protected endpoint without token
```bash
curl -X POST http://159.89.180.27:8001/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
# Should return 401
```

### Test with valid token
```bash
curl -X POST http://159.89.180.27:8001/memory/search \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
# Should return search results
```

---

## Troubleshooting

### "Auth not configured" in logs
Ensure `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set in `.env`

### Token validation fails
- Check token hasn't expired (default 1 hour)
- Verify SUPABASE_URL matches the token issuer
- Ensure user exists and is confirmed in Supabase

### CORS errors from frontend
Add your frontend URL to `ALLOWED_ORIGINS` environment variable

---

## Security Notes

1. **Never commit tokens** - Use environment variables
2. **Use HTTPS in production** - Tokens are sensitive
3. **Rotate service keys** - If compromised, regenerate in Supabase
4. **Enable RLS** - Row-Level Security in Supabase for data isolation
5. **Monitor auth logs** - Check Supabase dashboard for suspicious activity
