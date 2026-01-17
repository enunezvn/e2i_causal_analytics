# CopilotKit Chatbot Remediation Plan

**Created**: 2026-01-09
**Status**: In Progress
**Source**: CopilotKit Chatbot Audit (`.claude/plans/copilotkit-chatbot-audit.md`)

---

## Executive Summary

This plan addresses 5 bugs identified during the comprehensive CopilotKit chatbot audit. The fixes are organized into context-window friendly phases with incremental testing on the DigitalOcean droplet.

### Bugs to Fix

| Bug | Severity | Description | Root Cause |
|-----|----------|-------------|------------|
| #1 | Low | Documentation table names mismatch | Doc/code sync issue |
| #2 | High | Async Supabase client incompatible | `get_supabase_client()` returns sync client, code awaits it |
| #3 | Critical | No Redis checkpointer | `workflow.compile()` called without checkpointer param |
| #4 | Critical | Supabase message persistence fails | Same async issue as Bug #2 |
| #5 | High | Frontend CopilotKit disabled | `VITE_COPILOT_ENABLED=false` in build |

---

## Phase 1: Async Supabase Client Fix (Bug #2 + #4)

**Estimated Complexity**: Medium
**Files to Modify**: 3 files
**Testing**: Unit tests + droplet integration

### 1.1 Create Async Supabase Client Factory

**File**: `src/memory/services/factories.py`

**Task**: Add `get_async_supabase_client()` function

```python
# Add import at top
from supabase import create_async_client, AsyncClient

# Add new function after get_supabase_client()
_async_supabase_client: Optional[AsyncClient] = None

async def get_async_supabase_client() -> AsyncClient:
    """Get async Supabase client for use in async contexts."""
    global _async_supabase_client
    if _async_supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        _async_supabase_client = await create_async_client(url, key)
    return _async_supabase_client
```

### 1.2 Update Chatbot Tools

**File**: `src/api/routes/chatbot_tools.py`

**Changes**:
- Line 213: Change `client = await get_supabase_client()` to `client = await get_async_supabase_client()`
- Line 248: Same change in `_query_causal_chains()`
- Line 299: Same change in `_query_agent_analysis()`
- Line 330: Same change in `_query_triggers()`
- Line 647: Same change in `conversation_memory_tool()`

**Import Update**:
```python
from src.memory.services.factories import get_async_supabase_client
```

### 1.3 Update Chatbot Graph

**File**: `src/api/routes/chatbot_graph.py`

**Changes**:
- Line 184: Update `init_node()` to use `get_async_supabase_client()`
- Line 234: Update `load_context_node()` to use `get_async_supabase_client()`
- Line 478: Update `finalize_node()` to use `get_async_supabase_client()`

### 1.4 Testing Phase 1

- [ ] Run local unit tests for factories.py
- [ ] Run local unit tests for chatbot_tools.py
- [ ] Deploy to droplet
- [ ] Test `/api/copilotkit/` endpoint
- [ ] Verify no "object Client can't be used in 'await' expression" errors

---

## Phase 2: Redis Checkpointer Integration (Bug #3)

**Estimated Complexity**: Low
**Files to Modify**: 1 file
**Testing**: Integration test on droplet

### 2.1 Add Checkpointer to Workflow Compilation

**File**: `src/api/routes/chatbot_graph.py`

**Current Code** (line ~623):
```python
return workflow.compile()
```

**Fixed Code**:
```python
from src.memory.working_memory import get_langgraph_checkpointer

def build_chatbot_graph():
    # ... existing workflow definition ...

    checkpointer = get_langgraph_checkpointer()
    return workflow.compile(checkpointer=checkpointer)
```

### 2.2 Verify Redis Connection

**Pre-requisites**:
- Redis running on droplet at `redis://localhost:6382`
- `REDIS_URL` environment variable set (or use default)

### 2.3 Testing Phase 2

- [ ] Verify Redis is running on droplet: `redis-cli -p 6382 ping`
- [ ] Deploy updated code to droplet
- [ ] Send message with session_id
- [ ] Send follow-up message with same session_id
- [ ] Verify context is preserved (check Redis keys)
- [ ] Run: `redis-cli -p 6382 keys "*"`

---

## Phase 3: Frontend CopilotKit Enable (Bug #5)

**Estimated Complexity**: Low
**Files to Modify**: 1-2 files
**Testing**: Frontend rebuild + visual verification

### 3.1 Update Frontend Environment

**Option A: Update .env.production** (Recommended)

Create/update `frontend/.env.production`:
```
VITE_COPILOT_ENABLED=true
```

**Option B: Update Docker Compose Build Args**

**File**: `docker/docker-compose.yml`

Add build arg to frontend service:
```yaml
frontend:
  build:
    context: ../frontend
    dockerfile: ../docker/frontend/Dockerfile
    args:
      VITE_COPILOT_ENABLED: "true"
```

### 3.2 Rebuild Frontend

```bash
# SSH to droplet
ssh root@159.89.180.27

# Navigate to project
cd /root/e2i_causal_analytics

# Rebuild frontend
docker compose -f docker/docker-compose.yml build frontend

# Restart frontend service
docker compose -f docker/docker-compose.yml up -d frontend
```

### 3.3 Testing Phase 3

- [ ] Rebuild frontend container
- [ ] Access frontend at `http://159.89.180.27:5174`
- [ ] Verify CopilotKit chat widget appears
- [ ] Test sending a message through the chat interface
- [ ] Verify SSE streaming works (progressive response)

---

## Phase 4: Documentation Sync (Bug #1)

**Estimated Complexity**: Very Low
**Files to Modify**: Documentation only
**Testing**: Review only

### 4.1 Update Table Names in Documentation

**File**: `.claude/plans/copilotkit-chatbot-audit.md` (or relevant docs)

Align documentation table names with actual implementation:
- Verify `chat_sessions` vs actual table name
- Verify `chat_messages` vs actual table name
- Update any references to match `src/api/routes/chatbot_graph.py` finalize_node

### 4.2 Testing Phase 4

- [ ] Review documentation changes
- [ ] No deployment needed

---

## Phase 5: End-to-End Validation

**Purpose**: Verify all fixes work together

### 5.1 Complete Flow Test

```bash
# 1. SSH to droplet
ssh root@159.89.180.27

# 2. Check all services
docker compose -f docker/docker-compose.yml ps

# 3. Check Redis
redis-cli -p 6382 ping

# 4. Check API health
curl http://localhost:8000/api/health

# 5. Test CopilotKit endpoint
curl -X POST http://localhost:8000/api/copilotkit/ \
  -H "Content-Type: application/json" \
  -d '{"action": "chat", "message": "What agents are available?"}'
```

### 5.2 Frontend Integration Test

- [ ] Open frontend in browser
- [ ] CopilotKit widget visible
- [ ] Send: "What agents are available?"
- [ ] Verify streaming response
- [ ] Send follow-up question
- [ ] Verify context preserved from first message
- [ ] Check Redis for session data

### 5.3 Persistence Verification

```bash
# Check Redis for checkpointer data
redis-cli -p 6382 keys "*checkpoint*"

# Check Supabase for message persistence (via API or psql)
```

---

## Progress Tracker

### Phase 1: Async Supabase Client
- [ ] 1.1 Create `get_async_supabase_client()` in factories.py
- [ ] 1.2 Update chatbot_tools.py (5 locations)
- [ ] 1.3 Update chatbot_graph.py (3 locations)
- [ ] 1.4 Local testing
- [ ] 1.5 Droplet deployment
- [ ] 1.6 Integration testing

### Phase 2: Redis Checkpointer
- [ ] 2.1 Add checkpointer to workflow.compile()
- [ ] 2.2 Verify Redis running
- [ ] 2.3 Deploy to droplet
- [ ] 2.4 Test session persistence

### Phase 3: Frontend Enable
- [ ] 3.1 Update environment variable
- [ ] 3.2 Rebuild frontend
- [ ] 3.3 Deploy and verify

### Phase 4: Documentation
- [ ] 4.1 Update table name references

### Phase 5: E2E Validation
- [ ] 5.1 Full flow test
- [ ] 5.2 Frontend integration
- [ ] 5.3 Persistence verification

---

## Rollback Plan

If issues arise:

1. **Phase 1 Rollback**: Revert factories.py, chatbot_tools.py, chatbot_graph.py
2. **Phase 2 Rollback**: Remove checkpointer parameter (will use MemorySaver)
3. **Phase 3 Rollback**: Set `VITE_COPILOT_ENABLED=false`, rebuild

---

## Dependencies

- Redis running on port 6382
- Supabase credentials in environment
- Docker and docker-compose on droplet
- SSH access to droplet (159.89.180.27)

---

## Notes

- Each phase is designed to be completable in a single context window
- Phases 1-2 are backend, Phase 3 is frontend - can be parallelized
- Bug #4 is automatically fixed by Bug #2 fix (same root cause)
- Testing on droplet should use the existing SSH tunnel or direct access
