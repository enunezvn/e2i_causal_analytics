# CopilotKit Chatbot Debug Plan

**Status**: 🟢 RESOLVED - v1.20.0 Deployed and Verified
**Last Updated**: 2026-01-10
**Priority**: Closed

---

## Problem Statement

CopilotKit React SDK (v1.50.1) is not rendering AI assistant responses in the chat UI, despite the backend (Python copilotkit v0.1.74) correctly streaming events.

## Root Cause Analysis (Updated)

**The issue has THREE components:**

### 1. Streaming Format Mismatch (FIXED in v1.15.0)
- **Backend was using**: NDJSON format (`application/x-ndjson` with `{json}\n`)
- **SDK expects**: SSE format (`text/event-stream` with `data: {json}\n\n`)
- **Status**: ✅ FIXED

### 2. Event Type Casing Mismatch (FIXED in v1.16.0)
- **Backend sends (v1.13.0+)**: PascalCase types (`TextMessageStart`, `RunStarted`)
- **SDK expects**: SCREAMING_SNAKE_CASE types (`TEXT_MESSAGE_START`, `RUN_STARTED`)
- **Status**: ✅ FIXED

### 3. Missing Required Fields on ALL Event Types (FIXED in v1.19.0)
- **Backend was sending**: `timestamp: null`, `source: null` on non-lifecycle events
- **SDK expects**: `timestamp: number`, `source: string` on ALL events
- **Status**: ✅ FIXED - v1.19.0 adds timestamp and source to ALL events

### 4. Null Fields in MESSAGES_SNAPSHOT Messages (FIXED in v1.20.0)
- **Backend was sending**: `name: null`, `toolCalls: null` in message objects
- **SDK expects**: `name: string` (empty allowed), `toolCalls: array` (empty allowed)
- **Status**: ✅ FIXED - v1.20.0 converts null to empty string/array in messages

**Evidence**: Zod validation error from browser console during Jan 10 deployment test:
```
ZodError: [
  { "path": ["timestamp"], "message": "Expected number, received null" },
  { "path": ["source"], "message": "Expected string, received null" }
]
```

**Root Cause (Resolved in v1.19.0)**: The `_fix_lifecycle_event()` function in v1.18.0 only fixed `RUN_STARTED` and `RUN_FINISHED` events. Other event types (`TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`, `STATE_SNAPSHOT`, `MESSAGES_SNAPSHOT`, etc.) still emitted `null` for `timestamp` and `source` fields. **Fixed in v1.19.0** by renaming to `_fix_all_events()` and adding timestamp/source to ALL events.

---

## Version History

| Version | Change | Result |
|---------|--------|--------|
| v1.9.5 | Fixed 39s delay via fresh thread_id | Delay fixed, messages still not rendering |
| v1.10.0 | Refactored to use SDK's `copilotkit_emit_message()` | Did not deploy correctly |
| v1.13.0 | Changed event types from SCREAMING_SNAKE_CASE to PascalCase | **WRONG DIRECTION** |
| v1.14.0 | Changed `content` field to `delta` in TextMessageContent | Correct per AG-UI |
| v1.15.0 | Changed NDJSON to SSE format | Correct format, still fails due to casing |
| v1.16.0 | Reverted event types to SCREAMING_SNAKE_CASE | Still fails - Zod validation errors |
| v1.17.0 | Fixed null fields in RUN_STARTED/RUN_FINISHED (timestamp, parentRunId, input) | Still fails - input needs RunAgentInput structure |
| v1.18.0 | ✅ Fixed input field to contain full RunAgentInput structure | Curl tests pass, browser still fails |
| **v1.18.0 Deployment Test** | Deployed with Vite proxy, tested in browser | ❌ **FAILED** - Zod errors on timestamp/source |
| **v1.19.0** | ✅ Added timestamp/source to ALL events, renamed `_fix_lifecycle_event` to `_fix_all_events` | AI message renders, new Zod errors on messages |
| **v1.20.0** | ✅ Fixed null name/toolCalls in MESSAGES_SNAPSHOT messages | ✅ **RESOLVED** - Full functionality working |

---

## Jan 10 Deployment Test Results

A full 6-phase deployment was executed to test the v1.18.0 backend fixes with proper frontend proxy configuration.

### Deployment Plan Executed

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Pre-Flight Check | ✅ COMPLETE | SSH tunnels established (ports 5174, 8001) |
| Phase 2: Comprehensive Backup | ✅ COMPLETE | 7 files backed up to `/root/backups/copilot-fix-20260110-2036/` |
| Phase 3: Deploy Fix Files | ✅ COMPLETE | vite.config.ts, main.py, chatbot_graph.py deployed |
| Phase 4: Enable Test Mode | ✅ COMPLETE | Frontend switched to `npm run dev` with Vite proxy |
| Phase 5: Verification Testing | ❌ PARTIAL FAIL | See test results below |
| Phase 6B: Rollback | ✅ COMPLETE | All files restored from backup |

### Test Results

| Test | Result |
|------|--------|
| Frontend loads at localhost:5174 | ✅ PASS |
| Vite proxy forwards /api requests | ✅ PASS |
| CopilotKit Inspector shows "Connected" | ✅ PASS |
| Chat widget opens correctly | ✅ PASS |
| User message renders in chat | ✅ PASS |
| **AI message renders in chat** | ❌ FAIL |

### Files Deployed (Now Rolled Back)

```
/tmp/vite.config.ts.droplet → /root/Projects/.../frontend/vite.config.ts
Chatbot fix/main.py → /root/Projects/.../src/api/main.py
Chatbot fix/chatbot_graph.py → /root/Projects/.../src/api/routes/chatbot_graph.py
```

### Backup Location

All original files preserved at:
```
/root/backups/copilot-fix-20260110-2036/
├── vite.config.ts
├── main.py
├── chatbot_graph.py
├── .env
├── .env.production
├── .env.development.local
└── e2i-frontend.service
```

---

## Current State (v1.18.0) - ⚠️ PARTIAL FIX

### What's Correct
- SSE format (`data: {...}\n\n`)
- `text/event-stream` content type
- `delta` field for TextMessageContent
- All lifecycle events present (RUN_STARTED, RUN_FINISHED)
- MessagesSnapshot emitted
- **✅ Event types use SCREAMING_SNAKE_CASE (FIXED in v1.16.0)**
- **✅ Lifecycle events have all required fields (FIXED in v1.17.0)**
- **✅ RUN_STARTED.input contains full RunAgentInput structure (FIXED in v1.18.0)**
- **✅ Vite proxy configuration works (CONFIRMED in Jan 10 test)**
- **✅ CopilotKit SDK connects successfully (CONFIRMED in Jan 10 test)**

**Backend now emits (v1.18.0):**
```json
{"type": "RUN_STARTED", "timestamp": 1768074328117, "parentRunId": "", "threadId": "...", "runId": "...", "input": {...}}
{"type": "TEXT_MESSAGE_START", "messageId": "...", "role": "assistant", "timestamp": null, "source": null}  ← PROBLEM
{"type": "TEXT_MESSAGE_CONTENT", "messageId": "...", "delta": "...", "timestamp": null, "source": null}  ← PROBLEM
{"type": "TEXT_MESSAGE_END", "messageId": "...", "timestamp": null, "source": null}  ← PROBLEM
{"type": "RUN_FINISHED", "timestamp": ..., "parentRunId": "", "output": {"messages": []}}
```

### Remaining Issue: Missing Fields on Non-Lifecycle Events

The `_fix_lifecycle_event()` function only processes `RUN_STARTED` and `RUN_FINISHED` events. Other event types pass through with `null` values for `timestamp` and `source`, which causes Zod validation errors in the CopilotKit React SDK.

**Events needing fixes:**
- `TEXT_MESSAGE_START` - needs `timestamp`, `source`
- `TEXT_MESSAGE_CONTENT` - needs `timestamp`, `source`
- `TEXT_MESSAGE_END` - needs `timestamp`, `source`
- `STATE_SNAPSHOT` - needs `timestamp`, `source`
- `MESSAGES_SNAPSHOT` - needs `timestamp`, `source`
- Any other AG-UI event types emitted

---

## Fixes Applied (v1.16.0 - v1.18.0)

### File: `src/api/routes/copilotkit.py`

### v1.16.0 Changes (Event Type Casing)

1. **Deprecated `_screaming_snake_to_pascal()`** - No longer called
2. **TEXT_MESSAGE events use SCREAMING_SNAKE_CASE**
3. **Lifecycle events use SCREAMING_SNAKE_CASE**

### v1.17.0 Changes (Null Field Fixes)

Added `_fix_lifecycle_event()` helper to fix null fields:
- timestamp: number (milliseconds since epoch)
- parentRunId: string (empty string instead of null)
- threadId: string
- runId: string

### v1.18.0 Changes (RunAgentInput Structure)

Fixed `input` field in RUN_STARTED to contain full RunAgentInput structure:
```python
event_dict["input"] = {
    "threadId": thread_id,
    "runId": run_id,
    "messages": [],
    "tools": [],
    "context": []
}
```

Fixed `output` field in RUN_FINISHED to contain structure:
```python
event_dict["output"] = {
    "messages": []
}
```

---

## Environment

| Component | Version/Details |
|-----------|-----------------|
| Frontend SDK | @copilotkit/react-core v1.50.1 |
| Backend SDK | copilotkit (Python) v0.1.74 |
| Production IP | 159.89.180.27 |
| Frontend Port | 5174 (vite preview) |
| Backend Port | 8001 (uvicorn) |
| API Endpoint | `/api/copilotkit` |

---

## Testing Protocol

### 1. Deploy v1.16.0 to Droplet
```bash
scp -i ~/.ssh/replit src/api/routes/copilotkit.py root@159.89.180.27:/root/Projects/e2i_causal_analytics/src/api/routes/copilotkit.py
```

### 2. Restart Backend
```bash
ssh -i ~/.ssh/replit root@159.89.180.27 "pkill -9 -f 'uvicorn' && cd /root/Projects/e2i_causal_analytics && nohup /root/Projects/e2i_causal_analytics/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8001 > /root/fastapi.log 2>&1 &"
```

### 3. Verify via Curl
```bash
curl -X POST "http://localhost:8001/api/copilotkit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $E2I_API_KEY" \
  -d '{"messages":[{"role":"user","content":"test"}],"threadId":"test-123"}'
```

Expected output:
```
data: {"type": "RUN_STARTED", ...}

data: {"type": "TEXT_MESSAGE_START", "messageId": "...", "role": "assistant"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "...", "delta": "..."}

data: {"type": "TEXT_MESSAGE_END", "messageId": "..."}

data: {"type": "RUN_FINISHED", ...}
```

### 4. Test in Browser
- Establish SSH tunnels: `ssh -i ~/.ssh/replit -L 5174:localhost:5174 -L 8001:localhost:8001 root@159.89.180.27`
- Open http://localhost:5174
- Send a message in the chatbot
- Verify AI response renders

---

## Related Files

| File | Purpose |
|------|---------|
| `src/api/routes/copilotkit.py` | Backend CopilotKit endpoint (NEEDS FIX) |
| `frontend/src/providers/E2ICopilotProvider.tsx` | CopilotKit provider config |
| `src/agents/explainer/` | Explainer agent (handles chat) |

---

## Completed Tasks

- [x] Diagnose CopilotKit chatbot response delay (39s)
- [x] Fix delay by ensuring fresh thread_id per request (v1.9.5)
- [x] Verify frontend receives streaming events
- [x] Fix streaming format from NDJSON to SSE (v1.15.0)
- [x] Fix TextMessageContent field from `content` to `delta` (v1.14.0)
- [x] Identify root cause: SDK expects SCREAMING_SNAKE_CASE event types
- [x] Fix event type casing to SCREAMING_SNAKE_CASE (v1.16.0)
- [x] Fix null fields in RUN_STARTED/RUN_FINISHED events (v1.17.0)
- [x] Fix input field structure with full RunAgentInput (v1.18.0)
- [x] Verify backend fix via curl test ✅
- [x] Configure Vite proxy for SSH tunnel testing ✅ (Jan 10)
- [x] Deploy and test with Vite dev mode ✅ (Jan 10)
- [x] Verify CopilotKit SDK connection ✅ (Jan 10)
- [x] Identify NEW issue: timestamp/source fields on all events (Jan 10)

## Remaining Tasks

- [x] **Fix timestamp/source fields on ALL event types (v1.19.0)** ✅
- [x] **Fix null name/toolCalls in MESSAGES_SNAPSHOT (v1.20.0)** ✅
- [x] Re-deploy and verify chatbot renders messages in browser ✅

**All tasks completed. Issue resolved.**

---

## Key Discoveries

### Discovery 1: PascalCase vs SCREAMING_SNAKE_CASE (v1.13.0)

**The v1.13.0 change to PascalCase was based on an incorrect assumption.**

The CopilotKit documentation and some examples show PascalCase in JavaScript/TypeScript code, but the actual AG-UI protocol wire format and Zod schema validation use SCREAMING_SNAKE_CASE.

### Discovery 2: Required Fields on ALL Events (Jan 10 Test)

**The CopilotKit React SDK (v1.50.1) requires `timestamp` and `source` on ALL event types, not just lifecycle events.**

The v1.18.0 fix only addressed `RUN_STARTED` and `RUN_FINISHED`. The Zod validation error on `TEXT_MESSAGE_START` proves the SDK validates these fields on every event.

---

## How to Continue

**Backend needs v1.19.0 fix** to add `timestamp` and `source` fields to ALL events.

### Required Fix for v1.19.0

Modify `src/api/routes/copilotkit.py` to fix ALL events, not just lifecycle events:

```python
import time

def _fix_all_events(event_dict: dict, thread_id: str, run_id: str) -> dict:
    """Ensure ALL events have required timestamp and source fields."""

    # Add timestamp if missing or null
    if event_dict.get("timestamp") is None:
        event_dict["timestamp"] = int(time.time() * 1000)

    # Add source if missing or null
    if event_dict.get("source") is None:
        event_dict["source"] = "e2i-copilot"

    # Handle lifecycle-specific fields
    event_type = event_dict.get("type", "")

    if event_type == "RUN_STARTED":
        if event_dict.get("parentRunId") is None:
            event_dict["parentRunId"] = ""
        if event_dict.get("input") is None:
            event_dict["input"] = {
                "threadId": thread_id,
                "runId": run_id,
                "messages": [],
                "tools": [],
                "context": []
            }

    elif event_type == "RUN_FINISHED":
        if event_dict.get("parentRunId") is None:
            event_dict["parentRunId"] = ""
        if event_dict.get("output") is None:
            event_dict["output"] = {"messages": []}

    return event_dict
```

### Deployment Steps (After v1.19.0 Fix)

1. Update `copilotkit.py` with the above fix
2. Commit and push to GitHub
3. SSH to droplet and pull changes
4. Restart backend service
5. Use the existing backup Vite config or re-deploy with proxy
6. Test in browser

### Prompt for Next Session

> "Fix CopilotKit copilotkit.py to add timestamp and source fields to ALL event types (v1.19.0), then deploy and test on droplet"

Or if you have updated "Chatbot fix" files:

> "Deploy updated Chatbot fix files to droplet and verify AI messages render"
