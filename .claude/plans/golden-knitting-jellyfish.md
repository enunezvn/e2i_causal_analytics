# CopilotKit Chatbot Debug - Comprehensive Implementation Plan

**Created**: 2026-01-10
**Status**: In Progress
**Issue**: React SDK (v1.50.1) not rendering AI responses despite backend streaming correctly

---

## Executive Summary

After 2 days of debugging and extensive research into official CopilotKit SDK sources, this plan outlines a systematic approach to fix the message rendering issue. The backend (Python copilotkit v0.1.74) correctly streams TEXT_MESSAGE events, but the React frontend never renders AI responses.

### Critical Discovery: Wrong Event Pattern

**Root Cause Hypothesis (REVISED)**: After researching official CopilotKit SDK sources, we discovered that **the official SDK does NOT use TEXT_MESSAGE events directly**. Instead, it uses:

1. **State synchronization** via `copilotkit_emit_state(config, state)`
2. **LangChain serialization** via `langchain_dumps()` (not Pydantic's `model_dump_json`)
3. **State sync events** (`on_copilotkit_state_sync`) with `active=True/False`

**Our Current Approach (WRONG):**
```
TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT → TEXT_MESSAGE_END
```

**Official SDK Pattern:**
```
on_copilotkit_state_sync (active=True, state updates)
→ on_chat_model_stream (LangChain events)
→ on_copilotkit_state_sync (active=False, final state)
```

### Official SDK Resources Reviewed ✅

| Resource | Key Findings |
|----------|--------------|
| `langgraph_agui_agent.py` | Custom event dispatch via `_dispatch_event()`, handles ManuallyEmitMessage |
| `langgraph.py` | Uses `copilotkit_emit_state()`, `copilotkit_emit_message()`, `adispatch_custom_event()` |
| `langgraph_agent.py` | No RUN_STARTED/FINISHED, uses `on_copilotkit_state_sync`, `langchain_dumps()` |
| `copilotkit_lg_middleware.py` | Tool injection/interception, not event streaming |
| `protocol.py` | TypedDict event definitions (RunStarted, RunFinished, RunError) |
| `logging.py` | Simple logger with LOG_LEVEL env var |
| `parameter.py` | Parameter types for agent actions (not relevant to streaming) |
| `chat-with-your-data` example | Uses CopilotKit + Next.js, custom AssistantMessage component |
| `dev.to` tutorial | Uses `useCoAgentStateRender`, `copilotkit_emit_state()` for progress |

### Key Research Findings from Official Sources

1. **No RUN_STARTED/RUN_FINISHED in SDK**: The official `langgraph_agent.py` does NOT emit explicit RUN_STARTED/RUN_FINISHED events. It uses continuous state sync instead.

2. **Official Event Emission**: Uses `copilotkit_emit_state(config, state)` to push state updates, which the frontend receives via `useCoAgentStateRender` hook.

3. **Our Frontend Missing Hooks**:
   - Does NOT use `useCoAgent` for state binding
   - Does NOT use `useCoAgentStateRender` for conditional rendering
   - Uses only `useCopilotReadable` and `useCopilotAction`

4. **Serialization Mismatch**: Official SDK uses `langchain_dumps()`, not `model_dump_json(by_alias=True)`

### Two Possible Fix Approaches

**Option A: Align with Official SDK Pattern (Recommended)**
- Use CopilotKit SDK's `CopilotKitSDK` and `LangGraphAgent` classes properly
- Let the SDK handle event emission instead of manual TEXT_MESSAGE events
- Remove our custom streaming workarounds

**Option B: Debug Current TEXT_MESSAGE Approach**
- The TEXT_MESSAGE events may still work if formatted correctly
- Need to verify the React SDK CopilotChat component receives and parses events
- May need frontend hooks to process raw TEXT_MESSAGE events

---

## Phase 1: Codebase Exploration ✅ COMPLETE

**Goal**: Understand current implementation and identify root cause

### Findings

- [x] 1.1 Backend (v1.9.6) confirmed NOT emitting RUN_STARTED/RUN_FINISHED events
- [x] 1.2 Only TEXT_MESSAGE events (START, CONTENT, END) are being emitted
- [x] 1.3 AG-UI SDK event classes used with `by_alias=True` for camelCase
- [x] 1.4 Frontend uses default CopilotKit rendering (no custom message renderer)
- [x] 1.5 Frontend has `showDevConsole={true}` and `onError` handler already configured

### Root Cause Identified
**Missing RUN_STARTED and RUN_FINISHED lifecycle events** - the React SDK may require these to properly initialize message rendering

---

## Phase 2: Use Official SDK Pattern (PRIMARY FIX) ✅ SELECTED

**Goal**: Remove custom TEXT_MESSAGE workarounds and use SDK's native event handling

### Strategy

The official CopilotKit SDK already handles event emission internally. Our custom `execute()` method with manual TEXT_MESSAGE emission is working around SDK bugs from v0.1.74, but may be causing the rendering issue. We'll refactor to use the SDK properly.

### Tasks

- [ ] 2.1 Read current `/copilotkit` endpoint implementation
- [ ] 2.2 Identify what custom code can be removed
- [ ] 2.3 Refactor to use SDK's `CopilotKitSDK` and `LangGraphAgent` properly
- [ ] 2.4 Remove manual TEXT_MESSAGE event emission
- [ ] 2.5 Update to use SDK's native streaming handler
- [ ] 2.6 Update version to 1.10.0 (major refactor)

### Implementation Approach

**Step 1: Simplify the Agent**

Remove our custom `LangGraphAgent` class that overrides `execute()` with TEXT_MESSAGE workarounds. Use the SDK's built-in classes instead.

```python
# BEFORE (custom workaround):
class LangGraphAgent(LangGraphAGUIAgent):
    async def execute(self, ...):
        # 300+ lines of custom code with TEXT_MESSAGE emission
        ...

# AFTER (SDK native):
from copilotkit import CopilotKitSDK
from copilotkit.langgraph import LangGraphAgent as SDKLangGraphAgent

sdk = CopilotKitSDK(agents=[
    SDKLangGraphAgent(
        name="default",
        graph=e2i_graph,  # Our compiled LangGraph
    )
])
```

**Step 2: Use SDK's Streaming Handler**

```python
@app.post("/copilotkit")
async def copilotkit_handler(request: Request):
    # Let SDK handle everything - it knows how to emit events correctly
    return await sdk.handle_execute_agent(request)
```

**Step 3: Use Official Message Emission**

If we need to manually emit messages from within the graph, use SDK's helper:

```python
from copilotkit.langchain import copilotkit_emit_message

# Inside a LangGraph node:
async def my_node(state, config):
    # This emits messages the way React SDK expects
    await copilotkit_emit_message(config, "Hello from the agent!")
    return state
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/api/routes/copilotkit.py` | Remove custom LangGraphAgent class, use SDK's CopilotKitSDK |
| Graph nodes | Use `copilotkit_emit_message()` instead of custom events |

### Success Criteria
- Messages render in frontend using SDK's native event handling
- No custom TEXT_MESSAGE event emission code
- Backend uses SDK's `handle_execute_agent()` for streaming

---

## Phase 3: Deploy and Test on Droplet

**Goal**: Deploy lifecycle event fix and verify message rendering

### Tasks

- [ ] 3.1 SSH to droplet
- [ ] 3.2 Pull latest code changes
- [ ] 3.3 Restart e2i-api Docker container
- [ ] 3.4 Establish SSH tunnels for frontend access
- [ ] 3.5 Test chat in browser via http://localhost:5174
- [ ] 3.6 Verify AI responses render in chat UI

### Commands

```bash
# SSH to droplet
ssh -i ~/.ssh/replit root@159.89.180.27

# Pull and restart
cd /root/e2i_causal_analytics
git pull origin main
docker compose restart e2i-api

# Verify backend logs
docker compose logs -f e2i-api 2>&1 | head -50

# Exit and establish tunnels
exit
ssh -L 5174:localhost:5174 -L 8001:localhost:8001 root@159.89.180.27
```

### Success Criteria
- AI responses render in chat UI
- No `net::ERR_ABORTED` errors
- Console shows no CopilotKit errors

---

## Phase 4: Fallback Debugging (If Phase 3 Fails)

**Goal**: Deep investigation if lifecycle events don't fix the issue

### Tasks

- [ ] 4.1 Capture raw curl output from backend to verify event format
- [ ] 4.2 Add frontend fetch interceptor to log streaming data
- [ ] 4.3 Check browser DevTools Network tab for response body
- [ ] 4.4 Verify Zod validation isn't rejecting events
- [ ] 4.5 Check streaming headers (Content-Type, Connection)

### Debugging Commands

```bash
# Capture raw streaming response with curl
curl -X POST http://localhost:8001/copilotkit \
  -H "Content-Type: application/json" \
  -d '{"threadId":"test-123","runId":"run-456","messages":[{"role":"user","content":"Hello"}]}' \
  --no-buffer 2>&1
```

### Frontend Debug Code (if needed)

```typescript
// Add to E2ICopilotProvider.tsx temporarily
useEffect(() => {
  const originalFetch = window.fetch;
  window.fetch = async (...args) => {
    const [url] = args;
    if (url.toString().includes('copilotkit')) {
      console.log('[CopilotKit Fetch]', url);
      const response = await originalFetch(...args);
      console.log('[CopilotKit Response]', response.status);
      return response;
    }
    return originalFetch(...args);
  };
}, []);
```

### Success Criteria
- Identify exact point of failure
- Capture raw event data for analysis

---

## Phase 5: Cleanup and Documentation

**Goal**: Finalize fix and document solution

### Tasks

- [ ] 5.1 Remove any temporary debug logging
- [ ] 5.2 Ensure version is updated to 1.9.7
- [ ] 5.3 Document root cause and fix in copilotkit-chatbot-debug.md
- [ ] 5.4 Commit changes with descriptive message
- [ ] 5.5 Push to main branch

### Git Commands

```bash
git add src/api/routes/copilotkit.py
git commit -m "fix(copilotkit): add RUN_STARTED/RUN_FINISHED lifecycle events for React SDK rendering

- Added RunStartedEvent emission at beginning of streaming response
- Added RunFinishedEvent emission at end of streaming response
- Updated version to 1.9.7

Fixes: React SDK (v1.50.1) not rendering AI assistant responses

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push origin main
```

### Success Criteria
- Clean production code
- Solution documented for future reference
- Changes committed to git

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | ✅ Complete | Codebase + official SDK research - discovered event pattern mismatch |
| Phase 2 | ⏳ Ready | **USER SELECTED**: Use SDK's native event handling (remove workarounds) |
| Phase 3 | Pending | Deploy and test on droplet |
| Phase 4 | Pending | Fallback debugging (if Phase 2-3 fail) |
| Phase 5 | Pending | Cleanup and documentation |

### Key Discovery
Official CopilotKit SDK does NOT emit explicit TEXT_MESSAGE events. It uses:
- `on_copilotkit_state_sync` events for state updates
- `langchain_dumps()` for serialization (not `model_dump_json`)
- `copilotkit_emit_message()` helper for manual message emission

### User-Selected Approach
**Option A: Use SDK Properly** - Remove custom TEXT_MESSAGE workarounds and refactor to use CopilotKit SDK's native `handle_execute_agent` method.

### Next Step
Execute Phase 2: Refactor `src/api/routes/copilotkit.py` to use SDK's `CopilotKitSDK` and `LangGraphAgent` classes instead of our custom implementation.

---

## Quick Reference

### SSH Access
```bash
ssh -i ~/.ssh/replit root@159.89.180.27
```

### SSH Tunnels
```bash
ssh -L 5174:localhost:5174 -L 8001:localhost:8001 -L 5000:localhost:5000 root@159.89.180.27
```

### Key Files
- Backend: `src/api/routes/copilotkit.py` (v1.9.6)
- Frontend: `frontend/src/providers/E2ICopilotProvider.tsx`
- Debug Plan: `.claude/plans/copilotkit-chatbot-debug.md`

### Docker Commands (on droplet)
```bash
cd /root/e2i_causal_analytics
docker compose logs -f e2i-api 2>&1 | head -100
docker compose restart e2i-api
```
