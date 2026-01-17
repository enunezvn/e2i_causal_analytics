# Dynamic LLM Provider Configuration for CopilotKit

## Objective
Make the LLM provider (OpenAI vs Anthropic) dynamically configurable in CopilotKit, defaulting to OpenAI, with the ability to switch to Anthropic on demand.

## Current State
- **Factory exists**: `src/utils/llm_factory.py` already supports both providers via `LLM_PROVIDER` env var
- **CopilotKit ignores it**: `src/api/routes/copilotkit.py` hardcodes `ChatAnthropic` directly
- **Default is Anthropic**: Factory defaults to "anthropic" when `LLM_PROVIDER` not set

## Implementation Plan

### Phase 1: Update Factory Default to OpenAI
**File**: `src/utils/llm_factory.py`

Change line 71:
```python
# Before
provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# After
provider = os.environ.get("LLM_PROVIDER", "openai").lower()
```

Also update line 73-74 warning message to reflect new default.

### Phase 2: Integrate Factory into CopilotKit
**File**: `src/api/routes/copilotkit.py`

#### 2.1 Add Import (near line 186)
```python
from src.utils.llm_factory import get_chat_llm, get_llm_provider, LLMProvider
```

Remove the direct `ChatAnthropic` import if no longer needed elsewhere.

#### 2.2 Update `chat_node` (lines ~1595-1620)
Replace hardcoded ChatAnthropic with factory call:
```python
try:
    llm = get_chat_llm(
        model_tier="standard",
        max_tokens=2048,
        temperature=0.3,
    )
    provider = get_llm_provider()
    logger.info(f"[CopilotKit] Using {provider} LLM for chat")
except (ValueError, ImportError) as e:
    logger.warning(f"[CopilotKit] LLM not available: {e}, using fallback")
    fallback = generate_e2i_response(last_human_message)
    await copilotkit_emit_message(config, fallback)
    return {"messages": [AIMessage(content=fallback)]}
```

#### 2.3 Update `synthesize_node` (lines ~1790-1850)
Same pattern as chat_node - replace ChatAnthropic with factory call.

#### 2.4 Update Status Endpoint (lines ~2408-2420)
Add provider info to status response:
```python
"llm_provider": get_llm_provider(),
"llm_model": MODEL_MAPPINGS[get_llm_provider()]["standard"],
```

### Phase 3: Update Tests
**Files**:
- `tests/integration/test_chatbot_graph.py`
- `tests/integration/test_chatbot_streaming.py`

#### 3.1 Update Mock Targets
Change from:
```python
@patch("src.api.routes.copilotkit.ChatAnthropic")
```
To:
```python
@patch("src.utils.llm_factory.get_chat_llm")
```

#### 3.2 Add Provider Parameterization (optional enhancement)
```python
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_chat_with_provider(provider):
    with patch.dict("os.environ", {"LLM_PROVIDER": provider}):
        # test logic
```

### Phase 4: Update Environment Files
**Files**: `.env.example`, potentially `.env` on droplet

Add/update:
```bash
# LLM Provider: "openai" (default) or "anthropic"
LLM_PROVIDER=openai
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/utils/llm_factory.py` | Change default from "anthropic" to "openai" |
| `src/api/routes/copilotkit.py` | Replace hardcoded ChatAnthropic with factory (~3 locations) |
| `tests/integration/test_chatbot_graph.py` | Update mock targets |
| `tests/integration/test_chatbot_streaming.py` | Update mock targets |
| `.env.example` | Document LLM_PROVIDER variable |

## Configuration Options After Implementation

| Config | Value | Effect |
|--------|-------|--------|
| `LLM_PROVIDER=openai` | Default | Uses GPT-4o |
| `LLM_PROVIDER=anthropic` | Override | Uses Claude Sonnet |
| Unset | Falls back to openai | Uses GPT-4o |

## Verification

1. **Unit tests**: Run `pytest tests/integration/test_chatbot_graph.py -v`
2. **Local API test**:
   ```bash
   # Test with OpenAI (default)
   curl http://localhost:8001/api/copilotkit/status

   # Test with Anthropic
   LLM_PROVIDER=anthropic uvicorn src.api.main:app
   ```
3. **Droplet deployment**:
   - SSH to droplet
   - Update `.env` with `LLM_PROVIDER=openai`
   - Restart e2i-api service
   - Verify via `/api/copilotkit/status` endpoint

## Rollback
Set `LLM_PROVIDER=anthropic` in environment to revert to previous behavior.
