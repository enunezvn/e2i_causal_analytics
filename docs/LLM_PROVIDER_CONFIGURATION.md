# LLM Provider Configuration

This document describes how to configure the LLM (Large Language Model) provider for the E2I Causal Analytics platform.

## Overview

The platform supports two LLM providers:
- **OpenAI** (default): Uses `gpt-4o`
- **Anthropic**: Uses `claude-sonnet-4-20250514`

The LLM provider is configured via the `LLM_PROVIDER` environment variable and applies to:
- CopilotKit chat interactions
- CopilotKit synthesis operations
- Chatbot graph responses

## Configuration

### Environment Variable

Set the `LLM_PROVIDER` environment variable:

```bash
# Use OpenAI (default)
export LLM_PROVIDER=openai

# Use Anthropic
export LLM_PROVIDER=anthropic
```

If `LLM_PROVIDER` is not set, the system defaults to OpenAI.

### API Keys

Ensure the appropriate API key is set based on your provider:

```bash
# For OpenAI
export OPENAI_API_KEY=sk-...

# For Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### Model Mappings

Each provider supports three model tiers:

| Tier | OpenAI | Anthropic |
|------|--------|-----------|
| `fast` | gpt-4o-mini | claude-3-5-haiku-20241022 |
| `standard` | gpt-4o | claude-sonnet-4-20250514 |
| `reasoning` | o3-mini | claude-sonnet-4-20250514 |

## Verification

### Status Endpoint

Check the current LLM configuration via the CopilotKit status endpoint:

```bash
curl http://localhost:8001/api/copilotkit/status
```

Response includes:
```json
{
  "status": "active",
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "llm_configured": true,
  ...
}
```

### Logs

When the LLM is used, you'll see log messages indicating the provider:

```
[CopilotKit] Using openai LLM for chat
[CopilotKit] Using openai LLM for synthesis
Using openai LLM for chatbot
```

## Switching Providers

### On Local Development

1. Update your `.env` file:
   ```bash
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Restart the API server:
   ```bash
   make dev
   ```

### On Production (Droplet)

1. SSH into the droplet:
   ```bash
   ssh -i ~/.ssh/replit root@159.89.180.27
   ```

2. Edit the environment file:
   ```bash
   cd /root/Projects/e2i_causal_analytics
   nano .env
   # Update LLM_PROVIDER=anthropic
   ```

3. Restart the API service:
   ```bash
   systemctl restart e2i-api
   ```

4. Verify:
   ```bash
   curl http://localhost:8001/api/copilotkit/status | jq '.llm_provider'
   ```

## Architecture

### LLM Factory

The centralized LLM factory is located at `src/utils/llm_factory.py` and provides:

- `get_llm_provider()`: Returns current provider ("openai" or "anthropic")
- `get_chat_llm(model_tier, **kwargs)`: Returns configured LangChain chat model
- `get_fast_llm(**kwargs)`: Returns fast tier model
- `get_standard_llm(**kwargs)`: Returns standard tier model
- `get_reasoning_llm(**kwargs)`: Returns reasoning tier model

### Integration Points

The LLM factory is used by:

| Module | Function | Purpose |
|--------|----------|---------|
| `src/api/routes/copilotkit.py` | `chat_node()` | CopilotKit chat responses |
| `src/api/routes/copilotkit.py` | `synthesize_node()` | Tool result synthesis |
| `src/api/routes/chatbot_graph.py` | `generate_node()` | Chatbot responses |

## Fallback Behavior

If the LLM factory fails (missing API key, network error):

1. A `ValueError` is raised with descriptive message
2. The calling code catches the exception
3. A fallback response is generated without LLM
4. The user receives a helpful message explaining limited functionality

## Troubleshooting

### "OPENAI_API_KEY not set"

Ensure the API key is set in your environment:
```bash
export OPENAI_API_KEY=sk-...
```

### "ANTHROPIC_API_KEY not set"

When using Anthropic provider, ensure:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### Status shows wrong provider

After changing `LLM_PROVIDER`, restart the API service for changes to take effect.

## Changelog

- **2026-01-14**: Initial implementation
  - Changed default provider from Anthropic to OpenAI
  - Added dynamic provider switching via `LLM_PROVIDER` env var
  - Integrated factory into CopilotKit and chatbot_graph modules
  - Added `llm_provider` and `llm_model` to status endpoint
