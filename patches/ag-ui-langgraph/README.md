# ag-ui-langgraph (Patched Fork)

**PATCHED VERSION** - Relaxed fastapi constraint to allow >=0.116.0

This is a local fork of [ag-ui-langgraph](https://github.com/ag-ui-protocol/ag-ui/tree/main/integrations/langgraph/python) with a relaxed fastapi dependency constraint to allow security updates.

## Why This Fork Exists

The upstream `ag-ui-langgraph` package pins `fastapi = "^0.115.12"` (meaning `>=0.115.12, <0.116.0`), which blocks upgrading to fastapi 0.116+ and consequently blocks security fixes for starlette CVEs:

- **CVE-2025-54121** - starlette vulnerability
- **CVE-2025-62727** - starlette vulnerability

FastAPI 0.116.0 has **NO breaking changes** (only added cloud deployment feature), making this constraint unnecessarily restrictive.

## Changes Made

**pyproject.toml:**
```diff
- fastapi = { version = "^0.115.12", optional = true }
+ fastapi = { version = ">=0.115.12", optional = true }
```

## Installation

Install from local path:
```bash
pip install ./patches/ag-ui-langgraph[fastapi]
```

Or in requirements.txt:
```
./patches/ag-ui-langgraph[fastapi]
```

## Upstream Tracking

- **Upstream repo:** https://github.com/ag-ui-protocol/ag-ui
- **Upstream version:** 0.0.23
- **Fork version:** 0.0.23-patched
- **Last synced:** 2026-01-16

## When to Remove This Fork

Remove this fork and switch back to upstream when:
1. ag-ui-langgraph relaxes the fastapi constraint upstream
2. Or copilotkit switches to a different ag-ui integration

Monitor: https://github.com/ag-ui-protocol/ag-ui/blob/main/integrations/langgraph/python/pyproject.toml
