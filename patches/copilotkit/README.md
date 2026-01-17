# CopilotKit Fork (Patched)

This is a local fork of [copilotkit](https://github.com/CopilotKit/CopilotKit/tree/main/sdk-python) with a relaxed FastAPI version constraint.

## Why This Fork Exists

CopilotKit 0.1.76 requires `fastapi>=0.115.0,<0.116.0`, which prevents upgrading to FastAPI 0.116+ that includes fixes for critical starlette CVEs:
- **CVE-2025-54121**: Starlette request body memory exhaustion
- **CVE-2025-62727**: Starlette multipart parsing vulnerability

## Changes Made

1. **pyproject.toml**: Changed FastAPI constraint from `>=0.115.0,<0.116.0` to `>=0.115.12`
2. **Version**: Changed from `0.1.76` to `0.1.76.post1` to indicate patch release

## Installation

This package must be installed BEFORE the PyPI copilotkit:

```bash
# Install both local forks in order
pip install ./patches/ag-ui-langgraph[fastapi]
pip install ./patches/copilotkit
```

Or in requirements.txt:
```
./patches/ag-ui-langgraph[fastapi]
./patches/copilotkit
```

## When to Remove This Fork

Remove this fork when:
1. CopilotKit releases a version with relaxed FastAPI constraint (check their pyproject.toml for `fastapi>=0.116` or similar)
2. The upstream issue is resolved: https://github.com/CopilotKit/CopilotKit/issues (search for "fastapi constraint")

## Source Files

All source files are copied from copilotkit 0.1.76 without modification. Only the version constraint in pyproject.toml was changed.

## Related

- [ag-ui-langgraph fork](../ag-ui-langgraph/README.md) - Also patched for the same reason
