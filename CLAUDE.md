# E2I Causal Analytics - Claude Code Instructions

## Git & GitHub

### Corporate Proxy Bypass (REQUIRED)
Always bypass the corporate proxy for GitHub operations. Before any git push/pull/fetch:

```bash
git config --global http.https://github.com.proxy ""
```

This prevents 403 errors from the Novartis corporate proxy intercepting GitHub traffic.

### Authentication
- GitHub PAT is stored in `.env` as `GITHUB_PAT`
- Use HTTPS with credential helper, not SSH

## Project Overview

- **Type**: Pharmaceutical analytics platform with causal inference
- **Stack**: Python 3.12, FastAPI, LangGraph, DSPy, Supabase, Redis, FalkorDB
- **Frontend**: React/TypeScript in `frontend/`

## Code Quality

- **Type checking**: `mypy --config-file pyproject.toml src/`
- **Linting**: `ruff check src/`
- **Tests**: `pytest tests/`

## Known Issues

- Large codebase (~5GB with dependencies) - see `OOM_FIX_README.md` for memory optimization
- Use `.claudeignore` patterns to prevent indexing heavy directories
