# Generated API Types

This directory contains auto-generated TypeScript types from the FastAPI backend's OpenAPI specification.

## Usage

Generated types must be imported **directly from `./api`** (not from
`@/types/generated`). The `index.ts` re-export pattern was removed in
issue #281 (when `api.ts` was still untracked and could be absent) and
is deliberately not restored.

> ✅ **Committed contract baseline (since 2026-07):** `api.ts` is
> **tracked in git**. CI's `verify-types` workflow regenerates it from
> the backend schema on every schema-affecting PR and **fails if the
> fresh generation differs** from the committed file, so the checked-in
> copy is always in sync with the backend on `main`. Importing from
> `'@/types/generated/api'` is safe in committed code; hand-written
> types in `src/types/*.ts` remain the convention for application code.

```typescript
import type { paths, components, operations } from '@/types/generated/api';

// Access schema types directly
type CausalAnalysisRequest = components['schemas']['HierarchicalAnalysisRequest'];
type HealthResponse = components['schemas']['HealthCheckResponse'];

// Access endpoint response types
type GetHealthResponse = paths['/health']['get']['responses']['200']['content']['application/json'];

// The shared `ApiResponse<T>` envelope still lives on the index
import type { ApiResponse } from '@/types/generated';
```

## Regeneration

A backend PR that changes the API schema **must regenerate and commit
`api.ts`**, or the verify-types drift gate fails:

```bash
# Repo root — CI-identical path (static spec export, no running server).
# Use this one to fix a drift-gate failure.
make generate-types

# Alternatives (require a running backend; output should match when the
# running code equals your checkout):
npm run generate:types        # from localhost:8000
npm run generate:types:prod   # from the live deployment
```

## Files

- `api.ts` - Auto-generated contract baseline (DO NOT EDIT; committed, CI-enforced)
- `index.ts` - Hosts the local `ApiResponse<T>` helper only; no longer re-exports from `./api` (see issue #281)
- `index.test.ts` - Guard that fails CI if `./api` re-exports / imports reappear in `index.ts`
- `README.md` - This documentation

## Important Notes

1. **Never edit `api.ts` manually** - CI regenerates it and fails on any difference
2. **Regenerate after backend changes** - `make generate-types`, commit in the same PR
3. **Your Python env must match `requirements.txt`** - FastAPI/Pydantic versions
   change how duplicate schema names (e.g. `GraphNode`, `AnalysisStatus`) are
   disambiguated, producing a byte-different `api.ts`. CI regenerates with the
   pinned versions and is the arbiter; a drifted venv will fail the drift gate
4. **Import generated types from `./api`** - The `index.ts` no longer re-exports them (see issue #281); only `ApiResponse<T>` lives on the index
5. **Backward compatibility** - Hand-crafted types in the parent directory remain the convention for application code

## Integration with Existing Types

The generated types complement the existing hand-crafted types:

```typescript
// Existing types (manually maintained)
import { GraphNode, MemorySearchRequest } from '@/types';

// Generated types (CI-enforced sync with backend)
import type { components } from '@/types/generated/api';
type ApiGraphNode = components['schemas']['GraphNode'];
```

Over time, hand-crafted types can be migrated to use generated types as the source of truth.
