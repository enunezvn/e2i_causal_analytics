# Generated API Types

This directory contains auto-generated TypeScript types from the FastAPI backend's OpenAPI specification.

## Usage

Generated types must be imported **directly from `./api`** (not from
`@/types/generated`). The `index.ts` re-export pattern was removed in
issue #281 because the static re-export broke `tsc -b` whenever `api.ts`
was absent (it is gitignored and only present after running
`npm run generate:types`).

```typescript
// Import generated types directly from ./api after running `npm run generate:types`
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

Regenerate types when the backend API changes:

```bash
# From local development server (localhost:8000)
npm run generate:types

# From production server
npm run generate:types:prod
```

## Files

- `api.ts` - Auto-generated types from OpenAPI spec (DO NOT EDIT)
- `index.ts` - Re-exports and helper type utilities
- `README.md` - This documentation

## Important Notes

1. **Never edit `api.ts` manually** - Changes will be overwritten on regeneration
2. **Regenerate after backend changes** - Run generation after modifying Pydantic schemas
3. **Import generated types from `./api`** - The `index.ts` no longer re-exports them (see issue #281); only `ApiResponse<T>` lives on the index
4. **Backward compatibility** - Hand-crafted types in parent directory remain available for gradual migration

## Integration with Existing Types

The generated types complement the existing hand-crafted types:

```typescript
// Existing types (manually maintained)
import { GraphNode, MemorySearchRequest } from '@/types';

// Generated types (auto-sync with backend) — imported directly from ./api
import type { components } from '@/types/generated/api';
type ApiGraphNode = components['schemas']['GraphNode'];
```

Over time, hand-crafted types can be migrated to use generated types as the source of truth.
