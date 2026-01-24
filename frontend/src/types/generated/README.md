# Generated API Types

This directory contains auto-generated TypeScript types from the FastAPI backend's OpenAPI specification.

## Usage

```typescript
import type { paths, components, operations } from '@/types/generated';

// Access schema types directly
type CausalAnalysisRequest = components['schemas']['HierarchicalAnalysisRequest'];
type HealthResponse = components['schemas']['HealthCheckResponse'];

// Access endpoint response types
type GetHealthResponse = paths['/health']['get']['responses']['200']['content']['application/json'];

// Use helper types from index.ts
import { ExtractResponse, ExtractRequestBody } from '@/types/generated';
type AnalysisResponse = ExtractResponse<'/api/causal/hierarchical/analyze', 'post'>;
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
3. **Use helper types** - The `index.ts` provides `ExtractResponse` and `ExtractRequestBody` utilities
4. **Backward compatibility** - Hand-crafted types in parent directory remain available for gradual migration

## Integration with Existing Types

The generated types complement the existing hand-crafted types:

```typescript
// Existing types (manually maintained)
import { GraphNode, MemorySearchRequest } from '@/types';

// Generated types (auto-sync with backend)
import type { components } from '@/types/generated';
type ApiGraphNode = components['schemas']['GraphNode'];
```

Over time, hand-crafted types can be migrated to use generated types as the source of truth.
