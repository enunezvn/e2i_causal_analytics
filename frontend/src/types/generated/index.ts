/**
 * Auto-generated TypeScript types from FastAPI OpenAPI specification.
 *
 * The generated `api.ts` file is intentionally untracked (see root
 * `.gitignore`) and is produced by `npm run generate:types` against a
 * running FastAPI backend.
 *
 * This `index.ts` no longer re-exports from `./api` because the static
 * re-export broke `tsc -b` (and therefore the CI `build` job) any time
 * `api.ts` was absent — see issue #281. Consumers that need the
 * generated types should import directly from `./api` after running
 * `npm run generate:types`:
 *
 *   import type { paths, components, operations } from '@/types/generated/api';
 *
 *   type CausalAnalysisRequest =
 *     components['schemas']['HierarchicalAnalysisRequest'];
 *
 * The hand-crafted types in `frontend/src/types/*.ts` remain the
 * source of truth for the rest of the codebase until contract sync is
 * required.
 */

// Generic API response envelope. Kept here because it is independent of
// the generated `./api` module and is a useful shared helper.
export type ApiResponse<T> =
  | {
      data: T;
      error?: never;
    }
  | {
      data?: never;
      error: string;
    };
