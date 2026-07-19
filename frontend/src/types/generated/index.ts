/**
 * Auto-generated TypeScript types from FastAPI OpenAPI specification.
 *
 * `api.ts` is a COMMITTED contract baseline (re-tracked 2026-07): CI's
 * verify-types workflow regenerates it from the backend schema and fails
 * on any byte difference. Regenerate with `make generate-types` at the
 * repo root and commit alongside backend schema changes.
 *
 * This `index.ts` does not re-export from `./api` (the static re-export
 * was removed in issue #281 when `api.ts` was untracked and could be
 * absent). Import generated types directly:
 *
 *   import type { paths, components, operations } from '@/types/generated/api';
 *
 *   type CausalAnalysisRequest =
 *     components['schemas']['HierarchicalAnalysisRequest'];
 *
 * The hand-crafted types in `frontend/src/types/*.ts` remain the
 * convention for application code.
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
