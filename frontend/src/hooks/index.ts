/**
 * Hooks Index
 * ===========
 *
 * Central export for all custom hooks.
 *
 * @module hooks
 */

// Authentication hooks
export { useAuth } from './use-auth';
export type { UseAuthReturn } from './use-auth';

// Visualization hooks
export { useD3 } from './use-d3';
export { useCytoscape } from './use-cytoscape';

// UI hooks
export { useToast } from './use-toast';

// E2I CopilotKit hooks
export { useE2IFilters } from './use-e2i-filters';
export type { UseE2IFiltersReturn } from './use-e2i-filters';

export { useE2IHighlights } from './use-e2i-highlights';
export type { UseE2IHighlightsReturn, HighlightedPath } from './use-e2i-highlights';

export { useE2IValidation } from './use-e2i-validation';
export type {
  UseE2IValidationReturn,
  ValidationResult,
  ValidationConfig,
} from './use-e2i-validation';

export { useUserPreferences } from './use-user-preferences';
export type { UseUserPreferencesReturn } from './use-user-preferences';
