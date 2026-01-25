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

// Chat feedback hook
export { useChatFeedback } from './use-chat-feedback';
export type {
  FeedbackRating,
  FeedbackSubmission,
  FeedbackResult,
  FeedbackState,
  UseChatFeedbackReturn,
} from './use-chat-feedback';

// WebSocket hook with auto-reconnect
export { useWebSocket, useGraphWebSocket } from './use-websocket';
export type {
  WebSocketState,
  UseWebSocketOptions,
  UseWebSocketReturn,
} from './use-websocket';

// Query error handling hooks
export {
  useQueryError,
  useQueryErrorToast,
  useMutationError,
} from './use-query-error';
export type {
  UseQueryErrorOptions,
  UseQueryErrorReturn,
} from './use-query-error';

// WebSocket cache sync hook (Phase 4: Cache invalidation sync)
export {
  useWebSocketCacheSync,
  invalidateCacheForEvent,
} from './use-websocket-cache-sync';
export type {
  GraphEventType,
  GraphStreamPayload,
  UseWebSocketCacheSyncOptions,
  UseWebSocketCacheSyncReturn,
} from './use-websocket-cache-sync';

// Data freshness hooks (Phase C: Stale data indicators)
export {
  useDataFreshness,
  useQueryFreshness,
  formatTimeAgo,
  getFreshnessClassName,
  getFreshnessIconName,
} from './use-data-freshness';
export type {
  DataFreshnessOptions,
  DataFreshnessResult,
} from './use-data-freshness';

// Analytics hooks (Phase C: Metrics dashboard)
export {
  analyticsKeys,
  useAnalyticsDashboard,
  useAgentMetrics,
  useAgentTrend,
  useMetricsSummary,
} from './api/use-analytics';
