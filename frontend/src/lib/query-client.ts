/**
 * TanStack Query Client Configuration
 * ====================================
 *
 * Centralized QueryClient configuration with sensible defaults for
 * caching, stale time, and error handling.
 *
 * Usage:
 *   import { queryClient } from '@/lib/query-client'
 *   // Used in main.tsx with QueryClientProvider
 */

import { QueryClient } from '@tanstack/react-query';
import { env } from '@/config/env';
import { logger } from '@/lib/logger';

/**
 * Default stale time for queries (5 minutes)
 * Queries won't refetch if the data is fresher than this
 */
const DEFAULT_STALE_TIME = 5 * 60 * 1000;

/**
 * Default garbage collection time (10 minutes)
 * Inactive queries are garbage collected after this time
 */
const DEFAULT_GC_TIME = 10 * 60 * 1000;

/**
 * Default retry configuration
 * Retries failed queries up to 3 times with exponential backoff
 */
const DEFAULT_RETRY = 3;

/**
 * Custom retry delay function with exponential backoff
 * @param attemptIndex - The current retry attempt (0-indexed)
 * @returns Delay in milliseconds before next retry
 */
function getRetryDelay(attemptIndex: number): number {
  // Exponential backoff: 1s, 2s, 4s, 8s...
  const baseDelay = 1000;
  const maxDelay = 30000;
  const delay = Math.min(baseDelay * Math.pow(2, attemptIndex), maxDelay);
  return delay;
}

/**
 * Determine if a failed request should be retried
 * @param failureCount - Number of failures so far
 * @param error - The error that occurred
 * @returns Whether to retry the request
 */
function shouldRetry(failureCount: number, error: unknown): boolean {
  // Don't retry if we've hit the max retries
  if (failureCount >= DEFAULT_RETRY) {
    return false;
  }

  // Don't retry client errors (4xx) except 408 (Request Timeout) and 429 (Too Many Requests)
  if (error instanceof Error && 'status' in error) {
    const status = (error as { status: number }).status;
    if (status >= 400 && status < 500 && status !== 408 && status !== 429) {
      return false;
    }
  }

  return true;
}

/**
 * Global error handler for mutations
 * Logs errors in development mode
 */
function handleMutationError(error: unknown): void {
  if (env.isDev) {
    console.error('[Query] Mutation error:', error);
  }
}

/**
 * Create and configure the QueryClient instance
 */
function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Data freshness settings
        staleTime: DEFAULT_STALE_TIME,
        gcTime: DEFAULT_GC_TIME,

        // Retry configuration
        retry: shouldRetry,
        retryDelay: getRetryDelay,

        // Refetch behavior
        refetchOnWindowFocus: env.isProd, // Only refetch on focus in production
        refetchOnReconnect: true,
        refetchOnMount: true,

        // Network mode - online means queries will pause when offline
        networkMode: 'online',
      },
      mutations: {
        // Retry mutations only once
        retry: 1,
        retryDelay: getRetryDelay,

        // Error handling
        onError: handleMutationError,

        // Network mode
        networkMode: 'online',
      },
    },
  });
}

/**
 * Main QueryClient instance
 * Pre-configured with optimal defaults for the application
 */
export const queryClient = createQueryClient();

// =============================================================================
// TAB VISIBILITY HANDLING (Phase 4: Pause polling when tab hidden)
// =============================================================================

/**
 * Track tab visibility state for query management
 */
let isTabVisible = typeof document !== 'undefined' ? !document.hidden : true;

/**
 * Handle visibility change events.
 *
 * Background-polling pause is handled automatically by TanStack Query's
 * built-in `focusManager`: `refetchInterval` only fires while the tab is
 * focused (see queryObserver: it gates interval refetches on
 * `options.refetchIntervalInBackground || focusManager.isFocused()`, and
 * `refetchIntervalInBackground` defaults to `false`). The focusManager
 * already listens to `visibilitychange`, so polling is paused when the tab
 * is hidden and resumes on return without any work here.
 *
 * This handler therefore does NOT cancel queries on hide (which would only
 * abort an in-flight fetch — a one-shot, not a "pause" — and could drop a
 * load the user wants on return). It only: (1) tracks tab visibility for
 * `isTabCurrentlyVisible()`, (2) logs in dev, and (3) on return-to-visible
 * in production, nudges already-stale queries to refetch.
 */
function handleVisibilityChange(): void {
  const wasVisible = isTabVisible;
  isTabVisible = !document.hidden;

  logger.debug(`[Query] Tab visibility changed: ${wasVisible ? 'visible' : 'hidden'} → ${isTabVisible ? 'visible' : 'hidden'}`);

  if (!isTabVisible) {
    // Tab became hidden. Interval polling is paused by TanStack's
    // focusManager (refetchIntervalInBackground defaults to false); nothing
    // to cancel here. In-flight, non-interval fetches are left to complete.
    logger.debug('[Query] Tab hidden - background polling paused by focusManager');
  } else if (wasVisible === false && isTabVisible) {
    // Tab became visible again - invalidate stale queries to refetch
    // This triggers a refetch for queries that became stale while hidden
    logger.debug('[Query] Tab visible - resuming queries');
    // Refetch queries that may have become stale while hidden
    // Only refetch if refetchOnWindowFocus is enabled (production)
    if (env.isProd) {
      void queryClient.invalidateQueries({
        predicate: (query) => query.isStale(),
      });
    }
  }
}

/**
 * Initialize tab visibility listener
 * Call this once during app initialization
 */
export function initTabVisibilityListener(): void {
  if (typeof document === 'undefined') {
    return; // SSR safety
  }

  document.addEventListener('visibilitychange', handleVisibilityChange);

  logger.debug('[Query] Tab visibility listener initialized');
}

/**
 * Cleanup tab visibility listener
 * Call this during app teardown if needed
 */
export function cleanupTabVisibilityListener(): void {
  if (typeof document === 'undefined') {
    return;
  }

  document.removeEventListener('visibilitychange', handleVisibilityChange);
}

/**
 * Check if tab is currently visible
 * Useful for conditional query behavior
 */
export function isTabCurrentlyVisible(): boolean {
  return isTabVisible;
}

/**
 * Query key factory for consistent key generation
 * Use these helpers to create type-safe query keys
 */
export const queryKeys = {
  /**
   * Root key for all queries
   */
  all: ['e2i'] as const,

  /**
   * Graph-related queries
   */
  graph: {
    all: () => [...queryKeys.all, 'graph'] as const,
    nodes: () => [...queryKeys.graph.all(), 'nodes'] as const,
    node: (id: string) => [...queryKeys.graph.all(), 'node', id] as const,
    nodeNetwork: (id: string) =>
      [...queryKeys.graph.all(), 'node', id, 'network'] as const,
    relationships: () => [...queryKeys.graph.all(), 'relationships'] as const,
    stats: () => [...queryKeys.graph.all(), 'stats'] as const,
    search: (query: string) =>
      [...queryKeys.graph.all(), 'search', query] as const,
  },

  /**
   * Memory-related queries
   */
  memory: {
    all: () => [...queryKeys.all, 'memory'] as const,
    working: () => [...queryKeys.memory.all(), 'working'] as const,
    semantic: () => [...queryKeys.memory.all(), 'semantic'] as const,
    semanticPaths: () => [...queryKeys.memory.all(), 'semantic', 'paths'] as const,
    episodic: () => [...queryKeys.memory.all(), 'episodic'] as const,
    episodicMemory: (id: string) =>
      [...queryKeys.memory.all(), 'episodic', id] as const,
    search: (query: string) =>
      [...queryKeys.memory.all(), 'search', query] as const,
    stats: () => [...queryKeys.memory.all(), 'stats'] as const,
  },

  /**
   * Cognitive-related queries
   */
  cognitive: {
    all: () => [...queryKeys.all, 'cognitive'] as const,
    status: () => [...queryKeys.cognitive.all(), 'status'] as const,
    sessions: () => [...queryKeys.cognitive.all(), 'sessions'] as const,
    session: (id: string) =>
      [...queryKeys.cognitive.all(), 'session', id] as const,
    rag: (query: string) =>
      [...queryKeys.cognitive.all(), 'rag', query] as const,
  },

  /**
   * Executive insights (crystallized cross-agent narratives)
   */
  executiveInsights: {
    all: () => [...queryKeys.all, 'executive-insights'] as const,
    list: (brand?: string) =>
      [...queryKeys.executiveInsights.all(), 'list', brand ?? ''] as const,
  },

  /**
   * Explain-related queries (XAI)
   */
  explain: {
    all: () => [...queryKeys.all, 'explain'] as const,
    models: () => [...queryKeys.explain.all(), 'models'] as const,
    prediction: (predictionId: string) =>
      [...queryKeys.explain.all(), 'prediction', predictionId] as const,
    history: (patientId: string) =>
      [...queryKeys.explain.all(), 'history', patientId] as const,
    health: () => [...queryKeys.explain.all(), 'health'] as const,
    global: (modelType: string, brand: string, sampleSize: number) =>
      [...queryKeys.explain.all(), 'global', modelType, brand, sampleSize] as const,
    sampleEntities: (modelType: string, limit: number) =>
      [...queryKeys.explain.all(), 'sample-entities', modelType, limit] as const,
  },

  /**
   * RAG-related queries
   */
  rag: {
    all: () => [...queryKeys.all, 'rag'] as const,
    documents: () => [...queryKeys.rag.all(), 'documents'] as const,
    search: (query: string) =>
      [...queryKeys.rag.all(), 'search', query] as const,
    entities: (query: string) =>
      [...queryKeys.rag.all(), 'entities', query] as const,
    subgraph: (entity: string) =>
      [...queryKeys.rag.all(), 'subgraph', entity] as const,
    paths: (source: string, target: string) =>
      [...queryKeys.rag.all(), 'paths', source, target] as const,
    stats: () => [...queryKeys.rag.all(), 'stats'] as const,
    health: () => [...queryKeys.rag.all(), 'health'] as const,
  },

  /**
   * Health check queries
   */
  health: {
    all: () => [...queryKeys.all, 'health'] as const,
    api: () => [...queryKeys.health.all(), 'api'] as const,
  },

  /**
   * Monitoring-related queries (drift, alerts, health, performance)
   */
  monitoring: {
    all: () => [...queryKeys.all, 'monitoring'] as const,
    // Drift detection
    driftLatest: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'drift', 'latest', modelId] as const,
    driftHistory: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'drift', 'history', modelId] as const,
    driftStatus: (taskId: string) =>
      [...queryKeys.monitoring.all(), 'drift', 'status', taskId] as const,
    // Alerts
    alerts: () => [...queryKeys.monitoring.all(), 'alerts'] as const,
    alert: (alertId: string) =>
      [...queryKeys.monitoring.all(), 'alerts', alertId] as const,
    // Runs
    runs: () => [...queryKeys.monitoring.all(), 'runs'] as const,
    // Model health
    modelHealth: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'health', modelId] as const,
    // Performance
    performanceTrend: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'trend', modelId] as const,
    performanceAlerts: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'alerts', modelId] as const,
    performanceCompare: (modelId: string, otherModelId: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'compare', modelId, otherModelId] as const,
    performanceConfusion: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'confusion', modelId] as const,
    performanceRoc: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'roc', modelId] as const,
    brandSummary: (brand: string) =>
      [...queryKeys.monitoring.all(), 'performance', 'brand-summary', brand] as const,
    // Retraining
    retrainingStatus: (jobId: string) =>
      [...queryKeys.monitoring.all(), 'retraining', 'status', jobId] as const,
    retrainingEvaluate: (modelId: string) =>
      [...queryKeys.monitoring.all(), 'retraining', 'evaluate', modelId] as const,
  },

  /**
   * KPI-related queries
   */
  kpi: {
    all: () => [...queryKeys.all, 'kpi'] as const,
    list: () => [...queryKeys.kpi.all(), 'list'] as const,
    workstreams: () => [...queryKeys.kpi.all(), 'workstreams'] as const,
    health: () => [...queryKeys.kpi.all(), 'health'] as const,
    detail: (kpiId: string) =>
      [...queryKeys.kpi.all(), 'detail', kpiId] as const,
    history: (kpiId: string, brand: string) =>
      [...queryKeys.kpi.all(), 'history', kpiId, brand] as const,
  },

  /**
   * Model predictions queries
   */
  predictions: {
    all: () => [...queryKeys.all, 'predictions'] as const,
    modelHealth: (modelName: string) =>
      [...queryKeys.predictions.all(), 'health', modelName] as const,
    modelInfo: (modelName: string) =>
      [...queryKeys.predictions.all(), 'info', modelName] as const,
    modelsStatus: () =>
      [...queryKeys.predictions.all(), 'status'] as const,
  },

  /**
   * Digital Twin simulation queries
   */
  /**
   * Expert-review queue queries (R6-F2 human-in-the-loop)
   */
  expertReviews: {
    all: () => [...queryKeys.all, 'expert-reviews'] as const,
    pending: (params?: { brand?: string; reviewer_id?: string; limit?: number }) =>
      [
        ...queryKeys.expertReviews.all(),
        'pending',
        params?.brand ?? null,
        params?.reviewer_id ?? null,
        params?.limit ?? 50,
      ] as const,
    summary: (params?: { brand?: string }) =>
      [...queryKeys.expertReviews.all(), 'summary', params?.brand ?? null] as const,
  },

  digitalTwin: {
    all: () => [...queryKeys.all, 'digital-twin'] as const,
    simulation: (simulationId: string) =>
      [...queryKeys.digitalTwin.all(), 'simulation', simulationId] as const,
    // NOTE: the base ['…', 'history'] prefix (no params) is used for
    // partial-match invalidation; the parameterised form folds in
    // brand + limit/offset so a brand filter and paginated reads do not
    // collide in the cache (switching brand refetches).
    history: (params?: { brand?: string; limit?: number; offset?: number }) =>
      [
        ...queryKeys.digitalTwin.all(),
        'history',
        params?.brand ?? 'all',
        params?.limit ?? 20,
        params?.offset ?? 0,
      ] as const,
    health: () => [...queryKeys.digitalTwin.all(), 'health'] as const,
    // Brand-aware availability is folded into the key so switching brands
    // refetches (and does not collide with another brand's cached result).
    interventionTypes: (params?: { brand?: string; twin_type?: string }) =>
      [
        ...queryKeys.digitalTwin.all(),
        'intervention-types',
        params?.brand ?? null,
        params?.twin_type ?? 'hcp',
      ] as const,
  },

  /**
   * Gap Analysis queries
   */
  gaps: {
    all: () => [...queryKeys.all, 'gaps'] as const,
    analysis: (analysisId: string) =>
      [...queryKeys.gaps.all(), 'analysis', analysisId] as const,
    opportunities: () => [...queryKeys.gaps.all(), 'opportunities'] as const,
    health: () => [...queryKeys.gaps.all(), 'health'] as const,
  },

  /**
   * A/B Testing / Experiments queries
   */
  experiments: {
    all: () => [...queryKeys.all, 'experiments'] as const,
    assignments: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'assignments', experimentId] as const,
    enrollmentStats: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'enrollment', experimentId] as const,
    interimAnalyses: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'interim', experimentId] as const,
    results: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'results', experimentId] as const,
    segmentResults: (experimentId: string, segmentVar: string) =>
      [...queryKeys.experiments.all(), 'results', experimentId, 'segment', segmentVar] as const,
    srmChecks: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'srm', experimentId] as const,
    fidelityComparisons: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'fidelity', experimentId] as const,
    health: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'health', experimentId] as const,
    alerts: (experimentId: string) =>
      [...queryKeys.experiments.all(), 'alerts', experimentId] as const,
  },

  /**
   * Causal Inference queries
   */
  causal: {
    all: () => [...queryKeys.all, 'causal'] as const,
    variables: (dataset?: string) =>
      [...queryKeys.causal.all(), 'variables', dataset ?? 'patient_journeys'] as const,
    hierarchicalAnalysis: (analysisId: string) =>
      [...queryKeys.causal.all(), 'hierarchical', analysisId] as const,
    estimators: (library?: string) =>
      [...queryKeys.causal.all(), 'estimators', library ?? 'all'] as const,
    health: () => [...queryKeys.causal.all(), 'health'] as const,
    history: (limit?: number) =>
      [...queryKeys.causal.all(), 'history', limit ?? 20] as const,
    valueChains: (brand?: string, region?: string, limit?: number) =>
      [
        ...queryKeys.causal.all(),
        'value-chains',
        brand ?? 'All',
        region ?? 'All',
        limit ?? 3,
      ] as const,
    treatmentEffects: (cohort?: string, brand?: string) =>
      [
        ...queryKeys.causal.all(),
        'treatment-effects',
        cohort ?? '',
        brand ?? '',
      ] as const,
  },

  /**
   * Resource Optimization queries
   */
  resources: {
    all: () => [...queryKeys.all, 'resources'] as const,
    optimization: (optimizationId: string) =>
      [...queryKeys.resources.all(), 'optimization', optimizationId] as const,
    scenarios: () => [...queryKeys.resources.all(), 'scenarios'] as const,
    health: () => [...queryKeys.resources.all(), 'health'] as const,
  },

  /**
   * Segment Analysis queries
   */
  segments: {
    all: () => [...queryKeys.all, 'segments'] as const,
    analysis: (analysisId: string) =>
      [...queryKeys.segments.all(), 'analysis', analysisId] as const,
    policies: () => [...queryKeys.segments.all(), 'policies'] as const,
    health: () => [...queryKeys.segments.all(), 'health'] as const,
  },

  /**
   * Health Score queries (Tier 3 Fast Path)
   */
  healthScore: {
    all: () => [...queryKeys.all, 'health-score'] as const,
    check: (scope: string) =>
      [...queryKeys.healthScore.all(), 'check', scope] as const,
    quick: () => [...queryKeys.healthScore.all(), 'quick'] as const,
    full: () => [...queryKeys.healthScore.all(), 'full'] as const,
    components: () => [...queryKeys.healthScore.all(), 'components'] as const,
    models: () => [...queryKeys.healthScore.all(), 'models'] as const,
    pipelines: () => [...queryKeys.healthScore.all(), 'pipelines'] as const,
    agents: () => [...queryKeys.healthScore.all(), 'agents'] as const,
    history: (limit?: number) =>
      [...queryKeys.healthScore.all(), 'history', limit ?? 20] as const,
    status: () => [...queryKeys.healthScore.all(), 'status'] as const,
  },

  /**
   * Audit Chain queries (compliance and traceability)
   */
  audit: {
    all: () => [...queryKeys.all, 'audit'] as const,
    workflow: (workflowId: string) =>
      [...queryKeys.audit.all(), 'workflow', workflowId] as const,
    workflowSummary: (workflowId: string) =>
      [...queryKeys.audit.all(), 'workflow', workflowId, 'summary'] as const,
    workflowVerification: (workflowId: string) =>
      [...queryKeys.audit.all(), 'workflow', workflowId, 'verify'] as const,
    recent: () => [...queryKeys.audit.all(), 'recent'] as const,
  },

  /**
   * Feedback Learning queries (Tier 5 self-improvement)
   */
  feedback: {
    all: () => [...queryKeys.all, 'feedback'] as const,
    learning: (batchId: string) =>
      [...queryKeys.feedback.all(), 'learning', batchId] as const,
    patterns: () => [...queryKeys.feedback.all(), 'patterns'] as const,
    updates: () => [...queryKeys.feedback.all(), 'updates'] as const,
    health: () => [...queryKeys.feedback.all(), 'health'] as const,
  },
} as const;

export default queryClient;
