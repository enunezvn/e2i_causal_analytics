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
  digitalTwin: {
    all: () => [...queryKeys.all, 'digital-twin'] as const,
    simulation: (simulationId: string) =>
      [...queryKeys.digitalTwin.all(), 'simulation', simulationId] as const,
    history: (brand?: string) =>
      [...queryKeys.digitalTwin.all(), 'history', brand ?? 'all'] as const,
    health: () => [...queryKeys.digitalTwin.all(), 'health'] as const,
  },
} as const;

export default queryClient;
