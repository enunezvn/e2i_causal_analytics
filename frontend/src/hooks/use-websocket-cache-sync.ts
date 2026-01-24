/**
 * WebSocket Cache Sync Hook
 * ==========================
 *
 * Phase 4 Implementation: Bridge WebSocket broadcasts to React Query cache invalidation.
 *
 * Problem: WebSocket broadcasts graph updates in real-time, but React Query has
 * a 5-minute stale time. Users may see stale data until the next scheduled refetch.
 *
 * Solution: On WebSocket message, call `queryClient.invalidateQueries()` for
 * affected query keys based on the event type.
 *
 * @module hooks/use-websocket-cache-sync
 */

import { useCallback, useEffect, useRef } from 'react';
import { useQueryClient, type QueryKey } from '@tanstack/react-query';
import { useGraphWebSocket, type UseWebSocketReturn } from './use-websocket';
import { queryKeys } from '@/lib/query-client';
import { env } from '@/config/env';

// =============================================================================
// TYPES
// =============================================================================

/**
 * WebSocket event types that trigger cache invalidation.
 * Maps to GraphStreamMessage.event_type from the backend.
 */
export type GraphEventType =
  | 'episode_added'
  | 'node_added'
  | 'node_updated'
  | 'node_deleted'
  | 'edge_added'
  | 'relationship_added'
  | 'relationship_updated'
  | 'relationship_deleted'
  | 'graph_updated'
  | 'stats_changed'
  | 'subscription_updated'
  | 'pong'
  | 'error';

/**
 * Payload structure for graph WebSocket messages.
 * Based on GraphStreamMessage from the backend.
 */
export interface GraphStreamPayload {
  event_type: GraphEventType;
  payload: {
    episode_id?: string;
    node_id?: string;
    entity_id?: string;
    relationship_id?: string;
    edge_id?: string;
    type?: string;
    entity_type?: string;
    source?: string;
    entities_count?: number;
    relationships_count?: number;
    [key: string]: unknown;
  };
  timestamp?: string;
  session_id?: string;
}

/**
 * Configuration options for the cache sync hook.
 */
export interface UseWebSocketCacheSyncOptions {
  /** Enable cache sync (default: true) */
  enabled?: boolean;

  /** Custom handler for specific event types */
  onEvent?: (event: GraphStreamPayload) => void;

  /** Debounce invalidation in ms (default: 100) */
  debounceMs?: number;

  /** Additional query keys to invalidate on any event */
  additionalInvalidations?: QueryKey[];
}

/**
 * Return value from the cache sync hook.
 */
export interface UseWebSocketCacheSyncReturn extends UseWebSocketReturn {
  /** Number of invalidations triggered */
  invalidationCount: number;

  /** Last event received */
  lastEvent: GraphStreamPayload | null;
}

// =============================================================================
// EVENT TO QUERY KEY MAPPING
// =============================================================================

/**
 * Maps WebSocket event types to React Query keys that should be invalidated.
 *
 * Design principles:
 * 1. Be conservative - only invalidate what's likely affected
 * 2. Use broad keys for bulk operations (e.g., episode_added affects many nodes)
 * 3. Use specific keys when node/relationship ID is available
 */
function getQueryKeysForEvent(event: GraphStreamPayload): QueryKey[] {
  const keysToInvalidate: QueryKey[] = [];
  const { event_type, payload } = event;

  switch (event_type) {
    case 'episode_added':
      // Episodes add new nodes and relationships - invalidate graph-related queries
      keysToInvalidate.push(
        queryKeys.graph.all(),
        queryKeys.graph.nodes(),
        queryKeys.graph.relationships(),
        queryKeys.graph.stats(),
        queryKeys.memory.all(),
        queryKeys.memory.semantic(),
        queryKeys.memory.episodic(),
        queryKeys.rag.all(),
      );
      break;

    case 'node_added':
    case 'node_updated':
      // Invalidate node list and stats
      keysToInvalidate.push(
        queryKeys.graph.nodes(),
        queryKeys.graph.stats(),
      );
      // If specific node ID, invalidate that node's queries
      if (payload.node_id || payload.entity_id) {
        const nodeId = payload.node_id || payload.entity_id!;
        keysToInvalidate.push(
          queryKeys.graph.node(nodeId),
          queryKeys.graph.nodeNetwork(nodeId),
        );
      }
      break;

    case 'node_deleted':
      // Node removal affects the whole graph
      keysToInvalidate.push(
        queryKeys.graph.all(),
        queryKeys.graph.nodes(),
        queryKeys.graph.relationships(),
        queryKeys.graph.stats(),
      );
      break;

    case 'edge_added':
    case 'relationship_added':
    case 'relationship_updated':
      // Invalidate relationship list and stats
      keysToInvalidate.push(
        queryKeys.graph.relationships(),
        queryKeys.graph.stats(),
      );
      break;

    case 'relationship_deleted':
      // Relationship removal affects the whole graph
      keysToInvalidate.push(
        queryKeys.graph.all(),
        queryKeys.graph.relationships(),
        queryKeys.graph.stats(),
      );
      break;

    case 'graph_updated':
      // Generic graph update - invalidate all graph queries
      keysToInvalidate.push(queryKeys.graph.all());
      break;

    case 'stats_changed':
      // Only stats need refresh
      keysToInvalidate.push(queryKeys.graph.stats());
      break;

    case 'subscription_updated':
    case 'pong':
    case 'error':
      // These don't affect cached data
      break;

    default:
      // Unknown event type - log for debugging but don't invalidate
      if (env.isDev) {
        console.debug(`[CacheSync] Unknown event type: ${event_type}`);
      }
  }

  return keysToInvalidate;
}

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * React hook that syncs WebSocket graph updates with React Query cache.
 *
 * Automatically invalidates relevant queries when graph changes are broadcast
 * via WebSocket, ensuring the UI stays in sync with backend state.
 *
 * @param options - Configuration options
 * @returns WebSocket state, controls, and cache sync metadata
 *
 * @example
 * ```tsx
 * // Basic usage - automatically syncs cache on graph updates
 * const { isConnected, invalidationCount } = useWebSocketCacheSync();
 *
 * // With custom event handler
 * const { isConnected } = useWebSocketCacheSync({
 *   onEvent: (event) => {
 *     if (event.event_type === 'episode_added') {
 *       toast.success('New episode added to knowledge graph');
 *     }
 *   },
 * });
 *
 * // Disable when not needed
 * const { isConnected } = useWebSocketCacheSync({ enabled: false });
 * ```
 */
export function useWebSocketCacheSync(
  options: UseWebSocketCacheSyncOptions = {}
): UseWebSocketCacheSyncReturn {
  const {
    enabled = true,
    onEvent,
    debounceMs = 100,
    additionalInvalidations = [],
  } = options;

  const queryClient = useQueryClient();
  const invalidationCountRef = useRef(0);
  const lastEventRef = useRef<GraphStreamPayload | null>(null);
  const debounceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingInvalidationsRef = useRef<Set<string>>(new Set());

  /**
   * Process invalidations (debounced to batch rapid updates)
   */
  const processInvalidations = useCallback(() => {
    if (pendingInvalidationsRef.current.size === 0) return;

    const keysToInvalidate = Array.from(pendingInvalidationsRef.current);
    pendingInvalidationsRef.current.clear();

    // Parse keys back from JSON strings
    const queryKeys = keysToInvalidate.map((k) => JSON.parse(k) as QueryKey);

    if (env.isDev) {
      console.debug(`[CacheSync] Invalidating ${queryKeys.length} query keys:`, queryKeys);
    }

    // Invalidate each key
    for (const key of queryKeys) {
      void queryClient.invalidateQueries({ queryKey: key });
    }

    invalidationCountRef.current += queryKeys.length;
  }, [queryClient]);

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback(
    (data: unknown) => {
      if (!enabled) return;

      // Type guard for GraphStreamPayload
      if (
        typeof data !== 'object' ||
        data === null ||
        !('event_type' in data)
      ) {
        if (env.isDev) {
          console.debug('[CacheSync] Ignoring non-graph message:', data);
        }
        return;
      }

      const event = data as GraphStreamPayload;
      lastEventRef.current = event;

      if (env.isDev) {
        console.debug(`[CacheSync] Received event: ${event.event_type}`, event.payload);
      }

      // Call custom event handler
      onEvent?.(event);

      // Get query keys to invalidate
      const keysForEvent = getQueryKeysForEvent(event);
      const allKeys = [...keysForEvent, ...additionalInvalidations];

      if (allKeys.length === 0) return;

      // Add to pending invalidations (deduplicated by JSON key)
      for (const key of allKeys) {
        pendingInvalidationsRef.current.add(JSON.stringify(key));
      }

      // Debounce the actual invalidation
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      debounceTimeoutRef.current = setTimeout(processInvalidations, debounceMs);
    },
    [enabled, onEvent, additionalInvalidations, debounceMs, processInvalidations]
  );

  // Use the graph WebSocket hook
  const wsReturn = useGraphWebSocket(handleMessage, {
    connectOnMount: enabled,
  });

  // Cleanup debounce timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
        // Process any pending invalidations before unmount
        processInvalidations();
      }
    };
  }, [processInvalidations]);

  return {
    ...wsReturn,
    invalidationCount: invalidationCountRef.current,
    lastEvent: lastEventRef.current,
  };
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Manually trigger cache invalidation for a specific event type.
 * Useful for testing or programmatic invalidation.
 *
 * @param queryClient - React Query client
 * @param eventType - Event type to simulate
 * @param payload - Optional payload data
 *
 * @example
 * ```tsx
 * // Simulate an episode_added event
 * invalidateCacheForEvent(queryClient, 'episode_added', {
 *   episode_id: 'ep_123',
 *   entities_count: 5,
 * });
 * ```
 */
export function invalidateCacheForEvent(
  queryClient: ReturnType<typeof useQueryClient>,
  eventType: GraphEventType,
  payload: GraphStreamPayload['payload'] = {}
): void {
  const event: GraphStreamPayload = {
    event_type: eventType,
    payload,
  };

  const keysToInvalidate = getQueryKeysForEvent(event);

  if (env.isDev) {
    console.debug(`[CacheSync] Manual invalidation for ${eventType}:`, keysToInvalidate);
  }

  for (const key of keysToInvalidate) {
    void queryClient.invalidateQueries({ queryKey: key });
  }
}

export default useWebSocketCacheSync;
