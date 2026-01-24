/**
 * Tests for WebSocket Cache Sync Hook
 * ====================================
 *
 * Phase 4 Implementation: Tests for WebSocket → React Query cache invalidation bridge.
 *
 * @module hooks/use-websocket-cache-sync.test
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useWebSocketCacheSync, invalidateCacheForEvent } from './use-websocket-cache-sync';
import type { GraphStreamPayload } from './use-websocket-cache-sync';

// Mock the useGraphWebSocket hook
const mockSend = vi.fn().mockReturnValue(true);
const mockConnect = vi.fn();
const mockDisconnect = vi.fn();
const mockReset = vi.fn();

let mockOnMessage: ((data: unknown) => void) | undefined;

vi.mock('./use-websocket', () => ({
  useGraphWebSocket: (onMessage: (data: unknown) => void, _options?: unknown) => {
    mockOnMessage = onMessage;
    return {
      state: 'connected',
      isConnected: true,
      reconnectAttempts: 0,
      lastError: null,
      connect: mockConnect,
      disconnect: mockDisconnect,
      send: mockSend,
      reset: mockReset,
    };
  },
}));

// Mock the env config
vi.mock('@/config/env', () => ({
  env: {
    isDev: false,
    isProd: true,
  },
}));

// =============================================================================
// TEST HELPERS
// =============================================================================

function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
}

function createWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

function simulateWebSocketMessage(message: GraphStreamPayload): void {
  if (mockOnMessage) {
    act(() => {
      mockOnMessage!(message);
    });
  }
}

// =============================================================================
// TESTS
// =============================================================================

describe('useWebSocketCacheSync', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
    mockOnMessage = undefined;
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('initialization', () => {
    it('should connect to WebSocket when enabled', () => {
      renderHook(() => useWebSocketCacheSync({ enabled: true }), {
        wrapper: createWrapper(queryClient),
      });

      expect(mockOnMessage).toBeDefined();
    });

    it('should return connected state', () => {
      const { result } = renderHook(() => useWebSocketCacheSync(), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isConnected).toBe(true);
      expect(result.current.state).toBe('connected');
    });

    it('should initialize with zero invalidation count', () => {
      const { result } = renderHook(() => useWebSocketCacheSync(), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.invalidationCount).toBe(0);
      expect(result.current.lastEvent).toBeNull();
    });
  });

  describe('event handling', () => {
    it('should call onEvent callback when message received', async () => {
      const onEvent = vi.fn();

      renderHook(() => useWebSocketCacheSync({ onEvent }), {
        wrapper: createWrapper(queryClient),
      });

      const event: GraphStreamPayload = {
        event_type: 'episode_added',
        payload: { episode_id: 'ep_123', entities_count: 5 },
      };

      simulateWebSocketMessage(event);

      expect(onEvent).toHaveBeenCalledWith(event);
    });

    it('should ignore non-graph messages', async () => {
      const onEvent = vi.fn();
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      renderHook(() => useWebSocketCacheSync({ onEvent }), {
        wrapper: createWrapper(queryClient),
      });

      // Send a message without event_type
      act(() => {
        mockOnMessage?.({ type: 'pong' });
      });

      vi.advanceTimersByTime(200);

      expect(onEvent).not.toHaveBeenCalled();
      expect(invalidateSpy).not.toHaveBeenCalled();
    });

    it('should not process messages when disabled', async () => {
      const onEvent = vi.fn();

      renderHook(() => useWebSocketCacheSync({ enabled: false, onEvent }), {
        wrapper: createWrapper(queryClient),
      });

      const event: GraphStreamPayload = {
        event_type: 'episode_added',
        payload: {},
      };

      simulateWebSocketMessage(event);

      expect(onEvent).not.toHaveBeenCalled();
    });
  });

  describe('cache invalidation', () => {
    it('should invalidate graph queries on episode_added', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      renderHook(() => useWebSocketCacheSync(), {
        wrapper: createWrapper(queryClient),
      });

      simulateWebSocketMessage({
        event_type: 'episode_added',
        payload: { episode_id: 'ep_123' },
      });

      // Advance past debounce
      vi.advanceTimersByTime(200);

      // Should invalidate multiple graph-related keys
      expect(invalidateSpy).toHaveBeenCalled();
      const calls = invalidateSpy.mock.calls;

      // Check that graph-related keys were invalidated
      const invalidatedKeys = calls.map((call) => call[0]?.queryKey);
      expect(invalidatedKeys.some((k) => k?.includes('graph'))).toBe(true);
    });

    it('should invalidate specific node queries on node_updated', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      renderHook(() => useWebSocketCacheSync(), {
        wrapper: createWrapper(queryClient),
      });

      simulateWebSocketMessage({
        event_type: 'node_updated',
        payload: { node_id: 'node_456' },
      });

      vi.advanceTimersByTime(200);

      expect(invalidateSpy).toHaveBeenCalled();
    });

    it('should debounce rapid invalidations', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      renderHook(() => useWebSocketCacheSync({ debounceMs: 100 }), {
        wrapper: createWrapper(queryClient),
      });

      // Send multiple messages rapidly
      simulateWebSocketMessage({
        event_type: 'node_added',
        payload: { node_id: 'node_1' },
      });
      simulateWebSocketMessage({
        event_type: 'node_added',
        payload: { node_id: 'node_2' },
      });
      simulateWebSocketMessage({
        event_type: 'node_added',
        payload: { node_id: 'node_3' },
      });

      // Before debounce timeout
      expect(invalidateSpy).not.toHaveBeenCalled();

      // After debounce timeout
      vi.advanceTimersByTime(150);

      // Should batch into single invalidation run
      expect(invalidateSpy).toHaveBeenCalled();
    });

    it('should not invalidate on pong or error events', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      renderHook(() => useWebSocketCacheSync(), {
        wrapper: createWrapper(queryClient),
      });

      simulateWebSocketMessage({
        event_type: 'pong',
        payload: {},
      });

      simulateWebSocketMessage({
        event_type: 'error',
        payload: { message: 'test error' },
      });

      vi.advanceTimersByTime(200);

      expect(invalidateSpy).not.toHaveBeenCalled();
    });

    it('should include additional invalidations when provided', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');
      const customKey = ['custom', 'key'];

      renderHook(
        () =>
          useWebSocketCacheSync({
            additionalInvalidations: [customKey],
          }),
        {
          wrapper: createWrapper(queryClient),
        }
      );

      simulateWebSocketMessage({
        event_type: 'episode_added',
        payload: {},
      });

      vi.advanceTimersByTime(200);

      const calls = invalidateSpy.mock.calls;
      const invalidatedKeys = calls.map((call) => call[0]?.queryKey);

      // Check custom key was included
      expect(
        invalidatedKeys.some(
          (k) => JSON.stringify(k) === JSON.stringify(customKey)
        )
      ).toBe(true);
    });
  });

  describe('event type mapping', () => {
    const testCases: Array<{
      eventType: GraphStreamPayload['event_type'];
      shouldInvalidate: boolean;
      description: string;
    }> = [
      { eventType: 'episode_added', shouldInvalidate: true, description: 'episode_added' },
      { eventType: 'node_added', shouldInvalidate: true, description: 'node_added' },
      { eventType: 'node_updated', shouldInvalidate: true, description: 'node_updated' },
      { eventType: 'node_deleted', shouldInvalidate: true, description: 'node_deleted' },
      { eventType: 'edge_added', shouldInvalidate: true, description: 'edge_added' },
      { eventType: 'relationship_added', shouldInvalidate: true, description: 'relationship_added' },
      { eventType: 'relationship_updated', shouldInvalidate: true, description: 'relationship_updated' },
      { eventType: 'relationship_deleted', shouldInvalidate: true, description: 'relationship_deleted' },
      { eventType: 'graph_updated', shouldInvalidate: true, description: 'graph_updated' },
      { eventType: 'stats_changed', shouldInvalidate: true, description: 'stats_changed' },
      { eventType: 'subscription_updated', shouldInvalidate: false, description: 'subscription_updated' },
      { eventType: 'pong', shouldInvalidate: false, description: 'pong' },
      { eventType: 'error', shouldInvalidate: false, description: 'error' },
    ];

    testCases.forEach(({ eventType, shouldInvalidate, description }) => {
      it(`should ${shouldInvalidate ? '' : 'NOT '}invalidate for ${description}`, async () => {
        const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

        renderHook(() => useWebSocketCacheSync(), {
          wrapper: createWrapper(queryClient),
        });

        simulateWebSocketMessage({
          event_type: eventType,
          payload: {},
        });

        vi.advanceTimersByTime(200);

        if (shouldInvalidate) {
          expect(invalidateSpy).toHaveBeenCalled();
        } else {
          expect(invalidateSpy).not.toHaveBeenCalled();
        }

        invalidateSpy.mockClear();
      });
    });
  });
});

describe('invalidateCacheForEvent', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
  });

  it('should manually invalidate cache for event type', () => {
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    invalidateCacheForEvent(queryClient, 'episode_added', {
      episode_id: 'ep_manual',
    });

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('should work with empty payload', () => {
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    invalidateCacheForEvent(queryClient, 'graph_updated');

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('should not invalidate for non-data events', () => {
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    invalidateCacheForEvent(queryClient, 'pong');

    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
