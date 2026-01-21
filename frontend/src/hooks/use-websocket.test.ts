/**
 * useWebSocket Hook Tests
 * =======================
 *
 * Tests for the WebSocket hook with auto-reconnect functionality.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useWebSocket, useGraphWebSocket } from './use-websocket';

// =============================================================================
// MOCK WEBSOCKET
// =============================================================================

class MockWebSocket {
  // WebSocket readyState constants
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  static instances: MockWebSocket[] = [];
  static lastUrl: string | null = null;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  close = vi.fn();
  send = vi.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.lastUrl = url;
    MockWebSocket.instances.push(this);
  }

  // Helper to simulate open
  simulateOpen() {
    this.readyState = WebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  // Helper to simulate message
  simulateMessage(data: unknown) {
    this.onmessage?.({
      data: typeof data === 'string' ? data : JSON.stringify(data),
    } as MessageEvent);
  }

  // Helper to simulate error
  simulateError() {
    this.onerror?.(new Event('error'));
  }

  // Helper to simulate close
  simulateClose(code: number = 1000, reason: string = '') {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.({ code, reason } as CloseEvent);
  }
}

// Mock auth store
vi.mock('@/stores/auth-store', () => ({
  useAuthStore: {
    getState: vi.fn(() => ({
      session: { access_token: 'test-token' },
    })),
  },
}));

// Mock env
vi.mock('@/config/env', () => ({
  env: {
    apiUrl: 'http://localhost:8000',
    isDev: false,
  },
  buildApiUrl: (path: string) => `http://localhost:8000${path}`,
}));

// =============================================================================
// HELPER
// =============================================================================

/**
 * Helper to wait for WebSocket instance to be created.
 * Flushes React effects and waits for WebSocket constructor to be called.
 */
async function waitForConnection() {
  // First flush effects
  await act(async () => {
    // Give React time to schedule and run effects
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  // Then wait for WebSocket to be created
  await waitFor(
    () => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    },
    { timeout: 500 }
  );
}

// =============================================================================
// TESTS
// =============================================================================

describe('useWebSocket', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    MockWebSocket.lastUrl = null;
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  // Debug test - verify mock is set up correctly
  it('should have mocked WebSocket globally', () => {
    expect(globalThis.WebSocket).toBe(MockWebSocket);
    expect(window.WebSocket).toBe(MockWebSocket);

    // Create a WebSocket directly
    const ws = new WebSocket('ws://test');
    expect(MockWebSocket.instances.length).toBe(1);
    expect(MockWebSocket.lastUrl).toBe('ws://test');
  });

  describe('connection', () => {
    it('should connect on mount by default', async () => {
      const onMessage = vi.fn();
      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      expect(MockWebSocket.instances.length).toBe(1);
      expect(MockWebSocket.lastUrl).toContain('ws://localhost:8000/graph/stream');
    });

    it('should not connect on mount when connectOnMount is false', async () => {
      const onMessage = vi.fn();
      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          connectOnMount: false,
        })
      );

      // Wait a tick to ensure effect ran
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(MockWebSocket.instances.length).toBe(0);
    });

    it('should include auth token in URL when withAuth is true', async () => {
      const onMessage = vi.fn();
      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          withAuth: true,
        })
      );

      await waitForConnection();

      expect(MockWebSocket.instances.length).toBe(1);
      expect(MockWebSocket.lastUrl).toContain('token=test-token');
    });

    it('should not include auth token when withAuth is false', async () => {
      const onMessage = vi.fn();
      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          withAuth: false,
        })
      );

      await waitForConnection();

      expect(MockWebSocket.instances.length).toBe(1);
      expect(MockWebSocket.lastUrl).not.toContain('token=');
    });

    it('should set state to connected when WebSocket opens', async () => {
      const onMessage = vi.fn();
      const onOpen = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onOpen,
        })
      );

      await waitForConnection();

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      expect(result.current.state).toBe('connected');
      expect(result.current.isConnected).toBe(true);
      expect(onOpen).toHaveBeenCalled();
    });
  });

  describe('messages', () => {
    it('should parse JSON messages and call onMessage', async () => {
      const onMessage = vi.fn();

      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({ type: 'update', data: 'test' });
      });

      expect(onMessage).toHaveBeenCalledWith({ type: 'update', data: 'test' });
    });

    it('should pass non-JSON messages directly', async () => {
      const onMessage = vi.fn();

      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage('plain text');
      });

      expect(onMessage).toHaveBeenCalledWith('plain text');
    });

    it('should ignore pong messages', async () => {
      const onMessage = vi.fn();

      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({ type: 'pong' });
      });

      expect(onMessage).not.toHaveBeenCalled();
    });
  });

  describe('send', () => {
    it('should send string messages when connected', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      let success: boolean;
      act(() => {
        success = result.current.send('hello');
      });

      expect(success!).toBe(true);
      expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith('hello');
    });

    it('should stringify object messages when connected', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        result.current.send({ type: 'subscribe' });
      });

      expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'subscribe' })
      );
    });

    it('should return false when not connected', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          connectOnMount: false,
        })
      );

      // Wait a tick to ensure effect ran
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });

      let success: boolean;
      act(() => {
        success = result.current.send('hello');
      });

      expect(success!).toBe(false);
    });
  });

  describe('disconnect', () => {
    it('should close WebSocket on disconnect', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        result.current.disconnect();
      });

      expect(MockWebSocket.instances[0].close).toHaveBeenCalledWith(1000, 'Client disconnected');
      expect(result.current.state).toBe('disconnected');
    });

    it('should call onClose when connection closes', async () => {
      const onMessage = vi.fn();
      const onClose = vi.fn();

      renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onClose,
          autoReconnect: false,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateClose(1000, 'Normal closure');
      });

      expect(onClose).toHaveBeenCalled();
    });
  });

  describe('auto-reconnect', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should attempt to reconnect on abnormal close', async () => {
      const onMessage = vi.fn();
      const onReconnect = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onReconnect,
          autoReconnect: true,
          initialReconnectDelay: 100,
        })
      );

      // Flush initial effect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateClose(1006, 'Abnormal closure');
      });

      expect(result.current.state).toBe('reconnecting');
      expect(onReconnect).toHaveBeenCalledWith(1, 10);

      // Advance timers to trigger reconnect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(100);
      });

      // Should have created a new WebSocket instance
      expect(MockWebSocket.instances.length).toBe(2);
    });

    it('should not reconnect on normal close (1000)', async () => {
      const onMessage = vi.fn();
      const onReconnect = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onReconnect,
          autoReconnect: true,
        })
      );

      // Flush initial effect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateClose(1000, 'Normal closure');
      });

      expect(result.current.state).toBe('disconnected');
      expect(onReconnect).not.toHaveBeenCalled();
    });

    it('should stop reconnecting after max attempts', async () => {
      const onMessage = vi.fn();
      const onMaxReconnectAttemptsReached = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onMaxReconnectAttemptsReached,
          autoReconnect: true,
          maxReconnectAttempts: 2,
          initialReconnectDelay: 100,
        })
      );

      // Flush initial effect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(MockWebSocket.instances.length).toBe(1);

      // Simulate multiple failed connections
      for (let i = 0; i < 3; i++) {
        const instance = MockWebSocket.instances[MockWebSocket.instances.length - 1];
        act(() => {
          instance.simulateOpen();
        });
        act(() => {
          instance.simulateClose(1006, 'Abnormal closure');
        });
        await act(async () => {
          await vi.advanceTimersByTimeAsync(500);
        });
      }

      expect(result.current.state).toBe('error');
      expect(onMaxReconnectAttemptsReached).toHaveBeenCalled();
    });

    it('should reset reconnect attempts on successful connection', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          autoReconnect: true,
          initialReconnectDelay: 100,
        })
      );

      // Flush initial effect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateClose(1006, 'Abnormal closure');
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(100);
      });

      // Reconnected successfully
      expect(MockWebSocket.instances.length).toBe(2);

      act(() => {
        MockWebSocket.instances[1].simulateOpen();
      });

      expect(result.current.reconnectAttempts).toBe(0);
    });

    it('should not reconnect when autoReconnect is false', async () => {
      const onMessage = vi.fn();
      const onReconnect = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onReconnect,
          autoReconnect: false,
        })
      );

      // Flush initial effect
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateClose(1006, 'Abnormal closure');
      });

      expect(result.current.state).toBe('disconnected');
      expect(onReconnect).not.toHaveBeenCalled();
    });
  });

  describe('reset', () => {
    it('should reset connection state and reconnect', async () => {
      const onMessage = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      expect(MockWebSocket.instances.length).toBe(1);

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        result.current.reset();
      });

      await waitFor(() => {
        expect(MockWebSocket.instances.length).toBe(2);
      });

      expect(result.current.reconnectAttempts).toBe(0);
    });
  });

  describe('error handling', () => {
    it('should call onError and track lastError', async () => {
      const onMessage = vi.fn();
      const onError = vi.fn();

      const { result } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
          onError,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      act(() => {
        MockWebSocket.instances[0].simulateError();
      });

      expect(onError).toHaveBeenCalled();
      expect(result.current.lastError).not.toBeNull();
    });
  });

  describe('cleanup', () => {
    it('should close WebSocket on unmount', async () => {
      const onMessage = vi.fn();

      const { unmount } = renderHook(() =>
        useWebSocket({
          endpoint: '/graph/stream',
          onMessage,
        })
      );

      await waitForConnection();

      act(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      unmount();

      expect(MockWebSocket.instances[0].close).toHaveBeenCalledWith(
        1000,
        'Component unmounted'
      );
    });
  });
});

describe('useGraphWebSocket', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    MockWebSocket.lastUrl = null;
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('should use /graph/stream endpoint', async () => {
    const onMessage = vi.fn();

    renderHook(() => useGraphWebSocket(onMessage));

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    expect(MockWebSocket.lastUrl).toContain('/graph/stream');
  });

  it('should pass messages to callback', async () => {
    const onMessage = vi.fn();

    renderHook(() => useGraphWebSocket(onMessage));

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    act(() => {
      MockWebSocket.instances[0].simulateMessage({ event: 'node_added' });
    });

    expect(onMessage).toHaveBeenCalledWith({ event: 'node_added' });
  });
});
