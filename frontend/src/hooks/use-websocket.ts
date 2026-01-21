/**
 * WebSocket Hook with Auto-Reconnect
 * ===================================
 *
 * React hook for managing WebSocket connections with automatic reconnection,
 * exponential backoff, and connection state management.
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Connection state tracking (connecting, connected, disconnected, error)
 * - Configurable retry limits and intervals
 * - Manual connect/disconnect controls
 * - Heartbeat/ping support
 * - Cleanup on unmount
 *
 * @module hooks/use-websocket
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { env, buildApiUrl } from '@/config/env';
import { useAuthStore } from '@/stores/auth-store';

// =============================================================================
// TYPES
// =============================================================================

/**
 * WebSocket connection states
 */
export type WebSocketState =
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'reconnecting'
  | 'error';

/**
 * Configuration options for the WebSocket hook
 */
export interface UseWebSocketOptions {
  /** WebSocket endpoint path (relative to API base URL) */
  endpoint: string;

  /** Enable auto-reconnect on connection loss (default: true) */
  autoReconnect?: boolean;

  /** Maximum number of reconnection attempts (default: 10) */
  maxReconnectAttempts?: number;

  /** Initial reconnect delay in ms (default: 1000) */
  initialReconnectDelay?: number;

  /** Maximum reconnect delay in ms (default: 30000) */
  maxReconnectDelay?: number;

  /** Backoff multiplier for exponential backoff (default: 1.5) */
  backoffMultiplier?: number;

  /** Heartbeat interval in ms (0 to disable, default: 30000) */
  heartbeatInterval?: number;

  /** Message to send as heartbeat (default: '{"type":"ping"}') */
  heartbeatMessage?: string;

  /** Connect immediately on mount (default: true) */
  connectOnMount?: boolean;

  /** Include auth token in connection (default: true) */
  withAuth?: boolean;

  /** Callback when message is received */
  onMessage?: (data: unknown) => void;

  /** Callback when connection opens */
  onOpen?: () => void;

  /** Callback when connection closes */
  onClose?: (event: CloseEvent) => void;

  /** Callback when error occurs */
  onError?: (event: Event) => void;

  /** Callback when reconnection is attempted */
  onReconnect?: (attempt: number, maxAttempts: number) => void;

  /** Callback when max reconnection attempts reached */
  onMaxReconnectAttemptsReached?: () => void;
}

/**
 * Return value from the useWebSocket hook
 */
export interface UseWebSocketReturn {
  /** Current connection state */
  state: WebSocketState;

  /** Whether the socket is currently connected */
  isConnected: boolean;

  /** Number of reconnection attempts made */
  reconnectAttempts: number;

  /** Last error event (if any) */
  lastError: Event | null;

  /** Manually connect to the WebSocket */
  connect: () => void;

  /** Manually disconnect from the WebSocket */
  disconnect: () => void;

  /** Send a message through the WebSocket */
  send: (data: string | object) => boolean;

  /** Reset reconnection state and attempt to connect */
  reset: () => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_OPTIONS: Required<Omit<UseWebSocketOptions, 'endpoint' | 'onMessage' | 'onOpen' | 'onClose' | 'onError' | 'onReconnect' | 'onMaxReconnectAttemptsReached'>> = {
  autoReconnect: true,
  maxReconnectAttempts: 10,
  initialReconnectDelay: 1000,
  maxReconnectDelay: 30000,
  backoffMultiplier: 1.5,
  heartbeatInterval: 30000,
  heartbeatMessage: '{"type":"ping"}',
  connectOnMount: true,
  withAuth: true,
};

// Close codes that should trigger reconnection
const RECONNECTABLE_CLOSE_CODES = new Set([
  1001, // Going Away - server shutting down
  1006, // Abnormal Closure - connection lost
  1011, // Internal Error
  1012, // Service Restart
  1013, // Try Again Later
  1014, // Bad Gateway
]);

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * React hook for WebSocket connections with auto-reconnect
 *
 * @param options - Configuration options
 * @returns WebSocket state and controls
 *
 * @example
 * ```tsx
 * const { state, isConnected, send } = useWebSocket({
 *   endpoint: '/graph/stream',
 *   onMessage: (data) => console.log('Received:', data),
 *   onReconnect: (attempt, max) => console.log(`Reconnecting ${attempt}/${max}`),
 * });
 *
 * // Send a message
 * send({ type: 'subscribe', channel: 'updates' });
 *
 * // Check connection status
 * if (isConnected) {
 *   console.log('Connected!');
 * }
 * ```
 */
export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    endpoint,
    autoReconnect = DEFAULT_OPTIONS.autoReconnect,
    maxReconnectAttempts = DEFAULT_OPTIONS.maxReconnectAttempts,
    initialReconnectDelay = DEFAULT_OPTIONS.initialReconnectDelay,
    maxReconnectDelay = DEFAULT_OPTIONS.maxReconnectDelay,
    backoffMultiplier = DEFAULT_OPTIONS.backoffMultiplier,
    heartbeatInterval = DEFAULT_OPTIONS.heartbeatInterval,
    heartbeatMessage = DEFAULT_OPTIONS.heartbeatMessage,
    connectOnMount = DEFAULT_OPTIONS.connectOnMount,
    withAuth = DEFAULT_OPTIONS.withAuth,
    onMessage,
    onOpen,
    onClose,
    onError,
    onReconnect,
    onMaxReconnectAttemptsReached,
  } = options;

  // State
  const [state, setState] = useState<WebSocketState>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastError, setLastError] = useState<Event | null>(null);

  // Refs for mutable values that shouldn't trigger re-renders
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentReconnectDelayRef = useRef(initialReconnectDelay);
  const isUnmountedRef = useRef(false);
  const shouldReconnectRef = useRef(true);

  // Callback refs to avoid stale closures
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);
  const onReconnectRef = useRef(onReconnect);
  const onMaxReconnectAttemptsReachedRef = useRef(onMaxReconnectAttemptsReached);

  // Update callback refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
    onReconnectRef.current = onReconnect;
    onMaxReconnectAttemptsReachedRef.current = onMaxReconnectAttemptsReached;
  });

  /**
   * Clear heartbeat interval
   */
  const clearHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  /**
   * Clear reconnect timeout
   */
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  /**
   * Start heartbeat
   */
  const startHeartbeat = useCallback(() => {
    if (heartbeatInterval <= 0) return;

    clearHeartbeat();
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(heartbeatMessage);
        if (env.isDev) {
          console.debug('[WebSocket] Heartbeat sent');
        }
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, heartbeatMessage, clearHeartbeat]);

  /**
   * Calculate next reconnect delay with exponential backoff
   */
  const getNextReconnectDelay = useCallback((): number => {
    const delay = currentReconnectDelayRef.current;
    currentReconnectDelayRef.current = Math.min(
      delay * backoffMultiplier,
      maxReconnectDelay
    );
    return delay;
  }, [backoffMultiplier, maxReconnectDelay]);

  /**
   * Schedule reconnection attempt
   */
  const scheduleReconnect = useCallback(() => {
    if (isUnmountedRef.current || !shouldReconnectRef.current) return;
    if (reconnectAttempts >= maxReconnectAttempts) {
      if (env.isDev) {
        console.warn('[WebSocket] Max reconnection attempts reached');
      }
      setState('error');
      onMaxReconnectAttemptsReachedRef.current?.();
      return;
    }

    const delay = getNextReconnectDelay();
    const nextAttempt = reconnectAttempts + 1;

    if (env.isDev) {
      console.debug(`[WebSocket] Scheduling reconnect in ${delay}ms (attempt ${nextAttempt}/${maxReconnectAttempts})`);
    }

    setState('reconnecting');
    onReconnectRef.current?.(nextAttempt, maxReconnectAttempts);

    clearReconnectTimeout();
    reconnectTimeoutRef.current = setTimeout(() => {
      if (!isUnmountedRef.current && shouldReconnectRef.current) {
        setReconnectAttempts(nextAttempt);
        // Trigger connect on next render via effect
        setState('connecting');
      }
    }, delay);
  }, [reconnectAttempts, maxReconnectAttempts, getNextReconnectDelay, clearReconnectTimeout]);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    // Don't connect if unmounted or already connecting/connected
    if (isUnmountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    shouldReconnectRef.current = true;
    setState('connecting');

    // Build WebSocket URL
    let wsUrl = buildApiUrl(endpoint).replace(/^http/, 'ws');

    // Add auth token if configured
    if (withAuth) {
      const session = useAuthStore.getState().session;
      if (session?.access_token) {
        const separator = wsUrl.includes('?') ? '&' : '?';
        wsUrl += `${separator}token=${encodeURIComponent(session.access_token)}`;
      }
    }

    if (env.isDev) {
      console.debug('[WebSocket] Connecting to:', wsUrl.replace(/token=[^&]+/, 'token=***'));
    }

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (isUnmountedRef.current) {
          ws.close();
          return;
        }

        if (env.isDev) {
          console.debug('[WebSocket] Connected');
        }

        setState('connected');
        setReconnectAttempts(0);
        currentReconnectDelayRef.current = initialReconnectDelay;
        setLastError(null);

        startHeartbeat();
        onOpenRef.current?.();
      };

      ws.onmessage = (event) => {
        if (isUnmountedRef.current) return;

        try {
          const data = JSON.parse(event.data);

          // Ignore pong responses
          if (data.type === 'pong') {
            if (env.isDev) {
              console.debug('[WebSocket] Pong received');
            }
            return;
          }

          onMessageRef.current?.(data);
        } catch {
          // Not JSON, pass raw data
          onMessageRef.current?.(event.data);
        }
      };

      ws.onerror = (event) => {
        if (isUnmountedRef.current) return;

        if (env.isDev) {
          console.error('[WebSocket] Error:', event);
        }

        setLastError(event);
        onErrorRef.current?.(event);
      };

      ws.onclose = (event) => {
        if (isUnmountedRef.current) return;

        if (env.isDev) {
          console.debug('[WebSocket] Closed:', event.code, event.reason);
        }

        clearHeartbeat();
        wsRef.current = null;
        onCloseRef.current?.(event);

        // Determine if we should reconnect
        const shouldAutoReconnect =
          autoReconnect &&
          shouldReconnectRef.current &&
          (event.code !== 1000 || RECONNECTABLE_CLOSE_CODES.has(event.code));

        if (shouldAutoReconnect) {
          scheduleReconnect();
        } else {
          setState('disconnected');
        }
      };

      wsRef.current = ws;
    } catch (error) {
      if (env.isDev) {
        console.error('[WebSocket] Connection failed:', error);
      }
      setState('error');
      if (autoReconnect && shouldReconnectRef.current) {
        scheduleReconnect();
      }
    }
  }, [
    endpoint,
    withAuth,
    autoReconnect,
    initialReconnectDelay,
    startHeartbeat,
    clearHeartbeat,
    scheduleReconnect,
  ]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearReconnectTimeout();
    clearHeartbeat();

    if (wsRef.current) {
      if (env.isDev) {
        console.debug('[WebSocket] Disconnecting');
      }
      wsRef.current.close(1000, 'Client disconnected');
      wsRef.current = null;
    }

    setState('disconnected');
  }, [clearReconnectTimeout, clearHeartbeat]);

  /**
   * Send a message through the WebSocket
   */
  const send = useCallback((data: string | object): boolean => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      if (env.isDev) {
        console.warn('[WebSocket] Cannot send: not connected');
      }
      return false;
    }

    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(message);
      return true;
    } catch (error) {
      if (env.isDev) {
        console.error('[WebSocket] Send failed:', error);
      }
      return false;
    }
  }, []);

  /**
   * Reset reconnection state and connect
   */
  const reset = useCallback(() => {
    clearReconnectTimeout();
    clearHeartbeat();
    setReconnectAttempts(0);
    currentReconnectDelayRef.current = initialReconnectDelay;
    setLastError(null);
    shouldReconnectRef.current = true;

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    connect();
  }, [clearReconnectTimeout, clearHeartbeat, initialReconnectDelay, connect]);

  // Connect effect for reconnection state
  useEffect(() => {
    if (state === 'connecting' && !wsRef.current) {
      connect();
    }
  }, [state, connect]);

  // Initial connection on mount
  useEffect(() => {
    isUnmountedRef.current = false;

    if (connectOnMount) {
      connect();
    }

    return () => {
      isUnmountedRef.current = true;
      shouldReconnectRef.current = false;
      clearReconnectTimeout();
      clearHeartbeat();

      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [/* intentionally empty - only run on mount/unmount */]);

  return {
    state,
    isConnected: state === 'connected',
    reconnectAttempts,
    lastError,
    connect,
    disconnect,
    send,
    reset,
  };
}

// =============================================================================
// SPECIALIZED HOOKS
// =============================================================================

/**
 * Pre-configured hook for graph stream WebSocket
 *
 * @param onMessage - Callback when graph update is received
 * @param options - Additional options
 * @returns WebSocket state and controls
 *
 * @example
 * ```tsx
 * const { isConnected, state } = useGraphWebSocket((data) => {
 *   console.log('Graph update:', data);
 * });
 * ```
 */
export function useGraphWebSocket(
  onMessage: (data: unknown) => void,
  options?: Partial<Omit<UseWebSocketOptions, 'endpoint' | 'onMessage'>>
): UseWebSocketReturn {
  return useWebSocket({
    endpoint: '/graph/stream',
    onMessage,
    ...options,
  });
}

export default useWebSocket;
