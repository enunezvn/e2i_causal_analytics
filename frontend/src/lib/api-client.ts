/**
 * API Client Configuration
 * ========================
 *
 * Axios-based API client with interceptors for error handling,
 * request/response transformation, and authentication.
 *
 * Usage:
 *   import { apiClient } from '@/lib/api-client'
 *   const response = await apiClient.get('/graph/nodes')
 */

import axios, {
  AxiosError,
  AxiosInstance,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from 'axios';
import { env, buildApiUrl } from '@/config/env';

/**
 * Standard API error response structure
 */
export interface ApiErrorResponse {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp?: string;
}

/**
 * Custom API error class with typed error response
 */
export class ApiError extends Error {
  public readonly status: number;
  public readonly statusText: string;
  public readonly data: ApiErrorResponse | null;
  public readonly originalError: AxiosError;

  constructor(error: AxiosError<ApiErrorResponse>) {
    const message =
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';
    super(message);

    this.name = 'ApiError';
    this.status = error.response?.status ?? 0;
    this.statusText = error.response?.statusText ?? 'Unknown Error';
    this.data = error.response?.data ?? null;
    this.originalError = error;

    // Maintain proper stack trace for where our error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
  }

  /**
   * Check if error is a network error (no response received)
   */
  get isNetworkError(): boolean {
    return this.status === 0 && !this.originalError.response;
  }

  /**
   * Check if error is a client error (4xx)
   */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  /**
   * Check if error is a server error (5xx)
   */
  get isServerError(): boolean {
    return this.status >= 500;
  }

  /**
   * Check if error is a not found error (404)
   */
  get isNotFound(): boolean {
    return this.status === 404;
  }

  /**
   * Check if error is an unauthorized error (401)
   */
  get isUnauthorized(): boolean {
    return this.status === 401;
  }

  /**
   * Check if error is a forbidden error (403)
   */
  get isForbidden(): boolean {
    return this.status === 403;
  }
}

/**
 * Request interceptor for adding common headers and logging
 */
function requestInterceptor(
  config: InternalAxiosRequestConfig
): InternalAxiosRequestConfig {
  // Add common headers
  config.headers = config.headers || {};

  // Add request timestamp for latency tracking
  config.headers['X-Request-Time'] = new Date().toISOString();

  // Add correlation ID for request tracing
  const correlationId = crypto.randomUUID();
  config.headers['X-Correlation-ID'] = correlationId;

  // Log request in development
  if (env.isDev) {
    const method = config.method?.toUpperCase() ?? 'GET';
    const url = config.url ?? '';
    console.debug(`[API] ${method} ${url}`, {
      correlationId,
      params: config.params,
    });
  }

  return config;
}

/**
 * Response interceptor for logging and transformation
 */
function responseInterceptor(response: AxiosResponse): AxiosResponse {
  // Log response in development
  if (env.isDev) {
    const method = response.config.method?.toUpperCase() ?? 'GET';
    const url = response.config.url ?? '';
    const status = response.status;
    console.debug(`[API] ${method} ${url} -> ${status}`, {
      data: response.data,
    });
  }

  return response;
}

/**
 * Error interceptor for standardized error handling
 */
function errorInterceptor(error: AxiosError<ApiErrorResponse>): Promise<never> {
  // Log error in development
  if (env.isDev) {
    const method = error.config?.method?.toUpperCase() ?? 'GET';
    const url = error.config?.url ?? '';
    const status = error.response?.status ?? 'N/A';
    console.error(`[API] ${method} ${url} -> ${status}`, {
      error: error.message,
      data: error.response?.data,
    });
  }

  // Transform to ApiError
  const apiError = new ApiError(error);

  return Promise.reject(apiError);
}

/**
 * Create and configure the Axios instance
 */
function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: env.apiUrl,
    timeout: 30000, // 30 seconds
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });

  // Add interceptors
  client.interceptors.request.use(requestInterceptor, (error) =>
    Promise.reject(error)
  );
  client.interceptors.response.use(responseInterceptor, errorInterceptor);

  return client;
}

/**
 * Main API client instance
 * Pre-configured with base URL, interceptors, and error handling
 */
export const apiClient = createApiClient();

/**
 * Type-safe GET request helper
 */
export async function get<T>(
  endpoint: string,
  params?: Record<string, unknown>
): Promise<T> {
  const response = await apiClient.get<T>(endpoint, { params });
  return response.data;
}

/**
 * Type-safe POST request helper
 */
export async function post<T, D = unknown>(
  endpoint: string,
  data?: D,
  config?: { params?: Record<string, unknown> }
): Promise<T> {
  const response = await apiClient.post<T>(endpoint, data, config);
  return response.data;
}

/**
 * Type-safe PUT request helper
 */
export async function put<T, D = unknown>(
  endpoint: string,
  data?: D
): Promise<T> {
  const response = await apiClient.put<T>(endpoint, data);
  return response.data;
}

/**
 * Type-safe PATCH request helper
 */
export async function patch<T, D = unknown>(
  endpoint: string,
  data?: D
): Promise<T> {
  const response = await apiClient.patch<T>(endpoint, data);
  return response.data;
}

/**
 * Type-safe DELETE request helper
 */
export async function del<T>(endpoint: string): Promise<T> {
  const response = await apiClient.delete<T>(endpoint);
  return response.data;
}

/**
 * Health check helper
 * Returns true if API is healthy, false otherwise
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get('/health');
    return response.status === 200;
  } catch {
    return false;
  }
}

/**
 * Create a WebSocket connection to the graph stream endpoint
 * @param onMessage - Callback for incoming messages
 * @param onError - Callback for errors
 * @param onClose - Callback for connection close
 * @returns WebSocket instance
 */
export function createGraphWebSocket(
  onMessage: (data: unknown) => void,
  onError?: (error: Event) => void,
  onClose?: (event: CloseEvent) => void
): WebSocket {
  // Convert HTTP URL to WebSocket URL
  const wsUrl = buildApiUrl('/graph/stream').replace(/^http/, 'ws');
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      onMessage(event.data);
    }
  };

  ws.onerror = (event) => {
    if (env.isDev) {
      console.error('[WebSocket] Error:', event);
    }
    onError?.(event);
  };

  ws.onclose = (event) => {
    if (env.isDev) {
      console.debug('[WebSocket] Closed:', event.code, event.reason);
    }
    onClose?.(event);
  };

  return ws;
}

export default apiClient;
