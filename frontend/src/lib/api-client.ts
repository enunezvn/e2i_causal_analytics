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
import type { ZodTypeAny } from 'zod';
import { env, buildApiUrl } from '@/config/env';
import { useAuthStore } from '@/stores/auth-store';
import { validateApiResponse, ApiValidationError } from './api-schemas';

/**
 * Generate a UUID v4 with fallback for non-secure contexts (HTTP)
 * crypto.randomUUID() is only available in secure contexts (HTTPS/localhost)
 */
function generateUUID(): string {
  // Use native crypto.randomUUID if available (secure contexts only)
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  // Fallback: generate UUID v4 using crypto.getRandomValues or Math.random
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    // Set version (4) and variant (RFC4122)
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  // Last resort fallback using Math.random (less secure but functional)
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

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
 * Request interceptor for adding common headers, auth token, and logging
 */
function requestInterceptor(
  config: InternalAxiosRequestConfig
): InternalAxiosRequestConfig {
  // Add common headers
  config.headers = config.headers || {};

  // Add auth token if available
  const session = useAuthStore.getState().session;
  if (session?.access_token) {
    config.headers['Authorization'] = `Bearer ${session.access_token}`;
  }

  // Add request timestamp for latency tracking
  config.headers['X-Request-Time'] = new Date().toISOString();

  // Add correlation ID for request tracing
  const correlationId = generateUUID();
  config.headers['X-Correlation-ID'] = correlationId;

  // Log request in development
  if (env.isDev) {
    const method = config.method?.toUpperCase() ?? 'GET';
    const url = config.url ?? '';
    console.debug(`[API] ${method} ${url}`, {
      correlationId,
      params: config.params,
      hasAuth: !!session?.access_token,
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

  // Handle 401 Unauthorized - clear auth state (token expired or invalid)
  if (error.response?.status === 401) {
    const { clearAuth, setError } = useAuthStore.getState();
    clearAuth();
    setError({
      code: 'session_expired',
      message: 'Your session has expired. Please sign in again.',
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
export async function del<T>(
  endpoint: string,
  config?: { data?: unknown }
): Promise<T> {
  const response = await apiClient.delete<T>(endpoint, config);
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

// =============================================================================
// VALIDATED REQUEST HELPERS (Phase 3 - Type Safety)
// =============================================================================

/**
 * Configuration for validated requests
 */
export interface ValidatedRequestConfig {
  /** Log validation errors to console (default: true in dev) */
  logErrors?: boolean;
  /** Throw error on validation failure (default: true) */
  throwOnError?: boolean;
}

/**
 * Type-safe GET request with runtime Zod validation
 *
 * @param schema - Zod schema to validate response against
 * @param endpoint - API endpoint
 * @param params - Query parameters
 * @param config - Validation config
 * @returns Validated response data
 *
 * @example
 * ```typescript
 * import { KPIListResponseSchema } from '@/lib/api-schemas';
 *
 * const kpis = await getValidated(
 *   KPIListResponseSchema,
 *   '/kpis',
 *   { workstream: 'ws1_data_quality' }
 * );
 * ```
 */
export async function getValidated<T extends ZodTypeAny>(
  schema: T,
  endpoint: string,
  params?: Record<string, unknown>,
  config?: ValidatedRequestConfig
): Promise<T['_output']> {
  const response = await apiClient.get(endpoint, { params });
  return validateApiResponse(schema, response.data, endpoint, {
    logErrors: config?.logErrors ?? env.isDev,
    throwOnError: config?.throwOnError ?? true,
  });
}

/**
 * Type-safe POST request with runtime Zod validation
 *
 * @param schema - Zod schema to validate response against
 * @param endpoint - API endpoint
 * @param data - Request body
 * @param config - Additional axios config and validation config
 * @returns Validated response data
 *
 * @example
 * ```typescript
 * import { KPIResultSchema } from '@/lib/api-schemas';
 *
 * const result = await postValidated(
 *   KPIResultSchema,
 *   '/kpis/calculate',
 *   { kpi_id: 'WS1-DQ-001', force_refresh: true }
 * );
 * ```
 */
export async function postValidated<T extends ZodTypeAny, D = unknown>(
  schema: T,
  endpoint: string,
  data?: D,
  config?: { params?: Record<string, unknown> } & ValidatedRequestConfig
): Promise<T['_output']> {
  const response = await apiClient.post(endpoint, data, { params: config?.params });
  return validateApiResponse(schema, response.data, endpoint, {
    logErrors: config?.logErrors ?? env.isDev,
    throwOnError: config?.throwOnError ?? true,
  });
}

/**
 * Type-safe PUT request with runtime Zod validation
 */
export async function putValidated<T extends ZodTypeAny, D = unknown>(
  schema: T,
  endpoint: string,
  data?: D,
  config?: ValidatedRequestConfig
): Promise<T['_output']> {
  const response = await apiClient.put(endpoint, data);
  return validateApiResponse(schema, response.data, endpoint, {
    logErrors: config?.logErrors ?? env.isDev,
    throwOnError: config?.throwOnError ?? true,
  });
}

/**
 * Type-safe PATCH request with runtime Zod validation
 */
export async function patchValidated<T extends ZodTypeAny, D = unknown>(
  schema: T,
  endpoint: string,
  data?: D,
  config?: ValidatedRequestConfig
): Promise<T['_output']> {
  const response = await apiClient.patch(endpoint, data);
  return validateApiResponse(schema, response.data, endpoint, {
    logErrors: config?.logErrors ?? env.isDev,
    throwOnError: config?.throwOnError ?? true,
  });
}

/**
 * Type-safe DELETE request with runtime Zod validation
 */
export async function delValidated<T extends ZodTypeAny>(
  schema: T,
  endpoint: string,
  config?: { data?: unknown } & ValidatedRequestConfig
): Promise<T['_output']> {
  const response = await apiClient.delete(endpoint, { data: config?.data });
  return validateApiResponse(schema, response.data, endpoint, {
    logErrors: config?.logErrors ?? env.isDev,
    throwOnError: config?.throwOnError ?? true,
  });
}

// Re-export validation utilities for convenience
export { ApiValidationError } from './api-schemas';

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
