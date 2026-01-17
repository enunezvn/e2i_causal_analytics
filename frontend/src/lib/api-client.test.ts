/**
 * API Client Tests
 * ================
 *
 * Tests for the Axios-based API client with interceptors,
 * error handling, and request helpers.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios, { AxiosError, InternalAxiosRequestConfig, AxiosHeaders } from 'axios';

// Import the actual type from api-client
import type { ApiErrorResponse } from './api-client';

// Mock axios before importing the module
vi.mock('axios', async () => {
  const actual = await vi.importActual<typeof axios>('axios');
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: {
        use: vi.fn(),
      },
      response: {
        use: vi.fn(),
      },
    },
  };
  return {
    ...actual,
    default: {
      ...(actual as any).default,
      create: vi.fn(() => mockAxiosInstance),
    },
    create: vi.fn(() => mockAxiosInstance),
  };
});

// Mock env config
vi.mock('@/config/env', () => ({
  env: {
    apiUrl: 'http://localhost:8000',
    isDev: false,
  },
  buildApiUrl: (path: string) => `http://localhost:8000${path}`,
}));

// Import after mocks
import {
  ApiError,
  apiClient,
  get,
  post,
  put,
  patch,
  del,
  checkApiHealth,
  createGraphWebSocket,
} from './api-client';

// =============================================================================
// API ERROR TESTS
// =============================================================================

describe('ApiError', () => {
  function createMockAxiosError(
    status: number,
    statusText: string,
    data?: Partial<ApiErrorResponse> | null,
    hasResponse = true
  ): AxiosError<ApiErrorResponse> {
    const config: InternalAxiosRequestConfig = {
      headers: new AxiosHeaders(),
    };
    return {
      name: 'AxiosError',
      message: 'Request failed',
      config,
      isAxiosError: true,
      toJSON: () => ({}),
      response: hasResponse
        ? {
            status,
            statusText,
            data: data as ApiErrorResponse,
            headers: {},
            config,
          }
        : undefined,
    } as AxiosError<ApiErrorResponse>;
  }

  it('creates error with message from response data', () => {
    const error = createMockAxiosError(400, 'Bad Request', {
      message: 'Invalid input',
    });
    const apiError = new ApiError(error);

    expect(apiError.message).toBe('Invalid input');
    expect(apiError.status).toBe(400);
    expect(apiError.statusText).toBe('Bad Request');
  });

  it('uses axios message when response data has no message', () => {
    const error = createMockAxiosError(500, 'Internal Server Error');
    error.message = 'Network Error';
    const apiError = new ApiError(error);

    expect(apiError.message).toBe('Network Error');
  });

  it('uses default message when no message available', () => {
    const error = createMockAxiosError(0, '', undefined, false);
    error.message = '';
    const apiError = new ApiError(error);

    expect(apiError.message).toBe('An unexpected error occurred');
  });

  it('sets name to ApiError', () => {
    const error = createMockAxiosError(400, 'Bad Request');
    const apiError = new ApiError(error);

    expect(apiError.name).toBe('ApiError');
  });

  it('stores original error', () => {
    const error = createMockAxiosError(400, 'Bad Request');
    const apiError = new ApiError(error);

    expect(apiError.originalError).toBe(error);
  });

  it('stores error data', () => {
    const errorData = { message: 'Test', error: 'validation_error' };
    const error = createMockAxiosError(400, 'Bad Request', errorData);
    const apiError = new ApiError(error);

    expect(apiError.data).toEqual(errorData);
  });

  it('handles missing response data', () => {
    const error = createMockAxiosError(0, '', undefined, false);
    const apiError = new ApiError(error);

    expect(apiError.status).toBe(0);
    expect(apiError.statusText).toBe('Unknown Error');
    expect(apiError.data).toBeNull();
  });

  describe('isNetworkError', () => {
    it('returns true for network error (no response)', () => {
      const error = createMockAxiosError(0, '', undefined, false);
      const apiError = new ApiError(error);

      expect(apiError.isNetworkError).toBe(true);
    });

    it('returns false when response exists', () => {
      const error = createMockAxiosError(500, 'Server Error');
      const apiError = new ApiError(error);

      expect(apiError.isNetworkError).toBe(false);
    });
  });

  describe('isClientError', () => {
    it('returns true for 4xx status', () => {
      const error = createMockAxiosError(400, 'Bad Request');
      const apiError = new ApiError(error);

      expect(apiError.isClientError).toBe(true);
    });

    it('returns true for 404', () => {
      const error = createMockAxiosError(404, 'Not Found');
      const apiError = new ApiError(error);

      expect(apiError.isClientError).toBe(true);
    });

    it('returns false for 5xx status', () => {
      const error = createMockAxiosError(500, 'Server Error');
      const apiError = new ApiError(error);

      expect(apiError.isClientError).toBe(false);
    });
  });

  describe('isServerError', () => {
    it('returns true for 5xx status', () => {
      const error = createMockAxiosError(500, 'Server Error');
      const apiError = new ApiError(error);

      expect(apiError.isServerError).toBe(true);
    });

    it('returns true for 503', () => {
      const error = createMockAxiosError(503, 'Service Unavailable');
      const apiError = new ApiError(error);

      expect(apiError.isServerError).toBe(true);
    });

    it('returns false for 4xx status', () => {
      const error = createMockAxiosError(400, 'Bad Request');
      const apiError = new ApiError(error);

      expect(apiError.isServerError).toBe(false);
    });
  });

  describe('isNotFound', () => {
    it('returns true for 404 status', () => {
      const error = createMockAxiosError(404, 'Not Found');
      const apiError = new ApiError(error);

      expect(apiError.isNotFound).toBe(true);
    });

    it('returns false for other statuses', () => {
      const error = createMockAxiosError(400, 'Bad Request');
      const apiError = new ApiError(error);

      expect(apiError.isNotFound).toBe(false);
    });
  });

  describe('isUnauthorized', () => {
    it('returns true for 401 status', () => {
      const error = createMockAxiosError(401, 'Unauthorized');
      const apiError = new ApiError(error);

      expect(apiError.isUnauthorized).toBe(true);
    });

    it('returns false for other statuses', () => {
      const error = createMockAxiosError(403, 'Forbidden');
      const apiError = new ApiError(error);

      expect(apiError.isUnauthorized).toBe(false);
    });
  });

  describe('isForbidden', () => {
    it('returns true for 403 status', () => {
      const error = createMockAxiosError(403, 'Forbidden');
      const apiError = new ApiError(error);

      expect(apiError.isForbidden).toBe(true);
    });

    it('returns false for other statuses', () => {
      const error = createMockAxiosError(401, 'Unauthorized');
      const apiError = new ApiError(error);

      expect(apiError.isForbidden).toBe(false);
    });
  });
});

// =============================================================================
// API CLIENT CREATION TESTS
// =============================================================================

describe('apiClient', () => {
  it('is defined', () => {
    expect(apiClient).toBeDefined();
  });

  it('has get method', () => {
    expect(apiClient.get).toBeDefined();
  });

  it('has post method', () => {
    expect(apiClient.post).toBeDefined();
  });

  it('has put method', () => {
    expect(apiClient.put).toBeDefined();
  });

  it('has patch method', () => {
    expect(apiClient.patch).toBeDefined();
  });

  it('has delete method', () => {
    expect(apiClient.delete).toBeDefined();
  });
});

// =============================================================================
// REQUEST HELPER TESTS
// =============================================================================

describe('Request Helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('get', () => {
    it('makes GET request and returns data', async () => {
      const mockData = { users: [{ id: 1, name: 'Test' }] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockData,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      const result = await get<typeof mockData>('/users');

      expect(apiClient.get).toHaveBeenCalledWith('/users', { params: undefined });
      expect(result).toEqual(mockData);
    });

    it('passes params to request', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: [],
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      await get('/users', { page: 1, limit: 10 });

      expect(apiClient.get).toHaveBeenCalledWith('/users', {
        params: { page: 1, limit: 10 },
      });
    });
  });

  describe('post', () => {
    it('makes POST request and returns data', async () => {
      const requestData = { name: 'New User' };
      const responseData = { id: 1, name: 'New User' };
      vi.mocked(apiClient.post).mockResolvedValueOnce({
        data: responseData,
        status: 201,
        statusText: 'Created',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      const result = await post<typeof responseData>('/users', requestData);

      expect(apiClient.post).toHaveBeenCalledWith('/users', requestData, undefined);
      expect(result).toEqual(responseData);
    });

    it('handles undefined data', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({
        data: { success: true },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      await post('/trigger');

      expect(apiClient.post).toHaveBeenCalledWith('/trigger', undefined, undefined);
    });
  });

  describe('put', () => {
    it('makes PUT request and returns data', async () => {
      const requestData = { name: 'Updated User' };
      const responseData = { id: 1, name: 'Updated User' };
      vi.mocked(apiClient.put).mockResolvedValueOnce({
        data: responseData,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      const result = await put<typeof responseData>('/users/1', requestData);

      expect(apiClient.put).toHaveBeenCalledWith('/users/1', requestData);
      expect(result).toEqual(responseData);
    });
  });

  describe('patch', () => {
    it('makes PATCH request and returns data', async () => {
      const requestData = { name: 'Patched User' };
      const responseData = { id: 1, name: 'Patched User' };
      vi.mocked(apiClient.patch).mockResolvedValueOnce({
        data: responseData,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      const result = await patch<typeof responseData>('/users/1', requestData);

      expect(apiClient.patch).toHaveBeenCalledWith('/users/1', requestData);
      expect(result).toEqual(responseData);
    });
  });

  describe('del', () => {
    it('makes DELETE request and returns data', async () => {
      const responseData = { deleted: true };
      vi.mocked(apiClient.delete).mockResolvedValueOnce({
        data: responseData,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      });

      const result = await del<typeof responseData>('/users/1');

      expect(apiClient.delete).toHaveBeenCalledWith('/users/1');
      expect(result).toEqual(responseData);
    });
  });
});

// =============================================================================
// HEALTH CHECK TESTS
// =============================================================================

describe('checkApiHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns true when API responds with 200', async () => {
    vi.mocked(apiClient.get).mockResolvedValueOnce({
      status: 200,
      data: { status: 'healthy' },
      statusText: 'OK',
      headers: {},
      config: {} as InternalAxiosRequestConfig,
    });

    const result = await checkApiHealth();

    expect(result).toBe(true);
    expect(apiClient.get).toHaveBeenCalledWith('/health');
  });

  it('returns false when API request fails', async () => {
    vi.mocked(apiClient.get).mockRejectedValueOnce(new Error('Network Error'));

    const result = await checkApiHealth();

    expect(result).toBe(false);
  });
});

// =============================================================================
// WEBSOCKET TESTS
// =============================================================================

describe('createGraphWebSocket', () => {
  let mockInstance: {
    onmessage: ((event: MessageEvent) => void) | null;
    onerror: ((event: Event) => void) | null;
    onclose: ((event: CloseEvent) => void) | null;
    close: ReturnType<typeof vi.fn>;
    send: ReturnType<typeof vi.fn>;
  };

  // Mock WebSocket as a class
  class MockWebSocket {
    static lastUrl: string | null = null;
    static instances: MockWebSocket[] = [];

    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    close = vi.fn();
    send = vi.fn();

    constructor(url: string) {
      MockWebSocket.lastUrl = url;
      MockWebSocket.instances.push(this);
      mockInstance = this;
    }
  }

  beforeEach(() => {
    MockWebSocket.lastUrl = null;
    MockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('creates WebSocket with correct URL', () => {
    const onMessage = vi.fn();
    createGraphWebSocket(onMessage);

    expect(MockWebSocket.lastUrl).toBe('ws://localhost:8000/graph/stream');
  });

  it('parses JSON messages', () => {
    const onMessage = vi.fn();
    createGraphWebSocket(onMessage);

    const testData = { type: 'update', nodes: [] };
    mockInstance.onmessage?.({
      data: JSON.stringify(testData),
    } as MessageEvent);

    expect(onMessage).toHaveBeenCalledWith(testData);
  });

  it('passes non-JSON data directly', () => {
    const onMessage = vi.fn();
    createGraphWebSocket(onMessage);

    mockInstance.onmessage?.({
      data: 'plain text message',
    } as MessageEvent);

    expect(onMessage).toHaveBeenCalledWith('plain text message');
  });

  it('calls onError when error occurs', () => {
    const onMessage = vi.fn();
    const onError = vi.fn();
    createGraphWebSocket(onMessage, onError);

    const errorEvent = new Event('error');
    mockInstance.onerror?.(errorEvent);

    expect(onError).toHaveBeenCalledWith(errorEvent);
  });

  it('calls onClose when connection closes', () => {
    const onMessage = vi.fn();
    const onError = vi.fn();
    const onClose = vi.fn();
    createGraphWebSocket(onMessage, onError, onClose);

    const closeEvent = { code: 1000, reason: 'Normal closure' } as CloseEvent;
    mockInstance.onclose?.(closeEvent);

    expect(onClose).toHaveBeenCalledWith(closeEvent);
  });

  it('handles error without callback', () => {
    const onMessage = vi.fn();
    createGraphWebSocket(onMessage);

    // Should not throw when no error callback
    const errorEvent = new Event('error');
    expect(() => mockInstance.onerror?.(errorEvent)).not.toThrow();
  });

  it('handles close without callback', () => {
    const onMessage = vi.fn();
    createGraphWebSocket(onMessage);

    // Should not throw when no close callback
    const closeEvent = { code: 1000, reason: 'Normal closure' } as CloseEvent;
    expect(() => mockInstance.onclose?.(closeEvent)).not.toThrow();
  });

  it('returns WebSocket instance', () => {
    const onMessage = vi.fn();
    const ws = createGraphWebSocket(onMessage);

    expect(ws).toBeDefined();
    expect(ws.close).toBeDefined();
    expect(ws.send).toBeDefined();
  });
});
