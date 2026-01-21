/**
 * useQueryError Hook Tests
 */

import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  useQueryError,
  useQueryErrorToast,
  useMutationError,
} from './use-query-error';
import { toast } from '@/hooks/use-toast';
import { ApiError } from '@/lib/api-client';
import type { AxiosError } from 'axios';

// Mock the toast function
vi.mock('@/hooks/use-toast', () => ({
  toast: vi.fn(),
}));

const mockToast = toast as unknown as ReturnType<typeof vi.fn>;

// Mock ApiError factory
function createApiError(
  status: number,
  message: string,
  statusText = 'Error'
): ApiError {
  const axiosError = {
    response: {
      status,
      statusText,
      data: { message, error: statusText },
    },
    message,
    config: {},
    isAxiosError: true,
    toJSON: () => ({}),
    name: 'AxiosError',
  } as unknown as AxiosError<{ message: string; error: string }>;

  return new ApiError(axiosError);
}

// Create network error (no response)
function createNetworkError(): ApiError {
  const axiosError = {
    response: undefined,
    message: 'Network Error',
    config: {},
    isAxiosError: true,
    toJSON: () => ({}),
    name: 'AxiosError',
  } as unknown as AxiosError<{ message: string; error: string }>;

  return new ApiError(axiosError);
}

describe('useQueryError', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('showErrorToast', () => {
    it('shows toast for generic error', () => {
      const { result } = renderHook(() => useQueryError());
      const error = new Error('Something went wrong');

      act(() => {
        result.current.showErrorToast(error);
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: 'Error',
        description: 'Something went wrong',
      });
    });

    it('shows toast for network error', () => {
      const { result } = renderHook(() => useQueryError());
      const error = createNetworkError();

      act(() => {
        result.current.showErrorToast(error);
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: 'Connection Error',
        description: 'Unable to connect to the server. Please check your connection.',
      });
    });

    it('shows toast for server error', () => {
      const { result } = renderHook(() => useQueryError());
      const error = createApiError(500, 'Internal Server Error');

      act(() => {
        result.current.showErrorToast(error);
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: 'Server Error',
        description: 'Server error. Please try again later.',
      });
    });

    it('includes context in title when provided', () => {
      const { result } = renderHook(() => useQueryError());
      const error = createNetworkError();

      act(() => {
        result.current.showErrorToast(error, { context: 'loading KPIs' });
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: 'Connection Error: loading KPIs',
        description: expect.any(String),
      });
    });

    it('uses custom title when provided', () => {
      const { result } = renderHook(() => useQueryError());
      const error = new Error('Test');

      act(() => {
        result.current.showErrorToast(error, { title: 'Custom Title' });
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: 'Custom Title',
        description: expect.any(String),
      });
    });

    it('uses custom message when provided', () => {
      const { result } = renderHook(() => useQueryError());
      const error = new Error('Test');

      act(() => {
        result.current.showErrorToast(error, { message: 'Custom message' });
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'destructive',
        title: expect.any(String),
        description: 'Custom message',
      });
    });

    it('skips toast for 401 errors', () => {
      const { result } = renderHook(() => useQueryError());
      const error = createApiError(401, 'Unauthorized');

      act(() => {
        result.current.showErrorToast(error);
      });

      expect(mockToast).not.toHaveBeenCalled();
    });

    it('skips toast for specified status codes', () => {
      const { result } = renderHook(() => useQueryError());
      const error = createApiError(404, 'Not Found');

      act(() => {
        result.current.showErrorToast(error, { skipForStatus: [404] });
      });

      expect(mockToast).not.toHaveBeenCalled();
    });
  });

  describe('showSuccessToast', () => {
    it('shows success toast', () => {
      const { result } = renderHook(() => useQueryError());

      act(() => {
        result.current.showSuccessToast('Success!', 'Operation completed');
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'success',
        title: 'Success!',
        description: 'Operation completed',
      });
    });

    it('shows success toast without description', () => {
      const { result } = renderHook(() => useQueryError());

      act(() => {
        result.current.showSuccessToast('Done!');
      });

      expect(mockToast).toHaveBeenCalledWith({
        variant: 'success',
        title: 'Done!',
        description: undefined,
      });
    });
  });
});

describe('useQueryErrorToast', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows toast when error changes from null to error', () => {
    const { rerender } = renderHook(
      ({ error }) => useQueryErrorToast(error),
      { initialProps: { error: null as Error | null } }
    );

    expect(mockToast).not.toHaveBeenCalled();

    const error = new Error('Something went wrong');
    rerender({ error });

    expect(mockToast).toHaveBeenCalledWith({
      variant: 'destructive',
      title: 'Error',
      description: 'Something went wrong',
    });
  });

  it('does not show toast when error remains the same', () => {
    const error = new Error('Test error');
    const { rerender } = renderHook(
      ({ error }) => useQueryErrorToast(error),
      { initialProps: { error } }
    );

    expect(mockToast).toHaveBeenCalledTimes(1);

    // Rerender with same error
    rerender({ error });

    expect(mockToast).toHaveBeenCalledTimes(1);
  });

  it('shows toast for each new error', () => {
    const error1 = new Error('Error 1');
    const error2 = new Error('Error 2');

    const { rerender } = renderHook(
      ({ error }) => useQueryErrorToast(error),
      { initialProps: { error: error1 } }
    );

    expect(mockToast).toHaveBeenCalledTimes(1);

    rerender({ error: error2 });

    expect(mockToast).toHaveBeenCalledTimes(2);
  });

  it('calls onError callback when provided', () => {
    const onError = vi.fn();
    const error = new Error('Test');

    renderHook(() => useQueryErrorToast(error, { onError }));

    expect(onError).toHaveBeenCalledWith(error);
  });

  it('does not show toast when showToast is false', () => {
    const error = new Error('Test');

    renderHook(() => useQueryErrorToast(error, { showToast: false }));

    expect(mockToast).not.toHaveBeenCalled();
  });

  it('skips toast for 401 errors', () => {
    const error = createApiError(401, 'Unauthorized');

    renderHook(() => useQueryErrorToast(error));

    expect(mockToast).not.toHaveBeenCalled();
  });

  it('includes context in title', () => {
    const error = createNetworkError();

    renderHook(() => useQueryErrorToast(error, { context: 'fetching data' }));

    expect(mockToast).toHaveBeenCalledWith({
      variant: 'destructive',
      title: 'Connection Error: fetching data',
      description: expect.any(String),
    });
  });
});

describe('useMutationError', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns a function that shows toast on error', () => {
    const { result } = renderHook(() => useMutationError());
    const error = new Error('Mutation failed');

    act(() => {
      result.current(error);
    });

    expect(mockToast).toHaveBeenCalledWith({
      variant: 'destructive',
      title: 'Error',
      description: 'Mutation failed',
    });
  });

  it('includes context in error message', () => {
    const { result } = renderHook(() =>
      useMutationError({ context: 'saving experiment' })
    );
    const error = createApiError(500, 'Server Error');

    act(() => {
      result.current(error);
    });

    expect(mockToast).toHaveBeenCalledWith({
      variant: 'destructive',
      title: 'Server Error: saving experiment',
      description: expect.any(String),
    });
  });

  it('calls custom onError callback', () => {
    const onError = vi.fn();
    const { result } = renderHook(() => useMutationError({ onError }));
    const error = new Error('Test');

    act(() => {
      result.current(error);
    });

    expect(onError).toHaveBeenCalledWith(error);
    expect(mockToast).toHaveBeenCalled();
  });

  it('does not show toast when showToast is false', () => {
    const onError = vi.fn();
    const { result } = renderHook(() =>
      useMutationError({ showToast: false, onError })
    );
    const error = new Error('Test');

    act(() => {
      result.current(error);
    });

    expect(onError).toHaveBeenCalledWith(error);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it('skips toast for 401 errors', () => {
    const { result } = renderHook(() => useMutationError());
    const error = createApiError(401, 'Unauthorized');

    act(() => {
      result.current(error);
    });

    expect(mockToast).not.toHaveBeenCalled();
  });

  it('skips toast for specified status codes', () => {
    const { result } = renderHook(() =>
      useMutationError({ skipForStatus: [409] })
    );
    const error = createApiError(409, 'Conflict');

    act(() => {
      result.current(error);
    });

    expect(mockToast).not.toHaveBeenCalled();
  });
});
