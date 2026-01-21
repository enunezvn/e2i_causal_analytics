/**
 * QueryErrorState Component Tests
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryErrorState, InlineQueryError } from './query-error-state';
import { ApiError } from '@/lib/api-client';
import type { AxiosError } from 'axios';

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

describe('QueryErrorState', () => {
  describe('rendering', () => {
    it('renders nothing when error is null', () => {
      const { container } = render(<QueryErrorState error={null} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders nothing when error is undefined', () => {
      const { container } = render(<QueryErrorState error={undefined} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders error alert for generic Error', () => {
      const error = new Error('A specific error message');
      render(<QueryErrorState error={error} />);

      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('Something went wrong')).toBeInTheDocument(); // Title
      expect(screen.getByText('A specific error message')).toBeInTheDocument(); // Description
    });

    it('renders network error correctly', () => {
      const error = createNetworkError();
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Connection Error')).toBeInTheDocument();
      expect(
        screen.getByText(/Unable to connect to the server/i)
      ).toBeInTheDocument();
    });

    it('renders unauthorized error correctly', () => {
      const error = createApiError(401, 'Unauthorized');
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Session Expired')).toBeInTheDocument();
      expect(
        screen.getByText(/Your session has expired/i)
      ).toBeInTheDocument();
    });

    it('renders forbidden error correctly', () => {
      const error = createApiError(403, 'Forbidden');
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Access Denied')).toBeInTheDocument();
      expect(
        screen.getByText(/You do not have permission/i)
      ).toBeInTheDocument();
    });

    it('renders not found error correctly', () => {
      const error = createApiError(404, 'Not Found');
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Not Found')).toBeInTheDocument();
      expect(
        screen.getByText(/resource could not be found/i)
      ).toBeInTheDocument();
    });

    it('renders server error correctly', () => {
      const error = createApiError(500, 'Internal Server Error');
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Server Error')).toBeInTheDocument();
      expect(
        screen.getByText(/The server encountered an error/i)
      ).toBeInTheDocument();
    });

    it('uses API error message when available', () => {
      const error = createApiError(400, 'Invalid KPI ID format');
      render(<QueryErrorState error={error} />);

      expect(screen.getByText('Invalid KPI ID format')).toBeInTheDocument();
    });
  });

  describe('custom content', () => {
    it('uses custom title when provided', () => {
      const error = new Error('Test error');
      render(<QueryErrorState error={error} title="Custom Title" />);

      expect(screen.getByText('Custom Title')).toBeInTheDocument();
    });

    it('uses custom description when provided', () => {
      const error = new Error('Test error');
      render(<QueryErrorState error={error} description="Custom description" />);

      expect(screen.getByText('Custom description')).toBeInTheDocument();
    });
  });

  describe('retry functionality', () => {
    it('shows retry button when onRetry is provided and error is retryable', () => {
      const error = createNetworkError();
      const onRetry = vi.fn();
      render(<QueryErrorState error={error} onRetry={onRetry} />);

      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    });

    it('calls onRetry when retry button is clicked', () => {
      const error = createNetworkError();
      const onRetry = vi.fn();
      render(<QueryErrorState error={error} onRetry={onRetry} />);

      fireEvent.click(screen.getByRole('button', { name: /try again/i }));
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it('shows retrying state when isRetrying is true', () => {
      const error = createNetworkError();
      const onRetry = vi.fn();
      render(<QueryErrorState error={error} onRetry={onRetry} isRetrying />);

      expect(screen.getByRole('button', { name: /retrying/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /retrying/i })).toBeDisabled();
    });

    it('shows retry for server errors', () => {
      const error = createApiError(500, 'Server Error');
      const onRetry = vi.fn();
      render(<QueryErrorState error={error} onRetry={onRetry} />);

      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    });
  });

  describe('size variants', () => {
    it('renders sm size correctly', () => {
      const error = new Error('Test error');
      const { container } = render(<QueryErrorState error={error} size="sm" />);

      expect(container.querySelector('.p-3')).toBeInTheDocument();
    });

    it('renders default size correctly', () => {
      const error = new Error('Test error');
      const { container } = render(<QueryErrorState error={error} size="default" />);

      expect(container.querySelector('.p-4')).toBeInTheDocument();
    });

    it('renders lg size correctly', () => {
      const error = new Error('Test error');
      const { container } = render(<QueryErrorState error={error} size="lg" />);

      expect(container.querySelector('.p-6')).toBeInTheDocument();
    });
  });

  describe('compact mode', () => {
    it('hides description in compact mode', () => {
      const error = new Error('Test error');
      render(<QueryErrorState error={error} compact />);

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
      expect(screen.queryByText('Test error')).not.toBeInTheDocument();
    });
  });
});

describe('InlineQueryError', () => {
  it('renders nothing when error is null', () => {
    const { container } = render(<InlineQueryError error={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders inline error message', () => {
    const error = new Error('Something failed');
    render(<InlineQueryError error={error} />);

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('renders network error inline', () => {
    const error = createNetworkError();
    render(<InlineQueryError error={error} />);

    expect(screen.getByText('Connection Error')).toBeInTheDocument();
  });

  it('shows retry button when onRetry provided', () => {
    const error = createNetworkError();
    const onRetry = vi.fn();
    render(<InlineQueryError error={error} onRetry={onRetry} />);

    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('calls onRetry when clicked', () => {
    const error = createNetworkError();
    const onRetry = vi.fn();
    render(<InlineQueryError error={error} onRetry={onRetry} />);

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('shows retrying state', () => {
    const error = createNetworkError();
    const onRetry = vi.fn();
    render(<InlineQueryError error={error} onRetry={onRetry} isRetrying />);

    expect(screen.getByRole('button', { name: /retrying/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retrying/i })).toBeDisabled();
  });
});
