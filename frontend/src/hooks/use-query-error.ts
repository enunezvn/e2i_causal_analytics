/**
 * Query Error Hook
 * ================
 *
 * React hook for automatic error toast notifications on query/mutation errors.
 * Provides consistent error handling across the application.
 *
 * Features:
 * - Automatic toast on error
 * - Deduplication of error toasts
 * - Contextual error messages
 * - Integration with ApiError class
 *
 * @module hooks/use-query-error
 */

import { useEffect, useRef } from 'react';
import { toast } from '@/hooks/use-toast';
import { ApiError } from '@/lib/api-client';

// =============================================================================
// TYPES
// =============================================================================

export interface UseQueryErrorOptions {
  /** Whether to show toast on error (default: true) */
  showToast?: boolean;

  /** Custom error title */
  title?: string;

  /** Custom error message */
  message?: string;

  /** Context for the error (e.g., "loading KPIs", "saving experiment") */
  context?: string;

  /** Skip toast for specific error types */
  skipForStatus?: number[];

  /** Custom onError callback */
  onError?: (error: Error | ApiError) => void;
}

export interface UseQueryErrorReturn {
  /** Show an error toast manually */
  showErrorToast: (error: Error | ApiError, options?: UseQueryErrorOptions) => void;

  /** Show a success toast */
  showSuccessToast: (title: string, description?: string) => void;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Get user-friendly error title based on error type
 */
function getErrorTitle(error: Error | ApiError, context?: string): string {
  const baseTitle = (() => {
    if (error instanceof ApiError) {
      if (error.isNetworkError) return 'Connection Error';
      if (error.isUnauthorized) return 'Session Expired';
      if (error.isForbidden) return 'Access Denied';
      if (error.isNotFound) return 'Not Found';
      if (error.isServerError) return 'Server Error';
      if (error.isClientError) return 'Request Failed';
    }
    return 'Error';
  })();

  return context ? `${baseTitle}: ${context}` : baseTitle;
}

/**
 * Get user-friendly error description based on error type
 */
function getErrorDescription(error: Error | ApiError): string {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return 'Unable to connect to the server. Please check your connection.';
    }
    if (error.isUnauthorized) {
      return 'Your session has expired. Please sign in again.';
    }
    if (error.isForbidden) {
      return 'You do not have permission for this action.';
    }
    if (error.isNotFound) {
      return 'The requested resource was not found.';
    }
    if (error.isServerError) {
      return 'Server error. Please try again later.';
    }
    // Use API error message if available
    if (error.data?.message) {
      return error.data.message;
    }
  }
  return error.message || 'An unexpected error occurred.';
}

/**
 * Check if error should be skipped based on status
 */
function shouldSkipError(error: Error | ApiError, skipForStatus?: number[]): boolean {
  if (!skipForStatus?.length) return false;
  if (error instanceof ApiError) {
    return skipForStatus.includes(error.status);
  }
  return false;
}

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * Hook to show error toast manually or automatically
 *
 * @example
 * ```tsx
 * // Manual error toast
 * const { showErrorToast } = useQueryError();
 *
 * try {
 *   await someOperation();
 * } catch (error) {
 *   showErrorToast(error, { context: 'saving data' });
 * }
 * ```
 */
export function useQueryError(): UseQueryErrorReturn {
  const showErrorToast = (error: Error | ApiError, options: UseQueryErrorOptions = {}) => {
    const { title, message, context, skipForStatus } = options;

    // Skip if configured for this status
    if (shouldSkipError(error, skipForStatus)) {
      return;
    }

    // Skip 401 errors - they're handled by the API client
    if (error instanceof ApiError && error.isUnauthorized) {
      return;
    }

    toast({
      variant: 'destructive',
      title: title || getErrorTitle(error, context),
      description: message || getErrorDescription(error),
    });
  };

  const showSuccessToast = (title: string, description?: string) => {
    toast({
      variant: 'success',
      title,
      description,
    });
  };

  return {
    showErrorToast,
    showSuccessToast,
  };
}

/**
 * Hook to automatically show toast when error changes
 *
 * @param error - The error from a query/mutation
 * @param options - Configuration options
 *
 * @example
 * ```tsx
 * const { data, error } = useQuery(...);
 *
 * // Automatically shows toast when error occurs
 * useQueryErrorToast(error, { context: 'loading KPIs' });
 * ```
 */
export function useQueryErrorToast(
  error: Error | ApiError | null | undefined,
  options: UseQueryErrorOptions = {}
): void {
  const { showToast = true, onError, skipForStatus } = options;
  const prevErrorRef = useRef<Error | ApiError | null>(null);

  useEffect(() => {
    // Only trigger for new errors
    if (!error || error === prevErrorRef.current) {
      prevErrorRef.current = error || null;
      return;
    }

    prevErrorRef.current = error;

    // Call custom onError if provided
    onError?.(error);

    // Skip if configured for this status
    if (shouldSkipError(error, skipForStatus)) {
      return;
    }

    // Skip 401 errors - they're handled by the API client
    if (error instanceof ApiError && error.isUnauthorized) {
      return;
    }

    // Show toast
    if (showToast) {
      toast({
        variant: 'destructive',
        title: options.title || getErrorTitle(error, options.context),
        description: options.message || getErrorDescription(error),
      });
    }
  }, [error, showToast, onError, skipForStatus, options.title, options.message, options.context]);
}

/**
 * Hook for mutation error handling with toast and callback
 *
 * Returns an onError callback suitable for useMutation options.
 *
 * @param options - Configuration options
 * @returns onError callback
 *
 * @example
 * ```tsx
 * const onError = useMutationError({ context: 'saving experiment' });
 *
 * const mutation = useMutation({
 *   mutationFn: saveExperiment,
 *   onError,
 * });
 * ```
 */
export function useMutationError(
  options: UseQueryErrorOptions = {}
): (error: Error | ApiError) => void {
  const { showToast = true, onError, skipForStatus } = options;

  return (error: Error | ApiError) => {
    // Call custom onError if provided
    onError?.(error);

    // Skip if configured for this status
    if (shouldSkipError(error, skipForStatus)) {
      return;
    }

    // Skip 401 errors - they're handled by the API client
    if (error instanceof ApiError && error.isUnauthorized) {
      return;
    }

    // Show toast
    if (showToast) {
      toast({
        variant: 'destructive',
        title: options.title || getErrorTitle(error, options.context),
        description: options.message || getErrorDescription(error),
      });
    }
  };
}

export default useQueryError;
