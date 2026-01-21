/**
 * Query Error State Component
 * ===========================
 *
 * Reusable component for displaying query/mutation error states
 * with retry functionality and user-friendly error messages.
 *
 * Features:
 * - Consistent error display across the application
 * - Retry button for transient errors
 * - Contextual error messages based on error type
 * - Support for both ApiError and generic Error types
 *
 * @module components/ui/query-error-state
 */

import * as React from 'react';
import { AlertCircle, RefreshCw, WifiOff, Lock, ServerCrash } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { ApiError } from '@/lib/api-client';

// =============================================================================
// TYPES
// =============================================================================

export interface QueryErrorStateProps {
  /** The error to display */
  error: Error | ApiError | null | undefined;

  /** Optional retry function */
  onRetry?: () => void;

  /** Whether retry is currently in progress */
  isRetrying?: boolean;

  /** Optional custom title */
  title?: string;

  /** Optional custom description */
  description?: string;

  /** Size variant */
  size?: 'default' | 'sm' | 'lg';

  /** Additional CSS classes */
  className?: string;

  /** Show compact version without description */
  compact?: boolean;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Get appropriate icon based on error type
 */
function getErrorIcon(error: Error | ApiError): React.ReactNode {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return <WifiOff className="h-4 w-4" />;
    }
    if (error.isUnauthorized || error.isForbidden) {
      return <Lock className="h-4 w-4" />;
    }
    if (error.isServerError) {
      return <ServerCrash className="h-4 w-4" />;
    }
  }
  return <AlertCircle className="h-4 w-4" />;
}

/**
 * Get user-friendly error title based on error type
 */
function getErrorTitle(error: Error | ApiError): string {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return 'Connection Error';
    }
    if (error.isUnauthorized) {
      return 'Session Expired';
    }
    if (error.isForbidden) {
      return 'Access Denied';
    }
    if (error.isNotFound) {
      return 'Not Found';
    }
    if (error.isServerError) {
      return 'Server Error';
    }
    if (error.isClientError) {
      return 'Request Error';
    }
  }
  return 'Something went wrong';
}

/**
 * Get user-friendly error description based on error type
 */
function getErrorDescription(error: Error | ApiError): string {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return 'Unable to connect to the server. Please check your internet connection and try again.';
    }
    if (error.isUnauthorized) {
      return 'Your session has expired. Please sign in again to continue.';
    }
    if (error.isForbidden) {
      return 'You do not have permission to access this resource.';
    }
    if (error.isNotFound) {
      return 'The requested resource could not be found.';
    }
    if (error.isServerError) {
      return 'The server encountered an error. Please try again later.';
    }
    // Use the API error message if available
    if (error.data?.message) {
      return error.data.message;
    }
  }
  return error.message || 'An unexpected error occurred. Please try again.';
}

/**
 * Determine if error is likely transient and retry makes sense
 */
function isRetryable(error: Error | ApiError): boolean {
  if (error instanceof ApiError) {
    // Network errors and server errors are often transient
    return error.isNetworkError || error.isServerError;
  }
  return true; // Default to allowing retry for unknown errors
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * Query Error State Component
 *
 * Displays a consistent error state for failed queries/mutations.
 *
 * @example
 * ```tsx
 * const { data, error, refetch, isRefetching } = useQuery(...);
 *
 * if (error) {
 *   return (
 *     <QueryErrorState
 *       error={error}
 *       onRetry={refetch}
 *       isRetrying={isRefetching}
 *     />
 *   );
 * }
 * ```
 */
export function QueryErrorState({
  error,
  onRetry,
  isRetrying = false,
  title,
  description,
  size = 'default',
  className,
  compact = false,
}: QueryErrorStateProps) {
  if (!error) {
    return null;
  }

  const errorTitle = title || getErrorTitle(error);
  const errorDescription = description || getErrorDescription(error);
  const ErrorIcon = () => getErrorIcon(error);
  const canRetry = onRetry && isRetryable(error);

  const sizeClasses = {
    sm: 'p-3',
    default: 'p-4',
    lg: 'p-6',
  };

  const iconSizeClasses = {
    sm: '[&>svg]:h-3 [&>svg]:w-3',
    default: '[&>svg]:h-4 [&>svg]:w-4',
    lg: '[&>svg]:h-5 [&>svg]:w-5',
  };

  const buttonSizeMap = {
    sm: 'sm' as const,
    default: 'sm' as const,
    lg: 'default' as const,
  };

  return (
    <Alert
      variant="destructive"
      className={cn(sizeClasses[size], iconSizeClasses[size], className)}
    >
      <ErrorIcon />
      <AlertTitle className={cn(size === 'sm' && 'text-sm')}>
        {errorTitle}
      </AlertTitle>
      {!compact && (
        <AlertDescription className={cn(size === 'sm' && 'text-xs')}>
          <div className="flex flex-col gap-3">
            <span>{errorDescription}</span>
            {canRetry && (
              <Button
                variant="outline"
                size={buttonSizeMap[size]}
                onClick={onRetry}
                disabled={isRetrying}
                className="w-fit"
              >
                {isRetrying ? (
                  <>
                    <RefreshCw className="mr-2 h-3 w-3 animate-spin" />
                    Retrying...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-3 w-3" />
                    Try Again
                  </>
                )}
              </Button>
            )}
          </div>
        </AlertDescription>
      )}
    </Alert>
  );
}

// =============================================================================
// INLINE VARIANT
// =============================================================================

export interface InlineQueryErrorProps {
  /** The error to display */
  error: Error | ApiError | null | undefined;

  /** Optional retry function */
  onRetry?: () => void;

  /** Whether retry is currently in progress */
  isRetrying?: boolean;

  /** Additional CSS classes */
  className?: string;
}

/**
 * Inline Query Error
 *
 * Minimal inline error display for tight spaces.
 *
 * @example
 * ```tsx
 * <div className="flex items-center gap-2">
 *   <span>Status:</span>
 *   {error ? (
 *     <InlineQueryError error={error} onRetry={refetch} />
 *   ) : (
 *     <span>{data.status}</span>
 *   )}
 * </div>
 * ```
 */
export function InlineQueryError({
  error,
  onRetry,
  isRetrying = false,
  className,
}: InlineQueryErrorProps) {
  if (!error) {
    return null;
  }

  const errorTitle = getErrorTitle(error);
  const canRetry = onRetry && isRetryable(error);

  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 text-sm text-destructive',
        className
      )}
    >
      <AlertCircle className="h-3 w-3" />
      <span>{errorTitle}</span>
      {canRetry && (
        <button
          onClick={onRetry}
          disabled={isRetrying}
          className="inline-flex items-center gap-1 text-xs underline hover:no-underline disabled:opacity-50"
        >
          {isRetrying ? (
            <>
              <RefreshCw className="h-3 w-3 animate-spin" />
              Retrying
            </>
          ) : (
            'Retry'
          )}
        </button>
      )}
    </span>
  );
}

export default QueryErrorState;
