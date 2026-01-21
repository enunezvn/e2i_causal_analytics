/**
 * ErrorBoundary Component
 * =======================
 *
 * Catches JavaScript errors in child component tree and displays
 * a fallback UI instead of crashing the entire application.
 *
 * Usage:
 *   <ErrorBoundary fallback={<ErrorFallback />}>
 *     <YourComponent />
 *   </ErrorBoundary>
 *
 * Or with onError callback:
 *   <ErrorBoundary onError={(error, info) => logError(error, info)}>
 *     <YourComponent />
 *   </ErrorBoundary>
 *
 * @module components/ui/error-boundary
 */

import React, { Component, ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';

// =============================================================================
// TYPES
// =============================================================================

export interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Custom fallback UI to show on error */
  fallback?: ReactNode;
  /** Callback when an error is caught */
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  /** Whether to show the retry button */
  showRetry?: boolean;
  /** Section name for error message context */
  sectionName?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

// =============================================================================
// DEFAULT FALLBACK COMPONENT
// =============================================================================

interface ErrorFallbackProps {
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  onRetry?: () => void;
  showRetry?: boolean;
  sectionName?: string;
}

/**
 * Default fallback UI when an error occurs
 */
function ErrorFallback({
  error,
  errorInfo,
  onRetry,
  showRetry = true,
  sectionName = 'This section',
}: ErrorFallbackProps) {
  const isDev = process.env.NODE_ENV === 'development';

  return (
    <Card className="m-4 border-destructive/50 bg-destructive/5">
      <CardHeader>
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <CardTitle className="text-lg text-destructive">
            Something went wrong
          </CardTitle>
        </div>
        <CardDescription>
          {sectionName} encountered an error and couldn&apos;t load properly.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          {error?.message || 'An unexpected error occurred'}
        </p>

        {isDev && errorInfo && (
          <details className="rounded-md border border-border bg-muted/50 p-3">
            <summary className="cursor-pointer text-sm font-medium">
              Error Details (Development Only)
            </summary>
            <pre className="mt-2 max-h-40 overflow-auto text-xs text-muted-foreground">
              {error?.stack}
            </pre>
            <pre className="mt-2 max-h-40 overflow-auto text-xs text-muted-foreground">
              {errorInfo.componentStack}
            </pre>
          </details>
        )}

        {showRetry && onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// ERROR BOUNDARY CLASS COMPONENT
// =============================================================================

/**
 * ErrorBoundary - Catches errors in child components and displays fallback UI
 *
 * React requires error boundaries to be class components because they need
 * the getDerivedStateFromError and componentDidCatch lifecycle methods.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so next render shows fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Log error to console
    console.error('[ErrorBoundary] Caught error:', error);
    console.error('[ErrorBoundary] Component stack:', errorInfo.componentStack);

    // Store error info in state
    this.setState({ errorInfo });

    // Call optional onError callback
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleRetry = (): void => {
    // Reset error state to try rendering children again
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, showRetry = true, sectionName } = this.props;

    if (hasError) {
      // If custom fallback provided, use it
      if (fallback) {
        return fallback;
      }

      // Otherwise use default fallback
      return (
        <ErrorFallback
          error={error}
          errorInfo={errorInfo}
          onRetry={this.handleRetry}
          showRetry={showRetry}
          sectionName={sectionName}
        />
      );
    }

    return children;
  }
}

// =============================================================================
// SECTION-SPECIFIC ERROR BOUNDARIES
// =============================================================================

/**
 * Pre-configured error boundary for the Dashboard section
 */
export function DashboardErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      sectionName="Dashboard"
      onError={(error) => {
        console.error('[Dashboard] Error caught:', error.message);
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Pre-configured error boundary for the Chat section
 */
export function ChatErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      sectionName="Chat"
      onError={(error) => {
        console.error('[Chat] Error caught:', error.message);
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Pre-configured error boundary for Analytics sections
 */
export function AnalyticsErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      sectionName="Analytics"
      onError={(error) => {
        console.error('[Analytics] Error caught:', error.message);
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Pre-configured error boundary for the entire app
 * Shows a more prominent error message
 */
export function AppErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      sectionName="Application"
      showRetry={true}
      onError={(error) => {
        console.error('[App] Critical error caught:', error.message);
        // Could send to error tracking service here
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
