/**
 * Test Utilities
 * ==============
 *
 * Common utilities for testing React components and hooks.
 */

import { type ReactElement, type ReactNode } from 'react';
import { render, type RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

/**
 * Create a fresh QueryClient for each test
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

/**
 * Wrapper component that provides all necessary providers
 */
interface WrapperProps {
  children: ReactNode;
}

function createWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: WrapperProps): ReactElement {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

/**
 * Custom render function that wraps component with providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  const queryClient = createTestQueryClient();
  return {
    queryClient,
    ...render(ui, { wrapper: createWrapper(queryClient), ...options }),
  };
}

/**
 * Create a wrapper for testing hooks
 */
export function createHookWrapper() {
  const queryClient = createTestQueryClient();
  return {
    queryClient,
    wrapper: createWrapper(queryClient),
  };
}

export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
