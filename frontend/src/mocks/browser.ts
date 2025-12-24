/**
 * MSW Browser Setup
 * =================
 *
 * Configures the Mock Service Worker for browser environments.
 * Used in development to intercept API requests and return mock data.
 *
 * Usage:
 *   Import and call `initMSW()` before rendering the React app.
 *   The function returns a promise that resolves when the worker is ready.
 */

import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

/**
 * MSW browser worker instance
 *
 * The worker intercepts outgoing HTTP requests in the browser
 * and responds with mock data defined in handlers.
 */
export const worker = setupWorker(...handlers);

/**
 * Initialize MSW for development
 *
 * This function starts the service worker and should be called
 * before rendering the React application.
 *
 * @returns Promise that resolves when the worker is ready
 */
export async function initMSW(): Promise<void> {
  // Only initialize in development mode
  if (import.meta.env.MODE !== 'development') {
    return;
  }

  // Check if MSW is enabled via environment variable
  // Set VITE_MSW_ENABLED=false to disable mocking
  const mswEnabled = import.meta.env.VITE_MSW_ENABLED !== 'false';

  if (!mswEnabled) {
    console.info('[MSW] Mocking disabled via VITE_MSW_ENABLED=false');
    return;
  }

  try {
    await worker.start({
      // Don't log all requests by default, only warnings/errors
      onUnhandledRequest: 'bypass',

      // Service worker options
      serviceWorker: {
        url: '/mockServiceWorker.js',
      },
    });

    console.info('[MSW] Mock Service Worker started successfully');
    console.info('[MSW] API requests will be intercepted and return mock data');
  } catch (error) {
    console.error('[MSW] Failed to start Mock Service Worker:', error);
    // Don't throw - allow the app to continue without mocking
  }
}

/**
 * Stop the MSW worker
 *
 * Call this when you need to disable mocking at runtime.
 */
export function stopMSW(): void {
  worker.stop();
  console.info('[MSW] Mock Service Worker stopped');
}

/**
 * Reset handlers to defaults
 *
 * Useful for resetting state between tests.
 */
export function resetHandlers(): void {
  worker.resetHandlers();
}
