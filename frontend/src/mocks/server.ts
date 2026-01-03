/**
 * MSW Node Server Setup
 * =====================
 *
 * Configures the Mock Service Worker for Node.js test environments.
 * Used in Vitest to intercept API requests and return mock data.
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

/**
 * MSW node server instance for testing
 */
export const server = setupServer(...handlers);
