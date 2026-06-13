/**
 * MSW Browser Bootstrap Tests
 * ===========================
 *
 * Red-first tests for the window.__E2I_MSW_ACTIVE__ flag contract:
 * initMSW() must publish the flag when (and only when) the worker actually
 * starts, so MSWBanner can render the persistent mock-data banner.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const { startMock, stopMock, resetHandlersMock } = vi.hoisted(() => ({
  startMock: vi.fn(),
  stopMock: vi.fn(),
  resetHandlersMock: vi.fn(),
}));

vi.mock('msw/browser', () => ({
  setupWorker: () => ({
    start: startMock,
    stop: stopMock,
    resetHandlers: resetHandlersMock,
  }),
}));

vi.mock('./handlers', () => ({ handlers: [] }));

import { initMSW, stopMSW } from './browser';

beforeEach(() => {
  startMock.mockReset().mockResolvedValue(undefined);
  stopMock.mockReset();
  delete window.__E2I_MSW_ACTIVE__;
});

afterEach(() => {
  vi.unstubAllEnvs();
  delete window.__E2I_MSW_ACTIVE__;
});

describe('initMSW active flag', () => {
  it('does not set the flag outside development mode', async () => {
    // vitest runs with MODE=test
    await initMSW();
    expect(startMock).not.toHaveBeenCalled();
    expect(window.__E2I_MSW_ACTIVE__).toBeUndefined();
  });

  it('sets the flag after the worker starts in development mode', async () => {
    vi.stubEnv('MODE', 'development');
    await initMSW();
    expect(startMock).toHaveBeenCalledTimes(1);
    expect(window.__E2I_MSW_ACTIVE__).toBe(true);
  });

  it('does not set the flag when mocking is disabled via VITE_MSW_ENABLED=false', async () => {
    vi.stubEnv('MODE', 'development');
    vi.stubEnv('VITE_MSW_ENABLED', 'false');
    await initMSW();
    expect(startMock).not.toHaveBeenCalled();
    expect(window.__E2I_MSW_ACTIVE__).toBeUndefined();
  });

  it('does not set the flag when the worker fails to start', async () => {
    vi.stubEnv('MODE', 'development');
    startMock.mockRejectedValue(new Error('sw registration failed'));
    await initMSW();
    expect(window.__E2I_MSW_ACTIVE__).toBeUndefined();
  });

  it('stopMSW clears the flag', async () => {
    vi.stubEnv('MODE', 'development');
    await initMSW();
    expect(window.__E2I_MSW_ACTIVE__).toBe(true);
    stopMSW();
    expect(window.__E2I_MSW_ACTIVE__).toBe(false);
  });
});
