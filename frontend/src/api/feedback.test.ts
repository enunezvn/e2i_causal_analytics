/**
 * quickLearningCycle — lookback window regression tests.
 *
 * The 7-day window structurally missed all real signals older than a week
 * (signal accrual is bursty); the window is now 30 days. Same defect class
 * as the Fabhalta gap-floor bug (#1237): "no results" ≠ "correctly no results".
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('@/lib/api-client', () => ({
  get: vi.fn(),
  post: vi.fn().mockResolvedValue({}),
}));

import { get, post } from '@/lib/api-client';
import {
  quickLearningCycle,
  runLearningCycleAndWait,
  LEARNING_CYCLE_POLL_CEILING_MS,
} from './feedback';

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

describe('quickLearningCycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('requests a 30-day lookback, sync mode, auto_apply=false', async () => {
    const before = Date.now();
    await quickLearningCycle();
    const after = Date.now();

    expect(post).toHaveBeenCalledTimes(1);
    const [path, body, opts] = (post as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(path).toBe('/feedback/learn');
    expect(body.auto_apply).toBe(false);
    expect(body.min_feedback_count).toBe(5);
    expect(body.pattern_threshold).toBe(0.1);
    expect(opts.params.async_mode).toBe(false);

    const start = new Date(body.time_range_start).getTime();
    expect(start).toBeGreaterThanOrEqual(before - THIRTY_DAYS_MS - 2000);
    expect(start).toBeLessThanOrEqual(after - THIRTY_DAYS_MS + 2000);
  });

  it('passes focus agents through', async () => {
    await quickLearningCycle(['causal_impact']);
    const [, body] = (post as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(body.focus_agents).toEqual(['causal_impact']);
  });
});

/**
 * runLearningCycleAndWait — poll ceiling pinned to its measured basis
 * ===================================================================
 *
 * No page consumes this helper (only the hook re-export), so the ceiling has
 * no page test to pin it. Measured on prod from `feedback_learning_batches`
 * (n=163 completed): total_latency_ms avg 0.45 s, p95 0.85 s, worst 23.1 s.
 * 120 s stays; these tests keep it from silently regressing (#1839).
 */
describe('runLearningCycleAndWait — poll ceiling', () => {
  const PENDING = { batch_id: 'fb_pin', status: 'collecting', errors: [] };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('exports a ceiling no lower than the measured basis', () => {
    // `>=` so a later lift keeps passing; only a cut below the measurement fails.
    expect(LEARNING_CYCLE_POLL_CEILING_MS).toBeGreaterThanOrEqual(120_000);
  });

  it('keeps polling until the exported ceiling, then times out (default maxWaitMs)', async () => {
    vi.useFakeTimers();
    (post as ReturnType<typeof vi.fn>).mockResolvedValue({ ...PENDING, status: 'pending' });
    (get as ReturnType<typeof vi.fn>).mockResolvedValue(PENDING);

    let settled = false;
    const run = runLearningCycleAndWait({ focus_agents: ['gap_analyzer'] });
    run.catch(() => {
      settled = true;
    });

    // Just short of the ceiling: still polling, not rejected.
    await vi.advanceTimersByTimeAsync(LEARNING_CYCLE_POLL_CEILING_MS - 1);
    expect(settled).toBe(false);
    expect(get).toHaveBeenCalled();

    // Past the ceiling (one more poll interval): rejects with the timeout.
    await vi.advanceTimersByTimeAsync(2_001);
    await expect(run).rejects.toThrow(`timed out after ${LEARNING_CYCLE_POLL_CEILING_MS}ms`);
    expect(settled).toBe(true);
    // One POST; the rest were GET polls.
    expect(post).toHaveBeenCalledTimes(1);
  });
});
