/**
 * quickLearningCycle — lookback window regression tests.
 *
 * The 7-day window structurally missed all real signals older than a week
 * (signal accrual is bursty); the window is now 30 days. Same defect class
 * as the Fabhalta gap-floor bug (#1237): "no results" ≠ "correctly no results".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/lib/api-client', () => ({
  get: vi.fn(),
  post: vi.fn().mockResolvedValue({}),
}));

import { post } from '@/lib/api-client';
import { quickLearningCycle } from './feedback';

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
