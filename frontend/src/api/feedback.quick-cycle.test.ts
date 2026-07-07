/**
 * quickLearningCycle — window contract.
 *
 * The backend default window (last 24h) starves the cycle: real feedback
 * (chat thumbs + cognitive reward signals) accrues per active chat day, and
 * chat activity is not daily. The page button must request a 7-day window.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/lib/api-client', () => ({
  post: vi.fn(),
  get: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import { post } from '@/lib/api-client';
import { quickLearningCycle } from './feedback';

const postMock = post as ReturnType<typeof vi.fn>;

describe('quickLearningCycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    postMock.mockResolvedValue({ batch_id: 'b1', status: 'completed' });
  });

  it('requests a 7-day window (not the starving 24h backend default)', async () => {
    const before = Date.now();
    await quickLearningCycle();
    const after = Date.now();

    const body = postMock.mock.calls[0][1];
    expect(body.time_range_start).toBeDefined();
    const startMs = new Date(body.time_range_start).getTime();
    const sevenDays = 7 * 24 * 60 * 60 * 1000;
    expect(startMs).toBeGreaterThanOrEqual(before - sevenDays - 5000);
    expect(startMs).toBeLessThanOrEqual(after - sevenDays + 5000);
  });
});
