/**
 * Gap Analysis API Client Tests
 * =============================
 *
 * Focus: `runGapAnalysisAndWait` must honor its documented `@throws` and reject
 * when the analysis FAILS — including an IMMEDIATE failure on the very first
 * (kickoff) response, not just a failure observed later during polling.
 *
 * Previously the early-return guard treated `completed` and `failed` the same
 * way (`return initial`), so an immediate failure was returned as if it were a
 * successful result, letting callers silently consume a failed analysis.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnalysisStatus } from '@/types/gaps';

// Mock the shared api-client helpers so we control what `runGapAnalysis`
// (POST /gaps/analyze) and `getGapAnalysis` (GET /gaps/:id) return.
vi.mock('@/lib/api-client', () => ({
  get: vi.fn(),
  post: vi.fn(),
}));

import { runGapAnalysisAndWait } from './gaps';
import * as apiClient from '@/lib/api-client';

const baseRequest = {
  query: 'Find gaps',
  brand: 'kisqali',
} as Parameters<typeof runGapAnalysisAndWait>[0];

describe('runGapAnalysisAndWait', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('REJECTS when the initial response is already failed (immediate failure)', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      analysis_id: 'gap_1',
      status: AnalysisStatus.FAILED,
      warnings: ['no data for brand'],
    } as never);

    await expect(runGapAnalysisAndWait(baseRequest)).rejects.toThrow(
      /Gap analysis failed: no data for brand/
    );
    // Must fail fast on the kickoff response, never entering the poll loop.
    expect(apiClient.get).not.toHaveBeenCalled();
  });

  it('returns immediately when the initial response is already completed', async () => {
    const completed = {
      analysis_id: 'gap_2',
      status: AnalysisStatus.COMPLETED,
      warnings: [],
    };
    vi.mocked(apiClient.post).mockResolvedValueOnce(completed as never);

    const result = await runGapAnalysisAndWait(baseRequest);
    expect(result.status).toBe(AnalysisStatus.COMPLETED);
    expect(apiClient.get).not.toHaveBeenCalled();
  });

  it('still rejects when a failure is observed during polling (regression guard)', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      analysis_id: 'gap_3',
      status: AnalysisStatus.PENDING,
      warnings: [],
    } as never);
    vi.mocked(apiClient.get).mockResolvedValueOnce({
      analysis_id: 'gap_3',
      status: AnalysisStatus.FAILED,
      warnings: ['solver error'],
    } as never);

    await expect(
      // Short poll interval so the test does not wait 2s.
      runGapAnalysisAndWait(baseRequest, 1, 5000)
    ).rejects.toThrow(/Gap analysis failed: solver error/);
  });
});
