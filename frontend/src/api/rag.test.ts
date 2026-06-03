/**
 * RAG API Client Tests
 * ====================
 *
 * Unit tests for the RAG API client functions in `@/api/rag`.
 *
 * Focus: the request-side contract with the backend
 * (`src/api/routes/rag.py`). The backend `GET /v1/rag/stats` endpoint reads
 * the query parameter named `hours` (see `get_stats(hours: int = Query(...))`
 * and the pinned backend tests `test_get_stats_with_hours_param` /
 * `test_get_stats_custom_period`). The client must therefore send `hours`,
 * not `period_hours`, otherwise the period filter is silently ignored and the
 * backend falls back to its 24h default.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the shared api-client request helpers so we can assert the exact
// endpoint + query params each function sends.
vi.mock('@/lib/api-client', () => ({
  get: vi.fn(),
  post: vi.fn(),
}));

import { getRAGStats } from './rag';
import * as apiClient from '@/lib/api-client';

describe('RAG API Client - getRAGStats request contract', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockResolvedValue({} as never);
  });

  it('sends the period as the backend-canonical `hours` query param', async () => {
    await getRAGStats(48);

    // Backend reads `hours` (rag.py get_stats); sending `period_hours` would be
    // silently dropped, defaulting the backend to 24h.
    expect(apiClient.get).toHaveBeenCalledWith('/v1/rag/stats', { hours: 48 });
  });

  it('does NOT send the legacy `period_hours` query param', async () => {
    await getRAGStats(48);

    const params = vi.mocked(apiClient.get).mock.calls[0][1] as
      | Record<string, unknown>
      | undefined;
    expect(params).not.toHaveProperty('period_hours');
  });

  it('omits query params when no period is supplied (backend default applies)', async () => {
    await getRAGStats();

    expect(apiClient.get).toHaveBeenCalledWith('/v1/rag/stats', undefined);
  });
});
