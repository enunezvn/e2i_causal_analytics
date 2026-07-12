/**
 * audit API — getWorkflowDetails resilience
 * =========================================
 *
 * Regression for the audit-chain drill-down: getWorkflowDetails fetched
 * entries + summary + verification with `Promise.all`, so a single failing
 * sub-call (the chain-verify endpoint 500s when the pgcrypto `digest()`
 * function is unavailable) rejected the WHOLE call — blanking the Details /
 * Verification / Timeline tabs. Verification is best-effort: its failure must
 * degrade to `verification: null`, never break the required entries + summary.
 */
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';

vi.mock('@/lib/api-client', () => ({ get: vi.fn() }));

import { get } from '@/lib/api-client';
import { getWorkflowDetails, getLowConfidenceEntries } from './audit';

describe('getWorkflowDetails — resilient to verify failure', () => {
  beforeEach(() => vi.clearAllMocks());

  it('returns entries + summary with verification=null when /verify fails', async () => {
    (get as Mock).mockImplementation((url: string) => {
      if (url.includes('/verify')) return Promise.reject(new Error('500 digest does not exist'));
      if (url.endsWith('/summary')) return Promise.resolve({ workflow_id: 'wf1', total_entries: 2 });
      return Promise.resolve([{ entry_id: 'e1', sequence_number: 1, agent_name: 'a' }]);
    });

    const details = await getWorkflowDetails('wf1');

    expect(details.entries).toHaveLength(1);
    expect(details.summary.total_entries).toBe(2);
    expect(details.verification).toBeNull();
  });

  it('returns the real verification when all calls succeed', async () => {
    (get as Mock).mockImplementation((url: string) => {
      if (url.includes('/verify')) return Promise.resolve({ is_valid: true, entries_checked: 2 });
      if (url.endsWith('/summary')) return Promise.resolve({ total_entries: 2 });
      return Promise.resolve([{ entry_id: 'e1' }]);
    });

    const details = await getWorkflowDetails('wf1');

    expect(details.verification).toEqual({ is_valid: true, entries_checked: 2 });
  });

  it('still rejects when a REQUIRED call (entries) fails', async () => {
    (get as Mock).mockImplementation((url: string) => {
      if (url.endsWith('/summary')) return Promise.resolve({ total_entries: 2 });
      if (url.includes('/verify')) return Promise.resolve({ is_valid: true });
      return Promise.reject(new Error('entries 500'));
    });

    await expect(getWorkflowDetails('wf1')).rejects.toThrow();
  });
});

describe('getLowConfidenceEntries — null confidence is "not recorded", not zero', () => {
  beforeEach(() => vi.clearAllMocks());

  it('excludes entries whose confidence_score is null (most actions never record one)', async () => {
    // The backend serializes explicit null for actions that don't record
    // confidence; `null < 0.7` coerces null to 0, which used to flood the
    // low-confidence list with fabricated 0% entries.
    (get as Mock).mockResolvedValue([
      { entry_id: 'e1', agent_name: 'health_score', confidence_score: null },
      { entry_id: 'e2', agent_name: 'model_selector', confidence_score: 0.85 },
      { entry_id: 'e3', agent_name: 'estimator', confidence_score: 0.4 },
    ]);

    const low = await getLowConfidenceEntries('wf1', 0.7);

    expect(low.map((e) => e.entry_id)).toEqual(['e3']);
  });

  it('keeps a genuine measured 0 confidence', async () => {
    (get as Mock).mockResolvedValue([
      { entry_id: 'e1', agent_name: 'estimator', confidence_score: 0 },
    ]);

    const low = await getLowConfidenceEntries('wf1', 0.7);

    expect(low.map((e) => e.entry_id)).toEqual(['e1']);
  });
});
