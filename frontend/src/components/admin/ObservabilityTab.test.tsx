/**
 * ObservabilityTab tests — stat cards, per-user expansion, platform section,
 * honest states (tracking banner, unpriced "—", empty window). recharts SVG
 * is not asserted (visual — verified live).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { LlmUsageResponse } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  useLlmUsage: vi.fn(),
}));

import * as adminHooks from '@/hooks/api/use-admin';
import { ObservabilityTab } from './ObservabilityTab';

const U1 = '11111111-1111-1111-1111-111111111111';

const PAYLOAD: LlmUsageResponse = {
  summary: {
    total_cost_usd: 1.234567,
    input_tokens: 120000,
    output_tokens: 45000,
    calls: 42,
    distinct_users: 2,
    days: 30,
    // dynamic so the "tracking began inside this window" banner assertion
    // never rots as real time passes
    tracking_since: new Date().toISOString(),
  },
  daily: [
    { date: '2026-07-12', chat_cost_usd: 0.9, platform_cost_usd: 0.33, tokens: 165000 },
  ],
  by_user: [
    {
      user_id: U1,
      email: 'alice@x.com',
      sessions: 2,
      calls: 30,
      input_tokens: 100000,
      output_tokens: 40000,
      cost_usd: 0.9,
      models: ['claude-sonnet-4-6'],
    },
  ],
  sessions: {
    [U1]: [
      {
        session_id: `${U1}~conv-a`,
        title: 'Kisqali TRx dip',
        started_at: '2026-07-12T10:00:00+00:00',
        calls: 20,
        input_tokens: 80000,
        output_tokens: 30000,
        cost_usd: 0.7,
        models: ['claude-sonnet-4-6'],
      },
    ],
  },
  platform: [
    {
      surface: 'insights',
      component: 'ExecutiveBrief',
      model: 'gpt-4o',
      calls: 12,
      input_tokens: 20000,
      output_tokens: 5000,
      cost_usd: 0.33,
    },
  ],
  pricing_version: '2026-07-12',
  unpriced_models: [],
};

const mockHook = (over: Partial<ReturnType<typeof buildResult>> = {}) => {
  (adminHooks.useLlmUsage as ReturnType<typeof vi.fn>).mockReturnValue(
    buildResult(over)
  );
};

function buildResult(over: object) {
  return { data: PAYLOAD, isLoading: false, isError: false, ...over };
}

beforeEach(() => vi.clearAllMocks());

describe('ObservabilityTab', () => {
  it('renders stat cards and the tracking banner', () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.getByText('Total cost')).toBeInTheDocument();
    expect(screen.getByText('$1.23')).toBeInTheDocument();
    expect(screen.getByText('LLM calls')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText(/Usage tracking began/)).toBeInTheDocument();
  });

  it('expands a user row to session breakdown', async () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.queryByText('Kisqali TRx dip')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /alice@x.com/ }));
    expect(screen.getByText('Kisqali TRx dip')).toBeInTheDocument();
  });

  it('renders the platform (non-chat) section', () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.getByText('Platform LLM usage (non-chat)')).toBeInTheDocument();
    expect(screen.getByText('ExecutiveBrief')).toBeInTheDocument();
  });

  it('lists unpriced models honestly', () => {
    mockHook({
      data: { ...PAYLOAD, unpriced_models: ['mystery-lm-9'] },
    });
    render(<ObservabilityTab />);
    expect(screen.getByText(/mystery-lm-9/)).toBeInTheDocument();
  });

  it('shows an explicit empty state', () => {
    mockHook({
      data: {
        ...PAYLOAD,
        summary: { ...PAYLOAD.summary, calls: 0, tracking_since: null },
        daily: [],
        by_user: [],
        sessions: {},
        platform: [],
      },
    });
    render(<ObservabilityTab />);
    expect(screen.getByText(/No LLM usage recorded/)).toBeInTheDocument();
  });
});
