/**
 * AuditChain Page Tests
 * =====================
 *
 * H5: the Workflows tab must render live audit workflows, an honest empty state
 * when none exist, and an error state when the query fails — never the former
 * fabricated SAMPLE_WORKFLOWS / SAMPLE_TIER_DISTRIBUTION fallbacks.
 */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AuditChain from './AuditChain';

vi.mock('@/hooks/api', () => ({
  useRecentWorkflows: vi.fn(),
  useWorkflowDetails: vi.fn(),
  useTierDistribution: vi.fn(),
  useFailedValidationEntries: vi.fn(),
  useLowConfidenceEntries: vi.fn(),
}));

import {
  useRecentWorkflows,
  useWorkflowDetails,
  useTierDistribution,
  useFailedValidationEntries,
  useLowConfidenceEntries,
} from '@/hooks/api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  (useWorkflowDetails as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined, isLoading: false });
  (useTierDistribution as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  (useFailedValidationEntries as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  (useLowConfidenceEntries as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
});

describe('AuditChain (H5)', () => {
  it('renders live workflows when the API returns data', () => {
    (useRecentWorkflows as ReturnType<typeof vi.fn>).mockReturnValue({
      data: [
        { workflow_id: 'live-xyz', started_at: new Date().toISOString(), entry_count: 4, first_agent: 'orchestrator', last_agent: 'explainer', brand: 'Kisqali' },
      ],
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<AuditChain />, { wrapper: createWrapper() });

    expect(screen.getByText('live-xyz')).toBeInTheDocument();
    // The fabricated sample workflow ids must NOT render.
    expect(screen.queryByText('wf-001-abc123')).not.toBeInTheDocument();
  });

  it('renders an empty state when the API returns no workflows', () => {
    (useRecentWorkflows as ReturnType<typeof vi.fn>).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<AuditChain />, { wrapper: createWrapper() });

    expect(screen.getByText(/No workflows found/i)).toBeInTheDocument();
    expect(screen.queryByText('wf-001-abc123')).not.toBeInTheDocument();
  });

  it('renders an error state when the workflows query errors', () => {
    (useRecentWorkflows as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<AuditChain />, { wrapper: createWrapper() });

    expect(screen.getByText(/Failed to load workflows/i)).toBeInTheDocument();
  });
});

describe('AuditChain — execution timeline honesty with null validation/confidence', () => {
  // Most audit entries carry explicit null confidence_score and
  // validation_passed (only estimation nodes record confidence; only
  // refutation nodes record validation). Nulls must render as "not
  // applicable" — never as a fabricated "0% conf" badge or a red failed-X.
  const baseEntry = {
    workflow_id: 'live-xyz',
    agent_tier: 3,
    action_type: 'execute',
    created_at: new Date().toISOString(),
    entry_hash: 'hash',
  };

  beforeEach(() => {
    (useRecentWorkflows as ReturnType<typeof vi.fn>).mockReturnValue({
      data: [
        { workflow_id: 'live-xyz', started_at: new Date().toISOString(), entry_count: 3, first_agent: 'a', last_agent: 'b' },
      ],
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useWorkflowDetails as ReturnType<typeof vi.fn>).mockReturnValue({
      isLoading: false,
      data: {
        summary: { workflow_id: 'live-xyz', total_entries: 3, total_duration_ms: 1200, agents_involved: ['a', 'b'] },
        entries: [
          { ...baseEntry, entry_id: 'e1', sequence_number: 1, agent_name: 'health_score', confidence_score: null, validation_passed: null, duration_ms: 100 },
          { ...baseEntry, entry_id: 'e2', sequence_number: 2, agent_name: 'estimator', confidence_score: 0.85, validation_passed: true, duration_ms: 50 },
          { ...baseEntry, entry_id: 'e3', sequence_number: 3, agent_name: 'refuter', confidence_score: 0.4, validation_passed: false, duration_ms: 80 },
        ],
        verification: null,
      },
    });
  });

  function renderAndOpenDetails() {
    render(<AuditChain />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('live-xyz'));
  }

  it('renders no conf badge for null confidence, real badges for measured scores', () => {
    renderAndOpenDetails();

    expect(screen.queryByText('0% conf')).not.toBeInTheDocument();
    expect(screen.getByText('85% conf')).toBeInTheDocument();
    expect(screen.getByText('40% conf')).toBeInTheDocument();
  });

  it('renders a neutral not-applicable marker (not a failed X) for null validation', () => {
    renderAndOpenDetails();

    expect(screen.getByLabelText('No validation for this action')).toBeInTheDocument();
  });
});
