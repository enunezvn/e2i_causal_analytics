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
import { render, screen } from '@testing-library/react';
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
