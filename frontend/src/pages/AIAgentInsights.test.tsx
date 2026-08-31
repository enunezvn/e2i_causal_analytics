/**
 * AIAgentInsights Page Tests
 * ==========================
 *
 * Tests for the AI Agent Insights composite page (issue #304):
 * - brand is driven from context / URL, NOT hard-coded
 * - the agents badge shows the REAL health-score count, never an
 *   invented one
 * - error boundary wraps each insight component (one failing insight
 *   does not blank the page)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import type { ReactElement, ReactNode } from 'react';

// ----------------------------------------------------------------------------
// Mocks
// ----------------------------------------------------------------------------

// Provide a vi.fn-backed implementation of useE2ICopilot we can vary per-test.
// Default returns brand=Remibrutinib so legacy assertions still work.
vi.mock('@/providers/E2ICopilotProvider', () => ({
  useE2ICopilot: vi.fn(() => ({
    filters: { brand: 'Remibrutinib' },
  })),
}));

// Mock each insight component as a thin marker that surfaces the prop it
// receives, so we can assert what the page passes down.
vi.mock('@/components/insights', () => ({
  ExecutiveAIBrief: ({ brand }: { brand?: string }) => (
    <div data-testid="executive-ai-brief">brand:{brand ?? '__none__'}</div>
  ),
  PriorityActionsROI: ({ brand }: { brand?: string }) => (
    <div data-testid="priority-actions-roi">brand:{brand ?? '__none__'}</div>
  ),
  PredictiveAlerts: () => <div data-testid="predictive-alerts" />,
  ActiveCausalChains: () => <div data-testid="active-causal-chains" />,
  ExperimentRecommendations: () => <div data-testid="experiment-recommendations" />,
  HeterogeneousTreatmentEffects: () => <div data-testid="heterogeneous-treatment-effects" />,
}));

// The header badge shows REAL agent availability from the health-score
// service; mock the hook so each test controls (or withholds) the data.
vi.mock('@/hooks/api', () => ({
  useAgentHealth: vi.fn(() => ({ data: undefined })),
}));

import { useAgentHealth } from '@/hooks/api';
import { useE2ICopilot } from '@/providers/E2ICopilotProvider';
import { AIAgentInsights } from './AIAgentInsights';

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

function createWrapperWithUrl(url: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
    });
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>
      </QueryClientProvider>
    );
  };
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

describe('AIAgentInsights', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
      filters: { brand: 'Remibrutinib' },
    });
    (useAgentHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  });

  describe('Page rendering', () => {
    it('renders page header', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByText('Executive Insights')).toBeInTheDocument();
    });

    it('renders all six insight components', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByTestId('executive-ai-brief')).toBeInTheDocument();
      expect(screen.getByTestId('priority-actions-roi')).toBeInTheDocument();
      expect(screen.getByTestId('predictive-alerts')).toBeInTheDocument();
      expect(screen.getByTestId('active-causal-chains')).toBeInTheDocument();
      expect(screen.getByTestId('experiment-recommendations')).toBeInTheDocument();
      expect(screen.getByTestId('heterogeneous-treatment-effects')).toBeInTheDocument();
    });
  });

  describe('Agents badge (honest count)', () => {
    it('renders the real available/total count when the health service reports it', () => {
      (useAgentHealth as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { available_count: 19, total_agents: 21, data_provenance: 'measured' },
      });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByText('19/21 Agents Active')).toBeInTheDocument();
    });

    it('omits the badge entirely when agent health has not loaded (no invented count)', () => {
      (useAgentHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.queryByText(/Agents Active/i)).not.toBeInTheDocument();
    });

    it('omits the badge for untrusted (placeholder) provenance — sample counts are not live counts (codex PR-4 round 4)', () => {
      (useAgentHealth as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { available_count: 19, total_agents: 21, data_provenance: 'placeholder' },
      });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.queryByText(/Agents Active/i)).not.toBeInTheDocument();
    });
  });

  describe('Context-driven brand (AC: no hard-coded brand)', () => {
    it('passes brand from E2ICopilot context to ExecutiveAIBrief', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
        filters: { brand: 'Fabhalta' },
      });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Fabhalta');
    });

    it('honors ?brand=... URL override over context brand', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
        filters: { brand: 'Fabhalta' },
      });
      render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights?brand=Kisqali'),
      });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Kisqali');
    });

    it('does NOT pass hard-coded "Remibrutinib" when context brand is something else', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
        filters: { brand: 'Kisqali' },
      });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Kisqali');
      expect(screen.getByTestId('executive-ai-brief')).not.toHaveTextContent('brand:Remibrutinib');
    });
  });

  describe('Moved-capability bridge (codex PR-4 round 7)', () => {
    it('points stale ?modelId= operator links at /monitoring instead of silently ignoring them', () => {
      render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights?modelId=chatbot_v2'),
      });
      // The per-model health & drift card moved to /monitoring with this
      // consolidation; a #304-era link must land the user one click from it.
      const link = screen.getByRole('link', { name: /View chatbot_v2 in Monitoring/i });
      expect(link).toHaveAttribute('href', '/monitoring?modelId=chatbot_v2');
    });

    it('shows no bridge note when the URL carries no modelId', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.queryByRole('note')).not.toBeInTheDocument();
    });
  });

  describe('Source-text forcing function (AC: no hard-coded literals in JSX)', () => {
    it('does not contain the stale brand/modelId literals or a hard-coded agent count', () => {
      const here = dirname(fileURLToPath(import.meta.url));
      const src = readFileSync(resolve(here, 'AIAgentInsights.tsx'), 'utf8');
      // Forbid the JSX-attribute forms called out in issue #304 ...
      expect(src).not.toMatch(/brand=["']Remibrutinib["']/);
      // ... and forbid the stale modelId literal anywhere in the page (the
      // modelId-consuming card left with the consolidation; the only modelId
      // read that remains is the bridge pointing stale links at /monitoring).
      expect(src).not.toMatch(/propensity_v2\.1\.0/);
      // The agents badge must be data-driven, never the old invented count.
      expect(src).not.toMatch(/21 Agents Active/);
    });
  });

  describe('Error boundaries (AC: one failing insight does not blank the page)', () => {
    // Each row pairs the insight component to make throw with the data-testids
    // of the surviving insights expected to still render.
    const INSIGHTS: ReadonlyArray<{
      throwing: string;
      survivors: ReadonlyArray<string>;
    }> = [
      {
        throwing: 'ExecutiveAIBrief',
        survivors: [
          'priority-actions-roi',
          'predictive-alerts',
          'active-causal-chains',
          'experiment-recommendations',
          'heterogeneous-treatment-effects',
        ],
      },
      {
        throwing: 'PriorityActionsROI',
        survivors: [
          'executive-ai-brief',
          'predictive-alerts',
          'active-causal-chains',
          'experiment-recommendations',
          'heterogeneous-treatment-effects',
        ],
      },
      {
        throwing: 'PredictiveAlerts',
        survivors: [
          'executive-ai-brief',
          'priority-actions-roi',
          'active-causal-chains',
          'experiment-recommendations',
          'heterogeneous-treatment-effects',
        ],
      },
      {
        throwing: 'ActiveCausalChains',
        survivors: [
          'executive-ai-brief',
          'priority-actions-roi',
          'predictive-alerts',
          'experiment-recommendations',
          'heterogeneous-treatment-effects',
        ],
      },
      {
        throwing: 'ExperimentRecommendations',
        survivors: [
          'executive-ai-brief',
          'priority-actions-roi',
          'predictive-alerts',
          'active-causal-chains',
          'heterogeneous-treatment-effects',
        ],
      },
      {
        throwing: 'HeterogeneousTreatmentEffects',
        survivors: [
          'executive-ai-brief',
          'priority-actions-roi',
          'predictive-alerts',
          'active-causal-chains',
          'experiment-recommendations',
        ],
      },
    ];

    /**
     * Build the six insight mocks where exactly one throws. Each healthy
     * mock renders a data-testid we can later assert on.
     */
    function buildInsightMocks(throwName: string) {
      const ALL_NAMES = [
        'ExecutiveAIBrief',
        'PriorityActionsROI',
        'PredictiveAlerts',
        'ActiveCausalChains',
        'ExperimentRecommendations',
        'HeterogeneousTreatmentEffects',
      ] as const;
      const NAME_TO_TESTID: Record<(typeof ALL_NAMES)[number], string> = {
        ExecutiveAIBrief: 'executive-ai-brief',
        PriorityActionsROI: 'priority-actions-roi',
        PredictiveAlerts: 'predictive-alerts',
        ActiveCausalChains: 'active-causal-chains',
        ExperimentRecommendations: 'experiment-recommendations',
        HeterogeneousTreatmentEffects: 'heterogeneous-treatment-effects',
      };
      const out: Record<string, () => ReactElement> = {};
      for (const n of ALL_NAMES) {
        if (n === throwName) {
          out[n] = () => {
            throw new Error(`boom from ${throwName}`);
          };
        } else {
          const testid = NAME_TO_TESTID[n];
          out[n] = () => <div data-testid={testid} />;
        }
      }
      return out;
    }

    for (const { throwing, survivors } of INSIGHTS) {
      it(`isolates a render error in ${throwing} from the other five insights`, async () => {
        const consoleErrorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});

        vi.resetModules();
        vi.doMock('@/components/insights', () => buildInsightMocks(throwing));
        vi.doMock('@/providers/E2ICopilotProvider', () => ({
          useE2ICopilot: () => ({ filters: { brand: 'Remibrutinib' } }),
        }));

        const { AIAgentInsights: Page } = await import('./AIAgentInsights');
        render(<Page />, { wrapper: createWrapperWithUrl('/ai-insights') });

        // Page header still renders.
        expect(screen.getByText('Executive Insights')).toBeInTheDocument();

        // All other insights mount.
        for (const testid of survivors) {
          expect(
            screen.getByTestId(testid),
            `expected survivor ${testid} to render when ${throwing} throws`,
          ).toBeInTheDocument();
        }

        // The throwing insight's testid is absent and the fallback is shown.
        expect(screen.getAllByText(/Something went wrong/i).length).toBeGreaterThan(0);

        consoleErrorSpy.mockRestore();
        vi.doUnmock('@/components/insights');
        vi.doUnmock('@/providers/E2ICopilotProvider');
      });
    }
  });

  describe('Brand selector (T3)', () => {
    it('renders a brand selector in the header', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByRole('combobox', { name: /brand/i })).toBeInTheDocument();
    });

    it('routes a chosen brand to BOTH the executive brief and priority actions', async () => {
      const user = userEvent.setup();
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      await user.click(screen.getByRole('combobox', { name: /brand/i }));
      await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Kisqali');
      expect(screen.getByTestId('priority-actions-roi')).toHaveTextContent('brand:Kisqali');
    });

    it('"All brands" hands undefined to the children (component default kicks in)', async () => {
      const user = userEvent.setup();
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      await user.click(screen.getByRole('combobox', { name: /brand/i }));
      await user.click(await screen.findByRole('option', { name: /all brands/i }));
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:__none__');
      expect(screen.getByTestId('priority-actions-roi')).toHaveTextContent('brand:__none__');
    });

    it('stays reactive to a post-mount context brand change (no snapshot regression)', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
        filters: { brand: 'Fabhalta' },
      });
      const { rerender } = render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights'),
      });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Fabhalta');

      // The dashboard filter context changes (global filter / copilot) while the
      // page is mounted and the user has NOT used the local selector -> the page
      // must follow it (the brand is derived, not snapshotted into state).
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
        filters: { brand: 'Kisqali' },
      });
      rerender(<AIAgentInsights />);
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Kisqali');
    });

    it('a local selection overrides the ?brand= URL value', async () => {
      const user = userEvent.setup();
      render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights?brand=Fabhalta'),
      });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Fabhalta');
      await user.click(screen.getByRole('combobox', { name: /brand/i }));
      await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:Kisqali');
    });

    it('falls back to "All brands" (undefined) when neither URL nor context set a brand', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({ filters: undefined });
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:__none__');
    });

    it('coerces an unknown ?brand= to "All brands" (never routes an invisible brand)', () => {
      (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({ filters: undefined });
      render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights?brand=NotARealBrand'),
      });
      // The unknown brand is NOT forwarded to the children's API/RAG calls.
      expect(screen.getByTestId('executive-ai-brief')).toHaveTextContent('brand:__none__');
      expect(screen.getByTestId('priority-actions-roi')).toHaveTextContent('brand:__none__');
    });
  });
});
