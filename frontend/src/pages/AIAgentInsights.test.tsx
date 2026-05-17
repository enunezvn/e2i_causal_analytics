/**
 * AIAgentInsights Page Tests
 * ==========================
 *
 * Tests for the AI Agent Insights composite page (issue #304):
 * - brand + modelId are driven from context / URL, NOT hard-coded
 * - error boundary wraps each insight component (one failing insight
 *   does not blank the page)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import type { ReactNode } from 'react';

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
  PriorityActionsROI: () => <div data-testid="priority-actions-roi" />,
  PredictiveAlerts: () => <div data-testid="predictive-alerts" />,
  ActiveCausalChains: () => <div data-testid="active-causal-chains" />,
  ExperimentRecommendations: () => <div data-testid="experiment-recommendations" />,
  HeterogeneousTreatmentEffects: () => <div data-testid="heterogeneous-treatment-effects" />,
  SystemHealthScore: ({ modelId }: { modelId?: string }) => (
    <div data-testid="system-health-score">modelId:{modelId ?? '__none__'}</div>
  ),
}));

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
  });

  describe('Page rendering', () => {
    it('renders page header', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByText('AI Agent Insights')).toBeInTheDocument();
    });

    it('renders all five live insight components', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      expect(screen.getByTestId('executive-ai-brief')).toBeInTheDocument();
      expect(screen.getByTestId('predictive-alerts')).toBeInTheDocument();
      expect(screen.getByTestId('active-causal-chains')).toBeInTheDocument();
      expect(screen.getByTestId('heterogeneous-treatment-effects')).toBeInTheDocument();
      expect(screen.getByTestId('system-health-score')).toBeInTheDocument();
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

  describe('URL-driven modelId (AC: no hard-coded modelId)', () => {
    it('passes modelId from ?modelId=... URL query', () => {
      render(<AIAgentInsights />, {
        wrapper: createWrapperWithUrl('/ai-insights?modelId=churn_v3.0.0'),
      });
      expect(screen.getByTestId('system-health-score')).toHaveTextContent('modelId:churn_v3.0.0');
    });

    it('falls back to a non-empty default when no URL param is set', () => {
      render(<AIAgentInsights />, { wrapper: createWrapperWithUrl('/ai-insights') });
      // Issue #304's gripe is that the literal lived inline in JSX — not that the
      // value itself is wrong. A single-source-of-truth default constant is fine.
      const node = screen.getByTestId('system-health-score');
      expect(node.textContent).toMatch(/^modelId:.+/);
      expect(node.textContent).not.toBe('modelId:__none__');
    });
  });

  describe('Source-text forcing function (AC: no hard-coded literals in JSX)', () => {
    it('does not contain hard-coded brand or modelId attributes in the JSX', () => {
      const here = dirname(fileURLToPath(import.meta.url));
      const src = readFileSync(resolve(here, 'AIAgentInsights.tsx'), 'utf8');
      // The two literal-as-JSX-attribute forms called out in issue #304.
      // The bare strings may legitimately appear as constants — what we
      // forbid is them appearing inside an attribute assignment on JSX.
      expect(src).not.toMatch(/brand=["']Remibrutinib["']/);
      expect(src).not.toMatch(/modelId=["']propensity_v2\.1\.0["']/);
    });
  });

  describe('Error boundaries (AC: one failing insight does not blank the page)', () => {
    it('renders other insights when one insight throws during render', async () => {
      // Suppress noisy React error-boundary console output for this test.
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // Re-mock insights so ExecutiveAIBrief throws.
      vi.doMock('@/components/insights', () => ({
        ExecutiveAIBrief: () => {
          throw new Error('boom from ExecutiveAIBrief');
        },
        PriorityActionsROI: () => <div data-testid="priority-actions-roi" />,
        PredictiveAlerts: () => <div data-testid="predictive-alerts" />,
        ActiveCausalChains: () => <div data-testid="active-causal-chains" />,
        ExperimentRecommendations: () => <div data-testid="experiment-recommendations" />,
        HeterogeneousTreatmentEffects: () => <div data-testid="heterogeneous-treatment-effects" />,
        SystemHealthScore: ({ modelId }: { modelId?: string }) => (
          <div data-testid="system-health-score">modelId:{modelId ?? '__none__'}</div>
        ),
      }));

      // Re-import the page module so it picks up the new doMock.
      vi.resetModules();
      // Re-mock provider after resetModules so the import resolves correctly.
      vi.doMock('@/providers/E2ICopilotProvider', () => ({
        useE2ICopilot: () => ({ filters: { brand: 'Remibrutinib' } }),
      }));
      const { AIAgentInsights: Page } = await import('./AIAgentInsights');

      render(<Page />, { wrapper: createWrapperWithUrl('/ai-insights') });

      // Page header still renders.
      expect(screen.getByText('AI Agent Insights')).toBeInTheDocument();

      // The 4 healthy insights still mount.
      expect(screen.getByTestId('predictive-alerts')).toBeInTheDocument();
      expect(screen.getByTestId('active-causal-chains')).toBeInTheDocument();
      expect(screen.getByTestId('heterogeneous-treatment-effects')).toBeInTheDocument();
      expect(screen.getByTestId('system-health-score')).toBeInTheDocument();

      // The throwing insight is replaced with the boundary fallback.
      expect(screen.queryByTestId('executive-ai-brief')).not.toBeInTheDocument();
      expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

      consoleErrorSpy.mockRestore();
      vi.doUnmock('@/components/insights');
      vi.doUnmock('@/providers/E2ICopilotProvider');
    });
  });
});
