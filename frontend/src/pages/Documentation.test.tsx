/**
 * Documentation Page Tests
 * ========================
 * Page shell + interactive component behaviors (per the spec's testing
 * section, component behavior tests live here; content invariants live in
 * components/documentation/content.test.ts).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Documentation from './Documentation';

// The page's ONLY network hook. Mocked in every test; the "live chip" tests
// flip its return value.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
}));

import { useKPIList } from '@/hooks/api/use-kpi';

// jsdom has neither scrollIntoView nor IntersectionObserver.
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { kpis: [], total: 46 },
    isLoading: false,
    isError: false,
  });
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Documentation />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Documentation page shell', () => {
  it('renders the page header', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /understanding e2i/i })).toBeInTheDocument();
  });

  it('renders all four sections', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /^purpose/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^methodology/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^best practices/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^expected impact/i })).toBeInTheDocument();
  });

  it('renders the section nav and scrolls on click', async () => {
    renderPage();
    const nav = screen.getByRole('navigation', { name: /on this page/i });
    expect(nav).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^methodology$/i }));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});

describe('CausalScopeMap', () => {
  it('renders the three causal levels', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /hcp prescribing behavior/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /market & brand performance/i })).toBeInTheDocument();
  });

  it('defaults to the HCP level active on mount', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /hcp prescribing behavior/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByRole('button', { name: /market & brand performance/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByText(/rep detailing frequency/i)).toBeInTheDocument();
  });

  it('shows a level summary and its registry nodes on selection', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /patient journey outcomes/i }));
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByText(/treatment initiation, persistence, and discontinuation/i)).toBeInTheDocument();
    expect(screen.getByText('patient persistence')).toBeInTheDocument();
  });
});

describe('CorrelationCausationToggle', () => {
  it('starts on the correlation view, labeled illustrative', () => {
    renderPage();
    expect(screen.getByText(/calls correlate with trx/i)).toBeInTheDocument();
    expect(screen.getAllByText(/illustrative/i).length).toBeGreaterThan(0);
  });

  it('reveals the confounder on toggle', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /reveal the confounder/i }));
    expect(screen.getAllByText(/specialty drives both/i).length).toBeGreaterThan(0);
  });
});

describe('CapabilityIndex', () => {
  it('renders the five sidebar groups', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    for (const label of [
      'Causal Analytics',
      'Predictive Modeling',
      'Decisions & Optimization',
      'Data & Reference',
      'System & Platform',
    ]) {
      expect(within(index).getByRole('heading', { name: label })).toBeInTheDocument();
    }
  });

  it('links each card to its live page and excludes exempt/retired routes', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    expect(within(index).getByRole('link', { name: /segment analysis/i })).toHaveAttribute('href', '/segment-analysis');
    expect(within(index).queryByRole('link', { name: /causal discovery/i })).not.toBeInTheDocument();
    expect(within(index).queryByRole('link', { name: /^documentation$/i })).not.toBeInTheDocument();
  });

  it('shows causal-level badges on analysis surfaces', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    const twinCard = within(index).getByRole('link', { name: /digital twin/i }).closest('li');
    expect(twinCard).not.toBeNull();
    expect(within(twinCard as HTMLElement).getByText('HCP')).toBeInTheDocument();
  });
});

describe('CausalPipeline', () => {
  it('renders the five stages', () => {
    renderPage();
    for (const name of ['Frame', 'Identify', 'Estimate', 'Refute', 'Act']) {
      expect(screen.getByRole('button', { name: new RegExp(`^${name}`, 'i') })).toBeInTheDocument();
    }
  });

  it('expands a stage with plain language and a "For analysts" collapsible', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /^refute/i }));
    expect(screen.getByText(/attack the estimate before believing it/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /for analysts/i }));
    expect(screen.getByText(/placebo treatment/i)).toBeInTheDocument();
    expect(screen.getByText(/e-value/i)).toBeInTheDocument();
  });

  it('resets the "For analysts" layer when switching stages', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /^refute/i }));
    await userEvent.click(screen.getByRole('button', { name: /for analysts/i }));
    expect(screen.getByText(/placebo treatment/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^estimate/i }));
    expect(screen.getByRole('button', { name: /for analysts/i })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText(/econml/i)).not.toBeInTheDocument();
  });
});

describe('AgentTierStack', () => {
  it('renders all six tiers', () => {
    renderPage();
    for (const name of ['ML Foundation', 'Coordination', 'Causal Analytics', 'Monitoring', 'ML Predictions', 'Self-Improvement']) {
      expect(screen.getByRole('button', { name: new RegExp(name, 'i') })).toBeInTheDocument();
    }
  });

  it('expands a tier to list its agents', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /causal analytics.*3 agents/i }));
    expect(screen.getByText('causal_impact')).toBeInTheDocument();
    expect(screen.getByText('heterogeneous_optimizer')).toBeInTheDocument();
  });
});
