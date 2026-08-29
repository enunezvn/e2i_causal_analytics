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

  it('renders all five sections', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /^purpose/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^causal impact/i })).toBeInTheDocument();
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

  it('starts fully collapsed, closes the open tier when another opens, and toggles closed on re-click', async () => {
    renderPage();

    // All six tiers start collapsed
    for (const name of [
      /ml foundation/i,
      /coordination/i,
      /causal analytics/i,
      /monitoring/i,
      /ml predictions/i,
      /self-improvement/i,
    ]) {
      expect(screen.getByRole('button', { name })).toHaveAttribute('aria-expanded', 'false');
    }

    // Opening Monitoring, then Coordination: Monitoring must close (mutual exclusivity)
    const monitoring = screen.getByRole('button', { name: /monitoring/i });
    const coordination = screen.getByRole('button', { name: /coordination/i });
    await userEvent.click(monitoring);
    expect(monitoring).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(coordination);
    expect(coordination).toHaveAttribute('aria-expanded', 'true');
    expect(monitoring).toHaveAttribute('aria-expanded', 'false');

    // Clicking the open tier again closes it (toggle)
    await userEvent.click(coordination);
    expect(coordination).toHaveAttribute('aria-expanded', 'false');
  });
});

describe('ClinicalGrounding', () => {
  it('renders all five clinical sources with UMLS and OpenFDA present', () => {
    renderPage();
    const strip = screen.getByRole('region', { name: /grounded in clinical reality/i });
    for (const name of ['UMLS', 'OpenFDA', 'ClinicalTrials.gov', 'PubMed', 'ChEMBL']) {
      expect(within(strip).getByText(name)).toBeInTheDocument();
    }
  });
});

describe('PracticeCards', () => {
  it('renders do/don’t pairs', () => {
    renderPage();
    expect(screen.getByText(/what-if simulation inputs/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^all$/i })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText(/check the refutation gate/i)).toBeInTheDocument();
    expect(screen.getByText(/rerun an analysis with different settings/i)).toBeInTheDocument();
  });

  it('filters by role', async () => {
    renderPage();
    // whatif-ranges is analyst-only; informational-kpis is exec-only.
    await userEvent.click(screen.getByRole('button', { name: /^exec$/i }));
    expect(screen.getByRole('button', { name: /^exec$/i })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: /^all$/i })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.queryByText(/what-if simulation inputs/i)).not.toBeInTheDocument();
    expect(screen.getByText(/informational.*kpis as if they were performance targets/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^all$/i }));
    expect(screen.getByText(/what-if simulation inputs/i)).toBeInTheDocument();
  });
});

describe('ImpactPathways', () => {
  it('renders four pathway cards linking to live pages', () => {
    renderPage();
    const region = screen.getByRole('region', { name: /expected impact pathways/i });
    expect(within(region).getByRole('link', { name: /see your segments/i })).toHaveAttribute('href', '/segment-analysis');
    expect(within(region).getByRole('link', { name: /see your allocation/i })).toHaveAttribute('href', '/resource-optimization');
    expect(within(region).getByRole('link', { name: /run a simulation/i })).toHaveAttribute('href', '/digital-twin');
    expect(within(region).getByRole('link', { name: /open the dashboard/i })).toHaveAttribute('href', '/');
  });
});

describe('CausalVariableTypes', () => {
  it('renders the four color-coded variable types with their definitions', () => {
    renderPage();
    const region = screen.getByRole('region', { name: /four types of causal variables/i });
    // <dt> has role "term", which takes no name from content — query by text.
    for (const term of ['Treatment', 'Mediator', 'Outcome', 'Confounder']) {
      expect(within(region).getByText(term, { selector: 'dt' })).toBeInTheDocument();
    }
    expect(within(region).getByText(/opening a back-door path/i)).toBeInTheDocument();
    expect(within(region).getByText(/transmits part of the effect/i)).toBeInTheDocument();
  });
});

describe('CausalImpactDag', () => {
  it('renders the DAG labeled illustrative, with "All paths" pressed by default', () => {
    renderPage();
    const figure = screen.getByRole('figure', { name: /acceptance .* action .* revenue impact/i });
    expect(within(figure).getByText(/illustrative example/i)).toBeInTheDocument();
    expect(within(figure).getByRole('button', { name: /^all paths$/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(figure.querySelectorAll('[data-edge]').length).toBe(18);
    expect(figure.querySelectorAll('[data-edge][data-selected="true"]').length).toBe(0);
  });

  it('highlights exactly the edges of the chosen path and toggles back to all paths', async () => {
    renderPage();
    const figure = screen.getByRole('figure', { name: /acceptance .* action .* revenue impact/i });
    const confounders = within(figure).getByRole('button', { name: /backdoor confounders/i });
    await userEvent.click(confounders);
    expect(confounders).toHaveAttribute('aria-pressed', 'true');
    expect(within(figure).getByRole('button', { name: /^all paths$/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    const selected = figure.querySelectorAll('[data-edge][data-selected="true"]');
    expect(selected.length).toBe(4);
    for (const el of selected) expect(el.getAttribute('data-group')).toBe('confounders');
    // Clicking the pressed path again returns to the unfiltered view.
    await userEvent.click(confounders);
    expect(within(figure).getByRole('button', { name: /^all paths$/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(figure.querySelectorAll('[data-edge][data-selected="true"]').length).toBe(0);
  });
});

describe('live KPI chip degradation', () => {
  it('shows the governed-KPIs chip when the query succeeds', () => {
    renderPage();
    expect(screen.getByText('46')).toBeInTheDocument();
    expect(screen.getByText('governed KPIs')).toBeInTheDocument();
  });

  it('silently omits the chip on error — no error UI', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });
    renderPage();
    expect(screen.queryByText('governed KPIs')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    // Static chips unaffected:
    expect(screen.getByText('intervention channels')).toBeInTheDocument();
  });
});
