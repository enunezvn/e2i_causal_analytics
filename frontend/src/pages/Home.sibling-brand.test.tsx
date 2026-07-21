/**
 * Home Page — Sibling-Brand Badge + Hide Toggle Tests (Workstream A)
 * ==================================================================
 *
 * The 5 brand_specific KPIs are per-brand hard-bound (kpi_definitions.yaml:
 * BR-001/002 Remibrutinib, BR-003 Fabhalta, BR-004/005 Kisqali) but compute
 * portfolio-wide, so another brand's cards render under any brand-selector
 * value. The home-insights narrative deliberately references them tagged
 * "[sibling brand: X]" (src/insights/home_kpi.py) — so the grid fix is
 * visible-but-labeled, NOT a hard filter:
 *
 * - A1: the apiKPIs transform carries kpi.brand through (it used to drop it).
 * - A2: sibling-brand cards get a "sibling brand: {brand}" badge (same
 *       vocabulary as the narrative channel) — only under a selected brand.
 * - A3: a "hide other brands' KPIs" toggle scoped to the KPI grid, default
 *       OFF, persisted in localStorage.
 * - A4: tabs/counters stay coherent with the toggle on; the empty-state path
 *       renders (never a silent blank grid) if every computed KPI is hidden.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, within } from '@testing-library/react';
import { renderWithAllProviders } from '@/test/utils';
import Home from './Home';

// Same deterministic hook mocks as Home.test.tsx: the suite exercises the
// API-connected (live) branch via useKPIList + useBatchCalculateKPIs.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
  useKPIHealth: vi.fn(),
  useBatchCalculateKPIs: vi.fn(),
}));
vi.mock('@/hooks/api/use-health-score', () => ({
  useFullHealthCheck: vi.fn(),
}));
vi.mock('@/hooks/api/use-home-stats', () => ({
  useKpiSummary: vi.fn(),
  useActiveExperimentCount: vi.fn(),
}));
vi.mock('@/hooks/api/use-home-executive-insights', () => ({
  useHomeExecutiveInsights: vi.fn(),
}));
vi.mock('@/hooks/api/use-gaps', () => ({
  useOpportunities: vi.fn(),
}));
vi.mock('@/hooks/api/use-insights', () => ({
  useHomeKpiInsight: vi.fn(),
}));
vi.mock('@/hooks/api/use-monitoring', () => ({
  useAlerts: vi.fn(),
  useBrandModelSummary: vi.fn(),
}));
vi.mock('@/lib/api-client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api-client')>();
  return { ...actual, getValidated: vi.fn() };
});

import {
  useKPIList,
  useKPIHealth,
  useBatchCalculateKPIs,
} from '@/hooks/api/use-kpi';
import { useFullHealthCheck } from '@/hooks/api/use-health-score';
import { useKpiSummary, useActiveExperimentCount } from '@/hooks/api/use-home-stats';
import { useHomeExecutiveInsights } from '@/hooks/api/use-home-executive-insights';
import { useOpportunities } from '@/hooks/api/use-gaps';
import { useHomeKpiInsight } from '@/hooks/api/use-insights';
import { useAlerts, useBrandModelSummary } from '@/hooks/api/use-monitoring';
import { getValidated } from '@/lib/api-client';

const HIDE_SIBLING_KPIS_STORAGE_KEY = 'e2i-home-hide-sibling-kpis';

/** Reset all Home hooks to their honest "no data yet" defaults. */
function resetHomeHookDefaults() {
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  (useBatchCalculateKPIs as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    isError: false,
  });
  (useFullHealthCheck as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useKpiSummary as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useActiveExperimentCount as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  (useHomeExecutiveInsights as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useHomeKpiInsight as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    isPending: false,
    error: null,
  });
  (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useBrandModelSummary as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  (getValidated as ReturnType<typeof vi.fn>).mockResolvedValue({ agents: [], total: 0 });
}

interface LiveKpiFixture {
  id: string;
  name: string;
  workstream: string;
  /** Hard-bound brand (kpi_definitions.yaml) — absent for portfolio KPIs. */
  brand?: string;
  value: number;
  status?: string;
}

/** Put Home into live mode with KPI definitions (incl. brand) + batch values. */
function mockLiveKpis(kpis: LiveKpiFixture[]) {
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      kpis: kpis.map((k) => ({
        id: k.id,
        name: k.name,
        workstream: k.workstream,
        unit: undefined,
        definition: `${k.name} definition`,
        brand: k.brand ?? null,
      })),
      total: kpis.length,
    },
    isLoading: false,
    error: null,
  });
  (useBatchCalculateKPIs as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: vi.fn(),
    data: {
      results: kpis.map((k) => ({
        kpi_id: k.id,
        value: k.value,
        status: k.status ?? 'good',
        error: null,
        data_source: 'database',
      })),
    },
    isError: false,
  });
}

/** The three real brand-hard-bound KPIs, one per brand (values arbitrary). */
const BRAND_KPIS: LiveKpiFixture[] = [
  { id: 'BR-001', name: 'Remi - AH Uncontrolled %', workstream: 'brand_specific', brand: 'Remibrutinib', value: 42 },
  { id: 'BR-003', name: 'Fabhalta - % PNH Tested', workstream: 'brand_specific', brand: 'Fabhalta', value: 63 },
  { id: 'BR-004', name: 'Kisqali - Dx Adoption', workstream: 'brand_specific', brand: 'Kisqali', value: 21 },
];

/** Select a specific brand in the (first) brand combobox. */
async function selectBrand(name: string) {
  const brandSelector = screen.getAllByRole('combobox')[0];
  fireEvent.click(brandSelector);
  fireEvent.click(await screen.findByText(name));
}

/** Activate a KPI workstream tab (inactive TabsContent is unmounted). */
function activateTab(name: RegExp) {
  const tab = screen.getByRole('tab', { name });
  fireEvent.mouseDown(tab);
  fireEvent.click(tab);
}

/** The stat value rendered above the given brand-banner label. */
function bannerStat(label: string): string | null | undefined {
  const labelEl = screen.getByText(label);
  return labelEl.parentElement?.querySelector('.font-semibold')?.textContent;
}

describe('Home sibling-brand badge + hide toggle (Workstream A)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetHomeHookDefaults();
    localStorage.clear();
  });

  // ==========================================================================
  // A1 + A2 — badge on sibling-brand cards only, under a selected brand only
  // ==========================================================================

  it('shows no sibling-brand badge (and no hide toggle) under the All selector', () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);

    // All three brand cards render (visible-but-labeled semantic: nothing is
    // filtered), but under 'All' every brand is in scope — no badge, no toggle.
    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.getByText('Fabhalta - % PNH Tested')).toBeInTheDocument();
    expect(screen.getByText('Kisqali - Dx Adoption')).toBeInTheDocument();
    expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Hide other brands/)).not.toBeInTheDocument();
  });

  it('badges sibling-brand cards — and only those — under a selected brand', async () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    // Sibling cards stay visible AND get the narrative channel's vocabulary.
    expect(screen.getByText('sibling brand: Fabhalta')).toBeInTheDocument();
    expect(screen.getByText('sibling brand: Kisqali')).toBeInTheDocument();
    // The own-brand card is never badged.
    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.queryByText('sibling brand: Remibrutinib')).not.toBeInTheDocument();
  });

  it('does not badge brandless (portfolio) KPIs under a selected brand', async () => {
    mockLiveKpis([
      { id: 'WS3-BI-001', name: 'TRx Growth', workstream: 'ws3_business', value: 5.2 },
      ...BRAND_KPIS,
    ]);
    renderWithAllProviders(<Home />);
    await selectBrand('Fabhalta');

    // Business is the first (active) tab: the brandless KPI renders unbadged.
    expect(screen.getByText('TRx Growth')).toBeInTheDocument();
    expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();

    // The Brand tab carries the badges for the two non-Fabhalta cards.
    activateTab(/Brand$/);
    expect(screen.getByText('sibling brand: Remibrutinib')).toBeInTheDocument();
    expect(screen.getByText('sibling brand: Kisqali')).toBeInTheDocument();
    expect(screen.queryByText('sibling brand: Fabhalta')).not.toBeInTheDocument();
  });

  // ==========================================================================
  // A3 — hide toggle (default OFF, grid-scoped, persisted)
  // ==========================================================================

  it('hide toggle removes sibling-brand cards from grid + counts, and restores them', async () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    // Default OFF: everything visible-but-labeled.
    const toggle = screen.getByRole('checkbox', { name: /hide other brands/i });
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/Showing 3 of 3 defined KPIs/)).toBeInTheDocument();

    // ON: sibling cards leave the grid AND the shown-count together.
    fireEvent.click(toggle);
    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali - Dx Adoption')).not.toBeInTheDocument();
    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.getByText(/Showing 1 of 3 defined KPIs/)).toBeInTheDocument();

    // OFF again: sibling cards come back.
    fireEvent.click(toggle);
    expect(screen.getByText('Fabhalta - % PNH Tested')).toBeInTheDocument();
    expect(screen.getByText('Kisqali - Dx Adoption')).toBeInTheDocument();
    expect(screen.getByText(/Showing 3 of 3 defined KPIs/)).toBeInTheDocument();
  });

  it('persists the toggle choice to localStorage', async () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    const toggle = screen.getByRole('checkbox', { name: /hide other brands/i });
    fireEvent.click(toggle);
    expect(localStorage.getItem(HIDE_SIBLING_KPIS_STORAGE_KEY)).toBe('true');
    fireEvent.click(toggle);
    expect(localStorage.getItem(HIDE_SIBLING_KPIS_STORAGE_KEY)).toBe('false');
  });

  it('initializes hidden from a persisted preference', async () => {
    localStorage.setItem(HIDE_SIBLING_KPIS_STORAGE_KEY, 'true');
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    expect(screen.getByRole('checkbox', { name: /hide other brands/i })).toBeChecked();
    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
  });

  // ==========================================================================
  // A4 — tab derivation / counters / empty state with the toggle on
  // ==========================================================================

  it('brand banner counters count only visible cards with the toggle on', async () => {
    mockLiveKpis([
      { ...BRAND_KPIS[0], status: 'good' },
      { ...BRAND_KPIS[1], status: 'warning' },
      { ...BRAND_KPIS[2], status: 'good' },
    ]);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    // Toggle OFF: all three cards count (visible-but-labeled).
    expect(bannerStat('KPIs')).toBe('3');
    expect(bannerStat('On Track')).toBe('2');
    expect(bannerStat('Attention')).toBe('1');

    // Toggle ON: only the own-brand card remains counted.
    fireEvent.click(screen.getByRole('checkbox', { name: /hide other brands/i }));
    expect(bannerStat('KPIs')).toBe('1');
    expect(bannerStat('On Track')).toBe('1');
    expect(bannerStat('Attention')).toBe('0');
  });

  it('keeps tabs coherent with the toggle on: sibling-only workstreams drop out', async () => {
    mockLiveKpis([
      { id: 'WS3-BI-001', name: 'TRx Growth', workstream: 'ws3_business', value: 5.2 },
      ...BRAND_KPIS,
    ]);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    // Brand tab present while its own-brand card is visible.
    expect(screen.getByRole('tab', { name: /Brand$/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: /hide other brands/i }));
    // Remibrutinib keeps its own-brand card, so the Brand tab survives...
    expect(screen.getByRole('tab', { name: /Brand$/ })).toBeInTheDocument();
    activateTab(/Brand$/);
    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
  });

  it('renders an honest empty state (never a silent blank grid) when the toggle hides every computed KPI', async () => {
    // Pathological guard: a brand with NO own-brand computed card. Real data
    // keeps >=1 own-brand card per brand, but the empty-state path must hold.
    mockLiveKpis([BRAND_KPIS[1], BRAND_KPIS[2]]); // Fabhalta + Kisqali only
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    const toggle = screen.getByRole('checkbox', { name: /hide other brands/i });
    fireEvent.click(toggle);

    // Everything visible is gone — the grid must say so, not go blank...
    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali - Dx Adoption')).not.toBeInTheDocument();
    expect(screen.getByText('No KPIs available')).toBeInTheDocument();
    // ...the Brand tab drops out of the tablist...
    const tablist = screen.getByRole('tablist');
    expect(within(tablist).queryByText('Brand')).not.toBeInTheDocument();
    // ...and the toggle stays rendered so the user can turn the cards back on.
    expect(screen.getByRole('checkbox', { name: /hide other brands/i })).toBeChecked();
  });

  // ==========================================================================
  // Demo Mode (API offline) parity — codex HIGH: the demo-mode KPICard call
  // must apply the SAME badge/toggle logic as the live path. Each per-brand
  // SAMPLE_KPIS set carries one sibling-brand exemplar for this.
  // ==========================================================================

  describe('Demo Mode (API offline) parity', () => {
    beforeEach(() => {
      (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('API offline'),
      });
    });

    it('badges the sibling-brand sample card under a selected brand (parity with live path)', async () => {
      renderWithAllProviders(<Home />);

      // Demo mode is announced, and under 'All' nothing is badged.
      expect(screen.getByText('API Offline (using sample data)')).toBeInTheDocument();
      expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();

      await selectBrand('Remibrutinib');

      // The Remibrutinib demo set's Fabhalta exemplar renders badged; the
      // own-brand cards stay unbadged — identical logic to the live path.
      expect(screen.getByText('Fabhalta TRx')).toBeInTheDocument();
      expect(screen.getByText('sibling brand: Fabhalta')).toBeInTheDocument();
      expect(screen.queryByText('sibling brand: Remibrutinib')).not.toBeInTheDocument();
    });

    it('hide toggle removes the sibling-brand sample card via the same visible set', async () => {
      renderWithAllProviders(<Home />);
      await selectBrand('Remibrutinib');

      const toggle = screen.getByRole('checkbox', { name: /hide other brands/i });
      expect(toggle).not.toBeChecked();

      fireEvent.click(toggle);
      expect(screen.queryByText('Fabhalta TRx')).not.toBeInTheDocument();
      // Own-brand demo cards remain.
      expect(screen.getByText('TRx')).toBeInTheDocument();

      fireEvent.click(toggle);
      expect(screen.getByText('Fabhalta TRx')).toBeInTheDocument();
    });
  });
});
