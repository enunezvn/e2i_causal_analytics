/**
 * Home Page — Automatic Brand Scoping of the KPI Grid
 * ===================================================
 *
 * The 5 brand_specific KPIs are per-brand hard-bound (kpi_definitions.yaml:
 * BR-001/002 Remibrutinib, BR-003 Fabhalta, BR-004/005 Kisqali) but compute
 * portfolio-wide. Frontend review 2026-07-22: the brand selector itself is the
 * filter — under a selected brand, other brands' cards are scoped out of the
 * grid AUTOMATICALLY (no badge, no "hide other brands" toggle; supersedes the
 * earlier visible-but-labeled semantic). Under 'All' every brand's cards are
 * first-class. The backend insight grounding (src/insights/home_kpi.py)
 * applies the identical scope so narrative and grid stay coherent.
 *
 * - S1: sibling-brand cards leave the grid, tabs, and counts together under a
 *       selected brand; own-brand and portfolio (brandless) cards remain.
 * - S2: under 'All' nothing is filtered and nothing is badged.
 * - S3: the "Showing N of M" denominator counts only in-scope definitions.
 * - S4: honest empty state (never a silent blank grid) when every computed
 *       KPI belongs to another brand.
 * - S5: Demo Mode (API offline) applies the same scoping to SAMPLE_KPIS.
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

describe('Home automatic brand scoping of the KPI grid', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetHomeHookDefaults();
    localStorage.clear();
  });

  // ==========================================================================
  // S2 — 'All' scope: nothing filtered, nothing badged, no toggle anywhere
  // ==========================================================================

  it("shows every brand's cards first-class under the All selector", () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);

    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.getByText('Fabhalta - % PNH Tested')).toBeInTheDocument();
    expect(screen.getByText('Kisqali - Dx Adoption')).toBeInTheDocument();
    expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Hide other brands/)).not.toBeInTheDocument();
    expect(screen.getByText(/Showing 3 of 3 defined KPIs/)).toBeInTheDocument();
  });

  // ==========================================================================
  // S1 — selected brand: other brands' cards leave grid + counts automatically
  // ==========================================================================

  it("scopes other brands' cards out of the grid automatically under a selected brand", async () => {
    mockLiveKpis(BRAND_KPIS);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    expect(screen.getByText('Remi - AH Uncontrolled %')).toBeInTheDocument();
    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali - Dx Adoption')).not.toBeInTheDocument();
    // No residue of the superseded visible-but-labeled semantic.
    expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Hide other brands/)).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: /hide other brands/i })).not.toBeInTheDocument();
  });

  it('keeps brandless (portfolio) KPIs visible under a selected brand', async () => {
    mockLiveKpis([
      { id: 'WS3-BI-001', name: 'TRx Growth', workstream: 'ws3_business', value: 5.2 },
      ...BRAND_KPIS,
    ]);
    renderWithAllProviders(<Home />);
    await selectBrand('Fabhalta');

    // Business is the first (active) tab: the brandless KPI renders.
    expect(screen.getByText('TRx Growth')).toBeInTheDocument();

    // The Brand tab holds only the own-brand card.
    activateTab(/Brand$/);
    expect(screen.getByText('Fabhalta - % PNH Tested')).toBeInTheDocument();
    expect(screen.queryByText('Remi - AH Uncontrolled %')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali - Dx Adoption')).not.toBeInTheDocument();
  });

  it('brand banner counters count only in-scope cards', async () => {
    mockLiveKpis([
      { ...BRAND_KPIS[0], status: 'good' },
      { ...BRAND_KPIS[1], status: 'warning' },
      { ...BRAND_KPIS[2], status: 'good' },
    ]);
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    // Selected brand: only the own-brand card is counted.
    expect(bannerStat('KPIs')).toBe('1');
    expect(bannerStat('On Track')).toBe('1');
    expect(bannerStat('Attention')).toBe('0');
  });

  it('drops sibling-only workstreams from the tab list under a selected brand', async () => {
    mockLiveKpis([
      { id: 'WS3-BI-001', name: 'TRx Growth', workstream: 'ws3_business', value: 5.2 },
      { id: 'BR-003', name: 'Fabhalta - % PNH Tested', workstream: 'brand_specific', brand: 'Fabhalta', value: 63 },
    ]);
    renderWithAllProviders(<Home />);

    // 'All': the Brand tab is present.
    expect(screen.getByRole('tab', { name: /Brand$/ })).toBeInTheDocument();

    await selectBrand('Remibrutinib');
    // Remibrutinib has no own-brand card here → the Brand tab drops out with it.
    const tablist = screen.getByRole('tablist');
    expect(within(tablist).queryByText('Brand')).not.toBeInTheDocument();
    expect(screen.getByText('TRx Growth')).toBeInTheDocument();
  });

  // ==========================================================================
  // S3 — "Showing N of M": denominator counts only in-scope definitions
  // ==========================================================================

  it('scopes the "Showing N of M" denominator to the selected brand', async () => {
    mockLiveKpis([
      { id: 'WS3-BI-001', name: 'TRx Growth', workstream: 'ws3_business', value: 5.2 },
      ...BRAND_KPIS,
    ]);
    renderWithAllProviders(<Home />);
    expect(screen.getByText(/Showing 4 of 4 defined KPIs/)).toBeInTheDocument();

    await selectBrand('Remibrutinib');
    // 2 in scope (portfolio + own-brand); the other 2 brands' definitions
    // must not inflate the denominator.
    expect(screen.getByText(/Showing 2 of 2 defined KPIs for Remibrutinib/)).toBeInTheDocument();
  });

  // ==========================================================================
  // S4 — honest empty state when the scope holds only other brands' cards
  // ==========================================================================

  it('renders an honest empty state (never a silent blank grid) when every computed KPI is out of scope', async () => {
    // Pathological guard: a brand with NO own-brand computed card. Real data
    // keeps >=1 own-brand card per brand, but the empty-state path must hold.
    mockLiveKpis([BRAND_KPIS[1], BRAND_KPIS[2]]); // Fabhalta + Kisqali only
    renderWithAllProviders(<Home />);
    await selectBrand('Remibrutinib');

    expect(screen.queryByText('Fabhalta - % PNH Tested')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali - Dx Adoption')).not.toBeInTheDocument();
    expect(screen.getByText('No KPIs available')).toBeInTheDocument();
    expect(
      screen.getByText(/belongs to another brand — 2 scoped out by the Remibrutinib selection/)
    ).toBeInTheDocument();
    // ...and the Brand tab drops out of the tablist with the cards.
    const tablist = screen.getByRole('tablist');
    expect(within(tablist).queryByText('Brand')).not.toBeInTheDocument();
  });

  // ==========================================================================
  // S5 — Demo Mode (API offline) parity: SAMPLE_KPIS get the same scoping
  // ==========================================================================

  describe('Demo Mode (API offline) parity', () => {
    beforeEach(() => {
      (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('API offline'),
      });
    });

    it('filters the sibling-brand sample card under a selected brand (parity with live path)', async () => {
      renderWithAllProviders(<Home />);

      // Demo mode is announced, and under 'All' nothing is filtered.
      expect(screen.getByText('API Offline (using sample data)')).toBeInTheDocument();

      await selectBrand('Remibrutinib');

      // The Remibrutinib demo set's Fabhalta exemplar is scoped out; the
      // own-brand cards remain — identical logic to the live path.
      expect(screen.queryByText('Fabhalta TRx')).not.toBeInTheDocument();
      expect(screen.getByText('TRx')).toBeInTheDocument();
      expect(screen.queryByText(/sibling brand:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Hide other brands/)).not.toBeInTheDocument();
    });
  });
});
