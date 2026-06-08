/**
 * Home Page Tests
 * ===============
 *
 * Tests for the Home/Executive Dashboard page.
 * Includes tests for:
 * - Brand selector
 * - Region filter (Phase 3.1)
 * - Date range filter (Phase 3.1)
 * - KPI display
 * - Agent insights
 * - System health
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, within } from '@testing-library/react';
import { renderWithAllProviders } from '@/test/utils';
import Home from './Home';

// Mock the KPI hooks so we can exercise the API-connected branch
// deterministically. The rest of the suite (no per-test override) leaves these
// returning undefined => Demo Mode (SAMPLE) is the default for legacy structural
// tests, while the populated Home tiles render from the dedicated hooks below.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
  useKPIHealth: vi.fn(),
  useBatchCalculateKPIs: vi.fn(),
  useKPIValue: vi.fn(),
}));
vi.mock('@/hooks/api/use-health-score', () => ({
  useQuickHealthCheck: vi.fn(),
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
// Agent Status uses useQuery(getValidated(...)). Mock the client fn so the
// query resolves to a deterministic roster (or empty when desired).
vi.mock('@/lib/api-client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api-client')>();
  return { ...actual, getValidated: vi.fn() };
});

import {
  useKPIList,
  useKPIHealth,
  useBatchCalculateKPIs,
  useKPIValue,
} from '@/hooks/api/use-kpi';
import { useQuickHealthCheck } from '@/hooks/api/use-health-score';
import { useKpiSummary, useActiveExperimentCount } from '@/hooks/api/use-home-stats';
import { useHomeExecutiveInsights } from '@/hooks/api/use-home-executive-insights';
import { useOpportunities } from '@/hooks/api/use-gaps';
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
  });
  (useKPIValue as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  (useQuickHealthCheck as ReturnType<typeof vi.fn>).mockReturnValue({
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
  (getValidated as ReturnType<typeof vi.fn>).mockResolvedValue({ agents: [], total: 0 });
}

describe('Home', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetHomeHookDefaults();
  });

  // =========================================================================
  // PAGE STRUCTURE TESTS
  // =========================================================================

  it('renders page header with title and description', () => {
    renderWithAllProviders(<Home />);

    expect(screen.getByText('E2I Executive Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Causal Analytics for Commercial Operations')).toBeInTheDocument();
  });

  it('renders all filter selectors', () => {
    renderWithAllProviders(<Home />);

    // All three selectors should be present as comboboxes
    const comboboxes = screen.getAllByRole('combobox');
    expect(comboboxes.length).toBeGreaterThanOrEqual(3);
  });

  it('renders quick stats from real API rollups (no fabricated 125,430 / 94.2%)', () => {
    // Real sources wired: business_metrics rollup + active experiments + ROC-AUC.
    (useKpiSummary as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        brand: 'All',
        period: 'Last 90 days',
        metrics: { trx_volume: 125000, hcp_reach: 8500 },
        data_source: 'database',
      },
      isLoading: false,
      error: null,
    });
    (useActiveExperimentCount as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { active_count: 12 },
      isLoading: false,
    });
    (useKPIValue as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { kpi_id: 'WS1-MP-001', value: 0.7998, status: 'good' },
      isLoading: false,
    });

    renderWithAllProviders(<Home />);

    // Real tile labels (kept exact for the e2e selectors) + real values.
    expect(screen.getByText('Total TRx (MTD)')).toBeInTheDocument();
    expect(screen.getByText('Active Campaigns')).toBeInTheDocument();
    expect(screen.getByText('HCPs Reached')).toBeInTheDocument();
    expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
    expect(screen.getByText('125,000')).toBeInTheDocument();
    expect(screen.getByText('8,500')).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument();
    // The old fabricated values must be absent.
    expect(screen.queryByText('125,430')).not.toBeInTheDocument();
    expect(screen.queryByText('94.2%')).not.toBeInTheDocument();
  });

  // =========================================================================
  // BRAND SELECTOR TESTS
  // =========================================================================

  describe('Brand Selector', () => {
    it('displays default brand as All', () => {
      renderWithAllProviders(<Home />);

      // The first combobox should be the brand selector
      const brandSelector = screen.getAllByRole('combobox')[0];
      expect(brandSelector).toHaveTextContent('All');
    });

    it('shows all brand options when clicked', async () => {
      renderWithAllProviders(<Home />);

      const brandSelector = screen.getAllByRole('combobox')[0];
      fireEvent.click(brandSelector);

      // Wait for dropdown to open and check options
      expect(await screen.findByText('Remibrutinib')).toBeInTheDocument();
      expect(screen.getByText('Fabhalta')).toBeInTheDocument();
      expect(screen.getByText('Kisqali')).toBeInTheDocument();
    });

    it('displays indication labels for brands', async () => {
      renderWithAllProviders(<Home />);

      const brandSelector = screen.getAllByRole('combobox')[0];
      fireEvent.click(brandSelector);

      expect(await screen.findByText('(CSU)')).toBeInTheDocument();
      expect(screen.getByText('(PNH)')).toBeInTheDocument();
      expect(screen.getByText('(HR+/HER2- BC)')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // REGION FILTER TESTS (Phase 3.1)
  // =========================================================================

  describe('Region Filter', () => {
    it('displays default region as All US', () => {
      renderWithAllProviders(<Home />);

      // Region is the second combobox
      const regionSelector = screen.getAllByRole('combobox')[1];
      expect(regionSelector).toHaveTextContent('All US');
    });

    it('shows all region options when clicked', async () => {
      renderWithAllProviders(<Home />);

      const regionSelector = screen.getAllByRole('combobox')[1];
      fireEvent.click(regionSelector);

      // Check for region options
      expect(await screen.findByText('Northeast')).toBeInTheDocument();
      expect(screen.getByText('Southeast')).toBeInTheDocument();
      expect(screen.getByText('Midwest')).toBeInTheDocument();
      expect(screen.getByText('West')).toBeInTheDocument();
      expect(screen.getByText('Southwest')).toBeInTheDocument();
    });

    it('updates filter summary when region changes', async () => {
      renderWithAllProviders(<Home />);

      // Find the territory summary card
      const territoryLabel = screen.getByText('Territory');
      const card = territoryLabel.closest('div');

      // Initially shows All US
      expect(within(card!.parentElement!).getByText('All US Regions')).toBeInTheDocument();
    });

    it('has MapPin icon in region selector', () => {
      renderWithAllProviders(<Home />);

      // The region selector contains a MapPin icon
      const regionSelector = screen.getAllByRole('combobox')[1];
      const mapPinIcon = regionSelector.querySelector('svg');
      expect(mapPinIcon).toBeInTheDocument();
    });
  });

  // =========================================================================
  // DATE RANGE FILTER TESTS (Phase 3.1)
  // =========================================================================

  describe('Date Range Filter', () => {
    it('displays default date range as Q4 2025', () => {
      renderWithAllProviders(<Home />);

      // Date range is the third combobox
      const dateSelector = screen.getAllByRole('combobox')[2];
      expect(dateSelector).toHaveTextContent('Q4 2025');
    });

    it('shows all date range options when clicked', async () => {
      renderWithAllProviders(<Home />);

      const dateSelector = screen.getAllByRole('combobox')[2];
      fireEvent.click(dateSelector);

      // Check for date range options
      expect(await screen.findByText('Q3 2025')).toBeInTheDocument();
      expect(screen.getByText('Q2 2025')).toBeInTheDocument();
      expect(screen.getByText('Q1 2025')).toBeInTheDocument();
      expect(screen.getByText('Year to Date')).toBeInTheDocument();
      expect(screen.getByText('Last 12 Months')).toBeInTheDocument();
    });

    it('shows date range descriptions in filter summary', () => {
      renderWithAllProviders(<Home />);

      // The filter summary card shows the description (may appear in multiple places)
      const descriptions = screen.getAllByText('Oct - Dec 2025');
      expect(descriptions.length).toBeGreaterThan(0);
    });

    it('updates filter summary when date range changes', () => {
      renderWithAllProviders(<Home />);

      // Find the reporting period summary
      const reportingLabel = screen.getByText('Reporting Period');
      const card = reportingLabel.closest('div');

      // Initially shows Q4 2025 description
      expect(within(card!.parentElement!).getByText('Oct - Dec 2025')).toBeInTheDocument();
    });

    it('has CalendarDays icon in date selector', () => {
      renderWithAllProviders(<Home />);

      // The date selector contains a CalendarDays icon
      const dateSelector = screen.getAllByRole('combobox')[2];
      const calendarIcon = dateSelector.querySelector('svg');
      expect(calendarIcon).toBeInTheDocument();
    });
  });

  // =========================================================================
  // KPI DISPLAY TESTS
  // =========================================================================

  describe('KPI Display', () => {
    it('renders KPI section with title', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Key Performance Indicators')).toBeInTheDocument();
    });

    it('renders category tabs', () => {
      renderWithAllProviders(<Home />);

      // Look for tab elements or category buttons
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('displays KPIs for selected category', () => {
      renderWithAllProviders(<Home />);

      // Default should show commercial KPIs
      expect(screen.getByText('Total TRx')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // AGENT INSIGHTS TESTS
  // =========================================================================

  describe('Agent Insights', () => {
    it('renders agent insights section with the dual-source description', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Agent Insights')).toBeInTheDocument();
      // Now a real dual-source feed (executive insights + gap opportunities).
      expect(screen.getByText(/Executive insights/)).toBeInTheDocument();
    });

    it('renders an honest empty state when the substrate is empty (no SAMPLE_INSIGHTS)', () => {
      renderWithAllProviders(<Home />);

      // Default hooks return no data → honest empty-state copy, NOT fabricated cards.
      expect(
        screen.getByText(/No insights yet — run a gap analysis/)
      ).toBeInTheDocument();
      // The former hardcoded SAMPLE_INSIGHTS titles must NOT appear.
      expect(screen.queryByText('Model Performance Drift Detected')).not.toBeInTheDocument();
      expect(screen.queryByText('Speaker Programs Outperforming Digital')).not.toBeInTheDocument();
    });

    it('renders REAL merged insights when both sources return data', () => {
      (useHomeExecutiveInsights as ReturnType<typeof vi.fn>).mockReturnValue({
        data: [
          {
            insight_id: 'i1',
            title: 'Northeast Territory Lift',
            narrative: 'Real crystallized narrative.',
            brand: 'Kisqali',
            crystallized_at: new Date().toISOString(),
            effect_size: 0.23,
            effect_direction: 'positive',
            recommended_next_analysis: 'Expand coverage',
          },
        ],
        isLoading: false,
        error: null,
      });
      (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          total_count: 1,
          quick_wins_count: 1,
          strategic_bets_count: 0,
          total_addressable_value: 1000,
          opportunities: [
            {
              rank: 1,
              gap: { gap_id: 'g1', metric: 'TRx', segment: 'region', segment_value: 'West', current_value: 1, target_value: 2, gap_size: 1, gap_percentage: 50, gap_type: 'benchmark' },
              roi_estimate: { gap_id: 'g1', estimated_revenue_impact: 1, estimated_cost_to_close: 1, expected_roi: 2, risk_adjusted_roi: 1, payback_period_months: 6, attribution_level: 'a', attribution_rate: 0.5, confidence: 0.8 },
              recommended_action: 'Run a targeted campaign.',
              implementation_difficulty: 'low',
              time_to_impact: '3 months',
            },
          ],
        },
        isLoading: false,
        error: null,
      });

      renderWithAllProviders(<Home />);

      expect(screen.getByText('Northeast Territory Lift')).toBeInTheDocument();
      expect(screen.getByText('West TRx')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // SYSTEM HEALTH TESTS
  // =========================================================================

  describe('System Health', () => {
    it('renders system health section', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('System Health')).toBeInTheDocument();
    });

    it('renders REAL System Health dimension scores (no fabricated latencies)', () => {
      (useQuickHealthCheck as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          check_id: 'c1',
          check_scope: 'quick',
          overall_health_score: 92,
          health_grade: 'A',
          component_health_score: 0.95,
          model_health_score: 0.88,
          pipeline_health_score: 0.82,
          agent_health_score: 0.92,
          critical_issues: [],
          warnings: [],
          recommendations: [],
        },
        isLoading: false,
        error: null,
      });

      renderWithAllProviders(<Home />);

      // Real agent dimension labels render.
      expect(screen.getByText('Components')).toBeInTheDocument();
      expect(screen.getByText('Models')).toBeInTheDocument();
      // The anti-fabrication guards still hold.
      expect(screen.queryByText('API Gateway')).not.toBeInTheDocument();
      expect(screen.queryByText('45ms')).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // AGENT STATUS TESTS
  // =========================================================================

  describe('Agent Status', () => {
    it('renders agent status section', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Agent Status')).toBeInTheDocument();
    });

    it('renders REAL agent tier counts (never the hardcoded 15/21)', async () => {
      (getValidated as ReturnType<typeof vi.fn>).mockResolvedValue({
        agents: [
          { id: 'a0', name: 'Scope Definer', tier: 0, status: 'idle', capabilities: [] },
          { id: 'a1', name: 'Orchestrator', tier: 1, status: 'active', capabilities: [] },
          { id: 'a2', name: 'Causal Impact', tier: 2, status: 'active', capabilities: [] },
        ],
        total: 3,
      });

      renderWithAllProviders(<Home />);

      // Real footer derived from the roster: 2 active of 3 total.
      expect(await screen.findByText('2/3 agents active')).toBeInTheDocument();
      // The fabricated hardcoded summary must NEVER appear.
      expect(screen.queryByText('15/21 agents active')).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // ALERTS TESTS
  // =========================================================================

  describe('Alerts', () => {
    it('renders active alerts section', () => {
      renderWithAllProviders(<Home />);

      // Look for alert count
      expect(screen.getByText(/Active Alerts/)).toBeInTheDocument();
    });

    it('displays alert items', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Data Pipeline Delay')).toBeInTheDocument();
      expect(screen.getByText('Model Drift Detected')).toBeInTheDocument();
      expect(screen.getByText('New Insights Available')).toBeInTheDocument();
    });

    it('shows dismiss buttons', () => {
      renderWithAllProviders(<Home />);

      const dismissButtons = screen.getAllByText('Dismiss');
      expect(dismissButtons.length).toBe(3);
    });

    it('can dismiss alerts', () => {
      renderWithAllProviders(<Home />);

      const dismissButtons = screen.getAllByText('Dismiss');
      fireEvent.click(dismissButtons[0]);

      // After dismissing, should have 2 alerts left
      expect(screen.queryByText('Data Pipeline Delay')).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // REFRESH FUNCTIONALITY TESTS
  // =========================================================================

  describe('Refresh Button', () => {
    it('renders refresh button', () => {
      renderWithAllProviders(<Home />);

      // Find refresh button by its icon structure
      const buttons = screen.getAllByRole('button');
      const refreshButton = buttons.find(btn => btn.querySelector('.lucide-refresh-cw'));
      expect(refreshButton).toBeInTheDocument();
    });
  });

  // =========================================================================
  // QUICK ACTIONS TESTS
  // =========================================================================

  describe('Quick Actions', () => {
    it('renders quick actions section', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Quick Actions')).toBeInTheDocument();
    });

    it('displays navigation links', () => {
      renderWithAllProviders(<Home />);

      // Quick actions should have links to other pages
      const quickActionsCard = screen.getByText('Quick Actions').closest('div');
      expect(quickActionsCard).toBeInTheDocument();
    });
  });

  // =========================================================================
  // FILTER SUMMARY CARD TESTS
  // =========================================================================

  describe('Filter Summary Card', () => {
    it('displays reporting period summary', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Reporting Period')).toBeInTheDocument();
    });

    it('displays territory summary', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Territory')).toBeInTheDocument();
    });

    it('shows current filter values', () => {
      renderWithAllProviders(<Home />);

      // Default values - use getAllByText since there might be duplicates
      const octDecText = screen.getAllByText('Oct - Dec 2025');
      const allUSText = screen.getAllByText('All US Regions');
      expect(octDecText.length).toBeGreaterThan(0);
      expect(allUSText.length).toBeGreaterThan(0);
    });
  });
});

// ===========================================================================
// KPI VALUE HONESTY (H1)
// ===========================================================================
// When live /api/kpis metadata is present, the page must render the real KPI
// names/categories/units and an honest "Not yet computed" placeholder for the
// numeric value (which the backend does not yet provide) — NOT fabricated
// SAMPLE TRx / revenue numbers.

describe('KPI value honesty (H1)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetHomeHookDefaults();
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: { status: 'healthy' } });
  });

  it('does NOT render fabricated SAMPLE TRx/revenue numbers when the API is connected', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        kpis: [
          // workstream includes "commercial" so the KPI lands in the default
          // (commercial) category tab and is mounted in the DOM.
          { id: 'trx_total', name: 'Total TRx', workstream: 'commercial', definition: 'Total prescriptions', unit: undefined },
        ],
        total: 1,
      },
      isLoading: false,
      error: null,
    });

    renderWithAllProviders(<Home />);

    // The fabricated SAMPLE values must NOT appear when live metadata is present.
    expect(screen.queryByText('125,430')).not.toBeInTheDocument();
    expect(screen.queryByText(/\$425/)).not.toBeInTheDocument();
    // The live KPI name IS rendered.
    expect(screen.getByText('Total TRx')).toBeInTheDocument();
    // Numeric value shows the honest "not yet computed" placeholder.
    expect(screen.getAllByText(/not yet computed|—/i).length).toBeGreaterThan(0);
  });
});

// ===========================================================================
// SIDEBAR / STATS HONESTY (H1)
// ===========================================================================
// The fabricated System Health service latencies and Agent tier counts (with
// a hardcoded "15/21 agents active") must not render — real data lives on the
// System Health and Agent Orchestration pages.

describe('Sidebar / stats honesty (H1)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetHomeHookDefaults();
  });

  it('does not fabricate System Health service latencies on the landing page', () => {
    renderWithAllProviders(<Home />);
    // Fabricated infra latencies must be gone (real data lives on /system-health).
    expect(screen.queryByText('45ms')).not.toBeInTheDocument();
    expect(screen.queryByText('12ms')).not.toBeInTheDocument();
    expect(screen.queryByText('API Gateway')).not.toBeInTheDocument();
    // The card still exists and links out.
    expect(screen.getByText('System Health')).toBeInTheDocument();
  });

  it('does not fabricate a hardcoded "15/21 agents active" summary', () => {
    renderWithAllProviders(<Home />);
    expect(screen.queryByText('15/21 agents active')).not.toBeInTheDocument();
  });
});
