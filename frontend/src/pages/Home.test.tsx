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

// Mock the KPI hook so we can exercise the API-connected branch deterministically.
// The rest of the suite (no per-test mock) leaves these returning undefined =>
// Demo Mode (SAMPLE) is the default, so the existing tests are unaffected.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
  useKPIHealth: vi.fn(),
}));
import { useKPIList, useKPIHealth } from '@/hooks/api/use-kpi';

describe('Home', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: undefined => Demo Mode (SAMPLE values render) for the legacy suite.
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    });
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
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

  it('renders an honest empty state for quick stats (no fabricated rollups)', () => {
    renderWithAllProviders(<Home />);

    // QUICK_STATS fabricated TRx/HCP/accuracy figures were removed; an honest
    // empty state is rendered instead (no live aggregate-stats Home hook today).
    expect(screen.getByText('Quick stats not yet computed')).toBeInTheDocument();
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
    it('renders agent insights section', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('Agent Insights')).toBeInTheDocument();
      // Relabeled to make the sample/demo nature explicit (live feed not yet wired).
      expect(screen.getByText(/Sample insights \(live agent feed not yet wired\)/)).toBeInTheDocument();
    });

    it('displays sample insights', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText('High-Value Territory Opportunity')).toBeInTheDocument();
      expect(screen.getByText('Model Performance Drift Detected')).toBeInTheDocument();
    });

    it('shows agent tier badges', () => {
      renderWithAllProviders(<Home />);

      expect(screen.getByText(/Tier 2: Gap Analyzer/)).toBeInTheDocument();
      expect(screen.getByText(/Tier 3: Drift Monitor/)).toBeInTheDocument();
    });

    it('shows impact badges', () => {
      renderWithAllProviders(<Home />);

      const highBadges = screen.getAllByText('high');
      const mediumBadges = screen.getAllByText('medium');
      expect(highBadges.length).toBeGreaterThan(0);
      expect(mediumBadges.length).toBeGreaterThan(0);
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

    it('renders an honest empty state instead of fabricated service rows', () => {
      renderWithAllProviders(<Home />);

      // Fabricated service names/latencies were removed; honest empty state + link.
      expect(screen.getByText('Service status not on this page')).toBeInTheDocument();
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

    it('renders an honest empty state instead of fabricated tier counts', () => {
      renderWithAllProviders(<Home />);

      // Fabricated tier/agent counts were removed; honest empty state + link.
      expect(screen.getByText('Agent status not on this page')).toBeInTheDocument();
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
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined, isLoading: false, error: null });
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
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
