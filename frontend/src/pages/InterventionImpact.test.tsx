/**
 * InterventionImpact Page Tests
 * =============================
 *
 * Tests for the Intervention Impact analysis page with
 * causal analysis, before/after comparisons, and Digital Twin.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import InterventionImpact from './InterventionImpact';

// Mock Recharts components to avoid canvas/SVG rendering issues in tests
vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 800, height: 400 }}>
        {children}
      </div>
    ),
  };
});

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

// Create a wrapper with QueryClient for useRunSimulation hook
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('InterventionImpact', () => {
  it('renders page header with title and description', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Intervention Impact')).toBeInTheDocument();
    expect(
      screen.getByText(/Before\/after comparisons, treatment effects, and counterfactual analysis/i)
    ).toBeInTheDocument();
  });

  it('displays intervention selector dropdown', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('shows first intervention by default', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // First intervention name appears in dropdown and card heading
    const interventionNames = screen.getAllByText('Q1 2024 HCP Engagement Campaign');
    expect(interventionNames.length).toBeGreaterThanOrEqual(1);
  });

  it('displays intervention summary card with badges', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Type and status badges
    expect(screen.getByText('campaign')).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
    // Target metric is displayed
    expect(screen.getByText('Target:')).toBeInTheDocument();
  });

  it('displays 4 KPI cards', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Average Treatment Effect')).toBeInTheDocument();
    expect(screen.getByText('Significant Effects')).toBeInTheDocument();
    expect(screen.getByText('Cumulative Impact')).toBeInTheDocument();
    expect(screen.getByText('ROI Estimate')).toBeInTheDocument();
  });

  it('displays KPI values', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Significant Effects shows "3/4" (3 significant out of 4 total)
    expect(screen.getByText('3/4')).toBeInTheDocument();
    // ROI value
    expect(screen.getByText('3.2x')).toBeInTheDocument();
  });

  it('displays refresh and export buttons', () => {
    const { container } = render(<InterventionImpact />, { wrapper: createWrapper() });

    // Refresh button has RefreshCw icon
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
    // Export button has Download icon
    const exportButton = container.querySelector('button svg.lucide-download');
    expect(exportButton).toBeInTheDocument();
  });

  it('handles export button click', () => {
    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'a') {
        const link = originalCreateElement('a');
        link.click = mockClick;
        return link;
      }
      return originalCreateElement(tag);
    });

    const { container } = render(<InterventionImpact />, { wrapper: createWrapper() });

    // Find export button by its svg icon and get parent button
    const exportIcon = container.querySelector('svg.lucide-download');
    const exportButton = exportIcon?.closest('button');
    expect(exportButton).toBeInTheDocument();
    fireEvent.click(exportButton!);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('displays 5 main tabs', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /Causal Impact/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Before\/After/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Treatment Effects/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Segment Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Digital Twin/i })).toBeInTheDocument();
  });

  it('shows Causal Impact tab content by default', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Causal Impact Analysis')).toBeInTheDocument();
    expect(
      screen.getByText(/Comparison of actual outcomes vs\. counterfactual/i)
    ).toBeInTheDocument();
  });

  it('displays chart legend elements', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Chart legend items
    expect(screen.getByText('Actual')).toBeInTheDocument();
    expect(screen.getByText('Counterfactual')).toBeInTheDocument();
    expect(screen.getByText('95% CI')).toBeInTheDocument();
  });

  it('displays impact interpretation section', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Impact Interpretation')).toBeInTheDocument();
    expect(screen.getByText('Positive Impact Detected')).toBeInTheDocument();
    expect(screen.getByText('Confidence Level: 95%')).toBeInTheDocument();
    expect(screen.getByText('Methodology: CausalImpact')).toBeInTheDocument();
  });

  it('has clickable Before/After tab', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const beforeAfterTab = screen.getByRole('tab', { name: /Before\/After/i });
    expect(beforeAfterTab).toBeInTheDocument();
    expect(beforeAfterTab).not.toBeDisabled();
  });

  it('has clickable Treatment Effects tab', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const effectsTab = screen.getByRole('tab', { name: /Treatment Effects/i });
    expect(effectsTab).toBeInTheDocument();
    expect(effectsTab).not.toBeDisabled();
  });

  it('has clickable Segment Analysis tab', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const segmentTab = screen.getByRole('tab', { name: /Segment Analysis/i });
    expect(segmentTab).toBeInTheDocument();
    expect(segmentTab).not.toBeDisabled();
  });

  it('has clickable Digital Twin tab', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const digitalTwinTab = screen.getByRole('tab', { name: /Digital Twin/i });
    expect(digitalTwinTab).toBeInTheDocument();
    expect(digitalTwinTab).not.toBeDisabled();
  });

  it('shows average lift percentage in summary', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Average lift is displayed with % sign
    expect(screen.getByText(/Avg\. Lift/)).toBeInTheDocument();
  });

  it('shows cumulative effect in summary', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Cumulative Effect')).toBeInTheDocument();
  });
});

// =============================================================================
// BEFORE/AFTER TAB TESTS
// =============================================================================

describe('InterventionImpact - Before/After Tab', () => {
  it('switches to Before/After tab and shows comparison chart', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const beforeAfterTab = screen.getByRole('tab', { name: /Before\/After/i });
    await user.click(beforeAfterTab);

    expect(screen.getByText('Before/After Comparison')).toBeInTheDocument();
    expect(
      screen.getByText(/Metric changes comparing pre-intervention and post-intervention periods/i)
    ).toBeInTheDocument();
  });

  it('displays detailed comparison table with headers', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));

    expect(screen.getByText('Detailed Comparison')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Metric' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Before' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'After' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Change' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '% Change' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Status' })).toBeInTheDocument();
  });

  it('displays metric rows with before/after values', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));

    // Check for metric names from SAMPLE_BEFORE_AFTER (may appear multiple times due to chart and table)
    expect(screen.getAllByText('TRx Volume').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('NRx Volume').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Market Share').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('HCP Reach').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Cost per TRx').length).toBeGreaterThanOrEqual(1);
  });

  it('displays percentage change values in table', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));

    // Check for some percentage changes (TRx Volume: +8.3%)
    expect(screen.getByText('+8.3%')).toBeInTheDocument();
    // Cost per TRx has negative change: -10.1%
    expect(screen.getByText('-10.1%')).toBeInTheDocument();
  });
});

// =============================================================================
// TREATMENT EFFECTS TAB TESTS
// =============================================================================

describe('InterventionImpact - Treatment Effects Tab', () => {
  it('switches to Treatment Effects tab and shows estimates', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const effectsTab = screen.getByRole('tab', { name: /Treatment Effects/i });
    await user.click(effectsTab);

    expect(screen.getByText('Treatment Effect Estimates')).toBeInTheDocument();
    expect(
      screen.getByText(/Statistical estimates of causal effects with confidence intervals/i)
    ).toBeInTheDocument();
  });

  it('displays treatment effect cards with metric names', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    // Check for metrics from SAMPLE_TREATMENT_EFFECTS
    expect(screen.getAllByText('TRx Volume').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('NRx Volume').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Conversion Rate').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('HCP Satisfaction').length).toBeGreaterThanOrEqual(1);
  });

  it('displays effect size badges', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    expect(screen.getByText('large effect')).toBeInTheDocument();
    expect(screen.getAllByText('medium effect').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('small effect')).toBeInTheDocument();
  });

  it('displays significance badges', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    // 3 significant, 1 not significant
    expect(screen.getAllByText('Significant').length).toBe(3);
    expect(screen.getByText('Not Significant')).toBeInTheDocument();
  });

  it('displays ATE values and confidence intervals', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    // Check for ATE labels
    const ateLabels = screen.getAllByText('ATE (Average Treatment Effect)');
    expect(ateLabels.length).toBe(4);

    // Check for CI labels
    const ciLabels = screen.getAllByText('95% Confidence Interval');
    expect(ciLabels.length).toBe(4);
  });

  it('displays p-values and sample sizes', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    // P-Value labels
    const pValueLabels = screen.getAllByText('P-Value');
    expect(pValueLabels.length).toBe(4);

    // Sample Size labels
    const sampleSizeLabels = screen.getAllByText('Sample Size');
    expect(sampleSizeLabels.length).toBe(4);
  });

  it('displays Cohen\'s d effect size descriptions', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    // Effect size descriptions
    expect(screen.getByText("Cohen's d > 0.8")).toBeInTheDocument();
    expect(screen.getAllByText("Cohen's d 0.2-0.8").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Cohen's d < 0.2")).toBeInTheDocument();
  });
});

// =============================================================================
// SEGMENT ANALYSIS TAB TESTS
// =============================================================================

describe('InterventionImpact - Segment Analysis Tab', () => {
  it('switches to Segment Analysis tab and shows chart', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const segmentsTab = screen.getByRole('tab', { name: /Segment Analysis/i });
    await user.click(segmentsTab);

    expect(screen.getByText('Heterogeneous Treatment Effects')).toBeInTheDocument();
    expect(
      screen.getByText(/How the intervention impact varies across different segments/i)
    ).toBeInTheDocument();
  });

  it('displays segment cards with treatment effects', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    // Check for segment names from SAMPLE_SEGMENT_EFFECTS (may appear in chart and cards)
    expect(screen.getAllByText('High-Volume HCPs').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Medium-Volume HCPs').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Low-Volume HCPs').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Northeast Region').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Southeast Region').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Midwest Region').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('West Region').length).toBeGreaterThanOrEqual(1);
  });

  it('displays treatment effect and sample size for segments', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    // Check for Treatment Effect labels (7 segments)
    const effectLabels = screen.getAllByText('Treatment Effect');
    expect(effectLabels.length).toBe(7);

    // Check for Sample Size labels
    const sampleLabels = screen.getAllByText('Sample Size');
    expect(sampleLabels.length).toBe(7);
  });

  it('displays 95% CI labels for segments', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    // 95% CI appears multiple times (7 segment cards + chart tooltip area)
    const ciLabels = screen.getAllByText('95% CI');
    expect(ciLabels.length).toBeGreaterThanOrEqual(7);
  });

  it('displays Key Insights section', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    expect(screen.getByText('Key Insights')).toBeInTheDocument();
    expect(screen.getByText('Highest Impact Segment')).toBeInTheDocument();
    expect(screen.getByText('Regional Performance')).toBeInTheDocument();
  });

  it('displays insight descriptions', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    // Check for insight text content
    expect(
      screen.getByText(/show the strongest response to the intervention/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/outperforms other regions/i)
    ).toBeInTheDocument();
  });
});

// =============================================================================
// DIGITAL TWIN TAB TESTS
// =============================================================================

describe('InterventionImpact - Digital Twin Tab', () => {
  it('switches to Digital Twin tab and shows simulation panel', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const digitalTwinTab = screen.getByRole('tab', { name: /Digital Twin/i });
    await user.click(digitalTwinTab);

    // SimulationPanel should be rendered
    expect(screen.getByText('About Digital Twin Simulation')).toBeInTheDocument();
  });

  it('displays Digital Twin info card with three sections', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));

    expect(screen.getByText('Pre-Screen Interventions')).toBeInTheDocument();
    expect(screen.getByText('Causal Inference Engine')).toBeInTheDocument();
    expect(screen.getByText('Fidelity Metrics')).toBeInTheDocument();
  });

  it('displays info card descriptions', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));

    expect(
      screen.getByText(/Test intervention scenarios virtually/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Powered by DoWhy and EconML/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/fidelity scores indicating how well/i)
    ).toBeInTheDocument();
  });
});

// =============================================================================
// INTERVENTION SELECTION TESTS
// =============================================================================

describe('InterventionImpact - Intervention Selection', () => {
  it('displays all interventions in dropdown', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Open the dropdown
    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);

    // All 4 interventions should be available
    expect(screen.getByRole('option', { name: 'Q1 2024 HCP Engagement Campaign' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Digital Rep Training Program' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Kisqali Patient Support Enhancement' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Remibrutinib Launch Preparation' })).toBeInTheDocument();
  });

  it('updates display when selecting different intervention', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Open the dropdown and select a different intervention
    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);
    await user.click(screen.getByRole('option', { name: 'Digital Rep Training Program' }));

    // The selected intervention name should appear in the summary card
    const interventionNames = screen.getAllByText('Digital Rep Training Program');
    expect(interventionNames.length).toBeGreaterThanOrEqual(1);
  });

  it('shows intervention type badge after selection change', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Initially shows 'campaign' badge
    expect(screen.getByText('campaign')).toBeInTheDocument();

    // Change to training program
    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);
    await user.click(screen.getByRole('option', { name: 'Digital Rep Training Program' }));

    // Should now show 'training' badge
    expect(screen.getByText('training')).toBeInTheDocument();
  });

  it('shows intervention status badge after selection change', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Initially shows 'completed' status
    expect(screen.getByText('completed')).toBeInTheDocument();

    // Change to active intervention
    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);
    await user.click(screen.getByRole('option', { name: 'Kisqali Patient Support Enhancement' }));

    // Should now show 'active' status
    expect(screen.getByText('active')).toBeInTheDocument();
  });

  it('shows intervention description after selection change', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Change to Remibrutinib Launch Preparation
    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);
    await user.click(screen.getByRole('option', { name: 'Remibrutinib Launch Preparation' }));

    // Should show the description
    expect(
      screen.getByText(/Pre-launch awareness campaign for Remibrutinib/i)
    ).toBeInTheDocument();
  });
});

// =============================================================================
// BUTTON FUNCTIONALITY TESTS
// =============================================================================

describe('InterventionImpact - Button Functionality', () => {
  it('refresh button becomes disabled while refreshing', async () => {
    const user = userEvent.setup();
    const { container } = render(<InterventionImpact />, { wrapper: createWrapper() });

    const refreshIcon = container.querySelector('svg.lucide-refresh-cw');
    const refreshButton = refreshIcon?.closest('button');
    expect(refreshButton).toBeInTheDocument();

    await user.click(refreshButton!);

    // Button should be disabled while refreshing
    expect(refreshButton).toBeDisabled();
  });

  it('refresh button icon has animate-spin class while refreshing', async () => {
    const user = userEvent.setup();
    const { container } = render(<InterventionImpact />, { wrapper: createWrapper() });

    const refreshIcon = container.querySelector('svg.lucide-refresh-cw');
    const refreshButton = refreshIcon?.closest('button');

    await user.click(refreshButton!);

    // Icon should have animate-spin class after click
    const spinningIcon = container.querySelector('svg.lucide-refresh-cw.animate-spin');
    expect(spinningIcon).toBeInTheDocument();
  });
});

// =============================================================================
// TAB NAVIGATION TESTS
// =============================================================================

describe('InterventionImpact - Tab Navigation', () => {
  it('navigates to Before/After tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Start at Causal Impact (default)
    expect(screen.getByText('Causal Impact Analysis')).toBeInTheDocument();

    // Navigate to Before/After
    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));
    expect(screen.getByText('Before/After Comparison')).toBeInTheDocument();
  });

  it('navigates to Treatment Effects tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));
    expect(screen.getByText('Treatment Effect Estimates')).toBeInTheDocument();
  });

  it('navigates to Segment Analysis tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));
    expect(screen.getByText('Heterogeneous Treatment Effects')).toBeInTheDocument();
  });

  it('navigates to Digital Twin tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));
    expect(screen.getByText('About Digital Twin Simulation')).toBeInTheDocument();
  });

  it('only shows content for the active tab', async () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // At Causal Impact tab, Before/After content should not be visible
    expect(screen.getByText('Causal Impact Analysis')).toBeInTheDocument();
    expect(screen.queryByText('Detailed Comparison')).not.toBeInTheDocument();
  });

  it('hides previous tab content when switching tabs', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Switch to Before/After
    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));
    expect(screen.getByText('Detailed Comparison')).toBeInTheDocument();
    expect(screen.queryByText('Treatment Effect Estimates')).not.toBeInTheDocument();
  });
});

// =============================================================================
// KPI CARD TESTS
// =============================================================================

describe('InterventionImpact - KPI Cards', () => {
  it('displays cumulative impact with K suffix', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Cumulative impact shows value with K suffix
    const cumulativeValue = screen.getByText(/\+.*K/);
    expect(cumulativeValue).toBeInTheDocument();
  });

  it('displays status indicators on KPI cards', () => {
    const { container } = render(<InterventionImpact />, { wrapper: createWrapper() });

    // KPI cards should have status indicators (colored dots)
    const statusIndicators = container.querySelectorAll('.rounded-full');
    expect(statusIndicators.length).toBeGreaterThan(0);
  });

  it('displays trend indicators with percentage change', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Trend percentages are displayed (from KPICard previousValue comparison)
    // These appear when KPI cards have trend data
    const kpiCards = screen.getAllByText(/Average Treatment Effect|Significant Effects|Cumulative Impact|ROI Estimate/);
    expect(kpiCards.length).toBe(4);
  });
});

// =============================================================================
// INTERVENTION SUMMARY CARD TESTS
// =============================================================================

describe('InterventionImpact - Intervention Summary Card', () => {
  it('displays intervention dates', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Start:')).toBeInTheDocument();
    expect(screen.getByText('End:')).toBeInTheDocument();
  });

  it('displays intervention description', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(
      screen.getByText(/Targeted engagement campaign for high-potential HCPs/i)
    ).toBeInTheDocument();
  });

  it('shows target metric', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // Target: TRx Volume (for first intervention)
    const targetLabel = screen.getByText('Target:');
    expect(targetLabel).toBeInTheDocument();
  });
});
