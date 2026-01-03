/**
 * InterventionImpact Page Tests
 * =============================
 *
 * Tests for the Intervention Impact analysis page with
 * causal analysis, before/after comparisons, and Digital Twin.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import InterventionImpact from './InterventionImpact';

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
