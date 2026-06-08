/**
 * InterventionImpact Page Tests
 * =============================
 *
 * Tests for the Intervention Impact analysis page.
 *
 * F-002 note: SAMPLE_IMPACT_DATA / SAMPLE_TREATMENT_EFFECTS / SAMPLE_BEFORE_AFTER
 * / SAMPLE_SEGMENT_EFFECTS were removed from production rendering paths. Those
 * sections now render explicit empty states until the corresponding analysis
 * API hook is wired. Tests below assert the empty-state shape, not the former
 * fabricated values.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
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
      screen.getByText(/Before\/after comparisons, treatment effects, and counterfactual analysis/i),
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

    expect(screen.getByText('campaign')).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
  });

  it('displays 5 main tabs', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /Causal Impact/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Before\/After/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Treatment Effects/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Segment Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Digital Twin/i })).toBeInTheDocument();
  });
});

describe('InterventionImpact - F-002 empty states', () => {
  it('renders empty state on Causal Impact tab when no API data', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // F-002: causal impact chart now sourced from API; in its absence,
    // empty state is rendered (not fabricated counterfactual data).
    expect(
      screen.getByText(/No causal impact data available/),
    ).toBeInTheDocument();
    expect(screen.queryByText('Positive Impact Detected')).not.toBeInTheDocument();
  });

  it('renders empty state on Before/After tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));

    expect(
      screen.getByText(/No before\/after data available/),
    ).toBeInTheDocument();
    // Detailed comparison table (fabricated rows) is not present.
    expect(screen.queryByText('Detailed Comparison')).not.toBeInTheDocument();
    expect(screen.queryByText('Cost per TRx')).not.toBeInTheDocument();
  });

  it('renders empty state on Treatment Effects tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    expect(
      screen.getByText(/No treatment effect estimates available/),
    ).toBeInTheDocument();
    // Fabricated effect-size labels not present.
    expect(screen.queryByText('large effect')).not.toBeInTheDocument();
  });

  it('renders empty state on Segment Analysis tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    expect(
      screen.getByText(/No segment heterogeneity data available/),
    ).toBeInTheDocument();
    // Fabricated segments not present.
    expect(screen.queryByText('High-Volume HCPs')).not.toBeInTheDocument();
    expect(screen.queryByText('Northeast Region')).not.toBeInTheDocument();
  });

  it('renders top-level empty state for KPI section when no analysis data', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    // The KPI block now collapses to empty state when summaryMetrics is null.
    expect(
      screen.getByText(/No analysis data for this intervention/),
    ).toBeInTheDocument();
    // Fabricated KPI strings ("3/4", "3.2x", "+8.3%") must not be present.
    expect(screen.queryByText('3/4')).not.toBeInTheDocument();
    expect(screen.queryByText('3.2x')).not.toBeInTheDocument();
    expect(screen.queryByText('+8.3%')).not.toBeInTheDocument();
  });
});

describe('InterventionImpact - Digital Twin Tab', () => {
  it('navigates to Digital Twin tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));
    expect(screen.getByText('About Digital Twin Simulation')).toBeInTheDocument();
  });
});

describe('InterventionImpact - Intervention Selection', () => {
  it('displays all interventions in dropdown', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);

    expect(screen.getByRole('option', { name: 'Q1 2024 HCP Engagement Campaign' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Digital Rep Training Program' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Kisqali Patient Support Enhancement' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Remibrutinib Launch Preparation' })).toBeInTheDocument();
  });

  it('updates display when selecting different intervention', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    const dropdown = screen.getByRole('combobox');
    await user.click(dropdown);
    await user.click(screen.getByRole('option', { name: 'Digital Rep Training Program' }));

    const interventionNames = screen.getAllByText('Digital Rep Training Program');
    expect(interventionNames.length).toBeGreaterThanOrEqual(1);
  });

  // M4 regression guard: pins the honest state established by F-002 so a future
  // edit re-introducing fabricated SAMPLE_* analysis fixtures fails CI.
  it('source contains no SAMPLE_* analysis fixtures (M4 regression guard)', () => {
    const src = fs.readFileSync(
      path.join(process.cwd(), 'src/pages/InterventionImpact.tsx'),
      'utf8',
    );
    // The four analysis fixtures must stay empty-init; no fabricated SAMPLE_ analysis arrays.
    expect(src).not.toMatch(/SAMPLE_IMPACT_DATA|SAMPLE_TREATMENT_EFFECTS|SAMPLE_BEFORE_AFTER|SAMPLE_SEGMENT_EFFECTS/);
    // The legitimate static INTERVENTIONS selector catalog IS allowed to remain.
    expect(src).toMatch(/const INTERVENTIONS: Intervention\[\] = \[/);
  });
});
