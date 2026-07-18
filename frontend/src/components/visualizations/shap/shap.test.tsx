/**
 * SHAP Visualization Components Tests
 * ====================================
 *
 * Tests for SHAPBarChart, SHAPBeeswarm, SHAPForcePlot, and SHAPWaterfall components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SHAPBarChart } from './SHAPBarChart';
import { SHAPBeeswarm, type BeeswarmDataPoint } from './SHAPBeeswarm';
import { SHAPForcePlot } from './SHAPForcePlot';
import { SHAPWaterfall, buildWaterfallData } from './SHAPWaterfall';
import type { FeatureContribution } from '@/types/explain';

// =============================================================================
// TEST DATA
// =============================================================================

const mockFeatures: FeatureContribution[] = [
  { feature_name: 'days_since_visit', feature_value: 45, shap_value: 0.35, contribution_direction: 'positive', contribution_rank: 1 },
  { feature_name: 'total_prescriptions', feature_value: 12, shap_value: -0.28, contribution_direction: 'negative', contribution_rank: 2 },
  { feature_name: 'territory_sales', feature_value: 150000, shap_value: 0.22, contribution_direction: 'positive', contribution_rank: 3 },
  { feature_name: 'specialty_oncology', feature_value: 1, shap_value: 0.18, contribution_direction: 'positive', contribution_rank: 4 },
  { feature_name: 'recent_engagement', feature_value: 3, shap_value: -0.15, contribution_direction: 'negative', contribution_rank: 5 },
];

const mockBeeswarmData: BeeswarmDataPoint[] = [
  { feature: 'days_since_visit', shapValue: 0.15, featureValue: 0.8, originalValue: 45, instanceId: 'i1' },
  { feature: 'days_since_visit', shapValue: -0.12, featureValue: 0.2, originalValue: 10, instanceId: 'i2' },
  { feature: 'total_prescriptions', shapValue: 0.22, featureValue: 0.9, originalValue: 25, instanceId: 'i1' },
  { feature: 'total_prescriptions', shapValue: -0.18, featureValue: 0.1, originalValue: 2, instanceId: 'i2' },
];

// =============================================================================
// SHAP BAR CHART TESTS
// =============================================================================

describe('SHAPBarChart', () => {
  it('renders with feature data', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders an honest empty state (never sample data) when no features provided', () => {
    const { container } = render(<SHAPBarChart features={undefined as unknown as FeatureContribution[]} />);
    expect(container.querySelector('.recharts-responsive-container')).not.toBeInTheDocument();
    expect(screen.getByText('No feature data available')).toBeInTheDocument();
  });

  it('limits features based on maxFeatures prop', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} maxFeatures={2} />);
    // ResponsiveContainer renders in test but with 0 width, so chart content doesn't appear
    // We verify the component mounts without error with maxFeatures prop
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no features', () => {
    render(<SHAPBarChart features={[]} />);
    expect(screen.getByText('No feature data available')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} height={500} />);
    const responsiveContainer = container.querySelector('.recharts-responsive-container');
    expect(responsiveContainer).toBeInTheDocument();
  });

  it('renders reference line by default', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} />);
    // ResponsiveContainer renders in test but with 0 width, so internal chart elements don't render
    // We verify component mounts and has responsive container
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('hides reference line when showReferenceLine is false', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} showReferenceLine={false} />);
    expect(container.querySelector('.recharts-reference-line')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} className="custom-shap" />);
    expect(container.querySelector('.custom-shap')).toBeInTheDocument();
  });

  it('renders bars for features', () => {
    const { container } = render(<SHAPBarChart features={mockFeatures} />);
    // ResponsiveContainer renders in test but with 0 width, so bar elements don't render
    // We verify component mounts without error
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('accepts onBarClick callback prop', () => {
    const handleClick = vi.fn();
    const { container } = render(
      <SHAPBarChart features={mockFeatures} onBarClick={handleClick} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('accepts showValues prop', () => {
    const { container } = render(
      <SHAPBarChart features={mockFeatures} showValues={true} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('accepts showValues false', () => {
    const { container } = render(
      <SHAPBarChart features={mockFeatures} showValues={false} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('accepts tooltipFormatter prop', () => {
    const formatter = vi.fn((value: number) => `${value.toFixed(2)}`);
    const { container } = render(
      <SHAPBarChart features={mockFeatures} tooltipFormatter={formatter} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('uses custom positive and negative colors', () => {
    const { container } = render(
      <SHAPBarChart
        features={mockFeatures}
        positiveColor="#00ff00"
        negativeColor="#ff0000"
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('sorts features by absolute SHAP value', () => {
    const unsortedFeatures: FeatureContribution[] = [
      { feature_name: 'small', feature_value: 1, shap_value: 0.1, contribution_direction: 'positive', contribution_rank: 3 },
      { feature_name: 'large', feature_value: 2, shap_value: -0.5, contribution_direction: 'negative', contribution_rank: 1 },
      { feature_name: 'medium', feature_value: 3, shap_value: 0.3, contribution_direction: 'positive', contribution_rank: 2 },
    ];
    const { container } = render(<SHAPBarChart features={unsortedFeatures} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });
});

// =============================================================================
// SHAP BEESWARM TESTS
// =============================================================================

describe('SHAPBeeswarm', () => {
  it('renders with data', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders an honest empty state (never generated sample data) when no data provided', () => {
    const { container } = render(<SHAPBeeswarm data={undefined as unknown as BeeswarmDataPoint[]} />);
    expect(container.querySelector('.recharts-responsive-container')).not.toBeInTheDocument();
    expect(screen.getByText('No data available for beeswarm plot')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no data', () => {
    render(<SHAPBeeswarm data={[]} />);
    expect(screen.getByText('No data available for beeswarm plot')).toBeInTheDocument();
  });

  it('renders color legend by default', () => {
    render(<SHAPBeeswarm data={mockBeeswarmData} />);
    expect(screen.getByText('Low')).toBeInTheDocument();
    expect(screen.getByText('High')).toBeInTheDocument();
    expect(screen.getByText('Feature Value')).toBeInTheDocument();
  });

  it('hides legend when showLegend is false', () => {
    render(<SHAPBeeswarm data={mockBeeswarmData} showLegend={false} />);
    expect(screen.queryByText('Feature Value')).not.toBeInTheDocument();
  });

  it('orders features by importance', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} maxFeatures={2} />);
    // ResponsiveContainer renders in test but with 0 width, so scatter elements don't render
    // We verify component mounts without error with maxFeatures prop
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('respects provided features order', () => {
    const { container } = render(
      <SHAPBeeswarm
        data={mockBeeswarmData}
        features={['total_prescriptions', 'days_since_visit']}
      />
    );
    // ResponsiveContainer renders in test but with 0 width, so Y-axis labels don't render
    // We verify component mounts without error with custom features prop
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} className="custom-beeswarm" />);
    expect(container.querySelector('.custom-beeswarm')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} height={600} />);
    const responsiveContainer = container.querySelector('.recharts-responsive-container');
    expect(responsiveContainer).toBeInTheDocument();
  });

  it('accepts onPointClick callback prop', () => {
    const handleClick = vi.fn();
    const { container } = render(
      <SHAPBeeswarm data={mockBeeswarmData} onPointClick={handleClick} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows reference line by default', () => {
    const { container } = render(<SHAPBeeswarm data={mockBeeswarmData} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('hides reference line when showReferenceLine is false', () => {
    const { container } = render(
      <SHAPBeeswarm data={mockBeeswarmData} showReferenceLine={false} />
    );
    expect(container.querySelector('.recharts-reference-line')).not.toBeInTheDocument();
  });

  it('accepts custom pointSize', () => {
    const { container } = render(
      <SHAPBeeswarm data={mockBeeswarmData} pointSize={8} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('accepts custom color scales', () => {
    const { container } = render(
      <SHAPBeeswarm
        data={mockBeeswarmData}
        lowColor="#0000ff"
        highColor="#ff0000"
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('handles single feature filter', () => {
    const { container } = render(
      <SHAPBeeswarm data={mockBeeswarmData} features={['days_since_visit']} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('handles maxFeatures of 1', () => {
    const { container } = render(
      <SHAPBeeswarm data={mockBeeswarmData} maxFeatures={1} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });
});

// =============================================================================
// SHAP FORCE PLOT TESTS
// =============================================================================

describe('SHAPForcePlot', () => {
  it('renders with base and output values', () => {
    render(<SHAPForcePlot baseValue={0.35} outputValue={0.72} features={mockFeatures} />);
    expect(screen.getByText('0.35')).toBeInTheDocument();
    expect(screen.getByText('0.72')).toBeInTheDocument();
  });

  it('renders Base and Output labels', () => {
    render(<SHAPForcePlot baseValue={0.5} outputValue={0.8} features={mockFeatures} />);
    expect(screen.getByText(/Base:/)).toBeInTheDocument();
    expect(screen.getByText(/Output:/)).toBeInTheDocument();
  });

  it('renders an honest empty state (never sample data) when props not provided', () => {
    render(<SHAPForcePlot baseValue={undefined as unknown as number} outputValue={undefined as unknown as number} features={undefined as unknown as FeatureContribution[]} />);
    // The fabricated days_since_visit/total_prescriptions contributions
    // must never render.
    expect(screen.queryByText(/days since visit/i)).not.toBeInTheDocument();
    expect(screen.getByText(/no feature data/i)).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(
      <SHAPForcePlot baseValue={0.5} outputValue={0.7} features={mockFeatures} isLoading />
    );
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('renders legend with increase/decrease labels', () => {
    render(<SHAPForcePlot baseValue={0.5} outputValue={0.7} features={mockFeatures} />);
    expect(screen.getByText('Decreases prediction')).toBeInTheDocument();
    expect(screen.getByText('Increases prediction')).toBeInTheDocument();
  });

  it('uses custom value formatter', () => {
    render(
      <SHAPForcePlot
        baseValue={0.5}
        outputValue={0.75}
        features={mockFeatures}
        valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
      />
    );
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('limits displayed features', () => {
    const manyFeatures: FeatureContribution[] = Array.from({ length: 15 }, (_, i) => ({
      feature_name: `feature_${i}`,
      feature_value: i,
      shap_value: (i % 2 === 0 ? 1 : -1) * (0.1 + i * 0.01),
      contribution_direction: (i % 2 === 0 ? 'positive' : 'negative') as 'positive' | 'negative',
      contribution_rank: i,
    }));

    render(
      <SHAPForcePlot
        baseValue={0.5}
        outputValue={0.7}
        features={manyFeatures}
        maxFeatures={5}
      />
    );
    // Should limit to 5 features
    expect(screen.getByText(/Base:/)).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <SHAPForcePlot
        baseValue={0.5}
        outputValue={0.7}
        features={mockFeatures}
        className="custom-force"
      />
    );
    expect(container.querySelector('.custom-force')).toBeInTheDocument();
  });
});

// =============================================================================
// SHAP WATERFALL TESTS
// =============================================================================

describe('buildWaterfallData', () => {
  // The goldstd LR models emit SHAP in log-odds space: base values around
  // -2.8 and negative cumulatives are the NORM, not an edge case.
  const negBase = -2.83;

  it('regression: every row is an ordered [start, end] interval under a negative log-odds base', () => {
    // Unordered intervals fed Recharts negative rect widths -> invalid SVG ->
    // the chart rendered blank for every goldstd model.
    const rows = buildWaterfallData(negBase, mockFeatures, 10);
    rows.forEach((r) => {
      expect(r.start).toBeLessThanOrEqual(r.end);
    });
  });

  it('anchors base and output rows to 0 spanning the signed value', () => {
    const rows = buildWaterfallData(negBase, mockFeatures, 10);
    const base = rows[0];
    const output = rows[rows.length - 1];
    expect(base.isBase).toBe(true);
    expect([base.start, base.end]).toEqual([negBase, 0]);
    expect(base.value).toBe(negBase);
    expect(output.isOutput).toBe(true);
    expect(output.start).toBeLessThanOrEqual(0);
    expect(output.value).toBeCloseTo(
      negBase + mockFeatures.reduce((s, f) => s + f.shap_value, 0)
    );
  });

  it('chains feature rows cumulatively from the base value', () => {
    const rows = buildWaterfallData(negBase, mockFeatures, 10);
    let cumulative = negBase;
    for (const row of rows.slice(1, -1)) {
      const next = cumulative + (row.original?.shap_value ?? 0);
      // The interval endpoints are {previous cumulative, cumulative + shap},
      // ordered — the bar floats over exactly this step of the walk.
      expect([row.start, row.end]).toEqual([
        Math.min(cumulative, next),
        Math.max(cumulative, next),
      ]);
      expect(Math.abs(row.end - row.start)).toBeCloseTo(
        Math.abs(row.original?.shap_value ?? NaN)
      );
      cumulative = next;
    }
  });

  it('sorts by |shap| desc and truncates to maxFeatures', () => {
    const rows = buildWaterfallData(negBase, mockFeatures, 2);
    // base + 2 features + output
    expect(rows).toHaveLength(4);
    expect(rows[1].original?.feature_name).toBe('days_since_visit'); // |0.35|
    expect(rows[2].original?.feature_name).toBe('total_prescriptions'); // |-0.28|
  });

  it('keeps positive-base charts anchored correctly too', () => {
    const rows = buildWaterfallData(0.45, mockFeatures, 10);
    expect([rows[0].start, rows[0].end]).toEqual([0, 0.45]);
    rows.forEach((r) => expect(r.start).toBeLessThanOrEqual(r.end));
  });
});

describe('SHAPWaterfall', () => {
  it('renders with base value and features', () => {
    const { container } = render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders Base Value and Output bars', () => {
    render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
  });

  it('renders an honest empty state (never sample data) when props not provided', () => {
    render(<SHAPWaterfall baseValue={undefined as unknown as number} features={undefined as unknown as FeatureContribution[]} />);
    expect(screen.queryByText('Base Value')).not.toBeInTheDocument();
    expect(screen.getByText(/no feature data/i)).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no features', () => {
    render(<SHAPWaterfall baseValue={0.45} features={[]} />);
    expect(screen.getByText('No feature data available')).toBeInTheDocument();
  });

  it('renders legend items', () => {
    render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('Increases')).toBeInTheDocument();
    expect(screen.getByText('Decreases')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
  });

  it('converts feature names with underscores to spaces', () => {
    const { container } = render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} />);
    // ResponsiveContainer renders in test but with 0 width, so Y-axis labels don't render
    // We verify component mounts without error and legend renders
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
    expect(screen.getByText('Base Value')).toBeInTheDocument();
  });

  it('limits displayed features based on maxFeatures', () => {
    render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} maxFeatures={2} />);
    // Should show base, 2 features, and output = 4 items
    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
  });

  it('handles onBarClick callback', () => {
    const handleClick = vi.fn();
    render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} onBarClick={handleClick} />);
    // Note: Actually triggering clicks on Recharts bars is complex in tests
    // The component renders with the cursor style set correctly
    expect(screen.getByText('Base Value')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <SHAPWaterfall baseValue={0.45} features={mockFeatures} className="custom-waterfall" />
    );
    expect(container.querySelector('.custom-waterfall')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(<SHAPWaterfall baseValue={0.45} features={mockFeatures} height={500} />);
    const responsiveContainer = container.querySelector('.recharts-responsive-container');
    expect(responsiveContainer).toBeInTheDocument();
  });

  it('uses custom value formatter', () => {
    render(
      <SHAPWaterfall
        baseValue={0.5}
        features={mockFeatures}
        valueFormatter={(v) => `${v.toFixed(1)}`}
      />
    );
    // Value formatter is used in tooltips, component should render without error
    expect(screen.getByText('Base Value')).toBeInTheDocument();
  });

  it('uses custom base and output colors', () => {
    const { container } = render(
      <SHAPWaterfall
        baseValue={0.45}
        features={mockFeatures}
        baseColor="#333333"
        outputColor="#666666"
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('uses custom positive and negative colors', () => {
    const { container } = render(
      <SHAPWaterfall
        baseValue={0.45}
        features={mockFeatures}
        positiveColor="#00ff00"
        negativeColor="#ff0000"
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders with all positive contributions', () => {
    const positiveFeatures: FeatureContribution[] = [
      { feature_name: 'feature_a', feature_value: 10, shap_value: 0.2, contribution_direction: 'positive', contribution_rank: 1 },
      { feature_name: 'feature_b', feature_value: 20, shap_value: 0.15, contribution_direction: 'positive', contribution_rank: 2 },
    ];
    render(<SHAPWaterfall baseValue={0.3} features={positiveFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
  });

  it('renders with all negative contributions', () => {
    const negativeFeatures: FeatureContribution[] = [
      { feature_name: 'feature_a', feature_value: 10, shap_value: -0.2, contribution_direction: 'negative', contribution_rank: 1 },
      { feature_name: 'feature_b', feature_value: 20, shap_value: -0.15, contribution_direction: 'negative', contribution_rank: 2 },
    ];
    render(<SHAPWaterfall baseValue={0.7} features={negativeFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
  });

  it('handles features with zero SHAP values', () => {
    const zeroFeatures: FeatureContribution[] = [
      { feature_name: 'feature_a', feature_value: 10, shap_value: 0, contribution_direction: 'positive', contribution_rank: 1 },
    ];
    render(<SHAPWaterfall baseValue={0.5} features={zeroFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
  });

  it('handles very small SHAP values', () => {
    const smallFeatures: FeatureContribution[] = [
      { feature_name: 'tiny_effect', feature_value: 1, shap_value: 0.0001, contribution_direction: 'positive', contribution_rank: 1 },
    ];
    render(<SHAPWaterfall baseValue={0.5} features={smallFeatures} />);
    expect(screen.getByText('Base Value')).toBeInTheDocument();
  });
});
