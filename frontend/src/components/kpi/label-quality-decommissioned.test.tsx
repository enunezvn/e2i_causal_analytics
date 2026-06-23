/**
 * Regression guard for T8.
 *
 * T8 removed WS1-DQ-008 "Label Quality (IAA)" from the backend registry, the
 * data-quality calculator, the coverage tooling, and the dashboard Status Legend
 * by product decision — a *working* metric (corpus Fleiss κ ≈ 0.76) deprioritized
 * out of the live KPI set. The Status Legend is rendered on the KPI Dictionary
 * page, so it must not present a Label Quality (IAA) threshold row.
 *
 * Mirrors the WS1-MP-008 Fairness-Gap decommission guard (#1073). The DB objects
 * (`v_kpi_label_quality`, `ml_annotations`) are retained server-side, but no
 * user-facing KPI surface should advertise the metric.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { StatusLegend } from '@/components/visualizations/dashboard/StatusLegend';

describe('Label Quality (WS1-DQ-008) decommissioned from the FE', () => {
  it('is not shown as a Status Legend threshold row', () => {
    render(<StatusLegend />);
    expect(screen.queryByText('Label Quality (IAA)')).toBeNull();
  });
});
