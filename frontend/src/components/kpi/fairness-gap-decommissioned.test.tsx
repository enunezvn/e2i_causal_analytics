/**
 * Regression guard for issue #1073.
 *
 * PR #1068 decommissioned WS1-MP-008 "Fairness Gap (ΔRecall)" from the backend
 * registry and gold-standard scorer (it needs protected-group `fairness_metrics`
 * the synthetic substrate does not populate). The FE must therefore not present
 * "Fairness Gap" as a live KPI for which there is no data path:
 *  - the KPI-dictionary content table must not list it, and
 *  - the Status Legend (rendered on the KPI Dictionary page) must not show a
 *    Fairness Gap threshold row.
 *
 * The general "Fairness" *concept* in KeyConcepts is intentionally retained — it
 * explains an ML principle, not a data-backed KPI — so it is not asserted here.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { WS1_MODEL_PERFORMANCE_KPIS } from '@/data/kpi-dictionary-content';
import { StatusLegend } from '@/components/visualizations/dashboard/StatusLegend';

describe('Fairness Gap (WS1-MP-008) decommissioned from the FE', () => {
  it('is not listed in the KPI-dictionary model-performance content table', () => {
    const names = WS1_MODEL_PERFORMANCE_KPIS.map((k) => k.name);
    expect(names.some((n) => /fairness gap/i.test(n))).toBe(false);
  });

  it('is not shown as a Status Legend model-performance threshold row', () => {
    render(<StatusLegend />);
    expect(screen.queryByText('Fairness Gap')).toBeNull();
  });
});
