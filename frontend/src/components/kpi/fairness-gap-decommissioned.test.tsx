/**
 * Regression guard for issues #1073 / #1075.
 *
 * PR #1068 decommissioned WS1-MP-008 "Fairness Gap (ΔRecall)" from the backend
 * registry and gold-standard scorer (it needs protected-group `fairness_metrics`
 * the synthetic substrate does not populate). The Status Legend is rendered on the
 * KPI Dictionary page, so it must not present a Fairness Gap threshold row.
 *
 * (#1075: the static `kpi-dictionary-content.ts` table that also listed it was
 * deleted — it had zero importers; the KPI Dictionary page renders the live
 * registry via the API hooks, not those static tables. So the only remaining
 * user-facing surface to guard is the Status Legend.)
 *
 * The general "Fairness" *concept* in KeyConcepts is intentionally retained — it
 * explains an ML principle, not a data-backed KPI — so it is not asserted here.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { StatusLegend } from '@/components/visualizations/dashboard/StatusLegend';

describe('Fairness Gap (WS1-MP-008) decommissioned from the FE', () => {
  it('is not shown as a Status Legend model-performance threshold row', () => {
    render(<StatusLegend />);
    expect(screen.queryByText('Fairness Gap')).toBeNull();
  });
});
