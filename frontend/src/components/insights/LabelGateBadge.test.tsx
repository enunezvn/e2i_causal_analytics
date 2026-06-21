/**
 * LabelGateBadge Tests — label-segmentation gater verdict surfacing.
 *
 * Covers: off_label shows the warning + reason + de-prioritization notice; an
 * absent verdict renders nothing (gater off — honest empty state); indeterminate
 * is muted; on_label is a subtle confirmation; the FDA-label confirmation chip
 * shows only when label_evidence_confirmed is true.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { LabelGateBadge } from './LabelGateBadge';

describe('LabelGateBadge', () => {
  it('renders nothing when label_verdict is absent (gater off)', () => {
    const { container } = render(<LabelGateBadge />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows a warning badge with the reason and a de-prioritization notice for off_label', () => {
    render(
      <LabelGateBadge
        label_verdict="off_label"
        off_label
        off_label_reason="CSU specialty is outside the oncology label"
      />,
    );
    expect(screen.getByText('Off-label')).toBeInTheDocument();
    expect(
      screen.getByText('CSU specialty is outside the oncology label'),
    ).toBeInTheDocument();
    // Conveys it is de-prioritized.
    expect(screen.getByText(/de-prioritized/i)).toBeInTheDocument();
  });

  it('labels mixed verdicts "Partly off-label"', () => {
    render(<LabelGateBadge label_verdict="mixed" off_label_reason="some segments off-label" />);
    expect(screen.getByText('Partly off-label')).toBeInTheDocument();
    expect(screen.getByText(/de-prioritized/i)).toBeInTheDocument();
  });

  it('renders a muted "Label: review" for indeterminate (no de-prioritization claim)', () => {
    render(<LabelGateBadge label_verdict="indeterminate" />);
    expect(screen.getByText('Label: review')).toBeInTheDocument();
    expect(screen.queryByText('Off-label')).not.toBeInTheDocument();
    expect(screen.queryByText(/de-prioritized/i)).not.toBeInTheDocument();
  });

  it('renders a subtle "On-label" confirmation for on_label', () => {
    render(<LabelGateBadge label_verdict="on_label" />);
    expect(screen.getByText('On-label')).toBeInTheDocument();
    expect(screen.queryByText('Off-label')).not.toBeInTheDocument();
  });

  it('shows the "confirmed by FDA label" chip only when label_evidence_confirmed is true', () => {
    const { rerender } = render(
      <LabelGateBadge label_verdict="off_label" off_label_reason="r" />,
    );
    expect(screen.queryByText(/confirmed by FDA label/i)).not.toBeInTheDocument();

    rerender(
      <LabelGateBadge label_verdict="off_label" off_label_reason="r" label_evidence_confirmed />,
    );
    expect(screen.getByText(/confirmed by FDA label/i)).toBeInTheDocument();
  });
});
