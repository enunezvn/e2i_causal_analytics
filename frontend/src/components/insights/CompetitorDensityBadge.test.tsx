/**
 * CompetitorDensityBadge Tests — surface-only market-landscape density on a bet.
 *
 * Covers: a populated bet shows "N rivals" + the saturation label + competitor
 * name chips; an absent/zero/unknown density renders nothing (honest empty
 * state, mirroring the causal ClinicalContextPanel "N rivals" treatment); the
 * singular "1 rival" is not pluralised.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { CompetitorDensityBadge } from './CompetitorDensityBadge';

describe('CompetitorDensityBadge', () => {
  it('renders nothing when density is absent (honest empty state)', () => {
    const { container } = render(<CompetitorDensityBadge />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when count is 0 / unknown (backend fail-open state)', () => {
    const { container } = render(
      <CompetitorDensityBadge
        competitor_products_count={0}
        competitor_density_label="unknown"
        competitor_drug_names={[]}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('shows "N rivals", the saturation label, and competitor chips', () => {
    render(
      <CompetitorDensityBadge
        competitor_products_count={3}
        competitor_density_label="moderate"
        competitor_drug_names={['Verzenio', 'Ibrance', 'Kisqali']}
      />,
    );
    expect(screen.getByText(/Market landscape \(3 rivals\)/i)).toBeInTheDocument();
    expect(screen.getByText('moderate')).toBeInTheDocument();
    expect(screen.getByText('Verzenio')).toBeInTheDocument();
    expect(screen.getByText('Ibrance')).toBeInTheDocument();
    expect(screen.getByText('Kisqali')).toBeInTheDocument();
  });

  it('renders nothing when the saturation label is "unknown" regardless of count', () => {
    // The issue acceptance treats density "0 / unknown" as the empty state; the
    // backend only ever pairs "unknown" with count 0, so guard on both.
    const { container } = render(
      <CompetitorDensityBadge
        competitor_products_count={2}
        competitor_density_label="unknown"
        competitor_drug_names={['X', 'Y']}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('uses the singular "1 rival" when there is one competitor', () => {
    render(
      <CompetitorDensityBadge
        competitor_products_count={1}
        competitor_density_label="limited"
        competitor_drug_names={['Verzenio']}
      />,
    );
    expect(screen.getByText(/Market landscape \(1 rival\)/i)).toBeInTheDocument();
  });
});
