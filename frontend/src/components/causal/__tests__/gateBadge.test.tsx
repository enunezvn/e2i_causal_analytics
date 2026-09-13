import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { gateBadge } from '../CausalAnalysisDetail';

describe('gateBadge', () => {
  it('renders every gate', () => {
    render(
      <>
        {gateBadge('proceed')}
        {gateBadge('review')}
        {gateBadge('block')}
        {gateBadge(null)}
      </>
    );
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Review')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
    // A null/absent gate keeps the component's existing "not gated" wording (an
    // em dash), unchanged by #1991 debt 4's typing work.
    expect(screen.getByText('—')).toBeInTheDocument();
  });
});
