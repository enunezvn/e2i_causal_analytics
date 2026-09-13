import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { badgeVariants } from '@/components/ui/badge';
import { gateBadge } from './gateBadge';

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

  // Badge renders variant as a CVA class string, not a `data-variant`
  // attribute -- `badgeVariants({ variant })` is the reliable, exact hook
  // (Badge passes no extra className, so the rendered className matches it
  // verbatim). Each gate must carry ITS OWN variant, not just render a label.
  it('carries the destructive variant for a blocked gate', () => {
    render(gateBadge('block'));
    expect(screen.getByText('Blocked').className).toBe(badgeVariants({ variant: 'destructive' }));
  });

  it('carries the default variant for a proceed gate', () => {
    render(gateBadge('proceed'));
    expect(screen.getByText('Proceed').className).toBe(badgeVariants({ variant: 'default' }));
  });

  it('carries the secondary variant for a review gate', () => {
    render(gateBadge('review'));
    expect(screen.getByText('Review').className).toBe(badgeVariants({ variant: 'secondary' }));
  });

  it('carries the outline variant for a null gate', () => {
    render(gateBadge(null));
    expect(screen.getByText('—').className).toBe(badgeVariants({ variant: 'outline' }));
  });
});
