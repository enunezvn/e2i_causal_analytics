/**
 * MSWBanner Tests
 * ===============
 *
 * Red-first tests for the persistent MSW mock-data banner: whenever MSW is
 * actively intercepting API requests, the UI must show an unmissable,
 * non-dismissable banner so nobody mistakes mock pharma data for real data.
 *
 * The banner reads the window.__E2I_MSW_ACTIVE__ flag set by initMSW()
 * (src/mocks/browser.ts) after the worker starts. It deliberately does NOT
 * import anything from msw so it cannot drag MSW code into a bundle.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MSWBanner } from './MSWBanner';

afterEach(() => {
  delete window.__E2I_MSW_ACTIVE__;
});

describe('MSWBanner', () => {
  it('renders a visible banner when MSW is active', () => {
    window.__E2I_MSW_ACTIVE__ = true;
    render(<MSWBanner />);

    const banner = screen.getByRole('status');
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/mock/i);
    expect(banner).toHaveTextContent(/simulated|not real/i);
  });

  it('renders nothing when MSW is not active', () => {
    const { container } = render(<MSWBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when the flag is explicitly false', () => {
    window.__E2I_MSW_ACTIVE__ = false;
    const { container } = render(<MSWBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('is persistent: exposes no dismiss control', () => {
    window.__E2I_MSW_ACTIVE__ = true;
    render(<MSWBanner />);

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
