/**
 * WarningBanner Component Tests
 * ==============================
 *
 * F-010-frontend coverage — banner must render API warnings prominently
 * so users see backend fallback-mode disclosure.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { WarningBanner } from './WarningBanner';

describe('WarningBanner', () => {
  it('returns null when messages is empty', () => {
    const { container } = render(<WarningBanner messages={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('returns null when messages is undefined-equivalent (falsy)', () => {
    // Defensive: callers may pass through API responses where warnings is missing
    const { container } = render(
      <WarningBanner messages={undefined as unknown as string[]} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders banner with single warning message', () => {
    render(<WarningBanner messages={['Using mock data due to ImportError']} />);

    expect(screen.getByTestId('warning-banner')).toBeInTheDocument();
    expect(
      screen.getByText('Using mock data due to ImportError'),
    ).toBeInTheDocument();
  });

  it('renders multiple warning messages as a list', () => {
    const messages = [
      'Falling back to placeholder estimator',
      'CATE bounds approximate due to small sample',
    ];
    render(<WarningBanner messages={messages} />);

    expect(
      screen.getByText('Falling back to placeholder estimator'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('CATE bounds approximate due to small sample'),
    ).toBeInTheDocument();
  });

  it('renders default title "Analysis warnings"', () => {
    render(<WarningBanner messages={['x']} />);
    expect(screen.getByText('Analysis warnings')).toBeInTheDocument();
  });

  it('renders custom title when provided', () => {
    render(<WarningBanner messages={['x']} title="Backend fallback engaged" />);
    expect(screen.getByText('Backend fallback engaged')).toBeInTheDocument();
  });

  it('has alert role for accessibility', () => {
    render(<WarningBanner messages={['x']} />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });
});
