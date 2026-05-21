/**
 * EmptyState Component Tests
 * ==========================
 *
 * F-002 coverage — empty-state must replace hardcoded SAMPLE_ data
 * when API hooks return undefined.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EmptyState } from './EmptyState';

describe('EmptyState', () => {
  it('renders default title when no props provided', () => {
    render(<EmptyState />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders custom title', () => {
    render(<EmptyState title="No analysis result" />);
    expect(screen.getByText('No analysis result')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(
      <EmptyState
        title="Empty"
        description="Run an analysis to populate this section"
      />,
    );
    expect(
      screen.getByText('Run an analysis to populate this section'),
    ).toBeInTheDocument();
  });

  it('renders action button when provided', () => {
    render(
      <EmptyState
        title="Empty"
        action={<button type="button">Run analysis</button>}
      />,
    );
    expect(
      screen.getByRole('button', { name: 'Run analysis' }),
    ).toBeInTheDocument();
  });

  it('has status role for accessibility', () => {
    render(<EmptyState />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('exposes test id for downstream assertions', () => {
    render(<EmptyState />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
