/**
 * Footer Tests — all four quick links.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Footer } from './Footer';

function renderFooter() {
  return render(
    <MemoryRouter>
      <Footer />
    </MemoryRouter>
  );
}

describe('Footer', () => {
  it('renders all four quick links', () => {
    renderFooter();
    expect(screen.getByRole('link', { name: /dashboard/i })).toHaveAttribute('href', '/');
    expect(screen.getByRole('link', { name: /system status/i })).toHaveAttribute('href', '/system-health');
    expect(screen.getByRole('link', { name: /documentation/i })).toHaveAttribute('href', '/documentation');
    expect(screen.getByRole('link', { name: /api docs/i })).toHaveAttribute('href', '/api/docs');
  });

  it('keeps API Docs external (new tab) and Documentation internal', () => {
    renderFooter();
    expect(screen.getByRole('link', { name: /api docs/i })).toHaveAttribute('target', '_blank');
    expect(screen.getByRole('link', { name: /documentation/i })).not.toHaveAttribute('target');
  });
});
