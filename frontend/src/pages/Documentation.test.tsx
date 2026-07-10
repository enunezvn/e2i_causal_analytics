/**
 * Documentation Page Tests
 * ========================
 * Page shell + interactive component behaviors (per the spec's testing
 * section, component behavior tests live here; content invariants live in
 * components/documentation/content.test.ts).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Documentation from './Documentation';

// The page's ONLY network hook. Mocked in every test; the "live chip" tests
// flip its return value.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
}));

import { useKPIList } from '@/hooks/api/use-kpi';

// jsdom has neither scrollIntoView nor IntersectionObserver.
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { kpis: [], total: 46 },
    isLoading: false,
    isError: false,
  });
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Documentation />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Documentation page shell', () => {
  it('renders the page header', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /understanding e2i/i })).toBeInTheDocument();
  });

  it('renders all four sections', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /^purpose/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^methodology/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^best practices/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^expected impact/i })).toBeInTheDocument();
  });

  it('renders the section nav and scrolls on click', async () => {
    renderPage();
    const nav = screen.getByRole('navigation', { name: /on this page/i });
    expect(nav).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^methodology$/i }));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});

describe('CausalScopeMap', () => {
  it('renders the three causal levels', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /hcp prescribing behavior/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /market & brand performance/i })).toBeInTheDocument();
  });

  it('defaults to the HCP level active on mount', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /hcp prescribing behavior/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByRole('button', { name: /market & brand performance/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByText(/rep detailing frequency/i)).toBeInTheDocument();
  });

  it('shows a level summary and its registry nodes on selection', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /patient journey outcomes/i }));
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByText(/treatment initiation, persistence, and discontinuation/i)).toBeInTheDocument();
    expect(screen.getByText('patient persistence')).toBeInTheDocument();
  });
});
