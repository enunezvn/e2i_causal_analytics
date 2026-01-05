/**
 * CausalDiscovery Page Tests
 * ==========================
 *
 * Tests for the CausalDiscovery page component.
 * Includes tests for:
 * - Page header with technology badges
 * - CausalDiscovery visualization integration
 * - Refutation tests integration (Phase 3.2)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import CausalDiscovery from './CausalDiscovery';

// Mock the CausalDiscovery visualization component to avoid D3 complexities in tests
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({ showControls, showDetails, showEffectsTable, showRefutationTests }: {
    showControls?: boolean;
    showDetails?: boolean;
    showEffectsTable?: boolean;
    showRefutationTests?: boolean;
  }) => (
    <div data-testid="causal-discovery-viz">
      <div data-testid="show-controls">{String(showControls)}</div>
      <div data-testid="show-details">{String(showDetails)}</div>
      <div data-testid="show-effects-table">{String(showEffectsTable)}</div>
      <div data-testid="show-refutation-tests">{String(showRefutationTests)}</div>
    </div>
  ),
}));

// Wrapper for Router context
const renderWithRouter = (component: React.ReactNode) => {
  return render(<BrowserRouter>{component}</BrowserRouter>);
};

describe('CausalDiscovery Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // =========================================================================
  // PAGE HEADER TESTS
  // =========================================================================

  describe('Page Header', () => {
    it('renders page title', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText('Causal Discovery')).toBeInTheDocument();
    });

    it('renders page description', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText(/Causal analysis with DAG visualization/)).toBeInTheDocument();
      expect(screen.getByText(/effect estimates/)).toBeInTheDocument();
      expect(screen.getByText(/refutation tests/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // TECHNOLOGY BADGES TESTS (Phase 3.2)
  // =========================================================================

  describe('Technology Badges', () => {
    it('displays DoWhy badge', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText('DoWhy')).toBeInTheDocument();
    });

    it('displays EconML badge', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText('EconML')).toBeInTheDocument();
    });

    it('displays DAG badge', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText('DAG')).toBeInTheDocument();
    });

    it('displays Refutation badge', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByText('Refutation')).toBeInTheDocument();
    });

    it('renders all four technology badges', () => {
      renderWithRouter(<CausalDiscovery />);

      // Verify all 4 specific badges are present
      expect(screen.getByText('DoWhy')).toBeInTheDocument();
      expect(screen.getByText('EconML')).toBeInTheDocument();
      expect(screen.getByText('DAG')).toBeInTheDocument();
      expect(screen.getByText('Refutation')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // VISUALIZATION COMPONENT INTEGRATION TESTS
  // =========================================================================

  describe('CausalDiscovery Visualization', () => {
    it('renders the visualization component', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByTestId('causal-discovery-viz')).toBeInTheDocument();
    });

    it('passes showControls prop as true', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByTestId('show-controls')).toHaveTextContent('true');
    });

    it('passes showDetails prop as true', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByTestId('show-details')).toHaveTextContent('true');
    });

    it('passes showEffectsTable prop as true', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByTestId('show-effects-table')).toHaveTextContent('true');
    });

    it('passes showRefutationTests prop as true', () => {
      renderWithRouter(<CausalDiscovery />);

      expect(screen.getByTestId('show-refutation-tests')).toHaveTextContent('true');
    });
  });

  // =========================================================================
  // LAYOUT TESTS
  // =========================================================================

  describe('Layout', () => {
    it('has container with proper padding', () => {
      renderWithRouter(<CausalDiscovery />);

      const container = screen.getByText('Causal Discovery').closest('.container');
      expect(container).toBeInTheDocument();
      expect(container).toHaveClass('mx-auto');
    });

    it('header has responsive flex layout', () => {
      renderWithRouter(<CausalDiscovery />);

      const header = screen.getByText('Causal Discovery').closest('div');
      const headerParent = header?.parentElement;
      expect(headerParent).toHaveClass('flex');
    });
  });
});
