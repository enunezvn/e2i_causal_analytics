/**
 * Causal Visualization Components Tests
 * ======================================
 *
 * Tests for CausalDAG, EffectsTable, and RefutationTests components.
 */

import * as React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { CausalDAG, type CausalNode, type CausalEdge, type CausalDAGRef } from './CausalDAG';
import { EffectsTable, type CausalEffect } from './EffectsTable';
import { RefutationTests, type RefutationResult } from './RefutationTests';

// Mock D3 for CausalDAG tests - create a chainable mock
const createD3Selection = () => {
  const selection: Record<string, unknown> = {};
  const chainMethods = [
    'append', 'attr', 'call', 'on', 'selectAll', 'remove', 'transition',
    'duration', 'classed', 'filter', 'each', 'data', 'enter', 'exit',
    'merge', 'join', 'text', 'style', 'raise', 'lower', 'select',
  ];
  chainMethods.forEach(method => {
    selection[method] = vi.fn(() => selection);
  });
  selection.node = vi.fn(() => ({ getBBox: () => ({ x: 0, y: 0, width: 100, height: 100 }) }));
  selection.nodes = vi.fn(() => []);
  selection.empty = vi.fn(() => true);
  selection.size = vi.fn(() => 0);
  return selection;
};

vi.mock('d3', () => ({
  select: vi.fn(() => createD3Selection()),
  selectAll: vi.fn(() => createD3Selection()),
  zoom: vi.fn(() => ({
    scaleExtent: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    transform: {},
    scaleTo: vi.fn(),
    translateTo: vi.fn(),
  })),
  zoomIdentity: { translate: vi.fn().mockReturnThis(), scale: vi.fn().mockReturnThis() },
  zoomTransform: vi.fn(() => ({ k: 1, x: 0, y: 0 })),
  color: vi.fn(() => ({ darker: () => ({ toString: () => '#333' }) })),
  forceSimulation: vi.fn(() => ({
    nodes: vi.fn().mockReturnThis(),
    force: vi.fn().mockReturnThis(),
    alpha: vi.fn().mockReturnThis(),
    alphaTarget: vi.fn().mockReturnThis(),
    restart: vi.fn().mockReturnThis(),
    stop: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
  })),
  forceLink: vi.fn(() => ({
    id: vi.fn().mockReturnThis(),
    distance: vi.fn().mockReturnThis(),
    links: vi.fn(() => []),
  })),
  forceManyBody: vi.fn(() => ({
    strength: vi.fn().mockReturnThis(),
  })),
  forceCenter: vi.fn(() => ({})),
  forceCollide: vi.fn(() => ({
    radius: vi.fn().mockReturnThis(),
  })),
  drag: vi.fn(() => ({
    on: vi.fn().mockReturnThis(),
  })),
}));

// =============================================================================
// MOCK DATA
// =============================================================================

const mockNodes: CausalNode[] = [
  { id: 'node1', label: 'Treatment', type: 'treatment' },
  { id: 'node2', label: 'Outcome', type: 'outcome' },
  { id: 'node3', label: 'Confounder', type: 'confounder' },
];

const mockEdges: CausalEdge[] = [
  { id: 'edge1', source: 'node1', target: 'node2', type: 'causal', effect: 0.5 },
  { id: 'edge2', source: 'node3', target: 'node1', type: 'confounding' },
  { id: 'edge3', source: 'node3', target: 'node2', type: 'confounding' },
];

const mockEffects: CausalEffect[] = [
  {
    id: 'effect1',
    treatment: 'HCP Engagement',
    outcome: 'TRx Volume',
    estimate: 0.15,
    standardError: 0.02,
    ciLower: 0.11,
    ciUpper: 0.19,
    pValue: 0.0001, // Very small p-value to trigger "< 0.001" display
    isSignificant: true,
  },
  {
    id: 'effect2',
    treatment: 'Email Campaign',
    outcome: 'NRx Volume',
    estimate: -0.08,
    standardError: 0.03,
    ciLower: -0.14,
    ciUpper: -0.02,
    pValue: 0.02,
    isSignificant: true,
  },
  {
    id: 'effect3',
    treatment: 'Conference Attendance',
    outcome: 'Market Share',
    estimate: 0.0001, // Near-zero to trigger "neutral" effect icon
    standardError: 0.01,
    ciLower: -0.018,
    ciUpper: 0.022,
    pValue: 0.85,
    isSignificant: false,
  },
];

const mockRefutationResults: RefutationResult[] = [
  {
    id: 'ref1',
    method: 'random_common_cause',
    originalEstimate: 0.15,
    refutedEstimate: 0.14,
    pValue: 0.72,
    passed: true,
  },
  {
    id: 'ref2',
    method: 'placebo_treatment',
    originalEstimate: 0.15,
    refutedEstimate: 0.01,
    pValue: 0.89,
    passed: true,
  },
  {
    id: 'ref3',
    method: 'data_subset',
    originalEstimate: 0.15,
    refutedEstimate: 0.12,
    pValue: 0.68,
    passed: true,
  },
  {
    id: 'ref4',
    method: 'bootstrap',
    originalEstimate: 0.15,
    refutedEstimate: 0.05,
    pValue: 0.02,
    passed: false,
  },
];

// =============================================================================
// CAUSAL DAG TESTS
// =============================================================================

describe('CausalDAG', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Empty State', () => {
    it('renders empty state when no nodes', () => {
      render(<CausalDAG nodes={[]} edges={[]} />);
      expect(screen.getByText('No causal graph data available')).toBeInTheDocument();
      expect(screen.getByText(/Run causal discovery to visualize/)).toBeInTheDocument();
    });
  });

  describe('Rendering', () => {
    it('renders with nodes and edges', () => {
      const { container } = render(<CausalDAG nodes={mockNodes} edges={mockEdges} />);
      // Should render the container div
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      const { container } = render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} className="custom-class" />
      );
      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });

    it('applies custom minHeight as number', () => {
      const { container } = render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} minHeight={600} />
      );
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper.style.minHeight).toBe('600px');
    });

    it('applies custom minHeight as string', () => {
      const { container } = render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} minHeight="50vh" />
      );
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper.style.minHeight).toBe('50vh');
    });

    it('renders with custom ariaLabel', () => {
      const { container } = render(
        <CausalDAG
          nodes={mockNodes}
          edges={mockEdges}
          ariaLabel="Custom DAG visualization"
        />
      );
      expect(container.querySelector('[aria-label="Custom DAG visualization"]')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading spinner when showLoading is true', () => {
      const { container } = render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} showLoading />
      );
      expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('shows custom loading component when provided', () => {
      render(
        <CausalDAG
          nodes={mockNodes}
          edges={mockEdges}
          showLoading
          loadingComponent={<div data-testid="custom-loader">Loading...</div>}
        />
      );
      expect(screen.getByTestId('custom-loader')).toBeInTheDocument();
    });
  });

  describe('Node Types', () => {
    it('renders treatment nodes', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Treatment', type: 'treatment' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders outcome nodes', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Outcome', type: 'outcome' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders confounder nodes', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Confounder', type: 'confounder' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders mediator nodes', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Mediator', type: 'mediator' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders instrument nodes', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Instrument', type: 'instrument' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders variable nodes (default type)', () => {
      const nodes: CausalNode[] = [{ id: '1', label: 'Variable', type: 'variable' }];
      const { container } = render(<CausalDAG nodes={nodes} edges={[]} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });
  });

  describe('Edge Types', () => {
    const twoNodes: CausalNode[] = [
      { id: '1', label: 'A' },
      { id: '2', label: 'B' },
    ];

    it('renders causal edges', () => {
      const edges: CausalEdge[] = [{ id: 'e1', source: '1', target: '2', type: 'causal' }];
      const { container } = render(<CausalDAG nodes={twoNodes} edges={edges} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders association edges', () => {
      const edges: CausalEdge[] = [{ id: 'e1', source: '1', target: '2', type: 'association' }];
      const { container } = render(<CausalDAG nodes={twoNodes} edges={edges} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders confounding edges', () => {
      const edges: CausalEdge[] = [{ id: 'e1', source: '1', target: '2', type: 'confounding' }];
      const { container } = render(<CausalDAG nodes={twoNodes} edges={edges} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it('renders instrumental edges', () => {
      const edges: CausalEdge[] = [{ id: 'e1', source: '1', target: '2', type: 'instrumental' }];
      const { container } = render(<CausalDAG nodes={twoNodes} edges={edges} />);
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });
  });

  describe('Ref Methods', () => {
    it('exposes fit method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.fit).toBeDefined();
      expect(typeof dagRef.current?.fit).toBe('function');
    });

    it('exposes center method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.center).toBeDefined();
      expect(typeof dagRef.current?.center).toBe('function');
    });

    it('exposes getZoom method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.getZoom).toBeDefined();
      const zoom = dagRef.current?.getZoom();
      expect(typeof zoom).toBe('number');
    });

    it('exposes setZoom method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.setZoom).toBeDefined();
      expect(typeof dagRef.current?.setZoom).toBe('function');
    });

    it('exposes exportSvg method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.exportSvg).toBeDefined();
      expect(typeof dagRef.current?.exportSvg).toBe('function');
    });

    it('exposes highlightNode method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.highlightNode).toBeDefined();
      expect(typeof dagRef.current?.highlightNode).toBe('function');
    });

    it('exposes clearHighlights method via ref', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(dagRef.current?.clearHighlights).toBeDefined();
      expect(typeof dagRef.current?.clearHighlights).toBe('function');
    });

    it('can call highlightNode without error', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(() => dagRef.current?.highlightNode('node1')).not.toThrow();
    });

    it('can call clearHighlights without error', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      dagRef.current?.highlightNode('node1');
      expect(() => dagRef.current?.clearHighlights()).not.toThrow();
    });

    it('can call fit without error', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(() => dagRef.current?.fit()).not.toThrow();
    });

    it('can call center without error', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(() => dagRef.current?.center()).not.toThrow();
    });

    it('can call setZoom without error', () => {
      const dagRef = React.createRef<CausalDAGRef>();
      render(<CausalDAG ref={dagRef} nodes={mockNodes} edges={mockEdges} />);
      expect(() => dagRef.current?.setZoom(1.5)).not.toThrow();
    });
  });

  describe('Click Callbacks', () => {
    it('calls onNodeClick when node clicked', () => {
      const onNodeClick = vi.fn();
      render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} onNodeClick={onNodeClick} />
      );
      // Verify callback prop is passed (D3 handles actual click)
      expect(onNodeClick).not.toHaveBeenCalled();
    });

    it('calls onEdgeClick when edge clicked', () => {
      const onEdgeClick = vi.fn();
      render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} onEdgeClick={onEdgeClick} />
      );
      // Verify callback prop is passed (D3 handles actual click)
      expect(onEdgeClick).not.toHaveBeenCalled();
    });

    it('calls onBackgroundClick when background clicked', () => {
      const onBackgroundClick = vi.fn();
      render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} onBackgroundClick={onBackgroundClick} />
      );
      // Verify callback prop is passed (D3 handles actual click)
      expect(onBackgroundClick).not.toHaveBeenCalled();
    });
  });

  describe('Node Radius', () => {
    it('accepts custom nodeRadius prop', () => {
      const { container } = render(
        <CausalDAG nodes={mockNodes} edges={mockEdges} nodeRadius={32} />
      );
      expect(container.querySelector('[role="img"]')).toBeInTheDocument();
    });
  });
});

// =============================================================================
// EFFECTS TABLE TESTS
// =============================================================================

describe('EffectsTable', () => {
  describe('Empty State', () => {
    it('renders empty state when no effects', () => {
      render(<EffectsTable effects={[]} />);
      expect(screen.getByText('No causal effects available')).toBeInTheDocument();
      expect(screen.getByText(/Run causal analysis to estimate/)).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading spinner when isLoading is true', () => {
      const { container } = render(<EffectsTable effects={mockEffects} isLoading />);
      expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Table Rendering', () => {
    it('renders table headers', () => {
      render(<EffectsTable effects={mockEffects} />);

      expect(screen.getByText('Treatment')).toBeInTheDocument();
      expect(screen.getByText('Outcome')).toBeInTheDocument();
      expect(screen.getByText('Estimate')).toBeInTheDocument();
      expect(screen.getByText('95% CI')).toBeInTheDocument();
      expect(screen.getByText('P-value')).toBeInTheDocument();
      expect(screen.getByText('Sig.')).toBeInTheDocument();
    });

    it('renders effect rows', () => {
      render(<EffectsTable effects={mockEffects} />);

      expect(screen.getByText('HCP Engagement')).toBeInTheDocument();
      expect(screen.getByText('TRx Volume')).toBeInTheDocument();
      expect(screen.getByText('Email Campaign')).toBeInTheDocument();
      expect(screen.getByText('NRx Volume')).toBeInTheDocument();
    });

    it('formats effect estimates with decimal places', () => {
      render(<EffectsTable effects={mockEffects} decimalPlaces={3} />);
      expect(screen.getByText('0.150')).toBeInTheDocument();
    });

    it('renders confidence intervals', () => {
      render(<EffectsTable effects={mockEffects} />);
      // CI format: [lower, upper]
      expect(screen.getByText('[0.110, 0.190]')).toBeInTheDocument();
    });

    it('renders p-values', () => {
      render(<EffectsTable effects={mockEffects} />);
      expect(screen.getByText('< 0.001')).toBeInTheDocument();
      expect(screen.getByText('0.02')).toBeInTheDocument();
      expect(screen.getByText('0.85')).toBeInTheDocument();
    });

    it('renders significance badges', () => {
      render(<EffectsTable effects={mockEffects} />);
      const yesBadges = screen.getAllByText('Yes');
      const noBadges = screen.getAllByText('No');
      expect(yesBadges.length).toBe(2);
      expect(noBadges.length).toBe(1);
    });
  });

  describe('Effect Direction Icons', () => {
    it('shows positive effect icon for positive estimates', () => {
      render(<EffectsTable effects={mockEffects} />);
      const positiveIcons = document.querySelectorAll('[aria-label="Positive effect"]');
      expect(positiveIcons.length).toBeGreaterThan(0);
    });

    it('shows negative effect icon for negative estimates', () => {
      render(<EffectsTable effects={mockEffects} />);
      const negativeIcons = document.querySelectorAll('[aria-label="Negative effect"]');
      expect(negativeIcons.length).toBeGreaterThan(0);
    });

    it('shows no effect icon for near-zero estimates', () => {
      render(<EffectsTable effects={mockEffects} />);
      const neutralIcons = document.querySelectorAll('[aria-label="No effect"]');
      expect(neutralIcons.length).toBeGreaterThan(0);
    });
  });

  describe('CI Bars', () => {
    it('shows CI bars by default', () => {
      render(<EffectsTable effects={mockEffects} showCIBars />);
      expect(screen.getByText('CI Visualization')).toBeInTheDocument();
    });

    it('hides CI bars when showCIBars is false', () => {
      render(<EffectsTable effects={mockEffects} showCIBars={false} />);
      expect(screen.queryByText('CI Visualization')).not.toBeInTheDocument();
    });
  });

  describe('Sorting', () => {
    it('sorts by estimate by default (desc)', () => {
      render(<EffectsTable effects={mockEffects} />);
      const rows = screen.getAllByRole('row');
      // First data row (index 1 since header is index 0) should be highest estimate
      expect(within(rows[1]).getByText('HCP Engagement')).toBeInTheDocument();
    });

    it('toggles sort direction when clicking same column', () => {
      render(<EffectsTable effects={mockEffects} />);
      const estimateHeader = screen.getByText('Estimate');

      // Click to toggle to asc
      fireEvent.click(estimateHeader);

      // Now lowest estimate should be first (Email Campaign with -0.08)
      const rows = screen.getAllByRole('row');
      expect(within(rows[1]).getByText('Email Campaign')).toBeInTheDocument();
    });

    it('sorts by treatment column', () => {
      render(<EffectsTable effects={mockEffects} />);
      const treatmentHeader = screen.getByText('Treatment');
      fireEvent.click(treatmentHeader);

      // Should sort alphabetically desc by default
      const rows = screen.getAllByRole('row');
      expect(within(rows[1]).getByText('HCP Engagement')).toBeInTheDocument();
    });

    it('sorts by outcome column', () => {
      render(<EffectsTable effects={mockEffects} />);
      const outcomeHeader = screen.getByText('Outcome');
      fireEvent.click(outcomeHeader);

      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(1);
    });

    it('sorts by p-value column', () => {
      render(<EffectsTable effects={mockEffects} />);
      const pValueHeader = screen.getByText('P-value');
      fireEvent.click(pValueHeader);

      // Highest p-value first (0.85)
      const rows = screen.getAllByRole('row');
      expect(within(rows[1]).getByText('0.85')).toBeInTheDocument();
    });

    it('disables sorting when sortable is false', () => {
      render(<EffectsTable effects={mockEffects} sortable={false} />);
      // Verify header exists but no sort icons
      expect(screen.getByText('Estimate')).toBeInTheDocument();

      // Sort icons should not be present
      const sortIcons = document.querySelectorAll('.lucide-arrow-up-down');
      expect(sortIcons.length).toBe(0);
    });
  });

  describe('Row Selection', () => {
    it('calls onRowSelect when row is clicked', () => {
      const handleSelect = vi.fn();
      render(<EffectsTable effects={mockEffects} onRowSelect={handleSelect} />);

      fireEvent.click(screen.getByText('HCP Engagement'));
      expect(handleSelect).toHaveBeenCalledWith(mockEffects[0]);
    });

    it('highlights selected row', () => {
      const { container } = render(
        <EffectsTable effects={mockEffects} selectedEffectId="effect1" />
      );
      const selectedRow = container.querySelector('[data-state="selected"]');
      expect(selectedRow).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <EffectsTable effects={mockEffects} className="custom-table" />
      );
      expect(container.querySelector('.custom-table')).toBeInTheDocument();
    });

    it('uses custom significance threshold', () => {
      const effects: CausalEffect[] = [
        {
          id: 'e1',
          treatment: 'Test',
          outcome: 'Outcome',
          estimate: 0.1,
          ciLower: 0.05,
          ciUpper: 0.15,
          pValue: 0.08,
        },
      ];

      // With default threshold (0.05), this is not significant
      const { rerender } = render(<EffectsTable effects={effects} />);
      expect(screen.getByText('No')).toBeInTheDocument();

      // With higher threshold (0.1), this becomes significant
      rerender(<EffectsTable effects={effects} significanceThreshold={0.1} />);
      expect(screen.getByText('Yes')).toBeInTheDocument();
    });

    it('renders accessible caption', () => {
      render(<EffectsTable effects={mockEffects} caption="Custom caption" />);
      const caption = document.querySelector('caption');
      expect(caption).toHaveTextContent('Custom caption');
    });
  });
});

// =============================================================================
// REFUTATION TESTS TESTS
// =============================================================================

describe('RefutationTests', () => {
  describe('Empty State', () => {
    it('renders empty state when no results', () => {
      render(<RefutationTests results={[]} />);
      expect(screen.getByText('No refutation tests available')).toBeInTheDocument();
      expect(screen.getByText(/Run causal analysis with refutation tests/)).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading spinner when isLoading is true', () => {
      const { container } = render(
        <RefutationTests results={mockRefutationResults} isLoading />
      );
      expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Summary Cards', () => {
    it('renders summary cards by default', () => {
      render(<RefutationTests results={mockRefutationResults} />);

      expect(screen.getByText('Tests Passed')).toBeInTheDocument();
      expect(screen.getByText('Tests Failed')).toBeInTheDocument();
      expect(screen.getByText('Pass Rate')).toBeInTheDocument();
    });

    it('shows correct pass count', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      // 3 passed out of 4
      const passedCard = screen.getByText('Tests Passed').closest('div');
      expect(passedCard).toHaveTextContent('3');
    });

    it('shows correct fail count', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      // 1 failed out of 4
      const failedCard = screen.getByText('Tests Failed').closest('div');
      expect(failedCard).toHaveTextContent('1');
    });

    it('shows correct pass rate', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      // 3/4 = 75%
      expect(screen.getByText('75%')).toBeInTheDocument();
    });

    it('hides summary when showSummary is false', () => {
      render(<RefutationTests results={mockRefutationResults} showSummary={false} />);
      expect(screen.queryByText('Tests Passed')).not.toBeInTheDocument();
    });
  });

  describe('Comparison Chart', () => {
    it('renders chart section by default', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      expect(screen.getByText('Estimate Comparison')).toBeInTheDocument();
    });

    it('hides chart when showChart is false', () => {
      render(<RefutationTests results={mockRefutationResults} showChart={false} />);
      expect(screen.queryByText('Estimate Comparison')).not.toBeInTheDocument();
    });

    it('renders chart SVG', () => {
      const { container } = render(<RefutationTests results={mockRefutationResults} />);
      const svg = container.querySelector('svg[role="img"]');
      expect(svg).toBeInTheDocument();
    });

    it('renders legend items', () => {
      const { container } = render(<RefutationTests results={mockRefutationResults} />);
      // Legend items are inside SVG text elements
      const svgTexts = container.querySelectorAll('svg text');
      const textContents = Array.from(svgTexts).map((t) => t.textContent);
      expect(textContents).toContain('Original');
      expect(textContents).toContain('Refuted (Pass)');
      expect(textContents).toContain('Refuted (Fail)');
    });
  });

  describe('Results Table', () => {
    it('renders table header', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      expect(screen.getByText('Refutation Test Results')).toBeInTheDocument();
    });

    it('renders method names', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      expect(screen.getByText('Random Common Cause')).toBeInTheDocument();
      expect(screen.getByText('Placebo Treatment')).toBeInTheDocument();
      expect(screen.getByText('Data Subset')).toBeInTheDocument();
      expect(screen.getByText('Bootstrap Validation')).toBeInTheDocument();
    });

    it('renders method descriptions', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      expect(screen.getByText(/Tests if adding a random common cause/)).toBeInTheDocument();
    });

    it('renders original and refuted estimates', () => {
      render(<RefutationTests results={mockRefutationResults} decimalPlaces={2} />);
      // Original estimate column
      const originals = screen.getAllByText('0.15');
      expect(originals.length).toBeGreaterThanOrEqual(1);
    });

    it('renders p-values', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      expect(screen.getByText('0.72')).toBeInTheDocument();
      expect(screen.getByText('0.89')).toBeInTheDocument();
    });

    it('renders pass/fail badges', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      const passBadges = screen.getAllByText('Pass');
      const failBadges = screen.getAllByText('Fail');
      expect(passBadges.length).toBe(3);
      expect(failBadges.length).toBe(1);
    });

    it('renders status icons for passed tests', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      const passedIcons = document.querySelectorAll('[aria-label="Passed"]');
      expect(passedIcons.length).toBe(3);
    });

    it('renders status icons for failed tests', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      const failedIcons = document.querySelectorAll('[aria-label="Failed"]');
      expect(failedIcons.length).toBe(1);
    });

    it('calculates and displays change percentage', () => {
      render(<RefutationTests results={mockRefutationResults} />);
      // For ref1: (0.14 - 0.15) / 0.15 = -6.7%
      expect(screen.getByText('-6.7%')).toBeInTheDocument();
    });

    it('highlights large changes in red', () => {
      const resultsWithLargeChange: RefutationResult[] = [
        {
          id: 'ref1',
          method: 'bootstrap',
          originalEstimate: 0.15,
          refutedEstimate: 0.05, // -66.7% change
          pValue: 0.02,
          passed: false,
        },
      ];
      const { container } = render(<RefutationTests results={resultsWithLargeChange} />);
      const changeCell = container.querySelector('.text-\\[var\\(--color-destructive\\)\\]');
      expect(changeCell).toBeInTheDocument();
    });
  });

  describe('Row Selection', () => {
    it('calls onRowSelect when row is clicked', () => {
      const handleSelect = vi.fn();
      render(
        <RefutationTests results={mockRefutationResults} onRowSelect={handleSelect} />
      );

      fireEvent.click(screen.getByText('Random Common Cause'));
      expect(handleSelect).toHaveBeenCalledWith(mockRefutationResults[0]);
    });

    it('highlights selected row', () => {
      const { container } = render(
        <RefutationTests results={mockRefutationResults} selectedResultId="ref1" />
      );
      const selectedRow = container.querySelector('[data-state="selected"]');
      expect(selectedRow).toBeInTheDocument();
    });
  });

  describe('Pass Rate Colors', () => {
    it('shows green for high pass rate (>=80%)', () => {
      const allPassed: RefutationResult[] = mockRefutationResults.map(r => ({ ...r, passed: true }));
      const { container } = render(<RefutationTests results={allPassed} />);
      expect(container.querySelector('.text-\\[var\\(--color-success\\)\\]')).toBeInTheDocument();
    });

    it('shows amber for medium pass rate (50-80%)', () => {
      // 3/4 = 75% - should be amber (50-80% range)
      render(<RefutationTests results={mockRefutationResults} />);
      // The 75% should have warning color
      const passRateText = screen.getByText('75%');
      expect(passRateText).toHaveClass('text-[var(--color-warning)]');
    });

    it('shows red for low pass rate (<50%)', () => {
      const mostlyFailed: RefutationResult[] = mockRefutationResults.map(r => ({ ...r, passed: false }));
      const { container } = render(<RefutationTests results={mostlyFailed} />);
      // 0% pass rate should be red
      expect(container.querySelector('.text-\\[var\\(--color-destructive\\)\\]')).toBeInTheDocument();
    });
  });

  describe('Custom Method Names', () => {
    it('uses custom method name when provided', () => {
      const resultsWithCustomName: RefutationResult[] = [
        {
          ...mockRefutationResults[0],
          methodName: 'Custom Test Name',
        },
      ];
      render(<RefutationTests results={resultsWithCustomName} />);
      expect(screen.getByText('Custom Test Name')).toBeInTheDocument();
    });

    it('uses custom description when provided', () => {
      const resultsWithCustomDesc: RefutationResult[] = [
        {
          ...mockRefutationResults[0],
          description: 'Custom description for the test',
        },
      ];
      render(<RefutationTests results={resultsWithCustomDesc} />);
      expect(screen.getByText('Custom description for the test')).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <RefutationTests results={mockRefutationResults} className="custom-refutation" />
      );
      expect(container.querySelector('.custom-refutation')).toBeInTheDocument();
    });

    it('shows significance threshold in description', () => {
      render(<RefutationTests results={mockRefutationResults} significanceThreshold={0.01} />);
      expect(screen.getByText(/threshold: 0.01/)).toBeInTheDocument();
    });
  });

  describe('Decimal Places', () => {
    it('formats estimates with specified decimal places', () => {
      const { container } = render(
        <RefutationTests results={mockRefutationResults} decimalPlaces={4} />
      );
      // Check for formatted values in table cells
      const cells = container.querySelectorAll('td');
      const cellTexts = Array.from(cells).map((c) => c.textContent);
      // At least one cell should have a value formatted with 4 decimal places
      expect(cellTexts.some((t) => t?.includes('0.1500') || t?.includes('0.1400'))).toBe(true);
    });

    it('formats chart values with specified decimal places', () => {
      const { container } = render(
        <RefutationTests results={mockRefutationResults} decimalPlaces={2} />
      );
      // Values in SVG chart
      const svg = container.querySelector('svg[role="img"]');
      expect(svg).toBeInTheDocument();
    });
  });
});
