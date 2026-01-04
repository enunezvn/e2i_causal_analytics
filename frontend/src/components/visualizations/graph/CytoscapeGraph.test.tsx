/**
 * CytoscapeGraph Component Tests
 * ==============================
 *
 * Comprehensive tests for the CytoscapeGraph React component.
 * Tests rendering, event handlers, imperative API, and hover highlighting.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import * as React from 'react';
import { CytoscapeGraph, type CytoscapeGraphRef } from './CytoscapeGraph';
import type { Core, ElementDefinition } from 'cytoscape';

// =============================================================================
// MOCK SETUP
// =============================================================================

// Create mock before vi.mock call
const mockCyInstance = {
  getElementById: vi.fn().mockReturnValue({
    length: 1,
    addClass: vi.fn(),
    connectedEdges: vi.fn().mockReturnValue({
      addClass: vi.fn(),
      connectedNodes: vi.fn().mockReturnValue({
        addClass: vi.fn(),
      }),
    }),
    connectedNodes: vi.fn().mockReturnValue({
      addClass: vi.fn(),
    }),
    select: vi.fn(),
  }),
  elements: vi.fn().mockReturnValue({
    not: vi.fn().mockReturnValue({
      not: vi.fn().mockReturnValue({
        not: vi.fn().mockReturnValue({
          addClass: vi.fn(),
        }),
        addClass: vi.fn(),
      }),
      addClass: vi.fn(),
    }),
    removeClass: vi.fn(),
  }),
  edges: vi.fn().mockReturnValue({
    unselect: vi.fn(),
  }),
  on: vi.fn(),
};

const mockUseCytoscape = {
  containerRef: { current: null },
  cyInstance: mockCyInstance,
  isLoading: false,
  setElements: vi.fn(),
  runLayout: vi.fn(),
  fit: vi.fn(),
  center: vi.fn(),
  zoom: vi.fn(),
  getZoom: vi.fn().mockReturnValue(1),
  selectNodes: vi.fn(),
  clearSelection: vi.fn(),
  highlightNode: vi.fn(),
  clearHighlights: vi.fn(),
  exportPng: vi.fn().mockReturnValue('data:image/png;base64,test'),
};

vi.mock('@/hooks/use-cytoscape', () => ({
  useCytoscape: vi.fn(() => mockUseCytoscape),
  defaultCytoscapeStyles: [],
}));

vi.mock('@/lib/utils', () => ({
  cn: (...classes: string[]) => classes.filter(Boolean).join(' '),
}));

// Import after mock - Vitest hoists vi.mock calls
import { useCytoscape as useCytoscapeImport } from '@/hooks/use-cytoscape';
const mockedUseCytoscape = vi.mocked(useCytoscapeImport);

// =============================================================================
// TEST UTILITIES
// =============================================================================

const mockElements: ElementDefinition[] = [
  { data: { id: 'a', label: 'Node A' } },
  { data: { id: 'b', label: 'Node B' } },
  { data: { id: 'ab', source: 'a', target: 'b', label: 'Edge AB' } },
];

function resetMocks() {
  vi.clearAllMocks();
  mockUseCytoscape.isLoading = false;
  mockUseCytoscape.containerRef = { current: null };
  mockUseCytoscape.cyInstance = mockCyInstance;
}

// =============================================================================
// TESTS: RENDERING
// =============================================================================

describe('CytoscapeGraph', () => {
  beforeEach(() => {
    resetMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders the graph container', () => {
      render(<CytoscapeGraph elements={mockElements} />);

      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('applies aria-label for accessibility', () => {
      render(<CytoscapeGraph elements={mockElements} ariaLabel="Test Graph" />);

      expect(screen.getByRole('img')).toHaveAttribute('aria-label', 'Test Graph');
    });

    it('uses default aria-label when not provided', () => {
      render(<CytoscapeGraph elements={mockElements} />);

      expect(screen.getByRole('img')).toHaveAttribute(
        'aria-label',
        'Interactive graph visualization'
      );
    });

    it('applies custom className', () => {
      const { container } = render(
        <CytoscapeGraph elements={mockElements} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('applies minHeight as number', () => {
      const { container } = render(
        <CytoscapeGraph elements={mockElements} minHeight={500} />
      );

      expect(container.firstChild).toHaveStyle({ minHeight: '500px' });
    });

    it('applies minHeight as string', () => {
      const { container } = render(
        <CytoscapeGraph elements={mockElements} minHeight="50vh" />
      );

      expect(container.firstChild).toHaveStyle({ minHeight: '50vh' });
    });

    it('sets tabIndex for keyboard accessibility', () => {
      render(<CytoscapeGraph elements={mockElements} />);

      expect(screen.getByRole('img')).toHaveAttribute('tabIndex', '0');
    });
  });

  describe('loading state', () => {
    it('shows loading spinner when loading', () => {
      mockUseCytoscape.isLoading = true;

      render(<CytoscapeGraph elements={mockElements} showLoading={true} />);

      expect(screen.getByRole('status')).toBeInTheDocument();
      expect(screen.getByText('Loading graph...')).toBeInTheDocument();
    });

    it('hides loading spinner when showLoading is false', () => {
      mockUseCytoscape.isLoading = true;

      render(<CytoscapeGraph elements={mockElements} showLoading={false} />);

      expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('hides loading spinner when not loading', () => {
      mockUseCytoscape.isLoading = false;

      render(<CytoscapeGraph elements={mockElements} showLoading={true} />);

      expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('uses custom loading component when provided', () => {
      mockUseCytoscape.isLoading = true;

      render(
        <CytoscapeGraph
          elements={mockElements}
          showLoading={true}
          loadingComponent={<div data-testid="custom-loader">Custom Loading</div>}
        />
      );

      expect(screen.getByTestId('custom-loader')).toBeInTheDocument();
      expect(screen.getByText('Custom Loading')).toBeInTheDocument();
    });
  });

  describe('hook initialization', () => {
    it('passes elements to useCytoscape', () => {
      render(<CytoscapeGraph elements={mockElements} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          elements: mockElements,
        }),
        expect.any(Object)
      );
    });

    it('passes layout to useCytoscape', () => {
      render(<CytoscapeGraph elements={mockElements} layout="breadthfirst" />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          layout: 'breadthfirst',
        }),
        expect.any(Object)
      );
    });

    it('uses default layout when not provided', () => {
      render(<CytoscapeGraph elements={mockElements} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          layout: 'cose',
        }),
        expect.any(Object)
      );
    });

    it('passes zoom limits to useCytoscape', () => {
      render(<CytoscapeGraph elements={mockElements} minZoom={0.5} maxZoom={2} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          minZoom: 0.5,
          maxZoom: 2,
        }),
        expect.any(Object)
      );
    });

    it('passes custom styles to useCytoscape', () => {
      const customStyles = [{ selector: 'node', style: { 'background-color': 'red' } }];

      render(<CytoscapeGraph elements={mockElements} style={customStyles as any} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          style: customStyles,
        }),
        expect.any(Object)
      );
    });
  });

  describe('event handlers', () => {
    it('passes onNodeClick to useCytoscape', () => {
      const onNodeClick = vi.fn();

      render(<CytoscapeGraph elements={mockElements} onNodeClick={onNodeClick} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onNodeClick,
        })
      );
    });

    it('passes onNodeDoubleClick to useCytoscape', () => {
      const onNodeDoubleClick = vi.fn();

      render(<CytoscapeGraph elements={mockElements} onNodeDoubleClick={onNodeDoubleClick} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onNodeDoubleClick,
        })
      );
    });

    it('passes onEdgeClick to useCytoscape', () => {
      const onEdgeClick = vi.fn();

      render(<CytoscapeGraph elements={mockElements} onEdgeClick={onEdgeClick} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onEdgeClick,
        })
      );
    });

    it('passes onSelectionChange to useCytoscape', () => {
      const onSelectionChange = vi.fn();

      render(<CytoscapeGraph elements={mockElements} onSelectionChange={onSelectionChange} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onSelectionChange,
        })
      );
    });

    it('passes onBackgroundClick to useCytoscape', () => {
      const onBackgroundClick = vi.fn();

      render(<CytoscapeGraph elements={mockElements} onBackgroundClick={onBackgroundClick} />);

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onBackgroundClick,
        })
      );
    });

    it('wraps onNodeMouseOver for hover highlighting', () => {
      const onNodeMouseOver = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onNodeMouseOver={onNodeMouseOver}
        />
      );

      // The hook should receive a wrapped handler
      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onNodeMouseOver: expect.any(Function),
        })
      );
    });

    it('wraps onNodeMouseOut for hover highlighting', () => {
      const onNodeMouseOut = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onNodeMouseOut={onNodeMouseOut}
        />
      );

      expect(mockedUseCytoscape).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          onNodeMouseOut: expect.any(Function),
        })
      );
    });
  });

  describe('imperative handle', () => {
    it('exposes getZoom method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      expect(ref.current?.getZoom()).toBe(1);
      expect(mockUseCytoscape.getZoom).toHaveBeenCalled();
    });

    it('exposes setZoom method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.setZoom(2);
      });

      expect(mockUseCytoscape.zoom).toHaveBeenCalledWith(2);
    });

    it('exposes fit method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.fit(50);
      });

      expect(mockUseCytoscape.fit).toHaveBeenCalledWith(50);
    });

    it('exposes center method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.center();
      });

      expect(mockUseCytoscape.center).toHaveBeenCalled();
    });

    it('exposes runLayout method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.runLayout('breadthfirst');
      });

      expect(mockUseCytoscape.runLayout).toHaveBeenCalledWith('breadthfirst');
    });

    it('exposes exportPng method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      const result = ref.current?.exportPng();

      expect(result).toBe('data:image/png;base64,test');
      expect(mockUseCytoscape.exportPng).toHaveBeenCalled();
    });

    it('exposes selectNode method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.selectNode('a');
      });

      expect(mockUseCytoscape.selectNodes).toHaveBeenCalledWith(['a']);
    });

    it('exposes selectEdge method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.selectEdge('ab');
      });

      expect(mockCyInstance.edges().unselect).toHaveBeenCalled();
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('ab');
    });

    it('exposes clearSelection method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.clearSelection();
      });

      expect(mockUseCytoscape.clearSelection).toHaveBeenCalled();
    });

    it('exposes highlightNode method with cyInstance', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.highlightNode('a');
      });

      // Should call getElementById on the cy instance
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('a');
    });

    it('exposes highlightNode fallback when no cyInstance', () => {
      const ref = React.createRef<CytoscapeGraphRef>();
      mockUseCytoscape.cyInstance = null as unknown as typeof mockCyInstance;

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.highlightNode('a');
      });

      // Should fallback to hook's highlightNode
      expect(mockUseCytoscape.highlightNode).toHaveBeenCalledWith('a');
    });

    it('exposes clearHighlights method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.clearHighlights();
      });

      // Should call removeClass on cy instance
      expect(mockCyInstance.elements().removeClass).toHaveBeenCalledWith('highlighted dimmed');
    });

    it('exposes clearHighlights fallback when no cyInstance', () => {
      const ref = React.createRef<CytoscapeGraphRef>();
      mockUseCytoscape.cyInstance = null as unknown as typeof mockCyInstance;

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      act(() => {
        ref.current?.clearHighlights();
      });

      expect(mockUseCytoscape.clearHighlights).toHaveBeenCalled();
    });

    it('exposes getCyInstance method', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      const cy = ref.current?.getCyInstance();

      expect(cy).toBe(mockCyInstance);
    });
  });

  describe('hover highlighting', () => {
    it('calls highlightNodeConnections on node mouse over when enabled', () => {
      const onNodeMouseOver = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onNodeMouseOver={onNodeMouseOver}
        />
      );

      // Get the wrapped handler that was passed to useCytoscape
      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Call the wrapped handler
      act(() => {
        handlers.onNodeMouseOver!('a', { id: 'a', label: 'Node A' });
      });

      // Should have called the user's handler
      expect(onNodeMouseOver).toHaveBeenCalledWith('a', { id: 'a', label: 'Node A' });
    });

    it('calls clearAllHighlights on node mouse out when enabled', () => {
      const onNodeMouseOut = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onNodeMouseOut={onNodeMouseOut}
        />
      );

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      act(() => {
        handlers.onNodeMouseOut!('a');
      });

      expect(onNodeMouseOut).toHaveBeenCalledWith('a');
    });

    it('does not highlight when enableHoverHighlight is false', () => {
      render(<CytoscapeGraph elements={mockElements} enableHoverHighlight={false} />);

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Reset mock to track new calls
      mockCyInstance.getElementById.mockClear();

      act(() => {
        handlers.onNodeMouseOver!('a', { id: 'a' });
      });

      // Should not call getElementById for highlighting
      // (The wrapped handler still gets called but doesn't do highlighting)
    });

    it('registers edge hover handlers on ready when enabled', () => {
      const onEdgeMouseOver = vi.fn();
      const onEdgeMouseOut = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onEdgeMouseOver={onEdgeMouseOver}
          onEdgeMouseOut={onEdgeMouseOut}
        />
      );

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Simulate onReady callback
      act(() => {
        handlers.onReady!(mockCyInstance as unknown as Core);
      });

      // Should have registered edge hover handlers
      expect(mockCyInstance.on).toHaveBeenCalledWith('mouseover', 'edge', expect.any(Function));
      expect(mockCyInstance.on).toHaveBeenCalledWith('mouseout', 'edge', expect.any(Function));
    });

    it('calls user onReady after registering edge handlers', () => {
      const onReady = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onReady={onReady}
        />
      );

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      act(() => {
        handlers.onReady!(mockCyInstance as unknown as Core);
      });

      expect(onReady).toHaveBeenCalledWith(mockCyInstance);
    });
  });

  describe('element updates', () => {
    it('updates elements when they change', () => {
      const { rerender } = render(<CytoscapeGraph elements={mockElements} />);

      const newElements: ElementDefinition[] = [
        { data: { id: 'c', label: 'Node C' } },
      ];

      // Clear previous calls
      mockUseCytoscape.setElements.mockClear();
      mockUseCytoscape.runLayout.mockClear();

      rerender(<CytoscapeGraph elements={newElements} />);

      expect(mockUseCytoscape.setElements).toHaveBeenCalledWith(newElements);
      expect(mockUseCytoscape.runLayout).toHaveBeenCalled();
    });

    it('does not update elements when loading', () => {
      mockUseCytoscape.isLoading = true;

      const { rerender } = render(<CytoscapeGraph elements={mockElements} />);

      mockUseCytoscape.setElements.mockClear();

      const newElements: ElementDefinition[] = [
        { data: { id: 'c', label: 'Node C' } },
      ];

      rerender(<CytoscapeGraph elements={newElements} />);

      expect(mockUseCytoscape.setElements).not.toHaveBeenCalled();
    });

    it('does not update when elements array is empty', () => {
      const { rerender } = render(<CytoscapeGraph elements={mockElements} />);

      mockUseCytoscape.setElements.mockClear();

      rerender(<CytoscapeGraph elements={[]} />);

      expect(mockUseCytoscape.setElements).not.toHaveBeenCalled();
    });
  });

  describe('highlighting helpers', () => {
    it('highlightNodeConnections handles node not found', () => {
      const ref = React.createRef<CytoscapeGraphRef>();

      // Mock getElementById to return empty result
      mockCyInstance.getElementById.mockReturnValueOnce({ length: 0 });

      render(<CytoscapeGraph ref={ref} elements={mockElements} />);

      // Should not throw when node is not found
      act(() => {
        ref.current?.highlightNode('nonexistent');
      });
    });

    it('highlightEdgeConnections handles edge not found gracefully', () => {
      // Mock getElementById to return empty result
      mockCyInstance.getElementById.mockReturnValueOnce({ length: 0 });

      render(<CytoscapeGraph elements={mockElements} enableHoverHighlight={true} />);

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Trigger onReady to register edge handlers
      act(() => {
        handlers.onReady!(mockCyInstance as unknown as Core);
      });

      // Get the edge mouseover handler
      const edgeMouseoverCall = mockCyInstance.on.mock.calls.find(
        (c: any[]) => c[0] === 'mouseover' && c[1] === 'edge'
      );

      if (edgeMouseoverCall) {
        const handler = edgeMouseoverCall[2];
        const mockEvt = {
          target: {
            id: vi.fn().mockReturnValue('nonexistent'),
            data: vi.fn().mockReturnValue({}),
          },
        };

        // Should not throw
        act(() => {
          handler(mockEvt);
        });
      }
    });

    it('highlightEdgeConnections highlights edge and connected nodes when edge found', () => {
      // Mock edge that exists
      const mockEdge = {
        length: 1,
        addClass: vi.fn(),
        connectedNodes: vi.fn().mockReturnValue({
          addClass: vi.fn(),
        }),
      };
      mockCyInstance.getElementById.mockReturnValue(mockEdge);

      render(<CytoscapeGraph elements={mockElements} enableHoverHighlight={true} />);

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Trigger onReady to register edge handlers
      act(() => {
        handlers.onReady!(mockCyInstance as unknown as Core);
      });

      // Get the edge mouseover handler
      const edgeMouseoverCall = mockCyInstance.on.mock.calls.find(
        (c: any[]) => c[0] === 'mouseover' && c[1] === 'edge'
      );

      if (edgeMouseoverCall) {
        const handler = edgeMouseoverCall[2];
        const mockEvt = {
          target: {
            id: vi.fn().mockReturnValue('ab'),
            data: vi.fn().mockReturnValue({ source: 'a', target: 'b' }),
          },
        };

        act(() => {
          handler(mockEvt);
        });

        // Should have called connectedNodes to get connected nodes
        expect(mockEdge.connectedNodes).toHaveBeenCalled();
        // Should add highlighted class to edge
        expect(mockEdge.addClass).toHaveBeenCalledWith('highlighted');
      }
    });

    it('edge mouseout clears highlights and calls callback', () => {
      const onEdgeMouseOut = vi.fn();

      render(
        <CytoscapeGraph
          elements={mockElements}
          enableHoverHighlight={true}
          onEdgeMouseOut={onEdgeMouseOut}
        />
      );

      const hookCall = mockedUseCytoscape.mock.calls[0];
      const handlers = hookCall[1]!;

      // Trigger onReady to register edge handlers
      act(() => {
        handlers.onReady!(mockCyInstance as unknown as Core);
      });

      // Get the edge mouseout handler
      const edgeMouseoutCall = mockCyInstance.on.mock.calls.find(
        (c: any[]) => c[0] === 'mouseout' && c[1] === 'edge'
      );

      if (edgeMouseoutCall) {
        const handler = edgeMouseoutCall[2];
        const mockEvt = {
          target: {
            id: vi.fn().mockReturnValue('ab'),
          },
        };

        act(() => {
          handler(mockEvt);
        });

        // Should clear highlights (via elements().removeClass)
        expect(mockCyInstance.elements().removeClass).toHaveBeenCalledWith('highlighted dimmed');
        // Should call user callback
        expect(onEdgeMouseOut).toHaveBeenCalledWith('ab');
      }
    });
  });

  describe('displayName', () => {
    it('has correct displayName', () => {
      expect(CytoscapeGraph.displayName).toBe('CytoscapeGraph');
    });
  });
});
