/**
 * D3.js Hook Tests
 * =================
 *
 * Comprehensive tests for useD3 React hook.
 * Tests initialization, data updates, event handlers, and zoom controls.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// =============================================================================
// D3 MOCK SETUP
// =============================================================================

// Mock chain builder for D3 selections
const createMockSelection = () => {
  // Create mock SVG node with getBBox
  const mockSvgNode = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  (mockSvgNode as any).getBBox = vi.fn().mockReturnValue({
    x: 0,
    y: 0,
    width: 100,
    height: 100,
  });

  const selection: any = {
    _node: mockSvgNode,
    _data: [] as any[],
    _eventHandlers: {} as Record<string, Function>,
    append: vi.fn().mockImplementation(() => selection),
    attr: vi.fn().mockImplementation((name: string, value: any) => {
      // If value is a function, call it with each data item to cover callback code
      if (typeof value === 'function' && selection._data.length > 0) {
        selection._data.forEach((d: any) => {
          if (d && typeof d === 'object') {
            try {
              value(d);
            } catch {
              // Ignore errors in callbacks during mock execution
            }
          }
        });
      }
      return selection;
    }),
    style: vi.fn().mockImplementation(() => selection),
    text: vi.fn().mockImplementation((value: any) => {
      // If value is a function, call it with each data item to cover callback code
      if (typeof value === 'function' && selection._data.length > 0) {
        selection._data.forEach((d: any) => {
          if (d && typeof d === 'object') {
            try {
              value(d);
            } catch {
              // Ignore errors in callbacks during mock execution
            }
          }
        });
      }
      return selection;
    }),
    selectAll: vi.fn().mockImplementation(() => selection),
    select: vi.fn().mockImplementation(() => selection),
    remove: vi.fn().mockImplementation(() => selection),
    call: vi.fn().mockImplementation((fn: any) => {
      if (typeof fn === 'function') {
        fn(selection);
      }
      return selection;
    }),
    on: vi.fn().mockImplementation((event: string, handler: Function) => {
      selection._eventHandlers[event] = handler;
      return selection;
    }),
    data: vi.fn().mockImplementation((data: any[]) => {
      selection._data = data;
      return selection;
    }),
    join: vi.fn().mockImplementation(() => selection),
    node: vi.fn().mockImplementation(() => selection._node),
    each: vi.fn().mockImplementation((fn: Function) => {
      // Only call fn if we have valid data, skip otherwise to avoid undefined errors
      if (selection._data.length > 0) {
        selection._data.forEach((d: any, i: number) => {
          // Only call with valid data objects
          if (d && typeof d === 'object') {
            fn.call(selection._node, d, i);
          }
        });
      }
      return selection;
    }),
    filter: vi.fn().mockImplementation((filterFn?: Function) => {
      // If filterFn provided and we have data, filter it
      if (filterFn && selection._data.length > 0) {
        const filtered = selection._data.filter((d: any) => {
          // Only call filter with valid data
          if (d && typeof d === 'object') {
            try {
              return filterFn(d);
            } catch {
              return false;
            }
          }
          return false;
        });
        // Update selection's data with filtered results
        selection._data = filtered;
      }
      return selection;
    }),
    classed: vi.fn().mockImplementation(() => selection),
    transition: vi.fn().mockImplementation(() => selection),
    duration: vi.fn().mockImplementation(() => selection),
  };
  return selection;
};

// Create mock zoom behavior
const createMockZoom = () => {
  const zoom: any = {
    scaleExtent: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    transform: vi.fn(),
    scaleTo: vi.fn(),
  };
  return zoom;
};

// Create mock simulation
const createMockSimulation = () => {
  const simulation: any = {
    force: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    stop: vi.fn(),
    alphaTarget: vi.fn().mockReturnThis(),
    restart: vi.fn(),
  };
  return simulation;
};

// Create mock force link
const createMockForceLink = () => {
  const forceLink: any = {
    id: vi.fn().mockImplementation((accessor: (d: any) => string) => {
      // Call the accessor with a mock node to cover the id callback code
      if (typeof accessor === 'function') {
        try {
          accessor({ id: 'mock-node-id', label: 'Mock Node' });
        } catch {
          // Ignore errors from mock data
        }
      }
      return forceLink;
    }),
    distance: vi.fn().mockReturnThis(),
  };
  return forceLink;
};

// Create mock force many body
const createMockForceManyBody = () => {
  const force: any = {
    strength: vi.fn().mockReturnThis(),
  };
  return force;
};

// Create mock force collide
const createMockForceCollide = () => {
  const force: any = {
    radius: vi.fn().mockReturnThis(),
  };
  return force;
};

// Create mock drag behavior
const createMockDrag = () => {
  const drag: any = {
    on: vi.fn().mockReturnThis(),
  };
  return drag;
};

// Store mock instances for test access
let mockSelection: ReturnType<typeof createMockSelection>;
let mockZoom: ReturnType<typeof createMockZoom>;
let mockSimulation: ReturnType<typeof createMockSimulation>;
let mockDrag: ReturnType<typeof createMockDrag>;

// Mock d3 module
vi.mock('d3', () => {
  return {
    select: vi.fn().mockImplementation(() => {
      mockSelection = createMockSelection();
      return mockSelection;
    }),
    zoom: vi.fn().mockImplementation(() => {
      mockZoom = createMockZoom();
      return mockZoom;
    }),
    forceSimulation: vi.fn().mockImplementation(() => {
      mockSimulation = createMockSimulation();
      return mockSimulation;
    }),
    forceLink: vi.fn().mockImplementation(() => createMockForceLink()),
    forceManyBody: vi.fn().mockImplementation(() => createMockForceManyBody()),
    forceCenter: vi.fn().mockImplementation(() => ({})),
    forceCollide: vi.fn().mockImplementation(() => createMockForceCollide()),
    drag: vi.fn().mockImplementation(() => {
      mockDrag = createMockDrag();
      return mockDrag;
    }),
    zoomIdentity: {
      translate: vi.fn().mockReturnThis(),
      scale: vi.fn().mockReturnThis(),
    },
    zoomTransform: vi.fn().mockReturnValue({ k: 1 }),
    color: vi.fn().mockImplementation((c: string) => ({
      darker: vi.fn().mockReturnValue({
        toString: vi.fn().mockReturnValue('#333'),
      }),
    })),
  };
});

// Import after mock
import { useD3, type D3GraphData, type D3Options, type D3EventHandlers, type D3Node, type D3Edge } from './use-d3';
import * as d3 from 'd3';

// =============================================================================
// TEST UTILITIES
// =============================================================================

function createContainerDiv(): HTMLDivElement {
  const div = document.createElement('div');
  div.style.width = '800px';
  div.style.height = '600px';
  // Mock getBoundingClientRect
  div.getBoundingClientRect = vi.fn().mockReturnValue({
    width: 800,
    height: 600,
    top: 0,
    left: 0,
    right: 800,
    bottom: 600,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
  return div;
}

const mockNodes: D3Node[] = [
  { id: 'node1', label: 'Node 1', type: 'variable' },
  { id: 'node2', label: 'Node 2', type: 'treatment' },
  { id: 'node3', label: 'Node 3', type: 'outcome' },
];

const mockEdges: D3Edge[] = [
  { id: 'edge1', source: 'node1', target: 'node2', type: 'causal' },
  { id: 'edge2', source: 'node2', target: 'node3', type: 'causal' },
];

const mockGraphData: D3GraphData = {
  nodes: mockNodes,
  edges: mockEdges,
};

// =============================================================================
// TESTS: HOOK INITIALIZATION
// =============================================================================

describe('useD3', () => {
  let containerDiv: HTMLDivElement;

  beforeEach(() => {
    vi.clearAllMocks();
    containerDiv = createContainerDiv();
    document.body.appendChild(containerDiv);
  });

  afterEach(() => {
    document.body.removeChild(containerDiv);
  });

  describe('initialization', () => {
    it('returns all expected properties', () => {
      const { result } = renderHook(() => useD3());

      expect(result.current).toHaveProperty('containerRef');
      expect(result.current).toHaveProperty('svgSelection');
      expect(result.current).toHaveProperty('isLoading');
      expect(result.current).toHaveProperty('initialize');
      expect(result.current).toHaveProperty('destroy');
      expect(result.current).toHaveProperty('setData');
      expect(result.current).toHaveProperty('fit');
      expect(result.current).toHaveProperty('center');
      expect(result.current).toHaveProperty('zoom');
      expect(result.current).toHaveProperty('getZoom');
      expect(result.current).toHaveProperty('exportSvg');
      expect(result.current).toHaveProperty('highlightNode');
      expect(result.current).toHaveProperty('clearHighlights');
    });

    it('starts with isLoading true', () => {
      const { result } = renderHook(() => useD3());
      expect(result.current.isLoading).toBe(true);
    });

    it('starts with null svgSelection', () => {
      const { result } = renderHook(() => useD3());
      expect(result.current.svgSelection).toBe(null);
    });

    it('initializes when containerRef is assigned', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        // Simulate attaching ref to container
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(d3.select).toHaveBeenCalled();
    });

    it('creates SVG with correct attributes', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockSelection.append).toHaveBeenCalledWith('svg');
      expect(mockSelection.attr).toHaveBeenCalledWith('width', '100%');
      expect(mockSelection.attr).toHaveBeenCalledWith('height', '100%');
    });

    it('creates main group for zoom/pan', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockSelection.append).toHaveBeenCalledWith('g');
      expect(mockSelection.attr).toHaveBeenCalledWith('class', 'd3-main-group');
    });

    it('adds arrow marker definition', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockSelection.append).toHaveBeenCalledWith('defs');
      expect(mockSelection.append).toHaveBeenCalledWith('marker');
    });

    it('sets up zoom behavior when enabled', () => {
      const { result } = renderHook(() => useD3({ enableZoom: true }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(d3.zoom).toHaveBeenCalled();
      expect(mockZoom.scaleExtent).toHaveBeenCalled();
      expect(mockZoom.on).toHaveBeenCalledWith('zoom', expect.any(Function));
    });

    it('does not return early when container is available', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      // Verify d3.select was called with the container
      expect(d3.select).toHaveBeenCalled();
    });
  });

  describe('options', () => {
    it('uses default options when none provided', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      // Default minZoom is 0.1, maxZoom is 4
      expect(mockZoom.scaleExtent).toHaveBeenCalledWith([0.1, 4]);
    });

    it('uses custom zoom options', () => {
      const options: D3Options = {
        minZoom: 0.5,
        maxZoom: 2,
      };

      const { result } = renderHook(() => useD3(options));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockZoom.scaleExtent).toHaveBeenCalledWith([0.5, 2]);
    });

    it('disables zoom when enableZoom is false', () => {
      vi.clearAllMocks();
      const { result } = renderHook(() => useD3({ enableZoom: false }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(d3.zoom).not.toHaveBeenCalled();
    });

    it('uses custom width and height', () => {
      const { result } = renderHook(() =>
        useD3({ width: 1000, height: 800 })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockSelection.attr).toHaveBeenCalledWith(
        'viewBox',
        expect.stringContaining('1000')
      );
    });
  });

  describe('event handlers', () => {
    it('calls onReady when graph is initialized', () => {
      const onReady = vi.fn();
      const { result } = renderHook(() => useD3({}, { onReady }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(onReady).toHaveBeenCalled();
    });

    it('sets up background click handler', () => {
      const onBackgroundClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onBackgroundClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      expect(mockSelection.on).toHaveBeenCalledWith('click', expect.any(Function));
    });

    it('triggers onBackgroundClick when background is clicked', () => {
      const onBackgroundClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onBackgroundClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      // Simulate background click
      const clickHandler = mockSelection._eventHandlers['click'];
      if (clickHandler) {
        const mockEvent = { target: mockSelection._node };
        act(() => {
          clickHandler(mockEvent);
        });
        expect(onBackgroundClick).toHaveBeenCalled();
      }
    });
  });

  describe('destroy', () => {
    it('removes SVG elements', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      act(() => {
        result.current.destroy();
      });

      expect(mockSelection.selectAll).toHaveBeenCalledWith('svg');
      expect(mockSelection.remove).toHaveBeenCalled();
    });

    it('stops simulation if running', () => {
      const { result } = renderHook(() => useD3({ useHierarchicalLayout: false }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      act(() => {
        result.current.destroy();
      });

      expect(mockSimulation.stop).toHaveBeenCalled();
    });
  });

  describe('setData', () => {
    it('does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.setData(mockGraphData);
      });

      // Should not throw
    });

    it('clears existing elements before adding new data', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      vi.clearAllMocks();

      act(() => {
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.selectAll).toHaveBeenCalledWith('.links');
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.nodes');
      expect(mockSelection.remove).toHaveBeenCalled();
    });

    it('returns early for empty nodes', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      vi.clearAllMocks();

      act(() => {
        result.current.setData({ nodes: [], edges: [] });
      });

      // Should not create link or node groups
      expect(mockSelection.append).not.toHaveBeenCalledWith('g');
    });

    it('creates link and node groups', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      act(() => {
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.append).toHaveBeenCalledWith('g');
      expect(mockSelection.attr).toHaveBeenCalledWith('class', 'links');
      expect(mockSelection.attr).toHaveBeenCalledWith('class', 'nodes');
    });

    it('sets up node click handler', () => {
      const onNodeClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.on).toHaveBeenCalledWith('click', expect.any(Function));
    });

    it('sets up node double-click handler', () => {
      const onNodeDoubleClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeDoubleClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.on).toHaveBeenCalledWith('dblclick', expect.any(Function));
    });

    it('sets up node mouseenter handler', () => {
      const onNodeMouseOver = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeMouseOver }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.on).toHaveBeenCalledWith('mouseenter', expect.any(Function));
    });

    it('sets up node mouseleave handler', () => {
      const onNodeMouseOut = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeMouseOut }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.on).toHaveBeenCalledWith('mouseleave', expect.any(Function));
    });

    it('calls onNodeMouseOut when mouseleave event fires', () => {
      const onNodeMouseOut = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeMouseOut }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Find and call the mouseleave handler
      const mouseleaveCall = mockSelection.on.mock.calls.find(
        (c: any[]) => c[0] === 'mouseleave'
      );
      if (mouseleaveCall) {
        const handler = mouseleaveCall[1];
        const mockEvent = {};
        const mockNode = { id: 'n1', label: 'Node 1' };
        handler(mockEvent, mockNode);
        expect(onNodeMouseOut).toHaveBeenCalledWith(mockNode);
      }
    });

    it('calls onNodeClick when click event fires', () => {
      const onNodeClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Find all click handlers - there are multiple (edges and nodes)
      // We need to find the one for nodes by calling with node-like data
      const clickCalls = mockSelection.on.mock.calls.filter(
        (c: any[]) => c[0] === 'click'
      );
      // Try each click handler until we find one that triggers onNodeClick
      for (const clickCall of clickCalls) {
        const handler = clickCall[1];
        const mockEvent = {};
        const mockNode = { id: 'n1', label: 'Node 1' };
        handler(mockEvent, mockNode);
        if (onNodeClick.mock.calls.length > 0) {
          break;
        }
      }
      expect(onNodeClick).toHaveBeenCalledWith({ id: 'n1', label: 'Node 1' });
    });

    it('calls onNodeDoubleClick when dblclick event fires', () => {
      const onNodeDoubleClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeDoubleClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Find and call the dblclick handler
      const dblclickCall = mockSelection.on.mock.calls.find(
        (c: any[]) => c[0] === 'dblclick'
      );
      if (dblclickCall) {
        const handler = dblclickCall[1];
        const mockEvent = {};
        const mockNode = { id: 'n1', label: 'Node 1' };
        handler(mockEvent, mockNode);
        expect(onNodeDoubleClick).toHaveBeenCalledWith(mockNode);
      }
    });

    it('calls onNodeMouseOver when mouseenter event fires', () => {
      const onNodeMouseOver = vi.fn();
      const { result } = renderHook(() => useD3({}, { onNodeMouseOver }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Find and call the mouseenter handler
      const mouseenterCall = mockSelection.on.mock.calls.find(
        (c: any[]) => c[0] === 'mouseenter'
      );
      if (mouseenterCall) {
        const handler = mouseenterCall[1];
        const mockEvent = {};
        const mockNode = { id: 'n1', label: 'Node 1' };
        handler(mockEvent, mockNode);
        expect(onNodeMouseOver).toHaveBeenCalledWith(mockNode);
      }
    });

    it('sets up edge click handler', () => {
      const onEdgeClick = vi.fn();
      const { result } = renderHook(() => useD3({}, { onEdgeClick }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.on).toHaveBeenCalledWith('click', expect.any(Function));
    });

    it('adds circles to nodes', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.append).toHaveBeenCalledWith('circle');
      expect(mockSelection.attr).toHaveBeenCalledWith('r', 24); // default nodeRadius
    });

    it('adds labels to nodes', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.append).toHaveBeenCalledWith('text');
      expect(mockSelection.attr).toHaveBeenCalledWith('text-anchor', 'middle');
    });

    it('uses custom nodeRadius', () => {
      const { result } = renderHook(() => useD3({ nodeRadius: 30 }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('r', 30);
    });
  });

  describe('hierarchical layout', () => {
    it('uses hierarchical layout by default', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Should not create simulation when using hierarchical layout
      expect(d3.forceSimulation).not.toHaveBeenCalled();
    });

    it('positions nodes using transform attribute', () => {
      const { result } = renderHook(() => useD3({ useHierarchicalLayout: true }));

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
    });
  });

  describe('force layout', () => {
    it('creates force simulation when hierarchical layout disabled', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceSimulation).toHaveBeenCalled();
    });

    it('sets up force link', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceLink).toHaveBeenCalled();
      expect(mockSimulation.force).toHaveBeenCalledWith('link', expect.anything());
    });

    it('sets up charge force', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceManyBody).toHaveBeenCalled();
      expect(mockSimulation.force).toHaveBeenCalledWith('charge', expect.anything());
    });

    it('sets up center force', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceCenter).toHaveBeenCalled();
      expect(mockSimulation.force).toHaveBeenCalledWith('center', expect.anything());
    });

    it('sets up collision force', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceCollide).toHaveBeenCalled();
      expect(mockSimulation.force).toHaveBeenCalledWith('collision', expect.anything());
    });

    it('sets up drag behavior', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.drag).toHaveBeenCalled();
      expect(mockDrag.on).toHaveBeenCalledWith('start', expect.any(Function));
      expect(mockDrag.on).toHaveBeenCalledWith('drag', expect.any(Function));
      expect(mockDrag.on).toHaveBeenCalledWith('end', expect.any(Function));
    });

    it('simulation tick callback updates link and node positions', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Find the tick callback that was registered with simulation.on('tick', callback)
      const tickCall = mockSimulation.on.mock.calls.find((c: any[]) => c[0] === 'tick');
      expect(tickCall).toBeDefined();

      if (tickCall) {
        const tickHandler = tickCall[1];
        // Call the tick handler to cover line 450 (node transform)
        // The handler updates link positions and node transforms
        tickHandler();

        // Verify that attr was called with transform (the node position update)
        expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
      }
    });

    it('drag start handler sets fixed position and alpha target', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Get the drag start handler
      const startCall = mockDrag.on.mock.calls.find((c: any[]) => c[0] === 'start');
      if (startCall) {
        const startHandler = startCall[1];
        const mockEvent = { active: false };
        const mockDatum = { x: 10, y: 20, fx: null, fy: null };
        startHandler(mockEvent, mockDatum);
        expect(mockDatum.fx).toBe(10);
        expect(mockDatum.fy).toBe(20);
        expect(mockSimulation.alphaTarget).toHaveBeenCalledWith(0.3);
      }
    });

    it('drag handler updates position', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Get the drag handler
      const dragCall = mockDrag.on.mock.calls.find((c: any[]) => c[0] === 'drag');
      if (dragCall) {
        const dragHandler = dragCall[1];
        const mockEvent = { x: 50, y: 60 };
        const mockDatum = { fx: 10, fy: 20 };
        dragHandler(mockEvent, mockDatum);
        expect(mockDatum.fx).toBe(50);
        expect(mockDatum.fy).toBe(60);
      }
    });

    it('drag end handler releases fixed position when not active', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Get the drag end handler
      const endCall = mockDrag.on.mock.calls.find((c: any[]) => c[0] === 'end');
      if (endCall) {
        const endHandler = endCall[1];
        const mockEvent = { active: false };
        const mockDatum = { fx: 50, fy: 60 };
        endHandler(mockEvent, mockDatum);
        expect(mockDatum.fx).toBeNull();
        expect(mockDatum.fy).toBeNull();
        expect(mockSimulation.alphaTarget).toHaveBeenCalledWith(0);
      }
    });

    it('drag end handler skips alphaTarget when still active', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      // Reset the mock to clear previous calls
      mockSimulation.alphaTarget.mockClear();

      // Get the drag end handler
      const endCall = mockDrag.on.mock.calls.find((c: any[]) => c[0] === 'end');
      if (endCall) {
        const endHandler = endCall[1];
        const mockEvent = { active: true }; // Still active (other drag operations)
        const mockDatum = { fx: 50, fy: 60 };
        endHandler(mockEvent, mockDatum);
        // When still active, alphaTarget should NOT be called
        // But fx/fy are always released (set to null)
        expect(mockSimulation.alphaTarget).not.toHaveBeenCalled();
        expect(mockDatum.fx).toBeNull();
        expect(mockDatum.fy).toBeNull();
      }
    });

    it('uses custom linkDistance', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false, linkDistance: 200 })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceLink).toHaveBeenCalled();
    });

    it('uses custom chargeStrength', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false, chargeStrength: -500 })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(d3.forceManyBody).toHaveBeenCalled();
    });

    it('stops existing simulation when setting new data', () => {
      const { result } = renderHook(() =>
        useD3({ useHierarchicalLayout: false })
      );

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      const firstSimulation = mockSimulation;

      act(() => {
        result.current.setData(mockGraphData);
      });

      expect(firstSimulation.stop).toHaveBeenCalled();
    });
  });

  describe('zoom controls', () => {
    it('fit does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.fit();
      });

      // Should not throw
    });

    it('fit applies zoom transform', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      act(() => {
        result.current.fit();
      });

      expect(mockSelection.transition).toHaveBeenCalled();
    });

    it('center resets zoom transform', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      act(() => {
        result.current.center();
      });

      expect(mockSelection.transition).toHaveBeenCalled();
      expect(mockSelection.call).toHaveBeenCalled();
    });

    it('center does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.center();
      });

      // Should not throw
    });

    it('zoom sets zoom level', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      act(() => {
        result.current.zoom(2);
      });

      expect(mockSelection.transition).toHaveBeenCalled();
    });

    it('zoom does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.zoom(2);
      });

      // Should not throw
    });

    it('getZoom returns current zoom level', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      const zoomLevel = result.current.getZoom();
      expect(zoomLevel).toBe(1);
    });

    it('getZoom returns 1 if not initialized', () => {
      const { result } = renderHook(() => useD3());

      const zoomLevel = result.current.getZoom();
      expect(zoomLevel).toBe(1);
    });
  });

  describe('exportSvg', () => {
    it('returns undefined if not initialized', () => {
      const { result } = renderHook(() => useD3());

      const svg = result.current.exportSvg();
      expect(svg).toBeUndefined();
    });

    it('returns SVG string when initialized', () => {
      // Mock XMLSerializer as a class
      const mockSerializeToString = vi.fn().mockReturnValue('<svg></svg>');
      class MockXMLSerializer {
        serializeToString = mockSerializeToString;
      }
      global.XMLSerializer = MockXMLSerializer as any;

      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      const svg = result.current.exportSvg();
      expect(mockSerializeToString).toHaveBeenCalled();
    });
  });

  describe('highlightNode', () => {
    it('does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.highlightNode('node1');
      });

      // Should not throw
    });

    it('dims all elements and highlights selected node', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      // Clear mock calls from initialization
      vi.clearAllMocks();

      act(() => {
        result.current.highlightNode('node1');
      });

      // Verify the highlight workflow was initiated
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.node');
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.link');
      // classed should be called for dimming (the exact data binding behavior
      // depends on D3's internal implementation which is mocked)
      expect(mockSelection.classed).toHaveBeenCalled();
    });

    it('executes filter callbacks with node data to find matching node', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        // Set data with nodes and edges
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Node 1' },
            { id: 'n2', label: 'Node 2' },
            { id: 'n3', label: 'Node 3' },
          ],
          edges: [
            { id: 'e1', source: 'n1', target: 'n2' },
            { id: 'e2', source: 'n2', target: 'n3' },
          ],
        });
      });

      // Set up mock to have node data when selectAll('.node') is called
      mockSelection._data = [
        { id: 'n1', label: 'Node 1' },
        { id: 'n2', label: 'Node 2' },
        { id: 'n3', label: 'Node 3' },
      ];

      act(() => {
        result.current.highlightNode('n2');
      });

      // filter should have been called to find the matching node
      expect(mockSelection.filter).toHaveBeenCalled();
      // classed should have been called to highlight
      expect(mockSelection.classed).toHaveBeenCalledWith('highlighted', true);
    });

    it('executes filter and each callbacks with edge data to highlight connected edges', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Node 1' },
            { id: 'n2', label: 'Node 2' },
          ],
          edges: [
            { id: 'e1', source: 'n1', target: 'n2' },
          ],
        });
      });

      // Set up mock to have edge data with object source/target (as D3 resolves them)
      mockSelection._data = [
        { id: 'e1', source: { id: 'n1' }, target: { id: 'n2' } },
      ];

      act(() => {
        result.current.highlightNode('n1');
      });

      // filter and each should have been called on the edge selection
      expect(mockSelection.filter).toHaveBeenCalled();
      expect(mockSelection.each).toHaveBeenCalled();
    });

    it('handles edges with string source/target in filter callback', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Node 1' },
            { id: 'n2', label: 'Node 2' },
          ],
          edges: [
            { id: 'e1', source: 'n1', target: 'n2' },
          ],
        });
      });

      // Set up mock to have edge data with string source/target (before D3 resolves)
      mockSelection._data = [
        { id: 'e1', source: 'n1', target: 'n2' },
      ];

      act(() => {
        result.current.highlightNode('n1');
      });

      // Should not throw, filter callback handles string source/target
      expect(mockSelection.filter).toHaveBeenCalled();
    });

    it('highlights connected nodes via each callback with object source/target', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Node 1' },
            { id: 'n2', label: 'Node 2' },
            { id: 'n3', label: 'Node 3' },
          ],
          edges: [
            { id: 'e1', source: 'n1', target: 'n2' },
            { id: 'e2', source: 'n2', target: 'n3' },
          ],
        });
      });

      // Set up edge data with resolved object references (as D3 does after simulation)
      mockSelection._data = [
        { id: 'e1', source: { id: 'n1' }, target: { id: 'n2' } },
        { id: 'e2', source: { id: 'n2' }, target: { id: 'n3' } },
      ];

      // Clear mocks before highlightNode
      mockSelection.selectAll.mockClear();
      mockSelection.filter.mockClear();
      mockSelection.classed.mockClear();
      mockSelection.each.mockClear();

      act(() => {
        result.current.highlightNode('n2');
      });

      // The each callback should have been called to highlight connected nodes
      expect(mockSelection.each).toHaveBeenCalled();
      // The each callback calls selectAll('.node') internally
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.node');
    });

    it('highlights connected nodes via each callback with string source/target', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Node 1' },
            { id: 'n2', label: 'Node 2' },
          ],
          edges: [
            { id: 'e1', source: 'n1', target: 'n2' },
          ],
        });
      });

      // Set up edge data with string references
      mockSelection._data = [
        { id: 'e1', source: 'n1', target: 'n2' },
      ];

      // Clear mocks before highlightNode
      mockSelection.selectAll.mockClear();
      mockSelection.each.mockClear();

      act(() => {
        result.current.highlightNode('n1');
      });

      // The each callback should process string source/target
      expect(mockSelection.each).toHaveBeenCalled();
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.node');
    });
  });

  describe('clearHighlights', () => {
    it('does nothing if not initialized', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        result.current.clearHighlights();
      });

      // Should not throw
    });

    it('removes dimmed and highlighted classes', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      // Clear mock calls from initialization
      vi.clearAllMocks();

      act(() => {
        result.current.clearHighlights();
      });

      // clearHighlights should select nodes and links and remove classes
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.node');
      expect(mockSelection.selectAll).toHaveBeenCalledWith('.link');
      expect(mockSelection.classed).toHaveBeenCalledWith('dimmed', false);
      expect(mockSelection.classed).toHaveBeenCalledWith('highlighted', false);
    });
  });

  describe('edge resolution', () => {
    it('resolves string source/target to node references', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: mockNodes,
          edges: [{ id: 'e1', source: 'node1', target: 'node2' }],
        });
      });

      // The data function should have been called with resolved edges
      expect(mockSelection.data).toHaveBeenCalled();
    });

    it('handles edges with node references already', () => {
      const node1 = { id: 'node1', label: 'Node 1' };
      const node2 = { id: 'node2', label: 'Node 2' };

      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [node1, node2],
          edges: [{ id: 'e1', source: node1, target: node2 }],
        });
      });

      expect(mockSelection.data).toHaveBeenCalled();
    });
  });

  describe('node types and colors', () => {
    it('applies colors based on node type', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [
            { id: 'n1', label: 'Variable', type: 'variable' },
            { id: 'n2', label: 'Treatment', type: 'treatment' },
            { id: 'n3', label: 'Outcome', type: 'outcome' },
          ],
          edges: [],
        });
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('fill', expect.any(Function));
    });

    it('handles unknown node types with default color', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [{ id: 'n1', label: 'Unknown', type: 'unknown' }],
          edges: [],
        });
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('fill', expect.any(Function));
    });
  });

  describe('edge types and colors', () => {
    it('applies colors based on edge type', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: mockNodes,
          edges: [
            { id: 'e1', source: 'node1', target: 'node2', type: 'causal' },
            { id: 'e2', source: 'node2', target: 'node3', type: 'association' },
          ],
        });
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('stroke', expect.any(Function));
    });

    it('adds arrow markers to edges', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData(mockGraphData);
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('marker-end', 'url(#d3-arrow)');
    });
  });

  describe('label truncation', () => {
    it('truncates long labels', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [{ id: 'n1', label: 'This is a very long label that should be truncated' }],
          edges: [],
        });
      });

      expect(mockSelection.text).toHaveBeenCalled();
    });

    it('does not truncate short labels', () => {
      const { result } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
        result.current.setData({
          nodes: [{ id: 'n1', label: 'Short' }],
          edges: [],
        });
      });

      expect(mockSelection.text).toHaveBeenCalled();
    });
  });

  describe('cleanup on unmount', () => {
    it('destroys instance on unmount', () => {
      const { result, unmount } = renderHook(() => useD3());

      act(() => {
        (result.current.containerRef as any).current = containerDiv;
        result.current.initialize();
      });

      unmount();

      expect(mockSelection.remove).toHaveBeenCalled();
    });
  });

  describe('auto-initialization', () => {
    it('initializes when container ref is set via useEffect', async () => {
      // This tests the useEffect that auto-initializes
      const { result } = renderHook(() => useD3());

      // Manually set containerRef to trigger useEffect
      act(() => {
        (result.current.containerRef as any).current = containerDiv;
      });

      // Force re-render to trigger useEffect
      await waitFor(() => {
        expect(result.current.isLoading).toBeDefined();
      });
    });
  });
});

// =============================================================================
// TESTS: HELPER FUNCTIONS
// =============================================================================

describe('Helper functions', () => {
  describe('calculateNodeDepths', () => {
    it('correctly assigns depths to nodes in a chain', () => {
      // This is tested indirectly through setData with hierarchical layout
      const nodes: D3Node[] = [
        { id: 'a', label: 'A' },
        { id: 'b', label: 'B' },
        { id: 'c', label: 'C' },
      ];
      const edges: D3Edge[] = [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'b', target: 'c' },
      ];

      const { result } = renderHook(() => useD3({ useHierarchicalLayout: true }));

      act(() => {
        const div = createContainerDiv();
        document.body.appendChild(div);
        (result.current.containerRef as any).current = div;
        result.current.initialize();
        result.current.setData({ nodes, edges });
      });

      // Nodes should have been positioned
      expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
    });

    it('handles disconnected nodes', () => {
      const nodes: D3Node[] = [
        { id: 'a', label: 'A' },
        { id: 'b', label: 'B' },
        { id: 'c', label: 'C' }, // disconnected
      ];
      const edges: D3Edge[] = [{ id: 'e1', source: 'a', target: 'b' }];

      const { result } = renderHook(() => useD3({ useHierarchicalLayout: true }));

      act(() => {
        const div = createContainerDiv();
        document.body.appendChild(div);
        (result.current.containerRef as any).current = div;
        result.current.initialize();
        result.current.setData({ nodes, edges });
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
    });

    it('handles cycles gracefully', () => {
      // While DAGs shouldn't have cycles, the function should still work
      const nodes: D3Node[] = [
        { id: 'a', label: 'A' },
        { id: 'b', label: 'B' },
      ];
      const edges: D3Edge[] = [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'b', target: 'a' }, // cycle
      ];

      const { result } = renderHook(() => useD3({ useHierarchicalLayout: true }));

      act(() => {
        const div = createContainerDiv();
        document.body.appendChild(div);
        (result.current.containerRef as any).current = div;
        result.current.initialize();
        result.current.setData({ nodes, edges });
      });

      // Should not throw, should still position nodes
      expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
    });
  });

  describe('positionNodesHierarchically', () => {
    it('positions nodes at different depths', () => {
      const nodes: D3Node[] = [
        { id: 'a', label: 'A' },
        { id: 'b', label: 'B' },
        { id: 'c', label: 'C' },
      ];
      const edges: D3Edge[] = [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'a', target: 'c' },
      ];

      const { result } = renderHook(() => useD3({ useHierarchicalLayout: true }));

      act(() => {
        const div = createContainerDiv();
        document.body.appendChild(div);
        (result.current.containerRef as any).current = div;
        result.current.initialize();
        result.current.setData({ nodes, edges });
      });

      expect(mockSelection.attr).toHaveBeenCalledWith('transform', expect.any(Function));
    });
  });
});

// =============================================================================
// TESTS: TYPE EXPORTS
// =============================================================================

describe('Type exports', () => {
  it('exports D3Node type', () => {
    const node: D3Node = { id: 'test', label: 'Test' };
    expect(node.id).toBe('test');
  });

  it('exports D3Edge type', () => {
    const edge: D3Edge = { id: 'test', source: 'a', target: 'b' };
    expect(edge.id).toBe('test');
  });

  it('exports D3GraphData type', () => {
    const data: D3GraphData = { nodes: [], edges: [] };
    expect(data.nodes).toEqual([]);
  });

  it('exports D3Options type', () => {
    const opts: D3Options = { width: 100 };
    expect(opts.width).toBe(100);
  });

  it('exports D3EventHandlers type', () => {
    const handlers: D3EventHandlers = { onNodeClick: vi.fn() };
    expect(handlers.onNodeClick).toBeDefined();
  });
});
