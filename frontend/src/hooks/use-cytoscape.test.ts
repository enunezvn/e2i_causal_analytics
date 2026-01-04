/**
 * useCytoscape Hook Tests
 * ========================
 *
 * Tests for the Cytoscape.js graph visualization hook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// =============================================================================
// MOCK CYTOSCAPE
// =============================================================================

// Create mock Cytoscape instance with all methods
const createMockCyInstance = () => {
  const nodes = new Map<string, any>();
  const edges = new Map<string, any>();
  let zoomLevel = 1;
  const eventHandlers: Record<string, Function[]> = {};
  let readyCallback: Function | null = null;

  const createMockElement = (data: any, group: 'nodes' | 'edges') => ({
    data: () => data,
    id: () => data.id,
    addClass: vi.fn().mockReturnThis(),
    removeClass: vi.fn().mockReturnThis(),
    select: vi.fn().mockReturnThis(),
    unselect: vi.fn().mockReturnThis(),
    remove: vi.fn(() => {
      if (group === 'nodes') nodes.delete(data.id);
      else edges.delete(data.id);
    }),
    isNode: () => group === 'nodes',
    isEdge: () => group === 'edges',
  });

  const mockCollection = (elements: any[]) => {
    const collection: any = {
      length: elements.length,
      forEach: (fn: Function) => elements.forEach(fn),
      map: (fn: Function) => elements.map(fn),
      filter: (fn: Function) => mockCollection(elements.filter(fn)),
    };
    // Assign methods that return collection AFTER collection is defined
    // to avoid "Cannot access 'collection' before initialization" error
    collection.addClass = vi.fn().mockReturnValue(collection);
    collection.removeClass = vi.fn().mockReturnValue(collection);
    collection.select = vi.fn().mockReturnValue(collection);
    collection.unselect = vi.fn().mockReturnValue(collection);
    collection.remove = vi.fn(() => {
      elements.forEach(el => {
        if (el.isNode?.()) nodes.delete(el.id());
        else edges.delete(el.id?.());
      });
      return collection;
    });
    return collection;
  };

  const cyInstance: any = {
    destroy: vi.fn(),
    add: vi.fn((elements: any | any[]) => {
      const els = Array.isArray(elements) ? elements : [elements];
      els.forEach(el => {
        const group = el.group || (el.data.source ? 'edges' : 'nodes');
        const mockEl = createMockElement(el.data, group);
        if (group === 'nodes') nodes.set(el.data.id, mockEl);
        else edges.set(el.data.id, mockEl);
      });
      return mockCollection(els.map(el => {
        const group = el.group || (el.data.source ? 'edges' : 'nodes');
        return createMockElement(el.data, group);
      }));
    }),
    elements: vi.fn((selector?: string) => {
      const allElements = [...nodes.values(), ...edges.values()];
      if (!selector) return mockCollection(allElements);
      if (selector === 'node') return mockCollection([...nodes.values()]);
      if (selector === 'edge') return mockCollection([...edges.values()]);
      return mockCollection(allElements);
    }),
    nodes: vi.fn((selector?: string) => {
      const nodeList = [...nodes.values()];
      if (!selector) return mockCollection(nodeList);
      if (selector === ':selected') {
        return mockCollection(nodeList.filter(n => n._selected));
      }
      return mockCollection(nodeList);
    }),
    edges: vi.fn((selector?: string) => {
      const edgeList = [...edges.values()];
      if (selector === ':selected') {
        return mockCollection(edgeList.filter(e => e._selected));
      }
      return mockCollection(edgeList);
    }),
    getElementById: vi.fn((id: string) => {
      const el = nodes.get(id) || edges.get(id);
      return el || createMockElement({ id }, 'nodes');
    }),
    layout: vi.fn((config) => ({
      run: vi.fn(),
      stop: vi.fn(),
      _config: config,
    })),
    fit: vi.fn(),
    center: vi.fn(),
    zoom: vi.fn((level?: number) => {
      if (level !== undefined) {
        zoomLevel = level;
        return cyInstance;
      }
      return zoomLevel;
    }),
    pan: vi.fn(),
    resize: vi.fn(),
    on: vi.fn((event: string, selectorOrHandler: string | Function, handler?: Function) => {
      const eventName = typeof selectorOrHandler === 'string' ? `${event}:${selectorOrHandler}` : event;
      const fn = typeof selectorOrHandler === 'function' ? selectorOrHandler : handler!;
      if (!eventHandlers[eventName]) eventHandlers[eventName] = [];
      eventHandlers[eventName].push(fn);
      return cyInstance;
    }),
    off: vi.fn((event: string) => {
      delete eventHandlers[event];
      return cyInstance;
    }),
    ready: vi.fn((handler: Function) => {
      readyCallback = handler;
      // Call immediately to simulate ready state
      setTimeout(() => handler(), 0);
      return cyInstance;
    }),
    png: vi.fn(() => 'data:image/png;base64,mockPngData'),
    _private: { nodes, edges, eventHandlers, triggerReady: () => readyCallback?.() },
    _nodes: nodes,
    _edges: edges,
  };

  return cyInstance;
};

let mockCyInstance = createMockCyInstance();

// Mock the cytoscape module
vi.mock('cytoscape', () => ({
  default: vi.fn(() => mockCyInstance),
}));

// Import after mocking
import { useCytoscape, defaultCytoscapeStyles, LayoutName } from './use-cytoscape';
import cytoscape from 'cytoscape';

// =============================================================================
// TEST UTILITIES
// =============================================================================

const mockElements = [
  { group: 'nodes', data: { id: 'node1', label: 'Node 1', type: 'kpi' } },
  { group: 'nodes', data: { id: 'node2', label: 'Node 2', type: 'trigger' } },
  { group: 'nodes', data: { id: 'node3', label: 'Node 3', type: 'metric' } },
  { group: 'edges', data: { id: 'edge1', source: 'node1', target: 'node2', label: 'causes' } },
  { group: 'edges', data: { id: 'edge2', source: 'node2', target: 'node3', label: 'impacts' } },
];

// =============================================================================
// TESTS
// =============================================================================

describe('useCytoscape', () => {
  let containerDiv: HTMLDivElement;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCyInstance = createMockCyInstance();
    vi.mocked(cytoscape).mockReturnValue(mockCyInstance);
    containerDiv = document.createElement('div');
    document.body.appendChild(containerDiv);
  });

  afterEach(() => {
    document.body.removeChild(containerDiv);
  });

  // ===========================================================================
  // INITIALIZATION TESTS
  // ===========================================================================

  describe('initialization', () => {
    it('returns initial state with containerRef', () => {
      const { result } = renderHook(() => useCytoscape());

      expect(result.current.containerRef).toBeDefined();
      expect(result.current.containerRef.current).toBeNull();
      expect(result.current.isLoading).toBe(true); // Starts as true
    });

    it('initializes Cytoscape when container ref is set', async () => {
      const { result } = renderHook(() => useCytoscape());

      // Manually set the ref and call initialize
      act(() => {
        result.current.initialize(containerDiv);
      });

      // Wait for ready callback
      await waitFor(() => {
        expect(cytoscape).toHaveBeenCalled();
      });
    });

    it('sets isLoading to false after ready callback', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Wait for the ready callback to fire
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('accepts custom options during initialization', async () => {
      const customOptions = {
        minZoom: 0.5,
        maxZoom: 5,
        panningEnabled: false,
      };

      const { result } = renderHook(() => useCytoscape(customOptions));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(cytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          minZoom: 0.5,
          maxZoom: 5,
          panningEnabled: false,
        })
      );
    });

    it('sets up event handlers from options', async () => {
      const onNodeClick = vi.fn();
      const onEdgeClick = vi.fn();

      const { result } = renderHook(() =>
        useCytoscape({}, { onNodeClick, onEdgeClick })
      );

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('tap', 'node', expect.any(Function));
      expect(mockCyInstance.on).toHaveBeenCalledWith('tap', 'edge', expect.any(Function));
    });

    it('calls onReady handler when graph is ready', async () => {
      const onReady = vi.fn();

      const { result } = renderHook(() => useCytoscape({}, { onReady }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Trigger ready
      await waitFor(() => {
        expect(onReady).toHaveBeenCalledWith(mockCyInstance);
      });
    });

    it('applies initial elements from options', async () => {
      const initialElements = [
        { group: 'nodes', data: { id: 'n1', label: 'Test' } },
      ];

      const { result } = renderHook(() => useCytoscape({ elements: initialElements }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(cytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          elements: initialElements,
        })
      );
    });

    it('applies custom styles from options', async () => {
      const customStyles = [
        { selector: 'node', style: { 'background-color': 'red' } },
      ];

      const { result } = renderHook(() => useCytoscape({ style: customStyles }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(cytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          style: customStyles,
        })
      );
    });

    it('uses default styles when not provided', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(cytoscape).toHaveBeenCalledWith(
        expect.objectContaining({
          style: defaultCytoscapeStyles,
        })
      );
    });
  });

  // ===========================================================================
  // DESTRUCTION TESTS
  // ===========================================================================

  describe('destruction', () => {
    it('destroys Cytoscape instance', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.destroy();
      });

      expect(mockCyInstance.destroy).toHaveBeenCalled();
    });

    it('cleans up on unmount', async () => {
      const { result, unmount } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      unmount();

      expect(mockCyInstance.destroy).toHaveBeenCalled();
    });

    it('handles destroy when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      // Should not throw
      expect(() => {
        act(() => {
          result.current.destroy();
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // ELEMENT MANAGEMENT TESTS
  // ===========================================================================

  describe('element management', () => {
    it('sets elements on the graph', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.setElements(mockElements);
      });

      expect(mockCyInstance.elements).toHaveBeenCalled();
      expect(mockCyInstance.add).toHaveBeenCalledWith(mockElements);
    });

    it('clears existing elements before setting new ones', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Get initial call count for elements()
      const initialCallCount = mockCyInstance.elements.mock.calls.length;

      act(() => {
        result.current.setElements(mockElements);
      });

      // Verify elements() was called (to get collection for removal)
      expect(mockCyInstance.elements.mock.calls.length).toBeGreaterThan(initialCallCount);
      // Verify add was called with the new elements
      expect(mockCyInstance.add).toHaveBeenCalledWith(mockElements);
    });

    it('adds elements to the graph', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      const newElements = [
        { group: 'nodes', data: { id: 'newNode', label: 'New Node' } },
      ];

      act(() => {
        result.current.addElements(newElements);
      });

      expect(mockCyInstance.add).toHaveBeenCalledWith(newElements);
    });

    it('removes elements from the graph by ID', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.removeElements(['node1', 'node2']);
      });

      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node1');
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node2');
    });

    it('handles element operations when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      // These should not throw when not initialized
      expect(() => {
        act(() => {
          result.current.setElements([]);
          result.current.addElements([]);
          result.current.removeElements([]);
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // LAYOUT TESTS
  // ===========================================================================

  describe('layout operations', () => {
    it('runs layout with default options', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.runLayout();
      });

      expect(mockCyInstance.layout).toHaveBeenCalled();
    });

    it.each([
      'cose',
      'grid',
      'circle',
      'concentric',
      'breadthfirst',
      'random',
      'preset',
    ] as LayoutName[])('runs %s layout', async (layoutName) => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.runLayout(layoutName);
      });

      expect(mockCyInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({ name: layoutName })
      );
    });

    it('applies custom layout options', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      const customOptions = { animate: false, padding: 50 };

      act(() => {
        result.current.runLayout('cose', customOptions);
      });

      expect(mockCyInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({ animate: false, padding: 50 })
      );
    });

    it('handles layout when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      expect(() => {
        act(() => {
          result.current.runLayout('cose');
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // VIEWPORT CONTROL TESTS
  // ===========================================================================

  describe('viewport controls', () => {
    it('fits graph to viewport', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.fit();
      });

      expect(mockCyInstance.fit).toHaveBeenCalledWith(undefined, 30);
    });

    it('fits with custom padding', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.fit(50);
      });

      expect(mockCyInstance.fit).toHaveBeenCalledWith(undefined, 50);
    });

    it('centers the graph', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.center();
      });

      expect(mockCyInstance.center).toHaveBeenCalled();
    });

    it('sets zoom level', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.zoom(2);
      });

      expect(mockCyInstance.zoom).toHaveBeenCalledWith(2);
    });

    it('gets current zoom level', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      const zoom = result.current.getZoom();

      expect(mockCyInstance.zoom).toHaveBeenCalled();
      expect(zoom).toBe(1); // Default mock zoom
    });

    it('returns default zoom when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      const zoom = result.current.getZoom();

      expect(zoom).toBe(1); // Default fallback
    });

    it('handles viewport operations when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      expect(() => {
        act(() => {
          result.current.fit();
          result.current.center();
          result.current.zoom(1);
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // SELECTION TESTS
  // ===========================================================================

  describe('selection management', () => {
    it('selects nodes by IDs', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.selectNodes(['node1', 'node2']);
      });

      expect(mockCyInstance.nodes).toHaveBeenCalled();
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node1');
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node2');
    });

    it('clears existing selection before selecting new nodes', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Get initial call count for nodes()
      const initialCallCount = mockCyInstance.nodes.mock.calls.length;

      act(() => {
        result.current.selectNodes(['node1']);
      });

      // Verify nodes() was called (to get collection for unselect)
      // The hook calls nodes().unselect() to clear existing selection
      expect(mockCyInstance.nodes.mock.calls.length).toBeGreaterThan(initialCallCount);
      // And getElementById was called for the new selection
      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node1');
    });

    it('clears all selections', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.clearSelection();
      });

      expect(mockCyInstance.elements).toHaveBeenCalled();
    });

    it('gets selected node IDs', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      const selectedIds = result.current.getSelectedNodeIds();

      expect(mockCyInstance.nodes).toHaveBeenCalledWith(':selected');
      expect(Array.isArray(selectedIds)).toBe(true);
    });

    it('returns empty array for selected nodes when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      const selectedIds = result.current.getSelectedNodeIds();

      expect(selectedIds).toEqual([]);
    });
  });

  // ===========================================================================
  // HIGHLIGHTING TESTS
  // ===========================================================================

  describe('highlighting', () => {
    it('highlights a node by ID', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.highlightNode('node1');
      });

      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node1');
    });

    it('unhighlights a node by ID', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.unhighlightNode('node1');
      });

      expect(mockCyInstance.getElementById).toHaveBeenCalledWith('node1');
    });

    it('clears all highlights', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.clearHighlights();
      });

      expect(mockCyInstance.elements).toHaveBeenCalled();
    });

    it('handles highlight operations when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      expect(() => {
        act(() => {
          result.current.highlightNode('test');
          result.current.unhighlightNode('test');
          result.current.clearHighlights();
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // EXPORT TESTS
  // ===========================================================================

  describe('export', () => {
    it('exports graph as PNG', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      const pngData = result.current.exportPng();

      expect(mockCyInstance.png).toHaveBeenCalledWith({ full: true, scale: 2 });
      expect(pngData).toBe('data:image/png;base64,mockPngData');
    });

    it('returns undefined when instance not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      const pngData = result.current.exportPng();

      expect(pngData).toBeUndefined();
    });
  });

  // ===========================================================================
  // EVENT HANDLER TESTS
  // ===========================================================================

  describe('event handlers', () => {
    it('registers onNodeClick handler', async () => {
      const onNodeClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('tap', 'node', expect.any(Function));
    });

    it('registers onNodeDoubleClick handler', async () => {
      const onNodeDoubleClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeDoubleClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('dbltap', 'node', expect.any(Function));
    });

    it('registers onNodeMouseOver handler', async () => {
      const onNodeMouseOver = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeMouseOver }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('mouseover', 'node', expect.any(Function));
    });

    it('registers onNodeMouseOut handler', async () => {
      const onNodeMouseOut = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeMouseOut }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('mouseout', 'node', expect.any(Function));
    });

    it('registers onEdgeClick handler', async () => {
      const onEdgeClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onEdgeClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('tap', 'edge', expect.any(Function));
    });

    it('registers onSelectionChange handler', async () => {
      const onSelectionChange = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onSelectionChange }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.on).toHaveBeenCalledWith('select unselect', expect.any(Function));
    });

    it('registers onBackgroundClick handler', async () => {
      const onBackgroundClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onBackgroundClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Background click uses 'tap' on cy instance itself
      expect(mockCyInstance.on).toHaveBeenCalledWith('tap', expect.any(Function));
    });

    it('registers ready callback', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.ready).toHaveBeenCalled();
    });

    // Tests that trigger event callbacks to cover handler body code
    it('triggers onNodeClick callback with node id and data', async () => {
      const onNodeClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Get the stored handler and trigger it
      const handlers = mockCyInstance._private.eventHandlers['tap:node'];
      const mockEvent = {
        target: {
          id: () => 'test-node',
          data: () => ({ id: 'test-node', label: 'Test' }),
        },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onNodeClick).toHaveBeenCalledWith('test-node', { id: 'test-node', label: 'Test' });
    });

    it('triggers onNodeDoubleClick callback with node id and data', async () => {
      const onNodeDoubleClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeDoubleClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['dbltap:node'];
      const mockEvent = {
        target: {
          id: () => 'test-node',
          data: () => ({ id: 'test-node', label: 'Test' }),
        },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onNodeDoubleClick).toHaveBeenCalledWith('test-node', { id: 'test-node', label: 'Test' });
    });

    it('triggers onNodeMouseOver callback with node id and data', async () => {
      const onNodeMouseOver = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeMouseOver }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['mouseover:node'];
      const mockEvent = {
        target: {
          id: () => 'test-node',
          data: () => ({ id: 'test-node', label: 'Test' }),
        },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onNodeMouseOver).toHaveBeenCalledWith('test-node', { id: 'test-node', label: 'Test' });
    });

    it('triggers onNodeMouseOut callback with node id', async () => {
      const onNodeMouseOut = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onNodeMouseOut }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['mouseout:node'];
      const mockEvent = {
        target: {
          id: () => 'test-node',
        },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onNodeMouseOut).toHaveBeenCalledWith('test-node');
    });

    it('triggers onEdgeClick callback with edge id and data', async () => {
      const onEdgeClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onEdgeClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['tap:edge'];
      const mockEvent = {
        target: {
          id: () => 'test-edge',
          data: () => ({ id: 'test-edge', source: 'n1', target: 'n2' }),
        },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onEdgeClick).toHaveBeenCalledWith('test-edge', { id: 'test-edge', source: 'n1', target: 'n2' });
    });

    it('triggers onSelectionChange callback with selected node and edge ids', async () => {
      const onSelectionChange = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onSelectionChange }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['select unselect'];

      act(() => {
        handlers?.[0]?.();
      });

      // Callback should receive arrays from nodes(':selected') and edges(':selected')
      expect(onSelectionChange).toHaveBeenCalledWith(expect.any(Array), expect.any(Array));
    });

    it('triggers onBackgroundClick callback when clicking canvas', async () => {
      const onBackgroundClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onBackgroundClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['tap'];
      // evt.target === cy means background was clicked
      const mockEvent = {
        target: mockCyInstance,
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onBackgroundClick).toHaveBeenCalled();
    });

    it('does not trigger onBackgroundClick when clicking on element', async () => {
      const onBackgroundClick = vi.fn();
      const { result } = renderHook(() => useCytoscape({}, { onBackgroundClick }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      const handlers = mockCyInstance._private.eventHandlers['tap'];
      // evt.target !== cy means an element was clicked
      const mockEvent = {
        target: { id: () => 'some-node' },
      };

      act(() => {
        handlers?.[0]?.(mockEvent);
      });

      expect(onBackgroundClick).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // EDGE CASES
  // ===========================================================================

  describe('edge cases', () => {
    it('handles all operations when not initialized', () => {
      const { result } = renderHook(() => useCytoscape());

      // These should not throw
      expect(() => {
        result.current.setElements([]);
        result.current.addElements([]);
        result.current.removeElements([]);
        result.current.runLayout();
        result.current.fit();
        result.current.center();
        result.current.zoom(1);
        result.current.getZoom();
        result.current.selectNodes([]);
        result.current.clearSelection();
        result.current.getSelectedNodeIds();
        result.current.highlightNode('test');
        result.current.unhighlightNode('test');
        result.current.clearHighlights();
        result.current.exportPng();
        result.current.destroy();
      }).not.toThrow();
    });

    it('destroys existing instance on re-initialization', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      vi.clearAllMocks();

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(mockCyInstance.destroy).toHaveBeenCalled();
    });

    it('handles empty node IDs array for selection', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(() => {
        act(() => {
          result.current.selectNodes([]);
        });
      }).not.toThrow();
    });

    it('handles empty IDs array for removal', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      expect(() => {
        act(() => {
          result.current.removeElements([]);
        });
      }).not.toThrow();
    });
  });

  // ===========================================================================
  // AUTO-FIT BEHAVIOR TESTS
  // ===========================================================================

  describe('autoFit behavior', () => {
    it('fits graph on ready when autoFit is true', async () => {
      const { result } = renderHook(() => useCytoscape({ autoFit: true }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      await waitFor(() => {
        expect(mockCyInstance.fit).toHaveBeenCalled();
      });
    });

    it('does not fit graph when autoFit is false', async () => {
      const { result } = renderHook(() => useCytoscape({ autoFit: false }));

      act(() => {
        result.current.initialize(containerDiv);
      });

      // Clear fits from initialization
      const fitCallCount = mockCyInstance.fit.mock.calls.length;

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // No additional fits after ready
      expect(mockCyInstance.fit.mock.calls.length).toBe(fitCallCount);
    });
  });

  // ===========================================================================
  // LAYOUT CONFIGURATION TESTS
  // ===========================================================================

  describe('layout configurations', () => {
    it('cose layout includes animation options', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.runLayout('cose');
      });

      expect(mockCyInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'cose',
          animate: true,
        })
      );
    });

    it('grid layout has avoidOverlap enabled', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.runLayout('grid');
      });

      expect(mockCyInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'grid',
          avoidOverlap: true,
        })
      );
    });

    it('breadthfirst layout is directed', async () => {
      const { result } = renderHook(() => useCytoscape());

      act(() => {
        result.current.initialize(containerDiv);
      });

      act(() => {
        result.current.runLayout('breadthfirst');
      });

      expect(mockCyInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'breadthfirst',
          directed: true,
        })
      );
    });
  });
});

// =============================================================================
// DEFAULT STYLES TESTS
// =============================================================================

describe('defaultCytoscapeStyles', () => {
  it('exports default styles array', () => {
    expect(defaultCytoscapeStyles).toBeDefined();
    expect(Array.isArray(defaultCytoscapeStyles)).toBe(true);
    expect(defaultCytoscapeStyles.length).toBeGreaterThan(0);
  });

  it('contains node styles', () => {
    const nodeStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'node'
    );
    expect(nodeStyle).toBeDefined();
    expect(nodeStyle?.style).toHaveProperty('background-color');
    expect(nodeStyle?.style).toHaveProperty('label');
  });

  it('contains edge styles', () => {
    const edgeStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'edge'
    );
    expect(edgeStyle).toBeDefined();
    expect(edgeStyle?.style).toHaveProperty('width');
    expect(edgeStyle?.style).toHaveProperty('line-color');
    expect(edgeStyle?.style).toHaveProperty('target-arrow-shape');
  });

  it('contains selected node styles', () => {
    const selectedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'node:selected'
    );
    expect(selectedStyle).toBeDefined();
  });

  it('contains selected edge styles', () => {
    const selectedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'edge:selected'
    );
    expect(selectedStyle).toBeDefined();
  });

  it('contains highlighted node styles', () => {
    const highlightedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'node.highlighted'
    );
    expect(highlightedStyle).toBeDefined();
  });

  it('contains highlighted edge styles', () => {
    const highlightedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'edge.highlighted'
    );
    expect(highlightedStyle).toBeDefined();
  });

  it('contains dimmed node styles', () => {
    const dimmedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'node.dimmed'
    );
    expect(dimmedStyle).toBeDefined();
    expect(dimmedStyle?.style).toHaveProperty('opacity');
  });

  it('contains dimmed edge styles', () => {
    const dimmedStyle = defaultCytoscapeStyles.find(
      (s: any) => s.selector === 'edge.dimmed'
    );
    expect(dimmedStyle).toBeDefined();
    expect(dimmedStyle?.style).toHaveProperty('opacity');
  });
});

// =============================================================================
// TYPES EXPORT TESTS
// =============================================================================

describe('type exports', () => {
  it('LayoutName type includes expected values', () => {
    const validLayouts: LayoutName[] = [
      'grid',
      'circle',
      'concentric',
      'breadthfirst',
      'cose',
      'random',
      'preset',
    ];

    validLayouts.forEach((layout) => {
      expect(layout).toBeDefined();
    });
  });
});
