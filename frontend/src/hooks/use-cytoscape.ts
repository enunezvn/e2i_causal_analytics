/**
 * Cytoscape.js Hook
 * ==================
 *
 * React hook for managing Cytoscape.js graph instances.
 * Handles initialization, cleanup, and provides common graph operations.
 *
 * Features:
 * - Automatic instance lifecycle management
 * - Layout algorithms (cola, cose-bilkent, dagre, etc.)
 * - Event handling for node/edge interactions
 * - Zoom and pan controls
 * - Element selection management
 *
 * @module hooks/use-cytoscape
 */

import { useRef, useCallback, useEffect, useState } from 'react';
import cytoscape from 'cytoscape';
import type { Core, ElementDefinition, LayoutOptions, StylesheetStyle } from 'cytoscape';

// =============================================================================
// TYPES
// =============================================================================

/**
 * Layout algorithm names supported by the hook
 */
export type LayoutName =
  | 'grid'
  | 'circle'
  | 'concentric'
  | 'breadthfirst'
  | 'cose'
  | 'random'
  | 'preset';

/**
 * Options for initializing the Cytoscape instance
 */
export interface CytoscapeOptions {
  /** Initial elements (nodes and edges) */
  elements?: ElementDefinition[];
  /** Stylesheet for node/edge styling */
  style?: StylesheetStyle[];
  /** Initial layout algorithm */
  layout?: LayoutName;
  /** Layout-specific options */
  layoutOptions?: Partial<LayoutOptions>;
  /** Whether to fit the graph to the container on load */
  autoFit?: boolean;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** Whether panning is enabled */
  panningEnabled?: boolean;
  /** Whether user zooming is enabled */
  userZoomingEnabled?: boolean;
  /** Whether box selection is enabled */
  boxSelectionEnabled?: boolean;
}

/**
 * Event handlers for graph interactions
 */
export interface CytoscapeEventHandlers {
  /** Called when a node is clicked */
  onNodeClick?: (nodeId: string, nodeData: Record<string, unknown>) => void;
  /** Called when a node is double-clicked */
  onNodeDoubleClick?: (nodeId: string, nodeData: Record<string, unknown>) => void;
  /** Called when mouse enters a node */
  onNodeMouseOver?: (nodeId: string, nodeData: Record<string, unknown>) => void;
  /** Called when mouse leaves a node */
  onNodeMouseOut?: (nodeId: string) => void;
  /** Called when an edge is clicked */
  onEdgeClick?: (edgeId: string, edgeData: Record<string, unknown>) => void;
  /** Called when the selection changes */
  onSelectionChange?: (selectedNodeIds: string[], selectedEdgeIds: string[]) => void;
  /** Called when the graph is ready */
  onReady?: (cy: Core) => void;
  /** Called when the canvas background is clicked */
  onBackgroundClick?: () => void;
}

/**
 * Return value of the useCytoscape hook
 */
export interface UseCytoscapeReturn {
  /** Ref to attach to the container element */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** The Cytoscape instance (null before initialization) */
  cyInstance: Core | null;
  /** Whether the graph is currently loading/initializing */
  isLoading: boolean;
  /** Initialize or reinitialize the graph */
  initialize: (container: HTMLElement) => void;
  /** Destroy the graph instance */
  destroy: () => void;
  /** Update graph elements */
  setElements: (elements: ElementDefinition[]) => void;
  /** Add elements to the graph */
  addElements: (elements: ElementDefinition[]) => void;
  /** Remove elements from the graph */
  removeElements: (ids: string[]) => void;
  /** Run a layout algorithm */
  runLayout: (name?: LayoutName, options?: Partial<LayoutOptions>) => void;
  /** Fit the graph to the viewport */
  fit: (padding?: number) => void;
  /** Center the graph */
  center: () => void;
  /** Zoom to a specific level */
  zoom: (level: number) => void;
  /** Get the current zoom level */
  getZoom: () => number;
  /** Select specific nodes */
  selectNodes: (nodeIds: string[]) => void;
  /** Clear selection */
  clearSelection: () => void;
  /** Get selected node IDs */
  getSelectedNodeIds: () => string[];
  /** Highlight a node (add highlight class) */
  highlightNode: (nodeId: string) => void;
  /** Remove highlight from a node */
  unhighlightNode: (nodeId: string) => void;
  /** Clear all highlights */
  clearHighlights: () => void;
  /** Export the graph as PNG */
  exportPng: () => string | undefined;
}

// =============================================================================
// DEFAULT STYLES
// =============================================================================

/**
 * Default stylesheet for Cytoscape graphs following the app's design system.
 * Uses CSS custom properties for theming compatibility.
 */
export const defaultCytoscapeStyles: StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': '#3b82f6', // blue-500
      'label': 'data(label)',
      'text-valign': 'bottom',
      'text-halign': 'center',
      'font-size': '12px',
      'color': '#374151', // gray-700
      'text-margin-y': 8,
      'width': 40,
      'height': 40,
      'border-width': 2,
      'border-color': '#2563eb', // blue-600
    },
  },
  {
    selector: 'node:selected',
    style: {
      'background-color': '#1d4ed8', // blue-700
      'border-color': '#1e40af', // blue-800
      'border-width': 3,
    },
  },
  {
    selector: 'node.highlighted',
    style: {
      'background-color': '#f59e0b', // amber-500
      'border-color': '#d97706', // amber-600
      'border-width': 3,
    },
  },
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.3,
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#9ca3af', // gray-400
      'target-arrow-color': '#9ca3af',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 1.2,
    },
  },
  {
    selector: 'edge:selected',
    style: {
      'line-color': '#3b82f6', // blue-500
      'target-arrow-color': '#3b82f6',
      'width': 3,
    },
  },
  {
    selector: 'edge.highlighted',
    style: {
      'line-color': '#f59e0b', // amber-500
      'target-arrow-color': '#f59e0b',
      'width': 3,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      'opacity': 0.2,
    },
  },
];

// =============================================================================
// DEFAULT LAYOUT OPTIONS
// =============================================================================

/**
 * Get default options for a layout algorithm
 */
function getLayoutOptions(name: LayoutName): LayoutOptions {
  const baseOptions: Record<LayoutName, LayoutOptions> = {
    grid: {
      name: 'grid',
      fit: true,
      padding: 30,
      avoidOverlap: true,
    },
    circle: {
      name: 'circle',
      fit: true,
      padding: 30,
      avoidOverlap: true,
    },
    concentric: {
      name: 'concentric',
      fit: true,
      padding: 30,
      avoidOverlap: true,
      minNodeSpacing: 50,
    },
    breadthfirst: {
      name: 'breadthfirst',
      fit: true,
      padding: 30,
      directed: true,
      spacingFactor: 1.5,
    },
    cose: {
      name: 'cose',
      fit: true,
      padding: 30,
      animate: true,
      animationDuration: 500,
      nodeOverlap: 20,
      idealEdgeLength: 100,
      edgeElasticity: 100,
      nestingFactor: 1.2,
    },
    random: {
      name: 'random',
      fit: true,
      padding: 30,
    },
    preset: {
      name: 'preset',
      fit: true,
      padding: 30,
    },
  };

  return baseOptions[name];
}

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * React hook for managing a Cytoscape.js graph instance.
 *
 * @param options - Configuration options for the graph
 * @param eventHandlers - Event handlers for user interactions
 * @returns Object with refs, state, and control functions
 *
 * @example
 * ```tsx
 * function GraphComponent() {
 *   const {
 *     containerRef,
 *     isLoading,
 *     setElements,
 *     runLayout
 *   } = useCytoscape({
 *     layout: 'cose',
 *     autoFit: true
 *   }, {
 *     onNodeClick: (id, data) => console.log('Clicked:', id)
 *   });
 *
 *   useEffect(() => {
 *     setElements(myElements);
 *   }, [myElements, setElements]);
 *
 *   return <div ref={containerRef} style={{ width: '100%', height: '500px' }} />;
 * }
 * ```
 */
export function useCytoscape(
  options: CytoscapeOptions = {},
  eventHandlers: CytoscapeEventHandlers = {}
): UseCytoscapeReturn {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Store event handlers in a ref to avoid re-initialization on handler changes
  const handlersRef = useRef(eventHandlers);
  handlersRef.current = eventHandlers;

  const {
    elements = [],
    style = defaultCytoscapeStyles,
    layout = 'cose',
    layoutOptions = {},
    autoFit = true,
    minZoom = 0.1,
    maxZoom = 3,
    panningEnabled = true,
    userZoomingEnabled = true,
    boxSelectionEnabled = true,
  } = options;

  /**
   * Initialize the Cytoscape instance
   */
  const initialize = useCallback(
    (container: HTMLElement) => {
      // Destroy existing instance if any
      if (cyRef.current) {
        cyRef.current.destroy();
      }

      setIsLoading(true);

      // Create new Cytoscape instance
      const cy = cytoscape({
        container,
        elements,
        style,
        layout: { ...getLayoutOptions(layout), ...layoutOptions },
        minZoom,
        maxZoom,
        panningEnabled,
        userZoomingEnabled,
        boxSelectionEnabled,
        autounselectify: false,
        autoungrabify: false,
      });

      // Set up event listeners
      cy.on('tap', 'node', (evt) => {
        const node = evt.target;
        handlersRef.current.onNodeClick?.(node.id(), node.data());
      });

      cy.on('dbltap', 'node', (evt) => {
        const node = evt.target;
        handlersRef.current.onNodeDoubleClick?.(node.id(), node.data());
      });

      cy.on('mouseover', 'node', (evt) => {
        const node = evt.target;
        handlersRef.current.onNodeMouseOver?.(node.id(), node.data());
      });

      cy.on('mouseout', 'node', (evt) => {
        const node = evt.target;
        handlersRef.current.onNodeMouseOut?.(node.id());
      });

      cy.on('tap', 'edge', (evt) => {
        const edge = evt.target;
        handlersRef.current.onEdgeClick?.(edge.id(), edge.data());
      });

      cy.on('select unselect', () => {
        const selectedNodes = cy.nodes(':selected').map((n) => n.id());
        const selectedEdges = cy.edges(':selected').map((e) => e.id());
        handlersRef.current.onSelectionChange?.(selectedNodes, selectedEdges);
      });

      cy.on('tap', (evt) => {
        if (evt.target === cy) {
          handlersRef.current.onBackgroundClick?.();
        }
      });

      cy.ready(() => {
        if (autoFit) {
          cy.fit(undefined, 30);
        }
        setIsLoading(false);
        handlersRef.current.onReady?.(cy);
      });

      cyRef.current = cy;
    },
    [
      elements,
      style,
      layout,
      layoutOptions,
      autoFit,
      minZoom,
      maxZoom,
      panningEnabled,
      userZoomingEnabled,
      boxSelectionEnabled,
    ]
  );

  /**
   * Destroy the Cytoscape instance
   */
  const destroy = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }
  }, []);

  /**
   * Update graph elements (replace all)
   */
  const setElements = useCallback((newElements: ElementDefinition[]) => {
    if (cyRef.current) {
      cyRef.current.elements().remove();
      cyRef.current.add(newElements);
    }
  }, []);

  /**
   * Add elements to the graph
   */
  const addElements = useCallback((newElements: ElementDefinition[]) => {
    if (cyRef.current) {
      cyRef.current.add(newElements);
    }
  }, []);

  /**
   * Remove elements by ID
   */
  const removeElements = useCallback((ids: string[]) => {
    if (cyRef.current) {
      ids.forEach((id) => {
        cyRef.current?.getElementById(id).remove();
      });
    }
  }, []);

  /**
   * Run a layout algorithm
   */
  const runLayout = useCallback(
    (name: LayoutName = 'cose', opts: Partial<LayoutOptions> = {}) => {
      if (cyRef.current) {
        const layoutConfig = { ...getLayoutOptions(name), ...opts };
        cyRef.current.layout(layoutConfig).run();
      }
    },
    []
  );

  /**
   * Fit graph to viewport
   */
  const fit = useCallback((padding = 30) => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, padding);
    }
  }, []);

  /**
   * Center the graph
   */
  const center = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.center();
    }
  }, []);

  /**
   * Set zoom level
   */
  const zoom = useCallback((level: number) => {
    if (cyRef.current) {
      cyRef.current.zoom(level);
    }
  }, []);

  /**
   * Get current zoom level
   */
  const getZoom = useCallback(() => {
    return cyRef.current?.zoom() ?? 1;
  }, []);

  /**
   * Select specific nodes
   */
  const selectNodes = useCallback((nodeIds: string[]) => {
    if (cyRef.current) {
      cyRef.current.nodes().unselect();
      nodeIds.forEach((id) => {
        cyRef.current?.getElementById(id).select();
      });
    }
  }, []);

  /**
   * Clear selection
   */
  const clearSelection = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.elements().unselect();
    }
  }, []);

  /**
   * Get selected node IDs
   */
  const getSelectedNodeIds = useCallback(() => {
    return cyRef.current?.nodes(':selected').map((n) => n.id()) ?? [];
  }, []);

  /**
   * Highlight a node
   */
  const highlightNode = useCallback((nodeId: string) => {
    if (cyRef.current) {
      cyRef.current.getElementById(nodeId).addClass('highlighted');
    }
  }, []);

  /**
   * Remove highlight from a node
   */
  const unhighlightNode = useCallback((nodeId: string) => {
    if (cyRef.current) {
      cyRef.current.getElementById(nodeId).removeClass('highlighted');
    }
  }, []);

  /**
   * Clear all highlights
   */
  const clearHighlights = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.elements().removeClass('highlighted');
    }
  }, []);

  /**
   * Export graph as PNG
   */
  const exportPng = useCallback(() => {
    if (cyRef.current) {
      return cyRef.current.png({ full: true, scale: 2 });
    }
    return undefined;
  }, []);

  // Auto-initialize when container is available
  useEffect(() => {
    if (containerRef.current) {
      initialize(containerRef.current);
    }

    return () => {
      destroy();
    };
  }, [initialize, destroy]);

  return {
    containerRef,
    cyInstance: cyRef.current,
    isLoading,
    initialize,
    destroy,
    setElements,
    addElements,
    removeElements,
    runLayout,
    fit,
    center,
    zoom,
    getZoom,
    selectNodes,
    clearSelection,
    getSelectedNodeIds,
    highlightNode,
    unhighlightNode,
    clearHighlights,
    exportPng,
  };
}

export default useCytoscape;
