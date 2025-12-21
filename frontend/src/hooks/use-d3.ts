/**
 * D3.js Hook
 * ==========
 *
 * React hook for managing D3.js graph/visualization instances.
 * Handles initialization, cleanup, and provides common D3 operations
 * specifically optimized for DAG (Directed Acyclic Graph) rendering.
 *
 * Features:
 * - Automatic instance lifecycle management
 * - SVG container management
 * - Zoom and pan controls
 * - Force simulation for node layout
 * - DAG-specific hierarchical layout
 *
 * @module hooks/use-d3
 */

import { useRef, useCallback, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type {
  Selection,
  D3ZoomEvent,
  ZoomBehavior,
  Simulation,
  SimulationNodeDatum,
  SimulationLinkDatum,
} from 'd3';

// =============================================================================
// TYPES
// =============================================================================

/**
 * Node data for D3 DAG visualization
 */
export interface D3Node extends SimulationNodeDatum {
  /** Unique node identifier */
  id: string;
  /** Node display label */
  label: string;
  /** Node type for styling */
  type?: string;
  /** Node depth in DAG hierarchy */
  depth?: number;
  /** Additional node properties */
  properties?: Record<string, unknown>;
  /** Fixed x position (optional) */
  fx?: number | null;
  /** Fixed y position (optional) */
  fy?: number | null;
}

/**
 * Edge data for D3 DAG visualization
 */
export interface D3Edge extends SimulationLinkDatum<D3Node> {
  /** Unique edge identifier */
  id: string;
  /** Source node ID or node reference */
  source: string | D3Node;
  /** Target node ID or node reference */
  target: string | D3Node;
  /** Edge type for styling */
  type?: string;
  /** Edge weight/strength */
  weight?: number;
  /** Confidence score (0-1) */
  confidence?: number;
  /** Additional edge properties */
  properties?: Record<string, unknown>;
}

/**
 * Graph data structure for D3
 */
export interface D3GraphData {
  nodes: D3Node[];
  edges: D3Edge[];
}

/**
 * Options for initializing D3 visualization
 */
export interface D3Options {
  /** SVG width (defaults to container width) */
  width?: number;
  /** SVG height (defaults to container height) */
  height?: number;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** Whether to enable zoom/pan */
  enableZoom?: boolean;
  /** Node radius */
  nodeRadius?: number;
  /** Link distance for force simulation */
  linkDistance?: number;
  /** Charge strength for force simulation */
  chargeStrength?: number;
  /** Whether to use hierarchical layout (true) or force layout (false) */
  useHierarchicalLayout?: boolean;
}

/**
 * Event handlers for D3 interactions
 */
export interface D3EventHandlers {
  /** Called when a node is clicked */
  onNodeClick?: (node: D3Node) => void;
  /** Called when a node is double-clicked */
  onNodeDoubleClick?: (node: D3Node) => void;
  /** Called when mouse enters a node */
  onNodeMouseOver?: (node: D3Node) => void;
  /** Called when mouse leaves a node */
  onNodeMouseOut?: (node: D3Node) => void;
  /** Called when an edge is clicked */
  onEdgeClick?: (edge: D3Edge) => void;
  /** Called when the graph is ready */
  onReady?: () => void;
  /** Called when the background is clicked */
  onBackgroundClick?: () => void;
}

/**
 * Return value of the useD3 hook
 */
export interface UseD3Return {
  /** Ref to attach to the container element */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** The D3 SVG selection (null before initialization) */
  svgSelection: Selection<SVGSVGElement, unknown, null, undefined> | null;
  /** Whether the graph is currently loading/initializing */
  isLoading: boolean;
  /** Initialize or reinitialize the graph */
  initialize: () => void;
  /** Destroy the graph instance */
  destroy: () => void;
  /** Update graph data */
  setData: (data: D3GraphData) => void;
  /** Fit the graph to the viewport */
  fit: () => void;
  /** Center the graph */
  center: () => void;
  /** Zoom to a specific level */
  zoom: (level: number) => void;
  /** Get the current zoom level */
  getZoom: () => number;
  /** Export graph as SVG string */
  exportSvg: () => string | undefined;
  /** Highlight a node */
  highlightNode: (nodeId: string) => void;
  /** Clear all highlights */
  clearHighlights: () => void;
}

// =============================================================================
// DEFAULT VALUES
// =============================================================================

const DEFAULT_OPTIONS: Required<D3Options> = {
  width: 800,
  height: 600,
  minZoom: 0.1,
  maxZoom: 4,
  enableZoom: true,
  nodeRadius: 24,
  linkDistance: 150,
  chargeStrength: -400,
  useHierarchicalLayout: true,
};

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * React hook for managing a D3.js DAG visualization.
 *
 * @param options - Configuration options for the visualization
 * @param eventHandlers - Event handlers for user interactions
 * @returns Object with refs, state, and control functions
 *
 * @example
 * ```tsx
 * function DAGComponent() {
 *   const {
 *     containerRef,
 *     isLoading,
 *     setData,
 *   } = useD3({
 *     useHierarchicalLayout: true,
 *     nodeRadius: 30
 *   }, {
 *     onNodeClick: (node) => console.log('Clicked:', node.id)
 *   });
 *
 *   useEffect(() => {
 *     setData({ nodes: myNodes, edges: myEdges });
 *   }, [myNodes, myEdges, setData]);
 *
 *   return <div ref={containerRef} style={{ width: '100%', height: '500px' }} />;
 * }
 * ```
 */
export function useD3(
  options: D3Options = {},
  eventHandlers: D3EventHandlers = {}
): UseD3Return {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<Selection<SVGSVGElement, unknown, null, undefined> | null>(null);
  const gRef = useRef<Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const zoomBehaviorRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const simulationRef = useRef<Simulation<D3Node, D3Edge> | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Store event handlers in a ref to avoid re-initialization
  const handlersRef = useRef(eventHandlers);
  handlersRef.current = eventHandlers;

  // Merge options with defaults
  const mergedOptions = { ...DEFAULT_OPTIONS, ...options };
  const optionsRef = useRef(mergedOptions);
  optionsRef.current = mergedOptions;

  /**
   * Initialize the D3 SVG and zoom behavior
   */
  const initialize = useCallback(() => {
    if (!containerRef.current) return;

    // Clear existing SVG
    d3.select(containerRef.current).selectAll('svg').remove();

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const width = optionsRef.current.width || rect.width || 800;
    const height = optionsRef.current.height || rect.height || 600;

    setIsLoading(true);

    // Create SVG
    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('background', 'transparent');

    svgRef.current = svg;

    // Create main group for zoom/pan
    const g = svg.append('g').attr('class', 'd3-main-group');
    gRef.current = g;

    // Add arrow marker definition for directed edges
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'd3-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#9ca3af');

    // Set up zoom behavior
    if (optionsRef.current.enableZoom) {
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([optionsRef.current.minZoom, optionsRef.current.maxZoom])
        .on('zoom', (event: D3ZoomEvent<SVGSVGElement, unknown>) => {
          g.attr('transform', event.transform.toString());
        });

      svg.call(zoom);
      zoomBehaviorRef.current = zoom;

      // Handle background click
      svg.on('click', (event: MouseEvent) => {
        if (event.target === svg.node()) {
          handlersRef.current.onBackgroundClick?.();
        }
      });
    }

    setIsLoading(false);
    handlersRef.current.onReady?.();
  }, []);

  /**
   * Destroy the D3 instance
   */
  const destroy = useCallback(() => {
    if (simulationRef.current) {
      simulationRef.current.stop();
      simulationRef.current = null;
    }
    if (containerRef.current) {
      d3.select(containerRef.current).selectAll('svg').remove();
    }
    svgRef.current = null;
    gRef.current = null;
    zoomBehaviorRef.current = null;
  }, []);

  /**
   * Update graph data and render
   */
  const setData = useCallback((data: D3GraphData) => {
    if (!gRef.current || !svgRef.current) return;

    const g = gRef.current;
    const opts = optionsRef.current;

    // Clear existing elements
    g.selectAll('.links').remove();
    g.selectAll('.nodes').remove();

    // Stop any existing simulation
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    const { nodes, edges } = data;

    if (nodes.length === 0) return;

    // Get dimensions
    const rect = containerRef.current?.getBoundingClientRect();
    const width = rect?.width || opts.width;
    const height = rect?.height || opts.height;

    // Create a node map for edge resolution
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    // Resolve edge source/target to node references
    const resolvedEdges = edges.map((e) => ({
      ...e,
      source: typeof e.source === 'string' ? nodeMap.get(e.source) || e.source : e.source,
      target: typeof e.target === 'string' ? nodeMap.get(e.target) || e.target : e.target,
    })) as D3Edge[];

    // Calculate node depths for hierarchical layout
    if (opts.useHierarchicalLayout) {
      calculateNodeDepths(nodes, resolvedEdges);
      positionNodesHierarchically(nodes, width, height, opts.nodeRadius);
    }

    // Create links group
    const linkGroup = g.append('g').attr('class', 'links');

    // Create links
    const link = linkGroup
      .selectAll<SVGLineElement, D3Edge>('line')
      .data(resolvedEdges)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', (d) => getEdgeColor(d.type))
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', 'url(#d3-arrow)')
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        handlersRef.current.onEdgeClick?.(d);
      });

    // Create nodes group
    const nodeGroup = g.append('g').attr('class', 'nodes');

    // Create node groups
    const node = nodeGroup
      .selectAll<SVGGElement, D3Node>('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        handlersRef.current.onNodeClick?.(d);
      })
      .on('dblclick', (_event, d) => {
        handlersRef.current.onNodeDoubleClick?.(d);
      })
      .on('mouseenter', (_event, d) => {
        handlersRef.current.onNodeMouseOver?.(d);
      })
      .on('mouseleave', (_event, d) => {
        handlersRef.current.onNodeMouseOut?.(d);
      });

    // Add circles to nodes
    node
      .append('circle')
      .attr('r', opts.nodeRadius)
      .attr('fill', (d) => getNodeColor(d.type))
      .attr('stroke', (d) => d3.color(getNodeColor(d.type))?.darker(0.5)?.toString() || '#333')
      .attr('stroke-width', 2);

    // Add labels to nodes
    node
      .append('text')
      .text((d) => truncateLabel(d.label, 12))
      .attr('text-anchor', 'middle')
      .attr('dy', opts.nodeRadius + 16)
      .attr('font-size', '11px')
      .attr('fill', '#374151')
      .attr('pointer-events', 'none');

    // Position elements based on layout type
    if (opts.useHierarchicalLayout) {
      // Use pre-calculated positions
      node.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);

      link
        .attr('x1', (d) => (d.source as D3Node).x ?? 0)
        .attr('y1', (d) => (d.source as D3Node).y ?? 0)
        .attr('x2', (d) => (d.target as D3Node).x ?? 0)
        .attr('y2', (d) => (d.target as D3Node).y ?? 0);
    } else {
      // Use force simulation
      const simulation = d3
        .forceSimulation<D3Node>(nodes)
        .force(
          'link',
          d3
            .forceLink<D3Node, D3Edge>(resolvedEdges)
            .id((d) => d.id)
            .distance(opts.linkDistance)
        )
        .force('charge', d3.forceManyBody().strength(opts.chargeStrength))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(opts.nodeRadius * 1.5));

      simulationRef.current = simulation;

      simulation.on('tick', () => {
        link
          .attr('x1', (d) => (d.source as D3Node).x ?? 0)
          .attr('y1', (d) => (d.source as D3Node).y ?? 0)
          .attr('x2', (d) => (d.target as D3Node).x ?? 0)
          .attr('y2', (d) => (d.target as D3Node).y ?? 0);

        node.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
      });

      // Add drag behavior
      node.call(
        d3
          .drag<SVGGElement, D3Node>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );
    }
  }, []);

  /**
   * Fit the graph to the viewport
   */
  const fit = useCallback(() => {
    if (!svgRef.current || !gRef.current || !zoomBehaviorRef.current) return;

    const svg = svgRef.current;
    const g = gRef.current;
    const zoom = zoomBehaviorRef.current;

    const bounds = (g.node() as SVGGElement)?.getBBox();
    if (!bounds) return;

    const rect = containerRef.current?.getBoundingClientRect();
    const width = rect?.width || 800;
    const height = rect?.height || 600;

    const fullWidth = bounds.width;
    const fullHeight = bounds.height;

    const scale = 0.9 / Math.max(fullWidth / width, fullHeight / height);
    const translateX = width / 2 - scale * (bounds.x + fullWidth / 2);
    const translateY = height / 2 - scale * (bounds.y + fullHeight / 2);

    svg
      .transition()
      .duration(500)
      .call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
  }, []);

  /**
   * Center the graph
   */
  const center = useCallback(() => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;

    const svg = svgRef.current;
    const zoom = zoomBehaviorRef.current;

    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  }, []);

  /**
   * Set zoom level
   */
  const zoomTo = useCallback((level: number) => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;

    const svg = svgRef.current;
    const zoom = zoomBehaviorRef.current;

    svg.transition().duration(300).call(zoom.scaleTo, level);
  }, []);

  /**
   * Get current zoom level
   */
  const getZoom = useCallback(() => {
    if (!svgRef.current) return 1;
    const transform = d3.zoomTransform(svgRef.current.node() as Element);
    return transform.k;
  }, []);

  /**
   * Export graph as SVG string
   */
  const exportSvg = useCallback(() => {
    if (!svgRef.current) return undefined;
    const svgNode = svgRef.current.node();
    if (!svgNode) return undefined;

    const serializer = new XMLSerializer();
    return serializer.serializeToString(svgNode);
  }, []);

  /**
   * Highlight a node and its connections
   */
  const highlightNode = useCallback((nodeId: string) => {
    if (!gRef.current) return;

    const g = gRef.current;

    // Dim all elements
    g.selectAll('.node').classed('dimmed', true);
    g.selectAll('.link').classed('dimmed', true);

    // Highlight selected node
    g.selectAll<SVGGElement, D3Node>('.node')
      .filter((d) => d.id === nodeId)
      .classed('dimmed', false)
      .classed('highlighted', true);

    // Highlight connected edges and nodes
    g.selectAll<SVGLineElement, D3Edge>('.link')
      .filter((d) => {
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;
        return sourceId === nodeId || targetId === nodeId;
      })
      .classed('dimmed', false)
      .classed('highlighted', true)
      .each(function (d) {
        const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
        const targetId = typeof d.target === 'string' ? d.target : d.target.id;
        const connectedId = sourceId === nodeId ? targetId : sourceId;
        g.selectAll<SVGGElement, D3Node>('.node')
          .filter((n) => n.id === connectedId)
          .classed('dimmed', false);
      });
  }, []);

  /**
   * Clear all highlights
   */
  const clearHighlights = useCallback(() => {
    if (!gRef.current) return;

    gRef.current.selectAll('.node').classed('dimmed', false).classed('highlighted', false);
    gRef.current.selectAll('.link').classed('dimmed', false).classed('highlighted', false);
  }, []);

  // Auto-initialize when container is available
  useEffect(() => {
    if (containerRef.current) {
      initialize();
    }

    return () => {
      destroy();
    };
  }, [initialize, destroy]);

  return {
    containerRef,
    svgSelection: svgRef.current,
    isLoading,
    initialize,
    destroy,
    setData,
    fit,
    center,
    zoom: zoomTo,
    getZoom,
    exportSvg,
    highlightNode,
    clearHighlights,
  };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Calculate node depths using topological sort for DAG layout
 */
function calculateNodeDepths(nodes: D3Node[], edges: D3Edge[]): void {
  // Build adjacency list
  const incoming = new Map<string, Set<string>>();
  const outgoing = new Map<string, Set<string>>();

  nodes.forEach((n) => {
    incoming.set(n.id, new Set());
    outgoing.set(n.id, new Set());
  });

  edges.forEach((e) => {
    const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
    const targetId = typeof e.target === 'string' ? e.target : e.target.id;
    incoming.get(targetId)?.add(sourceId);
    outgoing.get(sourceId)?.add(targetId);
  });

  // Find root nodes (no incoming edges)
  const roots = nodes.filter((n) => (incoming.get(n.id)?.size ?? 0) === 0);

  // BFS to assign depths
  const visited = new Set<string>();
  const queue: Array<{ id: string; depth: number }> = roots.map((n) => ({
    id: n.id,
    depth: 0,
  }));

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!;
    if (visited.has(id)) continue;
    visited.add(id);

    const node = nodes.find((n) => n.id === id);
    if (node) {
      node.depth = Math.max(node.depth ?? 0, depth);
    }

    outgoing.get(id)?.forEach((targetId) => {
      if (!visited.has(targetId)) {
        queue.push({ id: targetId, depth: depth + 1 });
      }
    });
  }

  // Handle disconnected nodes
  nodes.forEach((n) => {
    if (n.depth === undefined) {
      n.depth = 0;
    }
  });
}

/**
 * Position nodes in a hierarchical layout
 */
function positionNodesHierarchically(
  nodes: D3Node[],
  width: number,
  height: number,
  nodeRadius: number
): void {
  // Group nodes by depth
  const levels = new Map<number, D3Node[]>();
  nodes.forEach((n) => {
    const depth = n.depth ?? 0;
    if (!levels.has(depth)) {
      levels.set(depth, []);
    }
    levels.get(depth)!.push(n);
  });

  const maxDepth = Math.max(...Array.from(levels.keys()));
  const levelHeight = height / (maxDepth + 2);
  const padding = nodeRadius * 2;

  levels.forEach((levelNodes, depth) => {
    const y = padding + (depth + 0.5) * levelHeight;
    const levelWidth = width - padding * 2;
    const nodeSpacing = levelWidth / (levelNodes.length + 1);

    levelNodes.forEach((node, i) => {
      node.x = padding + (i + 1) * nodeSpacing;
      node.y = y;
    });
  });
}

/**
 * Get color for a node type
 */
function getNodeColor(type?: string): string {
  const colors: Record<string, string> = {
    variable: '#3b82f6', // blue-500
    treatment: '#10b981', // emerald-500
    outcome: '#ef4444', // red-500
    confounder: '#f59e0b', // amber-500
    instrument: '#8b5cf6', // violet-500
    mediator: '#ec4899', // pink-500
    // Fallback for EntityType values
    Patient: '#3b82f6',
    HCP: '#10b981',
    Brand: '#f59e0b',
    Region: '#8b5cf6',
    KPI: '#ef4444',
    CausalPath: '#06b6d4',
    default: '#6b7280', // gray-500
  };

  return colors[type || 'default'] || colors.default;
}

/**
 * Get color for an edge type
 */
function getEdgeColor(type?: string): string {
  const colors: Record<string, string> = {
    causal: '#ef4444', // red-500
    CAUSES: '#ef4444',
    association: '#9ca3af', // gray-400
    IMPACTS: '#f97316', // orange-500
    INFLUENCES: '#f59e0b', // amber-500
    default: '#9ca3af',
  };

  return colors[type || 'default'] || colors.default;
}

/**
 * Truncate label to specified length
 */
function truncateLabel(label: string, maxLength: number): string {
  if (label.length <= maxLength) return label;
  return label.substring(0, maxLength - 1) + '\u2026';
}

export default useD3;
