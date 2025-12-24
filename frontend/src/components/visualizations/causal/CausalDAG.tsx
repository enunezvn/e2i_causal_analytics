/**
 * CausalDAG Component
 * ===================
 *
 * A reusable React component for rendering causal Directed Acyclic Graphs (DAGs)
 * using D3.js.
 *
 * @module components/visualizations/causal/CausalDAG
 */

import * as React from 'react';
import { useEffect, useImperativeHandle, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { Selection, D3ZoomEvent, ZoomBehavior } from 'd3';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export type CausalNodeType =
  | 'treatment'
  | 'outcome'
  | 'confounder'
  | 'mediator'
  | 'instrument'
  | 'variable';

export type CausalEdgeType = 'causal' | 'association' | 'confounding' | 'instrumental';

export interface CausalNode {
  id: string;
  label: string;
  type?: CausalNodeType;
  properties?: Record<string, unknown>;
}

export interface CausalEdge {
  id: string;
  source: string;
  target: string;
  type?: CausalEdgeType;
  effect?: number;
  confidence?: number;
  properties?: Record<string, unknown>;
}

interface PositionedNode extends CausalNode {
  x: number;
  y: number;
  depth: number;
}

export interface CausalDAGRef {
  fit: () => void;
  center: () => void;
  getZoom: () => number;
  setZoom: (level: number) => void;
  exportSvg: () => string | undefined;
  highlightNode: (nodeId: string) => void;
  clearHighlights: () => void;
}

export interface CausalDAGProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  className?: string;
  minHeight?: number | string;
  showLoading?: boolean;
  loadingComponent?: React.ReactNode;
  nodeRadius?: number;
  onNodeClick?: (node: CausalNode) => void;
  onEdgeClick?: (edge: CausalEdge) => void;
  onBackgroundClick?: () => void;
  ariaLabel?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const NODE_COLORS: Record<string, string> = {
  treatment: '#10b981',
  outcome: '#ef4444',
  confounder: '#f59e0b',
  mediator: '#ec4899',
  instrument: '#8b5cf6',
  variable: '#3b82f6',
};

const EDGE_COLORS: Record<string, string> = {
  causal: '#ef4444',
  association: '#9ca3af',
  confounding: '#f59e0b',
  instrumental: '#8b5cf6',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getNodeColor(type?: string): string {
  return NODE_COLORS[type || 'variable'] || NODE_COLORS.variable;
}

function getEdgeColor(type?: string): string {
  return EDGE_COLORS[type || 'causal'] || EDGE_COLORS.causal;
}

function truncateLabel(label: string, maxLength: number): string {
  if (label.length <= maxLength) return label;
  return label.substring(0, maxLength - 1) + '\u2026';
}

function calculateNodeDepths(
  nodes: CausalNode[],
  edges: CausalEdge[]
): Map<string, number> {
  const depths = new Map<string, number>();
  const incoming = new Map<string, Set<string>>();
  const outgoing = new Map<string, Set<string>>();

  nodes.forEach((n) => {
    incoming.set(n.id, new Set());
    outgoing.set(n.id, new Set());
    depths.set(n.id, 0);
  });

  edges.forEach((e) => {
    incoming.get(e.target)?.add(e.source);
    outgoing.get(e.source)?.add(e.target);
  });

  const roots = nodes.filter((n) => (incoming.get(n.id)?.size ?? 0) === 0);
  const visited = new Set<string>();
  const queue: Array<{ id: string; depth: number }> = roots.map((n) => ({
    id: n.id,
    depth: 0,
  }));

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!;
    if (visited.has(id)) continue;
    visited.add(id);

    depths.set(id, Math.max(depths.get(id) ?? 0, depth));

    outgoing.get(id)?.forEach((targetId) => {
      if (!visited.has(targetId)) {
        queue.push({ id: targetId, depth: depth + 1 });
      }
    });
  }

  return depths;
}

function positionNodes(
  nodes: CausalNode[],
  edges: CausalEdge[],
  width: number,
  height: number,
  nodeRadius: number
): PositionedNode[] {
  const depths = calculateNodeDepths(nodes, edges);
  const levels = new Map<number, CausalNode[]>();

  nodes.forEach((n) => {
    const depth = depths.get(n.id) ?? 0;
    if (!levels.has(depth)) {
      levels.set(depth, []);
    }
    levels.get(depth)!.push(n);
  });

  const maxDepth = Math.max(...Array.from(levels.keys()), 0);
  const levelHeight = height / (maxDepth + 2);
  const padding = nodeRadius * 2;

  const positioned: PositionedNode[] = [];

  levels.forEach((levelNodes, depth) => {
    const y = padding + (depth + 0.5) * levelHeight;
    const levelWidth = width - padding * 2;
    const nodeSpacing = levelWidth / (levelNodes.length + 1);

    levelNodes.forEach((node, i) => {
      positioned.push({
        ...node,
        x: padding + (i + 1) * nodeSpacing,
        y,
        depth,
      });
    });
  });

  return positioned;
}

// =============================================================================
// COMPONENT
// =============================================================================

const CausalDAG = React.forwardRef<CausalDAGRef, CausalDAGProps>(
  (
    {
      nodes,
      edges,
      className,
      minHeight = 500,
      showLoading = false,
      loadingComponent,
      nodeRadius = 24,
      onNodeClick,
      onEdgeClick,
      onBackgroundClick,
      ariaLabel = 'Causal DAG visualization',
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<Selection<SVGSVGElement, unknown, null, undefined> | null>(null);
    const gRef = useRef<Selection<SVGGElement, unknown, null, undefined> | null>(null);
    const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const initializedRef = useRef(false);

    // Store callbacks in refs
    const onNodeClickRef = useRef(onNodeClick);
    const onEdgeClickRef = useRef(onEdgeClick);
    const onBackgroundClickRef = useRef(onBackgroundClick);
    onNodeClickRef.current = onNodeClick;
    onEdgeClickRef.current = onEdgeClick;
    onBackgroundClickRef.current = onBackgroundClick;

    // Initialize D3 SVG once
    useEffect(() => {
      if (!containerRef.current || initializedRef.current) return;
      initializedRef.current = true;

      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const width = rect.width || 800;
      const height = rect.height || 500;

      // Create SVG
      const svg = d3
        .select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      svgRef.current = svg;

      // Create main group
      const g = svg.append('g').attr('class', 'd3-main-group');
      gRef.current = g;

      // Add arrow marker
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

      // Set up zoom
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 3])
        .on('zoom', (event: D3ZoomEvent<SVGSVGElement, unknown>) => {
          g.attr('transform', event.transform.toString());
        });

      svg.call(zoom);
      zoomRef.current = zoom;

      // Background click
      svg.on('click', (event: MouseEvent) => {
        if (event.target === svg.node()) {
          onBackgroundClickRef.current?.();
        }
      });

      return () => {
        d3.select(container).selectAll('svg').remove();
        svgRef.current = null;
        gRef.current = null;
        zoomRef.current = null;
        initializedRef.current = false;
      };
    }, []);

    // Render graph data when nodes/edges change
    useEffect(() => {
      if (!gRef.current || !containerRef.current || nodes.length === 0) return;

      const g = gRef.current;
      const rect = containerRef.current.getBoundingClientRect();
      const width = rect.width || 800;
      const height = rect.height || 500;

      // Clear existing elements
      g.selectAll('.links').remove();
      g.selectAll('.nodes').remove();

      // Position nodes
      const positionedNodes = positionNodes(nodes, edges, width, height, nodeRadius);
      const nodeMap = new Map(positionedNodes.map((n) => [n.id, n]));

      // Filter valid edges
      const validEdges = edges.filter(
        (e) => nodeMap.has(e.source) && nodeMap.has(e.target)
      );

      // Create links
      const linkGroup = g.append('g').attr('class', 'links');
      linkGroup
        .selectAll('line')
        .data(validEdges)
        .join('line')
        .attr('class', 'link')
        .attr('x1', (d) => nodeMap.get(d.source)?.x ?? 0)
        .attr('y1', (d) => nodeMap.get(d.source)?.y ?? 0)
        .attr('x2', (d) => nodeMap.get(d.target)?.x ?? 0)
        .attr('y2', (d) => nodeMap.get(d.target)?.y ?? 0)
        .attr('stroke', (d) => getEdgeColor(d.type))
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.7)
        .attr('marker-end', 'url(#d3-arrow)')
        .style('cursor', 'pointer')
        .on('click', (_event, d) => {
          const originalEdge = edges.find((e) => e.id === d.id);
          if (originalEdge) {
            onEdgeClickRef.current?.(originalEdge);
          }
        });

      // Create nodes
      const nodeGroup = g.append('g').attr('class', 'nodes');
      const nodeElements = nodeGroup
        .selectAll<SVGGElement, PositionedNode>('g')
        .data(positionedNodes)
        .join('g')
        .attr('class', 'node')
        .attr('transform', (d) => `translate(${d.x},${d.y})`)
        .style('cursor', 'pointer')
        .on('click', (_event, d) => {
          const originalNode = nodes.find((n) => n.id === d.id);
          if (originalNode) {
            onNodeClickRef.current?.(originalNode);
          }
        });

      // Add circles
      nodeElements
        .append('circle')
        .attr('r', nodeRadius)
        .attr('fill', (d) => getNodeColor(d.type))
        .attr('stroke', (d) => d3.color(getNodeColor(d.type))?.darker(0.5)?.toString() || '#333')
        .attr('stroke-width', 2);

      // Add labels
      nodeElements
        .append('text')
        .text((d) => truncateLabel(d.label, 12))
        .attr('text-anchor', 'middle')
        .attr('dy', nodeRadius + 16)
        .attr('font-size', '11px')
        .attr('fill', '#374151')
        .attr('pointer-events', 'none');
    }, [nodes, edges, nodeRadius]);

    // Imperative methods
    const fit = useCallback(() => {
      if (!svgRef.current || !gRef.current || !zoomRef.current) return;
      const bounds = (gRef.current.node() as SVGGElement)?.getBBox();
      if (!bounds) return;
      const rect = containerRef.current?.getBoundingClientRect();
      const width = rect?.width || 800;
      const height = rect?.height || 500;
      const scale = 0.9 / Math.max(bounds.width / width, bounds.height / height);
      const translateX = width / 2 - scale * (bounds.x + bounds.width / 2);
      const translateY = height / 2 - scale * (bounds.y + bounds.height / 2);
      svgRef.current
        .transition()
        .duration(500)
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    }, []);

    const center = useCallback(() => {
      if (!svgRef.current || !zoomRef.current) return;
      svgRef.current.transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity);
    }, []);

    const getZoom = useCallback(() => {
      if (!svgRef.current) return 1;
      return d3.zoomTransform(svgRef.current.node() as Element).k;
    }, []);

    const setZoom = useCallback((level: number) => {
      if (!svgRef.current || !zoomRef.current) return;
      svgRef.current.transition().duration(300).call(zoomRef.current.scaleTo, level);
    }, []);

    const exportSvg = useCallback(() => {
      if (!svgRef.current) return undefined;
      const node = svgRef.current.node();
      if (!node) return undefined;
      return new XMLSerializer().serializeToString(node);
    }, []);

    const highlightNode = useCallback((nodeId: string) => {
      if (!gRef.current) return;
      const g = gRef.current;
      g.selectAll('.node').classed('dimmed', true);
      g.selectAll('.link').classed('dimmed', true);
      g.selectAll<SVGGElement, PositionedNode>('.node')
        .filter((d) => d.id === nodeId)
        .classed('dimmed', false)
        .classed('highlighted', true);
      g.selectAll<SVGLineElement, CausalEdge>('.link')
        .filter((d) => d.source === nodeId || d.target === nodeId)
        .classed('dimmed', false)
        .classed('highlighted', true)
        .each(function (d) {
          const connectedId = d.source === nodeId ? d.target : d.source;
          g.selectAll<SVGGElement, PositionedNode>('.node')
            .filter((n) => n.id === connectedId)
            .classed('dimmed', false);
        });
    }, []);

    const clearHighlights = useCallback(() => {
      if (!gRef.current) return;
      gRef.current.selectAll('.node').classed('dimmed', false).classed('highlighted', false);
      gRef.current.selectAll('.link').classed('dimmed', false).classed('highlighted', false);
    }, []);

    useImperativeHandle(
      ref,
      () => ({ fit, center, getZoom, setZoom, exportSvg, highlightNode, clearHighlights }),
      [fit, center, getZoom, setZoom, exportSvg, highlightNode, clearHighlights]
    );

    const heightStyle = typeof minHeight === 'number' ? `${minHeight}px` : minHeight;

    if (nodes.length === 0) {
      return (
        <div
          className={cn(
            'flex items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]',
            className
          )}
          style={{ minHeight: heightStyle }}
        >
          <div className="text-center p-8">
            <div className="text-[var(--color-muted-foreground)] mb-2">
              No causal graph data available
            </div>
            <p className="text-sm text-[var(--color-muted-foreground)]/60">
              Run causal discovery to visualize the causal DAG
            </p>
          </div>
        </div>
      );
    }

    return (
      <div
        className={cn(
          'relative w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] overflow-hidden',
          className
        )}
        style={{ minHeight: heightStyle }}
      >
        {showLoading && (loadingComponent || (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-background)]/80 z-10">
            <div className="h-10 w-10 animate-spin rounded-full border-4 border-[var(--color-muted)] border-t-[var(--color-primary)]" />
          </div>
        ))}

        <div
          ref={containerRef}
          className="absolute inset-0"
          role="img"
          aria-label={ariaLabel}
          tabIndex={0}
        />

        <style>{`
          .node.highlighted circle { stroke: #fbbf24; stroke-width: 4px; }
          .node.dimmed { opacity: 0.25; }
          .link.highlighted { stroke: #fbbf24 !important; stroke-width: 3px; }
          .link.dimmed { opacity: 0.15; }
        `}</style>
      </div>
    );
  }
);

CausalDAG.displayName = 'CausalDAG';

export { CausalDAG };
export default CausalDAG;
