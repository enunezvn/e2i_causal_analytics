/**
 * KnowledgeGraph Component
 * ========================
 *
 * Main knowledge graph visualization component that integrates Cytoscape.js
 * with the E2I graph data types and API hooks.
 *
 * This component transforms API graph data (GraphNode, GraphRelationship)
 * into Cytoscape elements and provides the visualization wrapper.
 *
 * Features:
 * - Automatic data transformation from API types to Cytoscape elements
 * - Entity type-based node coloring
 * - Relationship type-based edge styling
 * - Loading and empty states
 * - Selection and interaction callbacks
 *
 * @module components/visualizations/KnowledgeGraph
 */

import * as React from 'react';
import { useMemo, useCallback } from 'react';
import type { ElementDefinition, StylesheetStyle } from 'cytoscape';
import { cn } from '@/lib/utils';
import { CytoscapeGraph } from './graph/CytoscapeGraph';
import type { LayoutName } from '@/hooks/use-cytoscape';
import type { GraphNode, GraphRelationship } from '@/types/graph';

// =============================================================================
// TYPES
// =============================================================================

export interface KnowledgeGraphProps {
  /** Graph nodes from the API */
  nodes: GraphNode[];
  /** Graph relationships from the API */
  relationships: GraphRelationship[];
  /** Layout algorithm */
  layout?: LayoutName;
  /** Additional CSS classes */
  className?: string;
  /** Minimum height of the graph */
  minHeight?: number | string;
  /** Whether the data is currently loading */
  isLoading?: boolean;
  /** Called when a node is clicked */
  onNodeSelect?: (node: GraphNode | null) => void;
  /** Called when an edge is clicked */
  onEdgeSelect?: (relationship: GraphRelationship | null) => void;
  /** Called when a node is hovered */
  onNodeHover?: (node: GraphNode | null) => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Color mapping for entity types
 * Uses the app's color palette for consistency
 */
const ENTITY_TYPE_COLORS: Record<string, string> = {
  Patient: '#3b82f6', // blue-500
  HCP: '#10b981', // emerald-500
  Brand: '#f59e0b', // amber-500
  Region: '#8b5cf6', // violet-500
  KPI: '#ef4444', // red-500
  CausalPath: '#06b6d4', // cyan-500
  Trigger: '#f97316', // orange-500
  Agent: '#ec4899', // pink-500
  Episode: '#6366f1', // indigo-500
  Community: '#14b8a6', // teal-500
  Treatment: '#84cc16', // lime-500
  Prediction: '#a855f7', // purple-500
  Experiment: '#22c55e', // green-500
  AgentActivity: '#64748b', // slate-500
};

/**
 * Color mapping for relationship types
 */
const RELATIONSHIP_TYPE_COLORS: Record<string, string> = {
  CAUSES: '#ef4444', // red-500 (causal)
  IMPACTS: '#f97316', // orange-500
  INFLUENCES: '#f59e0b', // amber-500
  TREATED_BY: '#10b981', // emerald-500
  PRESCRIBED: '#3b82f6', // blue-500
  PRESCRIBES: '#3b82f6', // blue-500
  DISCOVERED: '#8b5cf6', // violet-500
  GENERATED: '#ec4899', // pink-500
  MENTIONS: '#6b7280', // gray-500
  MEMBER_OF: '#14b8a6', // teal-500
  RELATES_TO: '#9ca3af', // gray-400 (default)
  RECEIVED: '#22c55e', // green-500
  LOCATED_IN: '#8b5cf6', // violet-500
  PRACTICES_IN: '#06b6d4', // cyan-500
  MEASURED_IN: '#a855f7', // purple-500
};

/**
 * Get color for an entity type
 */
function getEntityTypeColor(type: string): string {
  return ENTITY_TYPE_COLORS[type] || '#6b7280'; // gray-500 default
}

/**
 * Get color for a relationship type
 */
function getRelationshipTypeColor(type: string): string {
  return RELATIONSHIP_TYPE_COLORS[type] || '#9ca3af'; // gray-400 default
}

// =============================================================================
// DATA TRANSFORMATION
// =============================================================================

/**
 * Transform API GraphNode to Cytoscape node element
 */
function transformNode(node: GraphNode): ElementDefinition {
  return {
    data: {
      id: node.id,
      label: node.name,
      type: node.type,
      properties: node.properties,
      // Store original node for callbacks
      _original: node,
    },
  };
}

/**
 * Transform API GraphRelationship to Cytoscape edge element
 */
function transformRelationship(rel: GraphRelationship): ElementDefinition {
  return {
    data: {
      id: rel.id,
      source: rel.source_id,
      target: rel.target_id,
      type: rel.type,
      confidence: rel.confidence,
      properties: rel.properties,
      // Store original relationship for callbacks
      _original: rel,
    },
  };
}

/**
 * Transform API graph data to Cytoscape elements
 */
function transformGraphData(
  nodes: GraphNode[],
  relationships: GraphRelationship[]
): ElementDefinition[] {
  // Get set of valid node IDs for filtering orphan edges
  const validNodeIds = new Set(nodes.map((n) => n.id));

  // Transform nodes
  const nodeElements = nodes.map(transformNode);

  // Transform relationships, filtering out edges with missing endpoints
  const edgeElements = relationships
    .filter(
      (rel) => validNodeIds.has(rel.source_id) && validNodeIds.has(rel.target_id)
    )
    .map(transformRelationship);

  return [...nodeElements, ...edgeElements];
}

// =============================================================================
// STYLESHEET
// =============================================================================

/**
 * Generate dynamic stylesheet based on entity and relationship types
 */
function generateStylesheet(): StylesheetStyle[] {
  const baseStyles: StylesheetStyle[] = [
    // Base node style
    {
      selector: 'node',
      style: {
        'background-color': '#6b7280',
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'font-size': '11px',
        'color': '#374151',
        'text-margin-y': 6,
        'width': 36,
        'height': 36,
        'border-width': 2,
        'border-color': '#4b5563',
        'text-wrap': 'ellipsis',
        'text-max-width': '80px',
      },
    },
    // Selected node
    {
      selector: 'node:selected',
      style: {
        'border-width': 4,
        'border-color': '#1f2937',
        'width': 44,
        'height': 44,
      },
    },
    // Highlighted node (hover)
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 3,
        'border-color': '#fbbf24',
        'width': 42,
        'height': 42,
      },
    },
    // Dimmed node
    {
      selector: 'node.dimmed',
      style: {
        'opacity': 0.25,
      },
    },
    // Base edge style
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#9ca3af',
        'target-arrow-color': '#9ca3af',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 1,
        'opacity': 0.8,
      },
    },
    // Selected edge
    {
      selector: 'edge:selected',
      style: {
        'width': 3,
        'opacity': 1,
      },
    },
    // Highlighted edge
    {
      selector: 'edge.highlighted',
      style: {
        'width': 3,
        'opacity': 1,
        'line-color': '#fbbf24',
        'target-arrow-color': '#fbbf24',
      },
    },
    // Dimmed edge
    {
      selector: 'edge.dimmed',
      style: {
        'opacity': 0.15,
      },
    },
  ];

  // Add entity type-specific styles
  const entityTypeStyles: StylesheetStyle[] = Object.entries(ENTITY_TYPE_COLORS).map(
    ([type, color]) => ({
      selector: `node[type = "${type}"]`,
      style: {
        'background-color': color,
        'border-color': color,
      },
    })
  );

  // Add relationship type-specific styles
  const relationshipTypeStyles: StylesheetStyle[] = Object.entries(
    RELATIONSHIP_TYPE_COLORS
  ).map(([type, color]) => ({
    selector: `edge[type = "${type}"]`,
    style: {
      'line-color': color,
      'target-arrow-color': color,
    },
  }));

  return [...baseStyles, ...entityTypeStyles, ...relationshipTypeStyles];
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * KnowledgeGraph renders the E2I knowledge graph with interactive nodes and edges.
 *
 * @example
 * ```tsx
 * const { data } = useNodes();
 * const { data: rels } = useRelationships();
 *
 * <KnowledgeGraph
 *   nodes={data?.nodes ?? []}
 *   relationships={rels?.relationships ?? []}
 *   onNodeSelect={(node) => setSelectedNode(node)}
 * />
 * ```
 */
const KnowledgeGraph = React.forwardRef<HTMLDivElement, KnowledgeGraphProps>(
  (
    {
      nodes,
      relationships,
      layout = 'cose',
      className,
      minHeight = 500,
      isLoading = false,
      onNodeSelect,
      onEdgeSelect,
      onNodeHover,
    },
    ref
  ) => {
    // Memoize stylesheet (static)
    const stylesheet = useMemo(() => generateStylesheet(), []);

    // Memoize elements transformation
    const elements = useMemo(
      () => transformGraphData(nodes, relationships),
      [nodes, relationships]
    );

    // Create node lookup for callbacks
    const nodeMap = useMemo(() => {
      const map = new Map<string, GraphNode>();
      nodes.forEach((node) => map.set(node.id, node));
      return map;
    }, [nodes]);

    // Create relationship lookup for callbacks
    const relationshipMap = useMemo(() => {
      const map = new Map<string, GraphRelationship>();
      relationships.forEach((rel) => map.set(rel.id, rel));
      return map;
    }, [relationships]);

    // Handle node click
    const handleNodeClick = useCallback(
      (nodeId: string) => {
        const node = nodeMap.get(nodeId);
        onNodeSelect?.(node ?? null);
      },
      [nodeMap, onNodeSelect]
    );

    // Handle edge click
    const handleEdgeClick = useCallback(
      (edgeId: string) => {
        const rel = relationshipMap.get(edgeId);
        onEdgeSelect?.(rel ?? null);
      },
      [relationshipMap, onEdgeSelect]
    );

    // Handle node hover
    const handleNodeMouseOver = useCallback(
      (nodeId: string) => {
        const node = nodeMap.get(nodeId);
        onNodeHover?.(node ?? null);
      },
      [nodeMap, onNodeHover]
    );

    // Handle node hover end
    const handleNodeMouseOut = useCallback(() => {
      onNodeHover?.(null);
    }, [onNodeHover]);

    // Handle background click to deselect
    const handleBackgroundClick = useCallback(() => {
      onNodeSelect?.(null);
      onEdgeSelect?.(null);
    }, [onNodeSelect, onEdgeSelect]);

    // Show empty state if no data
    if (!isLoading && nodes.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]',
            className
          )}
          style={{ minHeight: typeof minHeight === 'number' ? `${minHeight}px` : minHeight }}
        >
          <div className="text-center p-8">
            <div className="text-[var(--color-muted-foreground)] mb-2">
              No graph data available
            </div>
            <p className="text-sm text-[var(--color-muted-foreground)]/60">
              Load nodes and relationships to visualize the knowledge graph
            </p>
          </div>
        </div>
      );
    }

    return (
      <CytoscapeGraph
        ref={ref}
        elements={elements}
        style={stylesheet}
        layout={layout}
        className={className}
        minHeight={minHeight}
        showLoading={isLoading}
        minZoom={0.1}
        maxZoom={3}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        onNodeMouseOver={handleNodeMouseOver}
        onNodeMouseOut={handleNodeMouseOut}
        onBackgroundClick={handleBackgroundClick}
        ariaLabel="E2I Knowledge Graph visualization showing entities and their relationships"
      />
    );
  }
);

KnowledgeGraph.displayName = 'KnowledgeGraph';

export { KnowledgeGraph, getEntityTypeColor, getRelationshipTypeColor };
export default KnowledgeGraph;
