/**
 * Active Causal Chains Component
 * ==============================
 *
 * Interactive Cytoscape.js graph visualization showing live causal relationships.
 * Displays the knowledge graph with nodes representing entities and edges
 * representing causal effects.
 *
 * @module components/insights/ActiveCausalChains
 */

import { useEffect, useState, useCallback } from 'react';
import { Share2, ZoomIn, ZoomOut, Maximize2, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useCytoscape, defaultCytoscapeStyles } from '@/hooks/use-cytoscape';
import { useCausalChains } from '@/hooks/api/use-graph';
import type { ElementDefinition, StylesheetStyle } from 'cytoscape';

// =============================================================================
// TYPES
// =============================================================================

interface ActiveCausalChainsProps {
  className?: string;
}

interface SelectedNode {
  id: string;
  label: string;
  type: string;
}

interface SelectedEdge {
  id: string;
  sourceLabel: string;
  targetLabel: string;
  relType: string;
  /** Edge confidence (0-1) when the API reported one; absent = unknown. */
  weight?: number;
  /** ATE estimate riding the terminal edge of a validated causal path. */
  ate?: number;
}

// NOTE: SAMPLE_ELEMENTS (a fabricated "Detailing Frequency -> ... -> TRx"
// demo graph with invented edge weights) was DELETED. The graph initializes
// EMPTY and renders only real chains from POST /api/graph/causal-chains;
// zero chains shows an honest empty state, failures a labeled error.

// =============================================================================
// NODE TYPE MAPPING
// =============================================================================

/**
 * Maps API node types to visualization categories for styling.
 * - driver: Sources of causal chains — interventions and exogenous causes (blue)
 * - mediator: Intermediate entities that transmit effects (violet)
 * - outcome: End results/metrics (emerald)
 *
 * There is deliberately NO moderator category: moderation (effect
 * modification) is not representable as a cause→effect edge in this DAG —
 * it lives in the HTE segment analysis, not the knowledge graph.
 */
const NODE_TYPE_MAP: Record<string, string> = {
  // API types → visualization types
  Trigger: 'driver',
  Action: 'driver',
  Campaign: 'driver',
  HCP: 'mediator',
  Brand: 'mediator',
  Treatment: 'mediator',
  Patient: 'mediator',
  Region: 'mediator',
  Segment: 'mediator',
  KPI: 'outcome',
  Metric: 'outcome',
  Conversion: 'outcome',
  // Identity mappings for already-canonical visualization types
  driver: 'driver',
  mediator: 'mediator',
  outcome: 'outcome',
};

/** Roles stamped on causal_paths variables by sync_causal_paths_to_falkordb. */
const KNOWN_ROLES = new Set(['driver', 'mediator', 'outcome']);

/**
 * Get the visualization type for a node. The SSOT `role` property stamped on
 * causal-path variables wins (it encodes the node's actual position across
 * validated chains); the entity-type map is the fallback, then 'mediator'.
 */
function getNodeVisualizationType(apiType: string | undefined, role?: unknown): string {
  if (typeof role === 'string' && KNOWN_ROLES.has(role)) return role;
  if (!apiType) return 'mediator';
  return NODE_TYPE_MAP[apiType] ?? 'mediator';
}

// =============================================================================
// CUSTOM STYLES
// =============================================================================

const customStyles: StylesheetStyle[] = [
  ...defaultCytoscapeStyles,
  {
    selector: 'node[vizType="driver"]',
    style: {
      'background-color': '#3b82f6', // blue-500
      'border-color': '#2563eb',
    },
  },
  {
    selector: 'node[vizType="mediator"]',
    style: {
      'background-color': '#8b5cf6', // violet-500
      'border-color': '#7c3aed',
    },
  },
  {
    selector: 'node[vizType="outcome"]',
    style: {
      'background-color': '#10b981', // emerald-500
      'border-color': '#059669',
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 'mapData(weight, 0, 1, 1, 4)',
      'line-color': '#6b7280',
      'target-arrow-color': '#6b7280',
      'opacity': 0.7,
    },
  },
  {
    // Edges whose relationship arrived WITHOUT a confidence value: thin and
    // dashed to signal unknown strength (the value is never invented).
    selector: 'edge[!weight]',
    style: {
      'width': 1,
      'line-style': 'dashed',
    },
  },
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ActiveCausalChains({ className }: ActiveCausalChainsProps) {
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<SelectedEdge | null>(null);

  // Fetch causal chains from API
  const { mutate: fetchChains, data: chainsResponse, error: chainsError, isPending } = useCausalChains();

  // Initialize Cytoscape — EMPTY until real chains arrive.
  const {
    containerRef,
    isLoading,
    setElements,
    runLayout,
    fit,
    zoom,
    getZoom,
  } = useCytoscape(
    {
      elements: [],
      style: customStyles,
      layout: 'cose',
      autoFit: true,
      minZoom: 0.3,
      maxZoom: 2.5,
    },
    {
      onNodeClick: (nodeId, nodeData) => {
        setSelectedEdge(null);
        setSelectedNode({
          id: nodeId,
          label: (nodeData.label as string) || nodeId,
          type: (nodeData.type as string) || 'unknown',
        });
      },
      onEdgeClick: (edgeId, edgeData) => {
        setSelectedNode(null);
        setSelectedEdge({
          id: edgeId,
          sourceLabel: (edgeData.sourceLabel as string) || (edgeData.source as string) || '',
          targetLabel: (edgeData.targetLabel as string) || (edgeData.target as string) || '',
          relType: (edgeData.relType as string) || 'CAUSES',
          weight: typeof edgeData.weight === 'number' ? edgeData.weight : undefined,
          ate: typeof edgeData.ate === 'number' ? edgeData.ate : undefined,
        });
      },
      onBackgroundClick: () => {
        setSelectedNode(null);
        setSelectedEdge(null);
      },
    }
  );

  // Transform API response to Cytoscape elements
  useEffect(() => {
    if (!chainsResponse) return;

    // Zero chains: explicitly clear the graph so stale elements from a
    // previous response cannot linger beneath the empty-state overlay.
    if (chainsResponse.chains.length === 0) {
      setElements([]);
      return;
    }

    const elements: ElementDefinition[] = [];
    const nodeIds = new Set<string>();
    const nodeLabels = new Map<string, string>();
    const edgeIds = new Set<string>();
    let skipped = 0;

    chainsResponse.chains.forEach((chain) => {
      chain.nodes.forEach((node) => {
        // Cytoscape throws on an empty-string id; skip the node and let the
        // endpoint check below drop its incident edges instead of killing
        // the whole card.
        if (!node.id) {
          skipped += 1;
          return;
        }
        if (!nodeIds.has(node.id)) {
          nodeIds.add(node.id);
          nodeLabels.set(node.id, node.name || node.id);
          const apiType = node.type || 'entity';
          elements.push({
            data: {
              id: node.id,
              label: node.name || node.id,
              type: apiType,
              vizType: getNodeVisualizationType(apiType, node.properties?.role),
            },
          });
        }
      });

      chain.relationships.forEach((rel) => {
        // Both endpoints must be non-empty AND refer to collected nodes —
        // Cytoscape also throws on edges referencing missing elements.
        if (
          !rel.source_id ||
          !rel.target_id ||
          !nodeIds.has(rel.source_id) ||
          !nodeIds.has(rel.target_id)
        ) {
          skipped += 1;
          return;
        }
        // Distinct chains can share hops (several chains start from the same
        // node), so key the edge on type + endpoints: the same causal edge
        // renders once and ids stay unique across chains (duplicate ids throw
        // just like empty ones).
        const edgeId = `edge-${rel.type}-${rel.source_id}->${rel.target_id}`;
        if (edgeIds.has(edgeId)) return;
        edgeIds.add(edgeId);
        // confidence is optional on the wire: when absent it stays absent
        // (rendered as a thin dashed unknown-strength edge), never invented.
        // relType/labels/ate feed the click panel — ate_estimate rides the
        // terminal edge of a validated causal path (absent elsewhere).
        const ate = rel.properties?.ate_estimate;
        elements.push({
          data: {
            id: edgeId,
            source: rel.source_id,
            target: rel.target_id,
            relType: rel.type,
            sourceLabel: nodeLabels.get(rel.source_id) ?? rel.source_id,
            targetLabel: nodeLabels.get(rel.target_id) ?? rel.target_id,
            ...(typeof rel.confidence === 'number'
              ? { weight: rel.confidence }
              : {}),
            ...(typeof ate === 'number' ? { ate } : {}),
          },
        });
      });
    });

    if (skipped > 0) {
      console.warn(
        `ActiveCausalChains: skipped ${skipped} element(s) with missing ids or unresolved endpoints from /api/graph/causal-chains`
      );
    }

    setElements(elements);
    runLayout('cose');
  }, [chainsResponse, setElements, runLayout]);

  // Fetch chains on mount
  useEffect(() => {
    fetchChains({
      min_confidence: 0.3,
      max_chain_length: 4,
    });
  }, [fetchChains]);

  const handleRefresh = useCallback(() => {
    fetchChains({
      min_confidence: 0.3,
      max_chain_length: 4,
    });
  }, [fetchChains]);

  const handleZoomIn = useCallback(() => {
    zoom(getZoom() * 1.2);
  }, [zoom, getZoom]);

  const handleZoomOut = useCallback(() => {
    zoom(getZoom() / 1.2);
  }, [zoom, getZoom]);

  const handleFit = useCallback(() => {
    fit(30);
  }, [fit]);

  const handleRelayout = useCallback(() => {
    runLayout('cose');
  }, [runLayout]);

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-blue-500/10">
              <Share2 className="h-5 w-5 text-blue-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Active Causal Chains</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Interactive knowledge graph visualization
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleFit}>
              <Maximize2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handleRefresh}
              disabled={isPending}
            >
              <RefreshCw className={cn('h-4 w-4', isPending && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Graph Container */}
        <div className="relative">
          <div
            ref={containerRef as React.RefObject<HTMLDivElement>}
            className="w-full h-[400px] rounded-lg border border-[var(--color-border)] bg-[var(--color-muted)]/20"
          />

          {/* Loading Overlay */}
          {(isLoading || isPending) && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-background)]/50 rounded-lg">
              <div className="flex items-center gap-2 text-[var(--color-muted-foreground)]">
                <RefreshCw className="h-5 w-5 animate-spin" />
                <span className="text-sm">Loading graph...</span>
              </div>
            </div>
          )}

          {/* Error Overlay — labeled, never replaced with a fake graph */}
          {!isPending && chainsError && (
            <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-[var(--color-background)]/70 p-6">
              <div className="text-center text-sm text-[var(--color-muted-foreground)]">
                <span className="font-medium text-rose-600">
                  Unable to load causal chains:
                </span>{' '}
                {chainsError.message}
              </div>
            </div>
          )}

          {/* Honest empty overlay when the KG has no chains */}
          {!isPending && !chainsError && chainsResponse && chainsResponse.chains.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center rounded-lg p-6">
              <div className="text-center text-sm text-[var(--color-muted-foreground)]">
                No causal chains found in the knowledge graph for the current
                confidence threshold.
              </div>
            </div>
          )}

          {/* Selected Node Info */}
          {selectedNode && (
            <div className="absolute bottom-4 left-4 p-3 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)] shadow-lg">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium text-sm">{selectedNode.label}</span>
                <Badge variant="outline" className="text-xs capitalize">
                  {selectedNode.type}
                </Badge>
              </div>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Click edges to see causal strength
              </p>
            </div>
          )}

          {/* Selected Edge Info — causal strength from real edge data only */}
          {selectedEdge && (
            <div className="absolute bottom-4 left-4 p-3 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)] shadow-lg">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium text-sm">
                  {selectedEdge.sourceLabel} → {selectedEdge.targetLabel}
                </span>
                <Badge variant="outline" className="text-xs">
                  {selectedEdge.relType}
                </Badge>
              </div>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                {typeof selectedEdge.weight === 'number'
                  ? `Causal strength (confidence): ${selectedEdge.weight.toFixed(2)}`
                  : 'Causal strength: unknown (no confidence recorded)'}
              </p>
              {typeof selectedEdge.ate === 'number' && (
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  ATE estimate: {selectedEdge.ate.toFixed(3)}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4 pt-4 border-t border-[var(--color-border)]">
          <span className="text-xs text-[var(--color-muted-foreground)]">Legend:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-xs">Driver</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-violet-500" />
            <span className="text-xs">Mediator</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-xs">Outcome</span>
          </div>
          <Button variant="ghost" size="sm" className="ml-auto text-xs" onClick={handleRelayout}>
            Re-layout
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default ActiveCausalChains;
