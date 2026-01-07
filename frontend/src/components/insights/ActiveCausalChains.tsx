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

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_ELEMENTS: ElementDefinition[] = [
  // Nodes - using vizType for styling selectors
  { data: { id: 'detailing', label: 'Detailing Frequency', type: 'intervention', vizType: 'intervention' } },
  { data: { id: 'awareness', label: 'HCP Awareness', type: 'mediator', vizType: 'mediator' } },
  { data: { id: 'prescribing', label: 'Prescribing Intent', type: 'mediator', vizType: 'mediator' } },
  { data: { id: 'trx', label: 'TRx Volume', type: 'outcome', vizType: 'outcome' } },
  { data: { id: 'samples', label: 'Sample Distribution', type: 'intervention', vizType: 'intervention' } },
  { data: { id: 'formulary', label: 'Formulary Status', type: 'moderator', vizType: 'moderator' } },
  { data: { id: 'access', label: 'Patient Access', type: 'mediator', vizType: 'mediator' } },
  { data: { id: 'adherence', label: 'Adherence Rate', type: 'outcome', vizType: 'outcome' } },
  // Edges
  { data: { id: 'e1', source: 'detailing', target: 'awareness', weight: 0.72 } },
  { data: { id: 'e2', source: 'awareness', target: 'prescribing', weight: 0.65 } },
  { data: { id: 'e3', source: 'prescribing', target: 'trx', weight: 0.81 } },
  { data: { id: 'e4', source: 'samples', target: 'awareness', weight: 0.45 } },
  { data: { id: 'e5', source: 'formulary', target: 'access', weight: 0.58 } },
  { data: { id: 'e6', source: 'access', target: 'trx', weight: 0.67 } },
  { data: { id: 'e7', source: 'trx', target: 'adherence', weight: 0.53 } },
];

// =============================================================================
// NODE TYPE MAPPING
// =============================================================================

/**
 * Maps API node types to visualization categories for styling.
 * - intervention: Actions/triggers that start causal chains (blue)
 * - mediator: Intermediate entities that transmit effects (violet)
 * - moderator: Factors that modify effect strength (amber)
 * - outcome: End results/metrics (emerald)
 */
const NODE_TYPE_MAP: Record<string, string> = {
  // API types → visualization types
  Trigger: 'intervention',
  Action: 'intervention',
  Campaign: 'intervention',
  HCP: 'mediator',
  Brand: 'mediator',
  Treatment: 'mediator',
  Patient: 'moderator',
  Region: 'moderator',
  Segment: 'moderator',
  KPI: 'outcome',
  Metric: 'outcome',
  Conversion: 'outcome',
  // Sample data types (already correct)
  intervention: 'intervention',
  mediator: 'mediator',
  moderator: 'moderator',
  outcome: 'outcome',
};

/**
 * Get the visualization type for a node, with fallback to 'mediator'
 */
function getNodeVisualizationType(apiType: string | undefined): string {
  if (!apiType) return 'mediator';
  return NODE_TYPE_MAP[apiType] ?? 'mediator';
}

// =============================================================================
// CUSTOM STYLES
// =============================================================================

const customStyles: StylesheetStyle[] = [
  ...defaultCytoscapeStyles,
  {
    selector: 'node[vizType="intervention"]',
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
    selector: 'node[vizType="moderator"]',
    style: {
      'background-color': '#f59e0b', // amber-500
      'border-color': '#d97706',
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
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ActiveCausalChains({ className }: ActiveCausalChainsProps) {
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);

  // Fetch causal chains from API
  const { mutate: fetchChains, data: chainsResponse, isPending } = useCausalChains();

  // Initialize Cytoscape
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
      elements: SAMPLE_ELEMENTS,
      style: customStyles,
      layout: 'cose',
      autoFit: true,
      minZoom: 0.3,
      maxZoom: 2.5,
    },
    {
      onNodeClick: (nodeId, nodeData) => {
        setSelectedNode({
          id: nodeId,
          label: (nodeData.label as string) || nodeId,
          type: (nodeData.type as string) || 'unknown',
        });
      },
      onBackgroundClick: () => {
        setSelectedNode(null);
      },
    }
  );

  // Transform API response to Cytoscape elements
  useEffect(() => {
    if (chainsResponse?.chains && chainsResponse.chains.length > 0) {
      const elements: ElementDefinition[] = [];
      const nodeIds = new Set<string>();

      chainsResponse.chains.forEach((chain) => {
        chain.nodes.forEach((node) => {
          if (!nodeIds.has(node.id)) {
            nodeIds.add(node.id);
            const apiType = node.type || 'entity';
            elements.push({
              data: {
                id: node.id,
                label: node.name || node.id,
                type: apiType,
                vizType: getNodeVisualizationType(apiType),
              },
            });
          }
        });

        chain.relationships.forEach((rel, idx) => {
          elements.push({
            data: {
              id: `edge-${chain.nodes[0]?.id}-${idx}`,
              source: rel.source_id,
              target: rel.target_id,
              weight: rel.confidence ?? 0.5,
            },
          });
        });
      });

      if (elements.length > 0) {
        setElements(elements);
        runLayout('cose');
      }
    }
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
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4 pt-4 border-t border-[var(--color-border)]">
          <span className="text-xs text-[var(--color-muted-foreground)]">Legend:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-xs">Intervention</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-violet-500" />
            <span className="text-xs">Mediator</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-xs">Moderator</span>
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
