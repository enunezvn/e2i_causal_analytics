/**
 * CausalDiscovery Component
 * =========================
 *
 * Main causal discovery visualization component that integrates the CausalDAG
 * component with controls and sample data for demonstration.
 *
 * This component provides:
 * - Causal DAG visualization using D3.js
 * - Effect estimates table with confidence intervals
 * - Sample/demo data for initial rendering
 * - Controls for zoom, fit, and export
 * - Node selection and details panel integration points
 * - Loading and empty states
 *
 * @module components/visualizations/CausalDiscovery
 */

import * as React from 'react';
import { useState, useCallback, useMemo, useRef } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CausalDAG, type CausalDAGRef, type CausalNode, type CausalEdge } from './causal/CausalDAG';
import { EffectsTable, type CausalEffect } from './causal/EffectsTable';
import { ZoomIn, ZoomOut, Maximize2, Download, RotateCcw } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export interface CausalDiscoveryProps {
  /** Causal graph nodes (uses sample data if not provided) */
  nodes?: CausalNode[];
  /** Causal graph edges (uses sample data if not provided) */
  edges?: CausalEdge[];
  /** Causal effect estimates (uses sample data if not provided) */
  effects?: CausalEffect[];
  /** Whether data is loading */
  isLoading?: boolean;
  /** Error object */
  error?: Error | null;
  /** Callback to retry loading data */
  onRetry?: () => void;
  /** Whether to show controls */
  showControls?: boolean;
  /** Whether to show node details panel */
  showDetails?: boolean;
  /** Whether to show effects table */
  showEffectsTable?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Called when a node is selected */
  onNodeSelect?: (node: CausalNode | null) => void;
  /** Called when an edge is selected */
  onEdgeSelect?: (edge: CausalEdge | null) => void;
  /** Called when an effect row is selected */
  onEffectSelect?: (effect: CausalEffect | null) => void;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

/**
 * Sample causal DAG nodes for demonstration
 */
const SAMPLE_NODES: CausalNode[] = [
  { id: 'treatment', label: 'Treatment', type: 'treatment' },
  { id: 'outcome', label: 'Health Outcome', type: 'outcome' },
  { id: 'age', label: 'Patient Age', type: 'confounder' },
  { id: 'severity', label: 'Disease Severity', type: 'confounder' },
  { id: 'adherence', label: 'Treatment Adherence', type: 'mediator' },
  { id: 'biomarker', label: 'Biomarker Level', type: 'mediator' },
  { id: 'genetics', label: 'Genetic Factors', type: 'instrument' },
];

/**
 * Sample causal DAG edges for demonstration
 */
const SAMPLE_EDGES: CausalEdge[] = [
  { id: 'e1', source: 'treatment', target: 'adherence', type: 'causal', effect: 0.7 },
  { id: 'e2', source: 'adherence', target: 'outcome', type: 'causal', effect: 0.5 },
  { id: 'e3', source: 'treatment', target: 'biomarker', type: 'causal', effect: 0.6 },
  { id: 'e4', source: 'biomarker', target: 'outcome', type: 'causal', effect: 0.4 },
  { id: 'e5', source: 'age', target: 'treatment', type: 'confounding', confidence: 0.8 },
  { id: 'e6', source: 'age', target: 'outcome', type: 'confounding', confidence: 0.7 },
  { id: 'e7', source: 'severity', target: 'treatment', type: 'confounding', confidence: 0.9 },
  { id: 'e8', source: 'severity', target: 'outcome', type: 'confounding', confidence: 0.85 },
  { id: 'e9', source: 'genetics', target: 'treatment', type: 'instrumental', confidence: 0.6 },
];

/**
 * Sample causal effect estimates for demonstration
 */
const SAMPLE_EFFECTS: CausalEffect[] = [
  {
    id: 'effect-1',
    treatment: 'Treatment',
    outcome: 'Health Outcome',
    estimate: 0.45,
    standardError: 0.12,
    ciLower: 0.21,
    ciUpper: 0.69,
    confidenceLevel: 0.95,
    pValue: 0.002,
    isSignificant: true,
  },
  {
    id: 'effect-2',
    treatment: 'Treatment Adherence',
    outcome: 'Health Outcome',
    estimate: 0.32,
    standardError: 0.08,
    ciLower: 0.16,
    ciUpper: 0.48,
    confidenceLevel: 0.95,
    pValue: 0.008,
    isSignificant: true,
  },
  {
    id: 'effect-3',
    treatment: 'Biomarker Level',
    outcome: 'Health Outcome',
    estimate: 0.18,
    standardError: 0.09,
    ciLower: 0.00,
    ciUpper: 0.36,
    confidenceLevel: 0.95,
    pValue: 0.051,
    isSignificant: false,
  },
  {
    id: 'effect-4',
    treatment: 'Patient Age',
    outcome: 'Health Outcome',
    estimate: -0.25,
    standardError: 0.11,
    ciLower: -0.47,
    ciUpper: -0.03,
    confidenceLevel: 0.95,
    pValue: 0.026,
    isSignificant: true,
  },
  {
    id: 'effect-5',
    treatment: 'Disease Severity',
    outcome: 'Health Outcome',
    estimate: -0.52,
    standardError: 0.14,
    ciLower: -0.79,
    ciUpper: -0.25,
    confidenceLevel: 0.95,
    pValue: 0.001,
    isSignificant: true,
  },
  {
    id: 'effect-6',
    treatment: 'Genetic Factors',
    outcome: 'Treatment Response',
    estimate: 0.12,
    standardError: 0.15,
    ciLower: -0.17,
    ciUpper: 0.41,
    confidenceLevel: 0.95,
    pValue: 0.42,
    isSignificant: false,
  },
];

// =============================================================================
// NODE TYPE COLORS
// =============================================================================

const NODE_TYPE_COLORS: Record<string, string> = {
  treatment: 'bg-emerald-500',
  outcome: 'bg-red-500',
  confounder: 'bg-amber-500',
  mediator: 'bg-pink-500',
  instrument: 'bg-violet-500',
  variable: 'bg-blue-500',
};

const NODE_TYPE_LABELS: Record<string, string> = {
  treatment: 'Treatment',
  outcome: 'Outcome',
  confounder: 'Confounder',
  mediator: 'Mediator',
  instrument: 'Instrument',
  variable: 'Variable',
};

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * CausalDiscovery renders a causal discovery dashboard with DAG visualization.
 *
 * @example
 * ```tsx
 * <CausalDiscovery
 *   nodes={causalNodes}
 *   edges={causalEdges}
 *   onNodeSelect={(node) => setSelectedNode(node)}
 * />
 * ```
 */
const CausalDiscovery = React.forwardRef<HTMLDivElement, CausalDiscoveryProps>(
  (
    {
      nodes: propNodes,
      edges: propEdges,
      effects: propEffects,
      isLoading = false,
      error = null,
      onRetry,
      showControls = true,
      showDetails = true,
      showEffectsTable = true,
      className,
      onNodeSelect,
      onEdgeSelect,
      onEffectSelect,
    },
    ref
  ) => {
    const dagRef = useRef<CausalDAGRef>(null);
    const [selectedNode, setSelectedNode] = useState<CausalNode | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<CausalEdge | null>(null);
    const [selectedEffect, setSelectedEffect] = useState<CausalEffect | null>(null);
    const [currentZoom, setCurrentZoom] = useState(1);

    // Use provided data or fall back to sample data
    const nodes = useMemo(() => propNodes ?? SAMPLE_NODES, [propNodes]);
    const edges = useMemo(() => propEdges ?? SAMPLE_EDGES, [propEdges]);
    const effects = useMemo(() => propEffects ?? SAMPLE_EFFECTS, [propEffects]);

    // Handle node click
    const handleNodeClick = useCallback(
      (node: CausalNode) => {
        setSelectedNode(node);
        setSelectedEdge(null);
        onNodeSelect?.(node);
      },
      [onNodeSelect]
    );

    // Handle edge click
    const handleEdgeClick = useCallback(
      (edge: CausalEdge) => {
        setSelectedEdge(edge);
        setSelectedNode(null);
        onEdgeSelect?.(edge);
      },
      [onEdgeSelect]
    );

    // Handle background click (deselect)
    const handleBackgroundClick = useCallback(() => {
      setSelectedNode(null);
      setSelectedEdge(null);
      setSelectedEffect(null);
      onNodeSelect?.(null);
      onEdgeSelect?.(null);
      onEffectSelect?.(null);
      dagRef.current?.clearHighlights();
    }, [onNodeSelect, onEdgeSelect, onEffectSelect]);

    // Handle effect row selection
    const handleEffectSelect = useCallback(
      (effect: CausalEffect) => {
        setSelectedEffect(effect);
        setSelectedNode(null);
        setSelectedEdge(null);
        onEffectSelect?.(effect);
      },
      [onEffectSelect]
    );


    // Control handlers
    const handleZoomIn = useCallback(() => {
      const newZoom = Math.min(currentZoom * 1.2, 3);
      setCurrentZoom(newZoom);
      dagRef.current?.setZoom(newZoom);
    }, [currentZoom]);

    const handleZoomOut = useCallback(() => {
      const newZoom = Math.max(currentZoom / 1.2, 0.1);
      setCurrentZoom(newZoom);
      dagRef.current?.setZoom(newZoom);
    }, [currentZoom]);

    const handleFit = useCallback(() => {
      dagRef.current?.fit();
      setCurrentZoom(dagRef.current?.getZoom() ?? 1);
    }, []);

    const handleReset = useCallback(() => {
      dagRef.current?.center();
      setCurrentZoom(1);
    }, []);

    const handleExport = useCallback(() => {
      const svgString = dagRef.current?.exportSvg();
      if (svgString) {
        const blob = new Blob([svgString], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = 'causal-dag.svg';
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
      }
    }, []);

    // Show error state
    if (error) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center rounded-lg border border-[var(--color-destructive)]/50 bg-[var(--color-card)]',
            className
          )}
          style={{ minHeight: '500px' }}
        >
          <div className="text-center p-8 max-w-md">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--color-destructive)]/10">
              <svg
                className="h-6 w-6 text-[var(--color-destructive)]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-[var(--color-foreground)] mb-2">
              Failed to Load Causal Data
            </h3>
            <p className="text-sm text-[var(--color-muted-foreground)] mb-4">
              {error.message || 'An unexpected error occurred while loading the causal graph.'}
            </p>
            {onRetry && (
              <Button onClick={onRetry} variant="default">
                Try Again
              </Button>
            )}
          </div>
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('flex flex-col gap-4', className)}>
        {/* Controls */}
        {showControls && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleZoomOut}
                disabled={isLoading}
                aria-label="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm text-[var(--color-muted-foreground)] min-w-[60px] text-center">
                {Math.round(currentZoom * 100)}%
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleZoomIn}
                disabled={isLoading}
                aria-label="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <div className="w-px h-6 bg-[var(--color-border)] mx-2" />
              <Button
                variant="outline"
                size="sm"
                onClick={handleFit}
                disabled={isLoading}
                aria-label="Fit to view"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
                disabled={isLoading}
                aria-label="Reset view"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleExport}
                disabled={isLoading}
              >
                <Download className="h-4 w-4 mr-2" />
                Export SVG
              </Button>
            </div>
          </div>
        )}

        <div className="flex gap-4">
          {/* DAG Visualization */}
          <div className="flex-1 min-w-0">
            <CausalDAG
              ref={dagRef}
              nodes={nodes}
              edges={edges}
              showLoading={isLoading}
              minHeight={500}
              onNodeClick={handleNodeClick}
              onEdgeClick={handleEdgeClick}
              onBackgroundClick={handleBackgroundClick}
            />
          </div>

          {/* Details Panel */}
          {showDetails && (
            <Card className="w-72 shrink-0">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Details</CardTitle>
                <CardDescription>
                  {selectedNode
                    ? 'Selected Node'
                    : selectedEdge
                      ? 'Selected Edge'
                      : 'Click a node or edge'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedNode ? (
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className={cn(
                            'w-3 h-3 rounded-full',
                            NODE_TYPE_COLORS[selectedNode.type || 'variable']
                          )}
                        />
                        <span className="font-medium">{selectedNode.label}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {NODE_TYPE_LABELS[selectedNode.type || 'variable']}
                      </Badge>
                    </div>
                    <div className="text-xs text-[var(--color-muted-foreground)]">
                      <p className="mb-1">
                        <span className="font-medium">ID:</span> {selectedNode.id}
                      </p>
                      {selectedNode.properties && (
                        <div className="mt-2">
                          <span className="font-medium">Properties:</span>
                          <pre className="mt-1 p-2 bg-[var(--color-muted)] rounded text-[10px] overflow-auto max-h-32">
                            {JSON.stringify(selectedNode.properties, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                ) : selectedEdge ? (
                  <div className="space-y-4">
                    <div>
                      <span className="font-medium text-sm">
                        {selectedEdge.source} → {selectedEdge.target}
                      </span>
                      <div className="mt-2">
                        <Badge variant="outline" className="text-xs capitalize">
                          {selectedEdge.type || 'causal'}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-xs text-[var(--color-muted-foreground)] space-y-1">
                      <p>
                        <span className="font-medium">ID:</span> {selectedEdge.id}
                      </p>
                      {selectedEdge.effect !== undefined && (
                        <p>
                          <span className="font-medium">Effect:</span>{' '}
                          {selectedEdge.effect.toFixed(2)}
                        </p>
                      )}
                      {selectedEdge.confidence !== undefined && (
                        <p>
                          <span className="font-medium">Confidence:</span>{' '}
                          {(selectedEdge.confidence * 100).toFixed(0)}%
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-[var(--color-muted-foreground)]">
                    <p className="mb-4">Select a node or edge to view details.</p>
                    <div className="space-y-2">
                      <p className="font-medium text-[var(--color-foreground)]">Legend:</p>
                      {Object.entries(NODE_TYPE_LABELS).map(([type, label]) => (
                        <div key={type} className="flex items-center gap-2">
                          <div
                            className={cn('w-3 h-3 rounded-full', NODE_TYPE_COLORS[type])}
                          />
                          <span className="text-xs">{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Effects Table */}
        {showEffectsTable && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Causal Effect Estimates</CardTitle>
              <CardDescription>
                Treatment effects with 95% confidence intervals
              </CardDescription>
            </CardHeader>
            <CardContent>
              <EffectsTable
                effects={effects}
                isLoading={isLoading}
                onRowSelect={handleEffectSelect}
                selectedEffectId={selectedEffect?.id}
                showCIBars
                sortable
              />
            </CardContent>
          </Card>
        )}
      </div>
    );
  }
);

CausalDiscovery.displayName = 'CausalDiscovery';

export { CausalDiscovery };
export default CausalDiscovery;
