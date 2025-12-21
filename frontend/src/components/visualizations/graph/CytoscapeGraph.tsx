/**
 * CytoscapeGraph Component
 * ========================
 *
 * A reusable React component for rendering interactive graph visualizations
 * using Cytoscape.js. This component wraps the useCytoscape hook and provides
 * a styled container with loading states.
 *
 * Features:
 * - Responsive container that fills parent
 * - Loading spinner during initialization
 * - Error boundary for graph errors
 * - Themeable through CSS custom properties
 * - Full keyboard and mouse interaction support
 * - Imperative API for external control
 *
 * @module components/visualizations/graph/CytoscapeGraph
 */

import * as React from 'react';
import { useEffect, useImperativeHandle } from 'react';
import type { ElementDefinition, StylesheetStyle } from 'cytoscape';
import { cn } from '@/lib/utils';
import {
  useCytoscape,
  defaultCytoscapeStyles,
  type LayoutName,
  type CytoscapeEventHandlers,
} from '@/hooks/use-cytoscape';

// =============================================================================
// IMPERATIVE HANDLE TYPES
// =============================================================================

/**
 * Methods exposed via ref for external control
 */
interface CytoscapeGraphRef {
  /** Get current zoom level */
  getZoom: () => number;
  /** Set zoom level */
  setZoom: (level: number) => void;
  /** Fit graph to viewport */
  fit: (padding?: number) => void;
  /** Center the graph */
  center: () => void;
  /** Run a layout algorithm */
  runLayout: (name: LayoutName) => void;
  /** Export graph as PNG data URL */
  exportPng: () => string | undefined;
}

// =============================================================================
// TYPES
// =============================================================================

export interface CytoscapeGraphProps {
  /** Graph elements (nodes and edges) to render */
  elements: ElementDefinition[];
  /** Custom stylesheet (uses default if not provided) */
  style?: StylesheetStyle[];
  /** Layout algorithm to use */
  layout?: LayoutName;
  /** Additional CSS classes for the container */
  className?: string;
  /** Minimum height of the graph container */
  minHeight?: number | string;
  /** Whether to show a loading indicator */
  showLoading?: boolean;
  /** Custom loading component */
  loadingComponent?: React.ReactNode;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** Event handler for node clicks */
  onNodeClick?: CytoscapeEventHandlers['onNodeClick'];
  /** Event handler for node double clicks */
  onNodeDoubleClick?: CytoscapeEventHandlers['onNodeDoubleClick'];
  /** Event handler for node hover start */
  onNodeMouseOver?: CytoscapeEventHandlers['onNodeMouseOver'];
  /** Event handler for node hover end */
  onNodeMouseOut?: CytoscapeEventHandlers['onNodeMouseOut'];
  /** Event handler for edge clicks */
  onEdgeClick?: CytoscapeEventHandlers['onEdgeClick'];
  /** Event handler for selection changes */
  onSelectionChange?: CytoscapeEventHandlers['onSelectionChange'];
  /** Event handler when graph is ready */
  onReady?: CytoscapeEventHandlers['onReady'];
  /** Event handler for background clicks */
  onBackgroundClick?: CytoscapeEventHandlers['onBackgroundClick'];
  /** Aria label for accessibility */
  ariaLabel?: string;
}

// =============================================================================
// LOADING SPINNER
// =============================================================================

/**
 * Default loading spinner component
 */
function LoadingSpinner() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-background)]/80 z-10">
      <div className="flex flex-col items-center gap-3">
        <div
          className="h-10 w-10 animate-spin rounded-full border-4 border-[var(--color-muted)] border-t-[var(--color-primary)]"
          role="status"
          aria-label="Loading graph"
        />
        <span className="text-sm text-[var(--color-muted-foreground)]">
          Loading graph...
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * CytoscapeGraph renders an interactive graph visualization.
 *
 * @example
 * ```tsx
 * const elements = [
 *   { data: { id: 'a', label: 'Node A' } },
 *   { data: { id: 'b', label: 'Node B' } },
 *   { data: { id: 'ab', source: 'a', target: 'b' } },
 * ];
 *
 * <CytoscapeGraph
 *   elements={elements}
 *   layout="cose"
 *   onNodeClick={(id, data) => console.log('Clicked:', id)}
 * />
 * ```
 */
const CytoscapeGraph = React.forwardRef<CytoscapeGraphRef, CytoscapeGraphProps>(
  (
    {
      elements,
      style = defaultCytoscapeStyles,
      layout = 'cose',
      className,
      minHeight = 400,
      showLoading = true,
      loadingComponent,
      minZoom = 0.1,
      maxZoom = 3,
      onNodeClick,
      onNodeDoubleClick,
      onNodeMouseOver,
      onNodeMouseOut,
      onEdgeClick,
      onSelectionChange,
      onReady,
      onBackgroundClick,
      ariaLabel = 'Interactive graph visualization',
    },
    ref
  ) => {
    // Initialize the Cytoscape hook
    const {
      containerRef,
      isLoading,
      setElements,
      runLayout,
      fit,
      center,
      zoom,
      getZoom,
      exportPng,
    } = useCytoscape(
      {
        elements,
        style,
        layout,
        autoFit: true,
        minZoom,
        maxZoom,
        panningEnabled: true,
        userZoomingEnabled: true,
        boxSelectionEnabled: true,
      },
      {
        onNodeClick,
        onNodeDoubleClick,
        onNodeMouseOver,
        onNodeMouseOut,
        onEdgeClick,
        onSelectionChange,
        onReady,
        onBackgroundClick,
      }
    );

    // Expose imperative methods via ref
    useImperativeHandle(
      ref,
      () => ({
        getZoom,
        setZoom: zoom,
        fit,
        center,
        runLayout,
        exportPng,
      }),
      [getZoom, zoom, fit, center, runLayout, exportPng]
    );

    // Update elements when they change
    useEffect(() => {
      if (!isLoading && elements.length > 0) {
        setElements(elements);
        runLayout(layout);
      }
    }, [elements, isLoading, setElements, runLayout, layout]);

    const heightStyle = typeof minHeight === 'number' ? `${minHeight}px` : minHeight;

    return (
      <div
        className={cn(
          'relative w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] overflow-hidden',
          className
        )}
        style={{ minHeight: heightStyle }}
      >
        {/* Loading Overlay */}
        {showLoading && isLoading && (loadingComponent || <LoadingSpinner />)}

        {/* Graph Container */}
        <div
          ref={containerRef as React.RefObject<HTMLDivElement>}
          className="absolute inset-0"
          role="img"
          aria-label={ariaLabel}
          tabIndex={0}
        />
      </div>
    );
  }
);

CytoscapeGraph.displayName = 'CytoscapeGraph';

export { CytoscapeGraph, type CytoscapeGraphRef };
export default CytoscapeGraph;
