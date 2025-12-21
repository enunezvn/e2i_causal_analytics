/**
 * GraphControls Component
 * =======================
 *
 * Controls for interacting with the knowledge graph visualization.
 * Provides zoom, pan, layout selection, and fit/center actions.
 *
 * Features:
 * - Zoom in/out with increment buttons
 * - Zoom slider for fine control
 * - Layout algorithm selection
 * - Fit to viewport and center actions
 * - Export to PNG
 *
 * @module components/visualizations/graph/GraphControls
 */

import * as React from 'react';
import { useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Move,
  Download,
  LayoutGrid,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import type { LayoutName } from '@/hooks/use-cytoscape';

// =============================================================================
// TYPES
// =============================================================================

export interface GraphControlsProps {
  /** Current zoom level */
  zoom: number;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** Current layout algorithm */
  layout: LayoutName;
  /** Available layout options */
  availableLayouts?: LayoutName[];
  /** Whether controls are disabled */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Called when zoom changes */
  onZoomChange: (zoom: number) => void;
  /** Called when layout changes */
  onLayoutChange: (layout: LayoutName) => void;
  /** Called when fit to viewport is requested */
  onFit: () => void;
  /** Called when center is requested */
  onCenter: () => void;
  /** Called when export is requested */
  onExport?: () => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

/** Default available layouts */
const DEFAULT_LAYOUTS: LayoutName[] = [
  'cose',
  'grid',
  'circle',
  'concentric',
  'breadthfirst',
  'random',
];

/** Layout display names */
const LAYOUT_LABELS: Record<LayoutName, string> = {
  cose: 'Force-Directed (COSE)',
  grid: 'Grid',
  circle: 'Circle',
  concentric: 'Concentric',
  breadthfirst: 'Hierarchical',
  random: 'Random',
  preset: 'Preset',
};

/** Zoom increment step */
const ZOOM_STEP = 0.2;

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * GraphControls provides UI for controlling graph zoom, layout, and actions.
 *
 * @example
 * ```tsx
 * <GraphControls
 *   zoom={1.0}
 *   layout="cose"
 *   onZoomChange={(z) => graphRef.current?.zoom(z)}
 *   onLayoutChange={(l) => graphRef.current?.runLayout(l)}
 *   onFit={() => graphRef.current?.fit()}
 *   onCenter={() => graphRef.current?.center()}
 * />
 * ```
 */
const GraphControls = React.forwardRef<HTMLDivElement, GraphControlsProps>(
  (
    {
      zoom,
      minZoom = 0.1,
      maxZoom = 3,
      layout,
      availableLayouts = DEFAULT_LAYOUTS,
      disabled = false,
      className,
      onZoomChange,
      onLayoutChange,
      onFit,
      onCenter,
      onExport,
    },
    ref
  ) => {
    // Handle zoom in
    const handleZoomIn = useCallback(() => {
      const newZoom = Math.min(zoom + ZOOM_STEP, maxZoom);
      onZoomChange(newZoom);
    }, [zoom, maxZoom, onZoomChange]);

    // Handle zoom out
    const handleZoomOut = useCallback(() => {
      const newZoom = Math.max(zoom - ZOOM_STEP, minZoom);
      onZoomChange(newZoom);
    }, [zoom, minZoom, onZoomChange]);

    // Format zoom percentage
    const zoomPercent = Math.round(zoom * 100);

    return (
      <div
        ref={ref}
        className={cn(
          'flex items-center gap-2 p-2 bg-[var(--color-card)] border border-[var(--color-border)] rounded-lg shadow-sm',
          className
        )}
        role="toolbar"
        aria-label="Graph controls"
      >
        {/* Zoom Controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleZoomOut}
            disabled={disabled || zoom <= minZoom}
            aria-label="Zoom out"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>

          <span
            className="min-w-[3rem] text-center text-sm text-[var(--color-muted-foreground)] tabular-nums"
            aria-label={`Zoom level ${zoomPercent}%`}
          >
            {zoomPercent}%
          </span>

          <Button
            variant="ghost"
            size="icon"
            onClick={handleZoomIn}
            disabled={disabled || zoom >= maxZoom}
            aria-label="Zoom in"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>

        {/* Separator */}
        <div className="w-px h-6 bg-[var(--color-border)]" />

        {/* Layout Selection */}
        <div className="flex items-center gap-2">
          <LayoutGrid className="h-4 w-4 text-[var(--color-muted-foreground)]" />
          <Select
            value={layout}
            onValueChange={(value) => onLayoutChange(value as LayoutName)}
            disabled={disabled}
          >
            <SelectTrigger className="w-[160px] h-8 text-sm">
              <SelectValue placeholder="Select layout" />
            </SelectTrigger>
            <SelectContent>
              {availableLayouts.map((layoutName) => (
                <SelectItem key={layoutName} value={layoutName}>
                  {LAYOUT_LABELS[layoutName] || layoutName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Separator */}
        <div className="w-px h-6 bg-[var(--color-border)]" />

        {/* View Actions */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={onFit}
            disabled={disabled}
            aria-label="Fit to viewport"
            title="Fit to viewport"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={onCenter}
            disabled={disabled}
            aria-label="Center graph"
            title="Center graph"
          >
            <Move className="h-4 w-4" />
          </Button>

          {onExport && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onExport}
              disabled={disabled}
              aria-label="Export as PNG"
              title="Export as PNG"
            >
              <Download className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    );
  }
);

GraphControls.displayName = 'GraphControls';

export { GraphControls, LAYOUT_LABELS, DEFAULT_LAYOUTS };
export default GraphControls;
