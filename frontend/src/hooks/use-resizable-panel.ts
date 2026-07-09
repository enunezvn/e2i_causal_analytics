/**
 * useResizablePanel Hook
 * ======================
 *
 * Drag-to-resize behavior for an edge-docked panel (e.g. the chat sidebar).
 * Returns the effective width plus props for a drag-handle element:
 * - Pointer drag resizes between `minWidth` and the full window width
 * - Double-click resets to the default width
 * - Arrow keys resize when the handle is focused (accessibility)
 *
 * The live width during a drag is local state (no persistence churn on every
 * pointermove); the final width is committed via `onWidthChange` on release.
 *
 * @module hooks/use-resizable-panel
 */

import * as React from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface UseResizablePanelOptions {
  /** Width in px used when no custom width is set */
  defaultWidth: number;
  /** Minimum width in px the panel can be dragged to */
  minWidth?: number;
  /** Which edge of the panel carries the drag handle ('left' for a right-docked panel) */
  edge: 'left' | 'right';
  /** Persisted custom width in px, or null to use defaultWidth */
  persistedWidth: number | null;
  /** Commit a new width (px) on drag end / keyboard resize; null resets to default */
  onWidthChange: (width: number | null) => void;
  /** Px per arrow-key press */
  keyboardStep?: number;
  /** Accessible label for the handle */
  ariaLabel?: string;
}

export interface ResizeHandleProps {
  role: 'separator';
  'aria-orientation': 'vertical';
  'aria-label': string;
  'aria-valuemin': number;
  'aria-valuemax': number;
  'aria-valuenow': number;
  tabIndex: number;
  onPointerDown: (e: React.PointerEvent<HTMLElement>) => void;
  onPointerMove: (e: React.PointerEvent<HTMLElement>) => void;
  onPointerUp: (e: React.PointerEvent<HTMLElement>) => void;
  onPointerCancel: (e: React.PointerEvent<HTMLElement>) => void;
  onDoubleClick: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLElement>) => void;
}

export interface UseResizablePanelReturn {
  /** Effective panel width in px (live during drag, persisted otherwise) */
  width: number;
  /** True while a pointer drag is in progress */
  isDragging: boolean;
  /** Spread onto the drag-handle element */
  handleProps: ResizeHandleProps;
}

// =============================================================================
// HOOK
// =============================================================================

function clampWidth(width: number, minWidth: number): number {
  return Math.round(Math.min(Math.max(width, minWidth), window.innerWidth));
}

export function useResizablePanel({
  defaultWidth,
  minWidth = 320,
  edge,
  persistedWidth,
  onWidthChange,
  keyboardStep = 32,
  ariaLabel = 'Resize panel',
}: UseResizablePanelOptions): UseResizablePanelReturn {
  // Non-null only while dragging; avoids writing to the (persisted) store on
  // every pointermove.
  const [dragWidth, setDragWidth] = React.useState<number | null>(null);
  const isDragging = dragWidth !== null;

  const width = dragWidth ?? persistedWidth ?? defaultWidth;

  // For a handle on the panel's left edge (right-docked panel), the width is
  // the distance from the pointer to the right viewport edge — and vice versa.
  const widthFromPointer = React.useCallback(
    (clientX: number) => (edge === 'left' ? window.innerWidth - clientX : clientX),
    [edge]
  );

  const onPointerDown = React.useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      if (e.button !== 0) return;
      // Prevent text selection from starting under the drag
      e.preventDefault();
      // Route subsequent pointer events to the handle even when the pointer
      // leaves it (optional-chained: not implemented in jsdom)
      e.currentTarget.setPointerCapture?.(e.pointerId);
      document.body.style.userSelect = 'none';
      setDragWidth(clampWidth(widthFromPointer(e.clientX), minWidth));
    },
    [widthFromPointer, minWidth]
  );

  const onPointerMove = React.useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      setDragWidth((current) =>
        current === null ? null : clampWidth(widthFromPointer(e.clientX), minWidth)
      );
    },
    [widthFromPointer, minWidth]
  );

  const endDrag = React.useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      document.body.style.userSelect = '';
      setDragWidth((current) => {
        if (current === null) return null;
        onWidthChange(clampWidth(widthFromPointer(e.clientX), minWidth));
        return null;
      });
    },
    [onWidthChange, widthFromPointer, minWidth]
  );

  const onDoubleClick = React.useCallback(() => {
    onWidthChange(null);
  }, [onWidthChange]);

  const onKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLElement>) => {
      // The key that visually moves the handle AWAY from the panel grows it
      const growKey = edge === 'left' ? 'ArrowLeft' : 'ArrowRight';
      const shrinkKey = edge === 'left' ? 'ArrowRight' : 'ArrowLeft';
      if (e.key !== growKey && e.key !== shrinkKey) return;
      e.preventDefault();
      const delta = e.key === growKey ? keyboardStep : -keyboardStep;
      onWidthChange(clampWidth(width + delta, minWidth));
    },
    [edge, keyboardStep, onWidthChange, width, minWidth]
  );

  // If unmounted mid-drag, don't leave text selection disabled page-wide
  React.useEffect(
    () => () => {
      document.body.style.userSelect = '';
    },
    []
  );

  return {
    width,
    isDragging,
    handleProps: {
      role: 'separator',
      'aria-orientation': 'vertical',
      'aria-label': ariaLabel,
      'aria-valuemin': minWidth,
      'aria-valuemax': typeof window !== 'undefined' ? window.innerWidth : minWidth,
      'aria-valuenow': width,
      tabIndex: 0,
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onDoubleClick,
      onKeyDown,
    },
  };
}

export default useResizablePanel;
