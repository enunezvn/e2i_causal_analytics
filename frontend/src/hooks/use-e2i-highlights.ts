/**
 * E2I Highlights Hook
 * ===================
 *
 * Provides access to highlighted causal paths on visualizations.
 * Used to synchronize highlights between AI chat and graph components.
 *
 * @module hooks/use-e2i-highlights
 */

import * as React from 'react';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface HighlightedPath {
  id: string;
  source: string;
  target: string;
  weight?: number;
  type?: 'causal' | 'correlation' | 'temporal';
}

export interface UseE2IHighlightsReturn {
  /** List of highlighted path IDs */
  highlightedPaths: string[];
  /** Whether CopilotKit is enabled */
  enabled: boolean;
  /** Add a path to highlights */
  addHighlight: (pathId: string) => void;
  /** Remove a path from highlights */
  removeHighlight: (pathId: string) => void;
  /** Toggle a path highlight */
  toggleHighlight: (pathId: string) => void;
  /** Set multiple highlights at once */
  setHighlights: (pathIds: string[]) => void;
  /** Clear all highlights */
  clearHighlights: () => void;
  /** Check if a path is highlighted */
  isHighlighted: (pathId: string) => boolean;
  /** Get highlight count */
  highlightCount: number;
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook for managing highlighted causal paths on visualizations.
 *
 * @example
 * ```tsx
 * const { highlightedPaths, addHighlight, clearHighlights, isHighlighted } = useE2IHighlights();
 *
 * // Highlight a specific path
 * addHighlight('path-123');
 *
 * // Check if path is highlighted for styling
 * const style = isHighlighted('path-123') ? 'highlighted' : 'normal';
 * ```
 */
export function useE2IHighlights(): UseE2IHighlightsReturn {
  const enabled = useCopilotEnabled();

  // Local state for when context is not available
  const [localHighlights, setLocalHighlights] = React.useState<string[]>([]);

  // Try to get context
  let contextHighlights: string[] | null = null;
  let setContextHighlights: React.Dispatch<React.SetStateAction<string[]>> | null = null;

  try {
    const context = useE2ICopilot();
    contextHighlights = context.highlightedPaths;
    setContextHighlights = context.setHighlightedPaths;
  } catch {
    // Context not available
  }

  const highlightedPaths = contextHighlights || localHighlights;
  const setHighlightedPaths = setContextHighlights || setLocalHighlights;

  const addHighlight = React.useCallback(
    (pathId: string) => {
      setHighlightedPaths((prev) => {
        if (prev.includes(pathId)) return prev;
        return [...prev, pathId];
      });
    },
    [setHighlightedPaths]
  );

  const removeHighlight = React.useCallback(
    (pathId: string) => {
      setHighlightedPaths((prev) => prev.filter((id) => id !== pathId));
    },
    [setHighlightedPaths]
  );

  const toggleHighlight = React.useCallback(
    (pathId: string) => {
      setHighlightedPaths((prev) => {
        if (prev.includes(pathId)) {
          return prev.filter((id) => id !== pathId);
        }
        return [...prev, pathId];
      });
    },
    [setHighlightedPaths]
  );

  const setHighlights = React.useCallback(
    (pathIds: string[]) => {
      setHighlightedPaths(pathIds);
    },
    [setHighlightedPaths]
  );

  const clearHighlights = React.useCallback(() => {
    setHighlightedPaths([]);
  }, [setHighlightedPaths]);

  const isHighlighted = React.useCallback(
    (pathId: string): boolean => {
      return highlightedPaths.includes(pathId);
    },
    [highlightedPaths]
  );

  return {
    highlightedPaths,
    enabled,
    addHighlight,
    removeHighlight,
    toggleHighlight,
    setHighlights,
    clearHighlights,
    isHighlighted,
    highlightCount: highlightedPaths.length,
  };
}

export default useE2IHighlights;
