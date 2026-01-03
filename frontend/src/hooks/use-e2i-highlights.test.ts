/**
 * useE2IHighlights Hook Tests
 * ===========================
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useE2IHighlights } from './use-e2i-highlights';

describe('useE2IHighlights', () => {
  it('should return empty highlights initially', () => {
    const { result } = renderHook(() => useE2IHighlights());

    expect(result.current.highlightedPaths).toEqual([]);
    expect(result.current.highlightCount).toBe(0);
  });

  it('should add a highlight', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.addHighlight('path-1');
    });

    expect(result.current.highlightedPaths).toContain('path-1');
    expect(result.current.highlightCount).toBe(1);
  });

  it('should not add duplicate highlights', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.addHighlight('path-1');
      result.current.addHighlight('path-1');
    });

    expect(result.current.highlightCount).toBe(1);
  });

  it('should remove a highlight', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.addHighlight('path-1');
      result.current.addHighlight('path-2');
    });

    act(() => {
      result.current.removeHighlight('path-1');
    });

    expect(result.current.highlightedPaths).not.toContain('path-1');
    expect(result.current.highlightedPaths).toContain('path-2');
  });

  it('should toggle a highlight', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.toggleHighlight('path-1');
    });

    expect(result.current.isHighlighted('path-1')).toBe(true);

    act(() => {
      result.current.toggleHighlight('path-1');
    });

    expect(result.current.isHighlighted('path-1')).toBe(false);
  });

  it('should set multiple highlights at once', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.setHighlights(['path-1', 'path-2', 'path-3']);
    });

    expect(result.current.highlightCount).toBe(3);
  });

  it('should clear all highlights', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.addHighlight('path-1');
      result.current.addHighlight('path-2');
    });

    act(() => {
      result.current.clearHighlights();
    });

    expect(result.current.highlightCount).toBe(0);
  });

  it('should check if path is highlighted', () => {
    const { result } = renderHook(() => useE2IHighlights());

    act(() => {
      result.current.addHighlight('path-1');
    });

    expect(result.current.isHighlighted('path-1')).toBe(true);
    expect(result.current.isHighlighted('path-2')).toBe(false);
  });
});
