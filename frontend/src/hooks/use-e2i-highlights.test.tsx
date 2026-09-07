/**
 * useE2IHighlights Hook Tests
 * ===========================
 */

import { describe, it, expect } from 'vitest';
import { render, renderHook, act, screen } from '@testing-library/react';
import { useE2IHighlights } from './use-e2i-highlights';
import {
  CopilotKitWrapper,
  E2ICopilotProvider,
  useE2ICopilot,
} from '@/providers/E2ICopilotProvider';

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

// #1942: the two cases the try/catch inside useE2IHighlights() must keep
// telling apart correctly. See use-e2i-filters.test.tsx for the full
// reasoning — same pattern, same provider.
describe('useE2IHighlights — provider-presence fallback (#1942)', () => {
  it('rendered with NO CopilotKit/E2I provider above it: does not throw and returns the local-state fallback', () => {
    expect(() => {
      const { result } = renderHook(() => useE2IHighlights());
      expect(result.current.enabled).toBe(false);
      expect(result.current.highlightedPaths).toEqual([]);
    }).not.toThrow();
  });

  it('rendered inside E2ICopilotProvider with CopilotKit disabled: still uses the REAL shared context, not an isolated local copy', () => {
    // Matches RootLayout's actual wiring (frontend/src/router/index.tsx):
    // <E2ICopilotProvider> is mounted unconditionally inside
    // <CopilotKitWrapper enabled={env.copilotEnabled}> — dev (default) and
    // the CI e2e build run with copilot disabled but the provider present.
    // useCopilotEnabled() reports `false` in this case too, indistinguishable
    // from "no provider at all" — a naive useCopilotEnabled()-gated rewrite
    // of the try/catch would silently stop sharing this state.
    function Setter() {
      const { addHighlight } = useE2IHighlights();
      return <button onClick={() => addHighlight('path-1')}>set</button>;
    }
    function DirectReader() {
      const { highlightedPaths } = useE2ICopilot();
      return <span data-testid="count">{highlightedPaths.length}</span>;
    }

    render(
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>
          <Setter />
          <DirectReader />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('count').textContent).toBe('0');

    act(() => {
      screen.getByText('set').click();
    });

    expect(screen.getByTestId('count').textContent).toBe('1');
  });
});
