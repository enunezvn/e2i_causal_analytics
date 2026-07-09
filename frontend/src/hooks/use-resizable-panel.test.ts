/**
 * useResizablePanel Hook Tests
 * ============================
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as React from 'react';
import { useResizablePanel, type UseResizablePanelOptions } from './use-resizable-panel';

// jsdom default viewport is 1024px wide
const WINDOW_WIDTH = 1024;

function pointerEvent(clientX: number): React.PointerEvent<HTMLElement> {
  return {
    button: 0,
    clientX,
    pointerId: 1,
    preventDefault: vi.fn(),
    currentTarget: { setPointerCapture: vi.fn() },
  } as unknown as React.PointerEvent<HTMLElement>;
}

function keyEvent(key: string): React.KeyboardEvent<HTMLElement> {
  return {
    key,
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLElement>;
}

function setup(overrides: Partial<UseResizablePanelOptions> = {}) {
  const onWidthChange = vi.fn();
  const options: UseResizablePanelOptions = {
    defaultWidth: 400,
    minWidth: 320,
    edge: 'left',
    persistedWidth: null,
    onWidthChange,
    ...overrides,
  };
  const view = renderHook((props: UseResizablePanelOptions) => useResizablePanel(props), {
    initialProps: options,
  });
  return { ...view, onWidthChange, options };
}

describe('useResizablePanel', () => {
  beforeEach(() => {
    window.innerWidth = WINDOW_WIDTH;
    document.body.style.userSelect = '';
  });

  it('uses defaultWidth when no persisted width is set', () => {
    const { result } = setup();
    expect(result.current.width).toBe(400);
    expect(result.current.isDragging).toBe(false);
  });

  it('uses persistedWidth when set', () => {
    const { result } = setup({ persistedWidth: 640 });
    expect(result.current.width).toBe(640);
  });

  it('tracks pointer drag live for a left-edge handle (right-docked panel)', () => {
    const { result, onWidthChange } = setup();

    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(700));
    });
    // width = window right edge - pointer x
    expect(result.current.width).toBe(WINDOW_WIDTH - 700);
    expect(result.current.isDragging).toBe(true);
    expect(onWidthChange).not.toHaveBeenCalled();

    act(() => {
      result.current.handleProps.onPointerMove(pointerEvent(500));
    });
    expect(result.current.width).toBe(WINDOW_WIDTH - 500);
  });

  it('commits the final width via onWidthChange only on pointer up', () => {
    const { result, onWidthChange } = setup();

    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(700));
      result.current.handleProps.onPointerMove(pointerEvent(424));
      result.current.handleProps.onPointerUp(pointerEvent(424));
    });

    expect(onWidthChange).toHaveBeenCalledTimes(1);
    expect(onWidthChange).toHaveBeenCalledWith(WINDOW_WIDTH - 424);
    expect(result.current.isDragging).toBe(false);
  });

  it('ignores pointer moves when no drag is in progress', () => {
    const { result } = setup();
    act(() => {
      result.current.handleProps.onPointerMove(pointerEvent(500));
    });
    expect(result.current.width).toBe(400);
  });

  it('clamps to minWidth', () => {
    const { result } = setup();
    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(WINDOW_WIDTH - 100));
    });
    expect(result.current.width).toBe(320);
  });

  it('clamps to full window width', () => {
    const { result, onWidthChange } = setup();
    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(700));
      result.current.handleProps.onPointerMove(pointerEvent(-50));
      result.current.handleProps.onPointerUp(pointerEvent(-50));
    });
    expect(onWidthChange).toHaveBeenCalledWith(WINDOW_WIDTH);
  });

  it('computes width from the pointer x for a right-edge handle (left-docked panel)', () => {
    const { result } = setup({ edge: 'right' });
    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(555));
    });
    expect(result.current.width).toBe(555);
  });

  it('resets to default on double-click', () => {
    const { result, onWidthChange } = setup({ persistedWidth: 640 });
    act(() => {
      result.current.handleProps.onDoubleClick();
    });
    expect(onWidthChange).toHaveBeenCalledWith(null);
  });

  it('grows and shrinks with arrow keys (left-edge handle)', () => {
    const { result, onWidthChange } = setup({ persistedWidth: 400 });

    act(() => {
      result.current.handleProps.onKeyDown(keyEvent('ArrowLeft'));
    });
    expect(onWidthChange).toHaveBeenLastCalledWith(432);

    act(() => {
      result.current.handleProps.onKeyDown(keyEvent('ArrowRight'));
    });
    expect(onWidthChange).toHaveBeenLastCalledWith(368);

    act(() => {
      result.current.handleProps.onKeyDown(keyEvent('Enter'));
    });
    expect(onWidthChange).toHaveBeenCalledTimes(2);
  });

  it('disables text selection during drag and restores it on release', () => {
    const { result } = setup();
    act(() => {
      result.current.handleProps.onPointerDown(pointerEvent(700));
    });
    expect(document.body.style.userSelect).toBe('none');
    act(() => {
      result.current.handleProps.onPointerUp(pointerEvent(700));
    });
    expect(document.body.style.userSelect).toBe('');
  });

  it('ignores non-primary buttons', () => {
    const { result } = setup();
    const e = pointerEvent(700);
    (e as { button: number }).button = 2;
    act(() => {
      result.current.handleProps.onPointerDown(e);
    });
    expect(result.current.isDragging).toBe(false);
  });

  it('exposes separator ARIA attributes reflecting the current width', () => {
    const { result } = setup({ persistedWidth: 500 });
    expect(result.current.handleProps.role).toBe('separator');
    expect(result.current.handleProps['aria-orientation']).toBe('vertical');
    expect(result.current.handleProps['aria-valuemin']).toBe(320);
    expect(result.current.handleProps['aria-valuemax']).toBe(WINDOW_WIDTH);
    expect(result.current.handleProps['aria-valuenow']).toBe(500);
  });
});
