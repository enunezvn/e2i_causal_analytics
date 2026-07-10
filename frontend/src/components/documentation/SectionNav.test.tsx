/**
 * SectionNav Tests
 * ================
 * Scroll-spy behavior unit tests. Page-level shell behavior (render, click →
 * scrollIntoView) lives in src/pages/Documentation.test.tsx.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { SectionNav } from './SectionNav';
import { DOC_SECTIONS } from './content';

type IOCallback = (entries: IntersectionObserverEntry[]) => void;

let capturedCallback: IOCallback | undefined;

class FakeIntersectionObserver {
  constructor(callback: IOCallback) {
    capturedCallback = callback;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
}

function entryFor(id: string, top: number, isIntersecting: boolean): IntersectionObserverEntry {
  return {
    isIntersecting,
    target: { id },
    boundingClientRect: { top },
  } as unknown as IntersectionObserverEntry;
}

beforeEach(() => {
  capturedCallback = undefined;
  vi.stubGlobal('IntersectionObserver', FakeIntersectionObserver);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('SectionNav scroll-spy', () => {
  it('activates the TOPMOST intersecting section, not the last in the batch', () => {
    render(<SectionNav sections={DOC_SECTIONS} />);
    expect(capturedCallback).toBeDefined();

    // Two sections intersect at once (e.g. the initial observer callback);
    // methodology is LAST in the array but purpose is topmost on screen.
    act(() => {
      capturedCallback!([
        entryFor('purpose', 10, true),
        entryFor('methodology', 200, true),
      ]);
    });

    expect(screen.getByRole('button', { name: /^purpose$/i })).toHaveAttribute(
      'aria-current',
      'true'
    );
    expect(screen.getByRole('button', { name: /^methodology$/i })).not.toHaveAttribute(
      'aria-current'
    );
  });

  it('ignores batches with no intersecting entries', () => {
    render(<SectionNav sections={DOC_SECTIONS} />);

    // Activate methodology first.
    act(() => {
      capturedCallback!([entryFor('methodology', 10, true)]);
    });
    expect(screen.getByRole('button', { name: /^methodology$/i })).toHaveAttribute(
      'aria-current',
      'true'
    );

    // A leave-only batch (nothing intersecting) must not change the active section.
    act(() => {
      capturedCallback!([entryFor('methodology', -500, false)]);
    });
    expect(screen.getByRole('button', { name: /^methodology$/i })).toHaveAttribute(
      'aria-current',
      'true'
    );
  });
});
