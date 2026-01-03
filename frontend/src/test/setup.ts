/**
 * Vitest Test Setup
 * =================
 *
 * Global test configuration for Vitest.
 * Sets up MSW server for API mocking and testing-library matchers.
 */

import '@testing-library/jest-dom';
import { afterAll, afterEach, beforeAll, vi } from 'vitest';
import { server } from '@/mocks/server';

// =============================================================================
// BROWSER API POLYFILLS
// =============================================================================

// ResizeObserver mock (required by Radix UI components)
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;

// PointerEvent mock (required by Radix UI Slider)
class MockPointerEvent extends MouseEvent {
  pointerId: number;
  pressure: number;
  tangentialPressure: number;
  tiltX: number;
  tiltY: number;
  twist: number;
  width: number;
  height: number;
  pointerType: string;
  isPrimary: boolean;

  constructor(type: string, props: PointerEventInit = {}) {
    super(type, props);
    this.pointerId = props.pointerId ?? 0;
    this.pressure = props.pressure ?? 0;
    this.tangentialPressure = props.tangentialPressure ?? 0;
    this.tiltX = props.tiltX ?? 0;
    this.tiltY = props.tiltY ?? 0;
    this.twist = props.twist ?? 0;
    this.width = props.width ?? 1;
    this.height = props.height ?? 1;
    this.pointerType = props.pointerType ?? 'mouse';
    this.isPrimary = props.isPrimary ?? true;
  }

  getCoalescedEvents(): PointerEvent[] {
    return [];
  }

  getPredictedEvents(): PointerEvent[] {
    return [];
  }
}

global.PointerEvent = MockPointerEvent as unknown as typeof PointerEvent;

// Element.scrollIntoView mock
Element.prototype.scrollIntoView = vi.fn();

// hasPointerCapture/setPointerCapture/releasePointerCapture mocks (Radix UI)
Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
Element.prototype.setPointerCapture = vi.fn();
Element.prototype.releasePointerCapture = vi.fn();

// Start MSW server before all tests
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));

// Reset handlers after each test for clean state
afterEach(() => server.resetHandlers());

// Close server after all tests
afterAll(() => server.close());
