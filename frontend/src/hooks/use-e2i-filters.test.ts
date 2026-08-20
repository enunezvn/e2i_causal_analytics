/**
 * useE2IFilters Hook Tests
 * ========================
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useE2IFilters } from './use-e2i-filters';

describe('useE2IFilters', () => {
  it('should return default filters', () => {
    const { result } = renderHook(() => useE2IFilters());

    expect(result.current.filters).toBeDefined();
    expect(result.current.filters.brand).toBe('All'); // #1749: no brand selected
    expect(result.current.filters.territory).toBeNull();
    expect(result.current.filters.hcpSegment).toBeNull();
  });

  it('should update brand filter', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setBrand('Kisqali');
    });

    expect(result.current.filters.brand).toBe('Kisqali');
  });

  it('should update territory filter', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setTerritory('Northeast');
    });

    expect(result.current.filters.territory).toBe('Northeast');
  });

  it('should update date range', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setDateRange('2024-01-01', '2024-06-30');
    });

    expect(result.current.filters.dateRange.start).toBe('2024-01-01');
    expect(result.current.filters.dateRange.end).toBe('2024-06-30');
  });

  it('should update HCP segment', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setHcpSegment('High Value');
    });

    expect(result.current.filters.hcpSegment).toBe('High Value');
  });

  it('should reset filters to defaults', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setBrand('Kisqali');
      result.current.setTerritory('West');
    });

    act(() => {
      result.current.resetFilters();
    });

    expect(result.current.filters.brand).toBe('All'); // #1749
    expect(result.current.filters.territory).toBeNull();
  });

  it('should generate filter summary', () => {
    const { result } = renderHook(() => useE2IFilters());

    const summary = result.current.getFilterSummary();

    expect(summary).toContain('All brands'); // #1749: sentinel rendered as prose
    expect(summary).toContain(' - '); // Date range separator
  });
});

// #1749: the old 'Remibrutinib' default leaked into chat pills and CoAgent
// state whenever the user had NOT picked that brand — no selection must mean
// 'All' (the value setBrandFilter and the backend filter note already speak).
describe('useE2IFilters — honest default (#1749)', () => {
  it("defaults the brand filter to 'All', not a hardcoded brand", () => {
    const { result } = renderHook(() => useE2IFilters());

    expect(result.current.filters.brand).toBe('All');
  });

  it("summarizes the 'All' sentinel as 'All brands', never as a pseudo-brand", () => {
    const { result } = renderHook(() => useE2IFilters());

    const summary = result.current.getFilterSummary();
    expect(summary).toContain('All brands');
    expect(summary).not.toContain('Remibrutinib');
  });

  it("resets back to the 'All' default", () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setBrand('Kisqali');
    });
    act(() => {
      result.current.resetFilters();
    });

    expect(result.current.filters.brand).toBe('All');
  });
});

// #1753: Home's Region selector re-scopes the KPI dashboard (KPI summary,
// batch-calculate context, value chains) but had NO channel to chat agent
// runs — E2IFilters carried no region field at all. 'All US' is the honest
// no-selection sentinel, mirroring the brand filter's 'All' (#1749).
describe('useE2IFilters — region filter (#1753)', () => {
  it("defaults the region filter to 'All US', not a specific region", () => {
    const { result } = renderHook(() => useE2IFilters());

    expect(result.current.filters.region).toBe('All US');
  });

  it('updates the region filter', () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setRegion('West');
    });

    expect(result.current.filters.region).toBe('West');
  });

  it("summarizes a selected region by name and the sentinel as 'All US regions'", () => {
    const { result } = renderHook(() => useE2IFilters());

    expect(result.current.getFilterSummary()).toContain('All US regions');

    act(() => {
      result.current.setRegion('Midwest');
    });

    expect(result.current.getFilterSummary()).toContain('Midwest');
  });

  it("resets back to the 'All US' default", () => {
    const { result } = renderHook(() => useE2IFilters());

    act(() => {
      result.current.setRegion('South');
    });
    act(() => {
      result.current.resetFilters();
    });

    expect(result.current.filters.region).toBe('All US');
  });
});
