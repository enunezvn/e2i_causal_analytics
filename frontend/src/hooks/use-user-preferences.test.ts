/**
 * useUserPreferences Hook Tests
 * =============================
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useUserPreferences } from './use-user-preferences';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

describe('useUserPreferences', () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it('should return default preferences', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    expect(result.current.preferences.detailLevel).toBe('detailed');
    expect(result.current.preferences.defaultBrand).toBe('Remibrutinib');
    expect(result.current.preferences.notificationsEnabled).toBe(true);
    expect(result.current.preferences.theme).toBe('system');
  });

  it('should update detail level', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    act(() => {
      result.current.setDetailLevel('expert');
    });

    expect(result.current.preferences.detailLevel).toBe('expert');
  });

  it('should update default brand', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    act(() => {
      result.current.setDefaultBrand('Kisqali');
    });

    expect(result.current.preferences.defaultBrand).toBe('Kisqali');
  });

  it('should update theme', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    act(() => {
      result.current.setTheme('dark');
    });

    expect(result.current.preferences.theme).toBe('dark');
  });

  it('should toggle notifications', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    const initialState = result.current.preferences.notificationsEnabled;

    act(() => {
      result.current.toggleNotifications();
    });

    expect(result.current.preferences.notificationsEnabled).toBe(!initialState);
  });

  it('should reset preferences to defaults', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    act(() => {
      result.current.setDetailLevel('expert');
      result.current.setTheme('dark');
    });

    act(() => {
      result.current.resetPreferences();
    });

    expect(result.current.preferences.detailLevel).toBe('detailed');
    expect(result.current.preferences.theme).toBe('system');
    expect(localStorageMock.removeItem).toHaveBeenCalledWith('e2i-user-preferences');
  });

  it('should persist preferences to localStorage', async () => {
    const { result } = renderHook(() => useUserPreferences());

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    act(() => {
      result.current.setDetailLevel('summary');
    });

    // Check that localStorage was called
    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalled();
    });
  });
});
