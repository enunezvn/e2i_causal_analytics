/**
 * useUserPreferences Hook Tests
 * =============================
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, renderHook, act, waitFor, screen } from '@testing-library/react';
import { useUserPreferences } from './use-user-preferences';
import {
  CopilotKitWrapper,
  E2ICopilotProvider,
  useE2ICopilot,
} from '@/providers/E2ICopilotProvider';

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

// #1942: the two cases the try/catch inside useUserPreferences() must keep
// telling apart correctly. See use-e2i-filters.test.tsx for the full
// reasoning — same pattern, same provider.
describe('useUserPreferences — provider-presence fallback (#1942)', () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it('rendered with NO CopilotKit/E2I provider above it: does not throw and returns the local-state fallback', async () => {
    const { result } = renderHook(() => useUserPreferences());

    expect(result.current.enabled).toBe(false);

    await waitFor(() => {
      expect(result.current.isLoaded).toBe(true);
    });

    expect(result.current.preferences.detailLevel).toBe('detailed');
  });

  it('rendered inside E2ICopilotProvider with CopilotKit disabled: still uses the REAL shared context, not an isolated local copy', async () => {
    // Matches RootLayout's actual wiring (frontend/src/router/index.tsx):
    // <E2ICopilotProvider> is mounted unconditionally inside
    // <CopilotKitWrapper enabled={env.copilotEnabled}> — dev (default) and
    // the CI e2e build run with copilot disabled but the provider present.
    // useCopilotEnabled() reports `false` in this case too, indistinguishable
    // from "no provider at all" — a naive useCopilotEnabled()-gated rewrite
    // of the try/catch would silently stop sharing this state.
    function Setter() {
      const { setDetailLevel, isLoaded } = useUserPreferences();
      return (
        <button disabled={!isLoaded} onClick={() => setDetailLevel('expert')}>
          set
        </button>
      );
    }
    function DirectReader() {
      const { preferences } = useE2ICopilot();
      return <span data-testid="level">{preferences.detailLevel}</span>;
    }

    render(
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>
          <Setter />
          <DirectReader />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('set')).not.toBeDisabled();
    });
    expect(screen.getByTestId('level').textContent).toBe('detailed');

    act(() => {
      screen.getByText('set').click();
    });

    expect(screen.getByTestId('level').textContent).toBe('expert');
  });
});
