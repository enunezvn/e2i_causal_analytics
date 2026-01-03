/**
 * User Preferences Hook
 * =====================
 *
 * Provides access to user preference state with localStorage persistence.
 * Manages detail level, default brand, theme, and notification settings.
 *
 * @module hooks/use-user-preferences
 */

import * as React from 'react';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';
import type { UserPreferences, E2IFilters } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface UseUserPreferencesReturn {
  /** Current preferences */
  preferences: UserPreferences;
  /** Whether CopilotKit is enabled */
  enabled: boolean;
  /** Set detail level */
  setDetailLevel: (level: UserPreferences['detailLevel']) => void;
  /** Set default brand */
  setDefaultBrand: (brand: E2IFilters['brand']) => void;
  /** Set theme */
  setTheme: (theme: UserPreferences['theme']) => void;
  /** Toggle notifications */
  toggleNotifications: () => void;
  /** Reset to defaults */
  resetPreferences: () => void;
  /** Check if preferences are loaded */
  isLoaded: boolean;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'e2i-user-preferences';

const DEFAULT_PREFERENCES: UserPreferences = {
  detailLevel: 'detailed',
  defaultBrand: 'Remibrutinib',
  notificationsEnabled: true,
  theme: 'system',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function loadPreferences(): UserPreferences {
  if (typeof window === 'undefined') return DEFAULT_PREFERENCES;

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) };
    }
  } catch {
    // Ignore parse errors
  }

  return DEFAULT_PREFERENCES;
}

function savePreferences(preferences: UserPreferences): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Ignore storage errors
  }
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook for managing user preferences with localStorage persistence.
 *
 * @example
 * ```tsx
 * const { preferences, setDetailLevel, setTheme } = useUserPreferences();
 *
 * // Change detail level
 * setDetailLevel('expert');
 *
 * // Change theme
 * setTheme('dark');
 * ```
 */
export function useUserPreferences(): UseUserPreferencesReturn {
  const enabled = useCopilotEnabled();
  const [isLoaded, setIsLoaded] = React.useState(false);

  // Local state with persistence
  const [localPreferences, setLocalPreferences] = React.useState<UserPreferences>(
    DEFAULT_PREFERENCES
  );

  // Try to get context
  let contextPreferences: UserPreferences | null = null;
  let setContextPreferences: React.Dispatch<React.SetStateAction<UserPreferences>> | null =
    null;

  try {
    const context = useE2ICopilot();
    contextPreferences = context.preferences;
    setContextPreferences = context.setPreferences;
  } catch {
    // Context not available
  }

  const preferences = contextPreferences || localPreferences;
  const setPreferences = setContextPreferences || setLocalPreferences;

  // Load from localStorage on mount
  React.useEffect(() => {
    const loaded = loadPreferences();
    setPreferences(loaded);
    setIsLoaded(true);
  }, [setPreferences]);

  // Save to localStorage when preferences change
  React.useEffect(() => {
    if (isLoaded) {
      savePreferences(preferences);
    }
  }, [preferences, isLoaded]);

  const setDetailLevel = React.useCallback(
    (level: UserPreferences['detailLevel']) => {
      setPreferences((prev) => ({ ...prev, detailLevel: level }));
    },
    [setPreferences]
  );

  const setDefaultBrand = React.useCallback(
    (brand: E2IFilters['brand']) => {
      setPreferences((prev) => ({ ...prev, defaultBrand: brand }));
    },
    [setPreferences]
  );

  const setTheme = React.useCallback(
    (theme: UserPreferences['theme']) => {
      setPreferences((prev) => ({ ...prev, theme }));
    },
    [setPreferences]
  );

  const toggleNotifications = React.useCallback(() => {
    setPreferences((prev) => ({
      ...prev,
      notificationsEnabled: !prev.notificationsEnabled,
    }));
  }, [setPreferences]);

  const resetPreferences = React.useCallback(() => {
    setPreferences(DEFAULT_PREFERENCES);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [setPreferences]);

  return {
    preferences,
    enabled,
    setDetailLevel,
    setDefaultBrand,
    setTheme,
    toggleNotifications,
    resetPreferences,
    isLoaded,
  };
}

export default useUserPreferences;
