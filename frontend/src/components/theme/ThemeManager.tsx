/**
 * ThemeManager
 * ============
 *
 * Applies the persisted theme preference to the document root. It toggles the
 * `.dark` class on <html> — the same hook Tailwind's `dark:` variants and the
 * `.dark { ... }` token block in index.css key off — and sets `color-scheme`
 * so native controls (scrollbars, form widgets) match.
 *
 * `'system'` resolves against the OS preference with a live listener, so the
 * app follows the OS only when the user explicitly picks System. The default
 * is light (see ui-store initial state).
 *
 * Renders nothing; mount once near the app root so it covers every route.
 */

import { useEffect } from 'react';
import { useTheme } from '@/stores/ui-store';

export function ThemeManager(): null {
  const { theme } = useTheme();

  useEffect(() => {
    const root = document.documentElement;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');

    const apply = () => {
      const dark = theme === 'dark' || (theme === 'system' && mql.matches);
      root.classList.toggle('dark', dark);
      root.style.colorScheme = dark ? 'dark' : 'light';
    };

    apply();

    // Only follow OS changes while the user is on "system".
    if (theme === 'system') {
      mql.addEventListener('change', apply);
      return () => mql.removeEventListener('change', apply);
    }
  }, [theme]);

  return null;
}

export default ThemeManager;
