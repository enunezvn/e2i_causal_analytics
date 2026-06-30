/**
 * ThemeToggle Tests
 * =================
 *
 * The button reflects the current mode and writes the opposite explicit choice
 * to the UI store. (Applying the class to <html> is ThemeManager's job and is
 * not exercised here.)
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeToggle } from './ThemeToggle';
import { useUIStore } from '@/stores/ui-store';

describe('ThemeToggle', () => {
  beforeEach(() => {
    useUIStore.getState().setTheme('light');
  });

  it('shows "switch to dark" while in light mode', () => {
    render(<ThemeToggle />);
    expect(
      screen.getByRole('button', { name: /switch to dark theme/i })
    ).toBeInTheDocument();
  });

  it('toggles the store theme light → dark → light', () => {
    render(<ThemeToggle />);
    fireEvent.click(screen.getByRole('button', { name: /switch to dark theme/i }));
    expect(useUIStore.getState().theme).toBe('dark');
    fireEvent.click(screen.getByRole('button', { name: /switch to light theme/i }));
    expect(useUIStore.getState().theme).toBe('light');
  });
});
