import { describe, it, expect, beforeEach } from 'vitest';
import { useUIStore } from './ui-store';

/**
 * Guards the mobile-nav-drawer UX fix: the drawer must default CLOSED so the
 * header hamburger opens it (instead of closing an already-open drawer). On
 * desktop the sidebar is always visible regardless of this flag.
 */
describe('ui-store: sidebar drawer state', () => {
  beforeEach(() => {
    useUIStore.getState().reset();
  });

  it('defaults the mobile drawer to closed', () => {
    expect(useUIStore.getState().sidebarOpen).toBe(false);
  });

  it('toggleSidebar opens then closes the drawer', () => {
    useUIStore.getState().toggleSidebar();
    expect(useUIStore.getState().sidebarOpen).toBe(true);
    useUIStore.getState().toggleSidebar();
    expect(useUIStore.getState().sidebarOpen).toBe(false);
  });

  it('setSidebarOpen sets the drawer explicitly', () => {
    useUIStore.getState().setSidebarOpen(true);
    expect(useUIStore.getState().sidebarOpen).toBe(true);
  });

  it('keeps the desktop collapse preference independent of the drawer', () => {
    useUIStore.getState().toggleSidebarCollapsed();
    expect(useUIStore.getState().sidebarCollapsed).toBe(true);
    // Toggling the mobile drawer must not touch the collapse preference.
    useUIStore.getState().toggleSidebar();
    expect(useUIStore.getState().sidebarCollapsed).toBe(true);
  });
});

describe('ui-store: chat panel width', () => {
  beforeEach(() => {
    useUIStore.getState().reset();
  });

  it('defaults to null (component default width)', () => {
    expect(useUIStore.getState().chatPanelWidth).toBeNull();
  });

  it('setChatPanelWidth stores a custom width', () => {
    useUIStore.getState().setChatPanelWidth(640);
    expect(useUIStore.getState().chatPanelWidth).toBe(640);
  });

  it('setChatPanelWidth(null) resets to the component default', () => {
    useUIStore.getState().setChatPanelWidth(640);
    useUIStore.getState().setChatPanelWidth(null);
    expect(useUIStore.getState().chatPanelWidth).toBeNull();
  });
});
