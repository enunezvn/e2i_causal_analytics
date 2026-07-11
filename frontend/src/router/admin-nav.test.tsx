/**
 * Admin nav gating — /admin appears in the sidebar ONLY for admins, and the
 * route registry carries adminOnly + requireAdmin protection.
 */
import { describe, it, expect } from 'vitest';
import { getNavigationSections, routeConfigs, routes } from './routes';

describe('admin nav registry', () => {
  it('has an adminOnly /admin entry in the system section', () => {
    const admin = routeConfigs.find((r) => r.path === '/admin');
    expect(admin).toBeDefined();
    expect(admin!.adminOnly).toBe(true);
    expect(admin!.section).toBe('system');
    expect(admin!.showInNav).toBe(true);
  });

  it('getNavigationSections(false) excludes adminOnly routes', () => {
    const sections = getNavigationSections(false);
    const paths = sections.flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).not.toContain('/admin');
  });

  it('getNavigationSections(true) includes /admin', () => {
    const sections = getNavigationSections(true);
    const paths = sections.flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).toContain('/admin');
  });

  it('defaults to excluding adminOnly (backwards-compatible no-arg call)', () => {
    const paths = getNavigationSections().flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).not.toContain('/admin');
  });

  it('registers /admin and /accept-invite as router routes', () => {
    const paths = routes.map((r) => r.path);
    expect(paths).toContain('/admin');
    expect(paths).toContain('/accept-invite');
  });
});
