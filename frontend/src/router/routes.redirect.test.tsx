// frontend/src/router/routes.redirect.test.tsx
import { describe, it, expect } from 'vitest';
import type { RouteObject } from 'react-router-dom';
import { Navigate } from 'react-router-dom';
import { routes, getNavigationRoutes } from './routes';

function findRoute(path: string): RouteObject | undefined {
  return routes.find((r) => r.path === path);
}

describe('/causal-discovery retirement (unified into /causal-analysis)', () => {
  it('still routes /causal-discovery (so the smoke spec gets HTML, not a 404)', () => {
    expect(findRoute('/causal-discovery')).toBeDefined();
  });

  it('redirects /causal-discovery to /causal-analysis (not NotFound)', () => {
    const route = findRoute('/causal-discovery');
    const el = route?.element as React.ReactElement;
    expect(el.type).toBe(Navigate);
    expect((el.props as { to: string }).to).toBe('/causal-analysis');
    expect((el.props as { replace?: boolean }).replace).toBe(true);
  });

  it('drops /causal-discovery from the sidebar nav (no dead link)', () => {
    const navPaths = getNavigationRoutes().map((r) => r.path);
    expect(navPaths).not.toContain('/causal-discovery');
    expect(navPaths).toContain('/causal-analysis');
  });
});
