/**
 * Route Configuration Tests
 * =========================
 *
 * Red-first tests for the password recovery routes. The login page has linked
 * /forgot-password since the auth feature landed (031ec2db) and AuthProvider
 * ships resetPassword/updatePassword actions plus a /reset-password
 * redirectTo, but the pages were never routed - both paths fell through to
 * the catch-all NotFound. Recovery routes must exist and be public
 * (reachable without authentication, like /login and /signup).
 */

import { describe, it, expect } from 'vitest';
import type { RouteObject } from 'react-router-dom';
import { routes } from './routes';

function findRoute(path: string): RouteObject | undefined {
  return routes.find((r) => r.path === path);
}

/** True when the route element tree is wrapped in ProtectedRoute */
function isProtected(route: RouteObject): boolean {
  const element = route.element as React.ReactElement | undefined;
  if (!element) return false;
  const name =
    typeof element.type === 'function' ? (element.type as { name?: string }).name : '';
  return name === 'ProtectedRoute';
}

describe('password recovery routes', () => {
  it('routes /forgot-password (linked from the login page)', () => {
    const route = findRoute('/forgot-password');
    expect(route).toBeDefined();
  });

  it('routes /reset-password (AuthProvider resetPassword redirect target)', () => {
    const route = findRoute('/reset-password');
    expect(route).toBeDefined();
  });

  it('keeps both recovery routes public like /login and /signup', () => {
    for (const path of ['/forgot-password', '/reset-password']) {
      const route = findRoute(path);
      expect(route, `${path} should be routed`).toBeDefined();
      expect(isProtected(route!), `${path} must not require auth`).toBe(false);
    }
  });
});
