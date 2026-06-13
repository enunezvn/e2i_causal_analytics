/**
 * Shared per-spec harness for the coverage-gap specs (#19).
 * ========================================================
 *
 * These helpers are intentionally SEPARATE from the broad
 * `fixtures/api-mocks.ts` graph (which 15+ existing specs depend on). The
 * e2e/README "do not edit api-mocks.ts from a per-spec PR" guidance applies —
 * so the new gap-page specs compose their own auth seed + CopilotKit stub +
 * page-specific endpoint stubs here and register page-specific routes inline.
 *
 * Design intent mirrors the rewritten gold-pattern specs
 * (`ai-insights.spec.ts`, `agent-orchestration.spec.ts`,
 * `intervention-impact.spec.ts`):
 *  - Seed a fake Supabase session so <ProtectedRoute> renders the page
 *    (local dev `.env` sets VITE_SUPABASE_ANON_KEY, so isSupabaseConfigured()
 *    is true and the gate is live — without a session every protected page
 *    redirects to /login).
 *  - Stub the CopilotKit runtime so the E2ICopilotProvider mounted by
 *    RootLayout (VITE_COPILOT_ENABLED=true in the dev `.env`) does not crash
 *    the page subtree with "Agent 'default' not found".
 *  - We stub the REAL backend endpoints each page calls and assert HONEST
 *    states (real data renders / honest empty state / labeled error). We do
 *    NOT assert against fabricated sample data — the gap pages were rewritten
 *    (F-002) to render explicit EmptyState / QueryErrorState instead of mock
 *    fallbacks, which is exactly what these specs lock in.
 */

import type { Page, Route } from '@playwright/test';

// ---------------------------------------------------------------------------
// Auth seeding — mirrors `seedAuthSession()` in fixtures/api-mocks.ts.
// ---------------------------------------------------------------------------

export async function seedAuthSession(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const now = new Date().toISOString();
    const fakeUser = {
      id: 'e2e-mock-user-id',
      aud: 'authenticated',
      role: 'authenticated',
      email: 'e2e@test.local',
      email_confirmed_at: now,
      phone: '',
      confirmed_at: now,
      last_sign_in_at: now,
      app_metadata: { provider: 'email' },
      user_metadata: { full_name: 'E2E User' },
      identities: [],
      created_at: now,
      updated_at: now,
    };
    const fakeSession = {
      access_token: 'e2e-mock-access-token',
      refresh_token: 'e2e-mock-refresh-token',
      expires_in: 3600,
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      token_type: 'bearer',
      user: fakeUser,
    };
    try {
      window.localStorage.setItem('e2i-auth-token', JSON.stringify(fakeSession));
      window.localStorage.setItem(
        'e2i-auth-store',
        JSON.stringify({
          state: {
            session: fakeSession,
            user: fakeUser,
            isInitialized: true,
            isLoading: false,
            redirectTo: null,
          },
          version: 0,
        }),
      );
    } catch {
      // localStorage may not be available; ignore.
    }
  });

  await page.route('**/auth/v1/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        access_token: 'e2e-mock-access-token',
        refresh_token: 'e2e-mock-refresh-token',
        token_type: 'bearer',
        expires_in: 3600,
        user: {
          id: 'e2e-mock-user-id',
          email: 'e2e@test.local',
          aud: 'authenticated',
          role: 'authenticated',
        },
      }),
    });
  });
}

// ---------------------------------------------------------------------------
// CopilotKit runtime stub — the RootLayout mounts E2ICopilotProvider, and the
// dev `.env` enables it (VITE_COPILOT_ENABLED=true). The runtime returns
// `agents` as a DICT keyed by name; an array (or a thread-shape for the
// "info" method) crashes every page that mounts the provider. Mirrors the
// shared fixture's copilot handling.
// ---------------------------------------------------------------------------

const COPILOT_INFO = {
  version: '1.0.0',
  agents: {
    default: { name: 'default', description: 'Default agent', className: 'MockAgent' },
    orchestrator: { name: 'orchestrator', description: 'Query routing', className: 'MockAgent' },
  },
  actions: [],
  copilotReadable: [],
  audioFileTranscriptionEnabled: false,
};

export async function stubCopilotRuntime(page: Page): Promise<void> {
  await page.route('**/api/copilotkit/info**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(COPILOT_INFO),
    });
  });

  // Match both `/api/copilotkit` and `/api/copilotkit/` (router/index.tsx
  // passes the trailing-slash form as `runtimeUrl`).
  await page.route(/\/api\/copilotkit\/?(\?.*)?$/, async (route: Route) => {
    if (route.request().method() === 'POST') {
      let method = '';
      try {
        const body = route.request().postDataJSON() as { method?: string } | null;
        method = body?.method ?? '';
      } catch {
        method = '';
      }
      if (method === 'info') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(COPILOT_INFO),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ threadId: 'test-thread-id', messages: [], agentState: {} }),
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Seed auth + stub CopilotKit. Call FIRST in beforeEach, then register any
 * page-specific endpoint stubs AFTER (Playwright resolves the most-recently
 * registered matching route first, so per-page stubs win over these).
 */
export async function harnessBase(page: Page): Promise<void> {
  await seedAuthSession(page);
  await stubCopilotRuntime(page);
}

/**
 * Robust navigation: goto → wait for the page heading → up to two
 * reload-retries if the SPA's lazy chunk hiccups under parallel-agent load
 * (the historical Cat-B blank-screen failure mode). Mirrors the
 * ai-insights / intervention-impact gold pattern.
 */
export async function gotoAndWaitForHeading(
  page: Page,
  url: string,
  headingRe: RegExp,
): Promise<void> {
  await page.goto(url);
  await page.waitForLoadState('domcontentloaded');
  const heading = page.getByRole('heading', { name: headingRe }).first();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await heading.waitFor({ state: 'visible', timeout: 7000 });
      return;
    } catch {
      const tryAgain = page.getByRole('button', { name: /Try Again/i });
      if (await tryAgain.isVisible().catch(() => false)) {
        await tryAgain.click();
        continue;
      }
      await page.reload();
    }
  }
  // Final swing — surfaces the real assertion error if it still fails.
  await heading.waitFor({ state: 'visible', timeout: 7000 });
}
