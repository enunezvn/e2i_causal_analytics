/**
 * AI Agent Insights Page E2E Tests
 * =================================
 *
 * Tests for the AI Agent Insights page functionality.
 *
 * Auth + API mocks are intentionally inlined here (not pulled from the shared
 * `fixtures/api-mocks.ts`) so this spec stays decoupled from the larger mock
 * graph (15+ other specs depend on the shared fixture).
 *
 * PR #312 live-wired this page through `ProtectedRoute` and three additional
 * data hooks (`useCognitiveRAG`, `useCausalChains`, `useAlerts`/`useModelHealth`/
 * `useMonitoringRuns`, `useBatchExplain`). Without auth seed every test redirects
 * to `/login`; without endpoint mocks the loading skeletons of `PredictiveAlerts`
 * and `SystemHealthScore` hide the landmarks the assertions look for. We seed
 * both below in `beforeEach` so the page renders deterministically.
 *
 * Refs #332.
 */

import { test, expect, type Page, type Route } from '@playwright/test';

// ---------------------------------------------------------------------------
// Auth seeding — mirrors `seedAuthSession()` in fixtures/api-mocks.ts so the
// SPA's AuthProvider finds a session and ProtectedRoute lets the page render.
// ---------------------------------------------------------------------------

async function seedAuthSession(page: Page): Promise<void> {
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
// Endpoint stubs. Every insight component falls back to its own SAMPLE_*
// fixture when the API returns empty/zero payloads, so these mocks need only
// be shape-correct — the rendered DOM is then driven by the sample data and
// the landmark assertions in this file pass deterministically.
// ---------------------------------------------------------------------------

async function mockInsightsEndpoints(page: Page): Promise<void> {
  // CopilotKit info — page mounts E2ICopilotProvider; must respond with an
  // agents dict (not array) or CopilotKit crashes the whole subtree.
  const copilotInfo = {
    version: '1.0.0',
    agents: {
      default: { name: 'default', description: 'Default agent', className: 'MockAgent' },
    },
    actions: [],
    copilotReadable: [],
    audioFileTranscriptionEnabled: false,
  };
  await page.route('**/api/copilotkit/info**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(copilotInfo),
    });
  });
  // Match both `/api/copilotkit` and `/api/copilotkit/` (router/index.tsx passes
  // the trailing-slash form as `runtimeUrl`; CopilotKit's POSTs go there).
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
          body: JSON.stringify(copilotInfo),
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

  // ExecutiveAIBrief → POST /api/cognitive/rag
  await page.route('**/api/cognitive/rag', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        response: '',
        evidence: [],
        hop_count: 0,
        visualization_config: {},
        routed_agents: [],
        entities: [],
        intent: 'unknown',
      }),
    });
  });

  // ActiveCausalChains → POST /api/graph/causal-chains
  await page.route('**/api/graph/causal-chains', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        chains: [],
        total_chains: 0,
        query_latency_ms: 1,
        timestamp: new Date().toISOString(),
      }),
    });
  });

  // PredictiveAlerts + SystemHealthScore → GET /api/monitoring/alerts
  // Empty `alerts` list trips the SAMPLE_ALERTS fallback in the component.
  await page.route('**/api/monitoring/alerts**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ total_count: 0, active_count: 0, alerts: [] }),
    });
  });

  // SystemHealthScore → GET /api/monitoring/health/<modelId>
  await page.route('**/api/monitoring/health/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model_id: 'propensity_v2.1.0',
        overall_health: 'healthy',
        last_check: new Date().toISOString(),
        drift_score: 0.05,
        active_alerts: 0,
        performance_trend: 'stable',
        recommendations: [],
      }),
    });
  });

  // SystemHealthScore → GET /api/monitoring/runs
  await page.route('**/api/monitoring/runs**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ total_runs: 0, runs: [] }),
    });
  });

  // SystemHealthScore → GET /api/health-score/full (real Tier-3 health-score
  // agent; HealthScoreResponse schema verified against the live OpenAPI spec).
  // The widget renders Component/Model/Pipeline/Agent Health rows from these
  // scores — '—' + "Not measured in this check" when a dimension is null.
  await page.route('**/api/health-score/full**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        check_id: 'e2e-health-check',
        check_scope: 'full',
        // overall is 0-100; per-dimension scores are 0-1 fractions
        // (None = unmeasured) per the HealthScoreResult contract.
        overall_health_score: 87.5,
        health_grade: 'B',
        component_health_score: 0.92,
        model_health_score: 0.84,
        pipeline_health_score: 0.885,
        agent_health_score: 0.85,
        // HealthScoreResponseWireSchema requires arrays here (nullable)
        component_statuses: [],
        model_metrics: [],
        pipeline_statuses: [],
        agent_statuses: [],
        critical_issues: [],
        warnings: [],
        recommendations: [],
        health_summary: 'All systems nominal',
        check_latency_ms: 1240,
        timestamp: new Date().toISOString(),
        data_provenance: 'measured',
      }),
    });
  });

  // HeterogeneousTreatmentEffects → POST /api/explain/predict/batch
  await page.route('**/api/explain/predict/batch', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        batch_id: 'e2e-batch',
        total_requests: 0,
        successful: 0,
        failed: 0,
        explanations: [],
        errors: [],
        total_time_ms: 1,
      }),
    });
  });
}

test.describe('AI Agent Insights Page', () => {
  test.beforeEach(async ({ page }) => {
    // Seed auth BEFORE registering route handlers; ProtectedRoute would
    // otherwise redirect us to /login the moment the SPA boots when
    // Supabase is configured. When it is *not* configured `useAuth`
    // bypasses the gate (see `frontend/src/hooks/use-auth.ts:117-121`),
    // so this seed is defensive — it covers both CI shapes.
    await seedAuthSession(page);
    await mockInsightsEndpoints(page);

    await page.goto('/ai-insights');
    // `networkidle` is unreliable on CopilotKit-mounted pages (long-poll
    // streams); wait for the page-level heading instead. The page lazy-
    // loads `AIAgentInsights-*.js`; under flaky chunk fetches (most often
    // a static-file-server hiccup on the served `dist/` artifact in CI)
    // the SPA-level ErrorBoundary catches and renders "Something went
    // wrong" with a "Try Again" button. We give it up to two chances
    // before failing so the spec isn't pinned to a transient asset fetch.
    const heading = page.getByRole('heading', { name: /AI Agent Insights/i });
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        await expect(heading).toBeVisible({ timeout: 6000 });
        return;
      } catch {
        const tryAgain = page.getByRole('button', { name: /Try Again/i });
        if (await tryAgain.isVisible().catch(() => false)) {
          await tryAgain.click();
          continue;
        }
        // No retry button — give the chunk fetcher one more swing.
        await page.reload();
      }
    }
    // Last attempt — surfaces the proper assertion error if it still fails.
    await expect(heading).toBeVisible();
  });

  test.describe('Page Load', () => {
    test('should load successfully', async ({ page }) => {
      await expect(page).toHaveURL(/ai-insights/);
    });

    test('should display page title', async ({ page }) => {
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should display page description', async ({ page }) => {
      await expect(page.getByText(/GPT-powered executive summaries/i)).toBeVisible();
    });

    test('should show active agents badge', async ({ page }) => {
      await expect(page.getByText(/Agents Active/i)).toBeVisible();
    });
  });

  test.describe('Executive AI Brief', () => {
    test('should display Executive AI Brief section', async ({ page }) => {
      await expect(page.getByText(/Executive AI Brief/i)).toBeVisible();
    });
  });

  test.describe('Priority Actions', () => {
    test('should display Priority Actions section', async ({ page }) => {
      await expect(page.getByText(/Priority Actions/i)).toBeVisible();
    });
  });

  test.describe('Predictive Alerts', () => {
    test('should display Predictive Alerts section', async ({ page }) => {
      await expect(page.getByText('Predictive Alerts', { exact: true })).toBeVisible();
    });
  });

  test.describe('Active Causal Chains', () => {
    test('should display Active Causal Chains section', async ({ page }) => {
      await expect(page.getByText(/Active Causal Chains/i)).toBeVisible();
    });

    test('should have zoom controls', async ({ page }) => {
      // Scope to the ActiveCausalChains Card. The component (see
      // `frontend/src/components/insights/ActiveCausalChains.tsx`) renders
      // four `<Button size="icon" className="h-8 w-8">` zoom controls inside
      // the same Card whose `CardTitle` reads "Active Causal Chains". We
      // locate the title, climb to the enclosing `<div>` that holds *both*
      // the title and its sibling button row (the CardHeader's inner
      // flex/justify-between container is the deepest common ancestor),
      // then assert the icon-size button class is present and visible.
      const title = page.getByText('Active Causal Chains', { exact: true }).first();
      const headerRow = title.locator(
        'xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " justify-between ")][1]',
      );
      const zoomButton = headerRow.locator('button.h-8.w-8').first();
      await expect(zoomButton).toBeVisible();
    });
  });

  test.describe('Experiment Recommendations', () => {
    test('should display Experiment Recommendations section', async ({ page }) => {
      await expect(page.getByText(/Experiment Recommendations/i)).toBeVisible();
    });
  });

  test.describe('Heterogeneous Treatment Effects', () => {
    test('should display HTE section', async ({ page }) => {
      await expect(
        page.getByText('Heterogeneous Treatment Effects', { exact: true }),
      ).toBeVisible();
    });

    test('should show CATE info', async ({ page }) => {
      await expect(page.getByText('CATE Analysis:', { exact: true })).toBeVisible();
    });
  });

  test.describe('System Health Score', () => {
    test('should display System Health Score section', async ({ page }) => {
      // CardTitle renders as div; use first() since the string appears in
      // both the header and a `last check` aria-label internally.
      await expect(page.getByText('System Health Score').first()).toBeVisible();
    });

    test('should display health metrics', async ({ page }) => {
      // The widget renders REAL dimension rows from the stubbed
      // /api/health-score/full response (SAMPLE_METRICS is gone) —
      // assert the rows and one stubbed score so a regression back to
      // fabricated values cannot pass.
      await expect(page.getByText('Pipeline Health', { exact: true })).toBeVisible();
      await expect(page.getByText('Model Health', { exact: true })).toBeVisible();
      await expect(page.getByText('89%', { exact: true })).toBeVisible();
    });
  });

  // ---------------------------------------------------------------------------
  // Falsifiability anchor — the assertions above intentionally exercise the
  // SAMPLE_* fallback paths so the rest of the spec stays stable on
  // empty-API CI runs. That makes them weak against a regression where the
  // wave PRs' `useAlerts` wiring is reverted: SAMPLE_ALERTS would still
  // render and the smoke tests would still pass.
  //
  // This block re-registers the `/api/monitoring/alerts` route with a real,
  // non-empty payload whose title is unique (does NOT appear in
  // SAMPLE_ALERTS). If `useAlerts` no longer drives the DOM, the title
  // below will not render and this assertion fails — closing the
  // falsifiability gap surfaced by codex gate-on-diff.
  // ---------------------------------------------------------------------------
  test.describe('Live wiring (falsifiability)', () => {
    const LIVE_TITLE = 'E2E-LIVE-ALERT-9f3a2b'; // not present in SAMPLE_ALERTS

    test.beforeEach(async ({ page }) => {
      await page.route('**/api/monitoring/alerts**', async (route: Route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            total_count: 1,
            active_count: 1,
            alerts: [
              {
                id: 'live-alert-1',
                model_version: 'propensity_v2.1.0',
                alert_type: 'drift',
                severity: 'critical',
                title: LIVE_TITLE,
                description: 'Synthetic alert proving useAlerts -> DOM is wired.',
                status: 'active',
                triggered_at: new Date().toISOString(),
              },
            ],
          }),
        });
      });
      await page.reload();
      await expect(
        page.getByRole('heading', { name: /AI Agent Insights/i }),
      ).toBeVisible({ timeout: 10000 });
    });

    test('renders alert title from live useAlerts hook', async ({ page }) => {
      // If PR #312's wiring is reverted (PredictiveAlerts stops consuming
      // `useAlerts().data.alerts`), the page falls back to SAMPLE_ALERTS
      // and this assertion fails.
      await expect(page.getByText(LIVE_TITLE)).toBeVisible({ timeout: 10000 });
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });
  });
});
