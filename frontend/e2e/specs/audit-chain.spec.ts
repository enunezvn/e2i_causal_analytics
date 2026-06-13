/**
 * Audit Chain Page E2E Tests (#19 coverage gap)
 * =============================================
 *
 * `/audit-chain` was a routed data page with NO e2e coverage. It is the
 * cryptographic audit-trail / workflow-verification dashboard. These specs
 * stub the REAL endpoint the page calls (`GET /api/audit/recent`, returning a
 * `RecentWorkflowResponse[]`) and assert HONEST states:
 *   - empty list -> EmptyState "No workflows found"
 *   - endpoint 500 -> EmptyState "Failed to load workflows" (labeled error)
 *   - real list -> the workflow's last_agent label renders
 *
 * We do NOT assert against fabricated workflow data — the empty/error states
 * ARE the honest contract.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { AuditChainPage } from '../pages/audit-chain.page'
import { harnessBase } from '../fixtures/page-harness'

async function stubAuditRecent(
  page: Page,
  opts: { status?: number; body?: unknown } = {},
): Promise<void> {
  await page.route('**/api/audit/recent**', async (route: Route) => {
    if (opts.status && opts.status >= 400) {
      await route.fulfill({
        status: opts.status,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'audit service unavailable' }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opts.body ?? []),
    })
  })
}

test.describe('Audit Chain Page', () => {
  let auditPage: AuditChainPage

  test.describe('Empty (honest) state', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubAuditRecent(page, { body: [] })
      auditPage = new AuditChainPage(page)
      await auditPage.goto()
    })

    test('loads at /audit-chain', async ({ page }) => {
      await expect(page).toHaveURL(/audit-chain/)
    })

    test('displays the page header', async () => {
      await expect(auditPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(auditPage.pageDescription).toBeVisible()
    })

    test('shows honest empty state when no workflows are returned', async () => {
      await expect(auditPage.emptyState).toBeVisible()
    })
  })

  test.describe('Error state', () => {
    test('shows a labeled error when the workflows endpoint fails', async ({ page }) => {
      await harnessBase(page)
      await stubAuditRecent(page, { status: 500 })
      auditPage = new AuditChainPage(page)
      await auditPage.goto()

      await expect(auditPage.errorState).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('Loaded state (falsifiability)', () => {
    test('renders a workflow row from the live audit endpoint', async ({ page }) => {
      await harnessBase(page)
      await stubAuditRecent(page, {
        body: [
          {
            workflow_id: 'e2e-wf-1',
            started_at: new Date().toISOString(),
            entry_count: 5,
            first_agent: 'orchestrator',
            last_agent: 'E2ELiveAuditAgent',
            brand: 'kisqali',
          },
        ],
      })
      auditPage = new AuditChainPage(page)
      await auditPage.goto()

      // The unique last_agent name proves the row is driven by the real API.
      await expect(page.getByText(/E2ELiveAuditAgent/i)).toBeVisible({ timeout: 10000 })
      await expect(auditPage.emptyState).toBeHidden()
    })
  })
})
