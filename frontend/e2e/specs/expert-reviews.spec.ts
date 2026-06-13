/**
 * Expert Reviews Page E2E Tests (#19 coverage gap)
 * ================================================
 *
 * `/expert-reviews` was a routed data page with NO e2e coverage. It is the
 * human-in-the-loop review queue for REVIEW-band causal DAGs. The page
 * documents its honest states (loading / error / empty — no hardcoded
 * fallback). These specs stub the REAL endpoints
 * (`GET /api/expert-reviews/pending`, `GET /api/expert-reviews/summary`) and
 * assert HONEST states:
 *   - empty queue -> EmptyState "No pending reviews"
 *   - pending endpoint 500 -> WarningBanner "Failed to load pending reviews"
 *   - real queue -> the review's treatment_variable renders in the table
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { ExpertReviewsPage } from '../pages/expert-reviews.page'
import { harnessBase } from '../fixtures/page-harness'

const SUMMARY = { pending: 0, approved: 0, rejected: 0, expired: 0, expiring_soon: 0 }

async function stubExpertReviewEndpoints(
  page: Page,
  opts: { pendingStatus?: number; reviews?: unknown[] } = {},
): Promise<void> {
  await page.route('**/api/expert-reviews/summary**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(SUMMARY),
    })
  })

  await page.route('**/api/expert-reviews/pending**', async (route: Route) => {
    if (opts.pendingStatus && opts.pendingStatus >= 400) {
      await route.fulfill({
        status: opts.pendingStatus,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'expert-review service unavailable' }),
      })
      return
    }
    const reviews = opts.reviews ?? []
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ reviews, total: reviews.length }),
    })
  })
}

test.describe('Expert Reviews Page', () => {
  let reviewsPage: ExpertReviewsPage

  test.describe('Empty (honest) state', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubExpertReviewEndpoints(page, { reviews: [] })
      reviewsPage = new ExpertReviewsPage(page)
      await reviewsPage.goto()
    })

    test('loads at /expert-reviews', async ({ page }) => {
      await expect(page).toHaveURL(/expert-reviews/)
    })

    test('displays the page header', async () => {
      await expect(reviewsPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(reviewsPage.pageDescription).toBeVisible()
    })

    test('shows honest empty state when the queue is empty', async () => {
      await expect(reviewsPage.emptyState).toBeVisible()
    })
  })

  test.describe('Error state', () => {
    test('shows a labeled error when the pending endpoint fails', async ({ page }) => {
      await harnessBase(page)
      await stubExpertReviewEndpoints(page, { pendingStatus: 500 })
      reviewsPage = new ExpertReviewsPage(page)
      await reviewsPage.goto()

      await expect(reviewsPage.errorState).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('Loaded state (falsifiability)', () => {
    test('renders a pending-review row from the live endpoint', async ({ page }) => {
      await harnessBase(page)
      await stubExpertReviewEndpoints(page, {
        reviews: [
          {
            review_id: 'rev-e2e-1',
            review_type: 'causal_dag',
            dag_version_hash: 'abc123def456',
            brand: 'kisqali',
            treatment_variable: 'E2ELiveTreatmentVar',
            outcome_variable: 'trx',
            analysis_context: 'e2e',
            created_at: new Date().toISOString(),
            days_pending: 3,
          },
        ],
      })
      reviewsPage = new ExpertReviewsPage(page)
      await reviewsPage.goto()

      // The unique treatment variable proves the row is driven by the real API.
      await expect(page.getByText(/E2ELiveTreatmentVar/i)).toBeVisible({ timeout: 10000 })
      await expect(reviewsPage.emptyState).toBeHidden()
    })
  })
})
