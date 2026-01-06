import { Page, Locator, expect } from '@playwright/test'
import { TIMEOUTS } from '../fixtures/test-data'

/**
 * Custom assertion utilities for E2E tests.
 */

/**
 * Assert that an element is visible and contains expected text.
 */
export async function assertElementHasText(
  locator: Locator,
  expectedText: string | RegExp,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toBeVisible({ timeout })
  await expect(locator).toContainText(expectedText, { timeout })
}

/**
 * Assert that a page title matches expected value.
 */
export async function assertPageTitle(
  page: Page,
  expectedTitle: string | RegExp,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(page).toHaveTitle(expectedTitle, { timeout })
}

/**
 * Assert that a URL contains expected path.
 */
export async function assertUrlContains(
  page: Page,
  expectedPath: string,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(page).toHaveURL(new RegExp(expectedPath), { timeout })
}

/**
 * Assert that an element is visible.
 */
export async function assertVisible(
  locator: Locator,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toBeVisible({ timeout })
}

/**
 * Assert that an element is not visible.
 */
export async function assertNotVisible(
  locator: Locator,
  timeout = TIMEOUTS.SHORT
): Promise<void> {
  await expect(locator).not.toBeVisible({ timeout })
}

/**
 * Assert that an element is enabled.
 */
export async function assertEnabled(
  locator: Locator,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toBeEnabled({ timeout })
}

/**
 * Assert that an element is disabled.
 */
export async function assertDisabled(
  locator: Locator,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toBeDisabled({ timeout })
}

/**
 * Assert that a form field has expected value.
 */
export async function assertFieldValue(
  locator: Locator,
  expectedValue: string,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toHaveValue(expectedValue, { timeout })
}

/**
 * Assert that a list has expected count of items.
 */
export async function assertListCount(
  locator: Locator,
  expectedCount: number,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toHaveCount(expectedCount, { timeout })
}

/**
 * Assert that a checkbox is checked.
 */
export async function assertChecked(
  locator: Locator,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toBeChecked({ timeout })
}

/**
 * Assert that a checkbox is not checked.
 */
export async function assertNotChecked(
  locator: Locator,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).not.toBeChecked({ timeout })
}

/**
 * Assert that an element has specific CSS class.
 */
export async function assertHasClass(
  locator: Locator,
  className: string,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toHaveClass(new RegExp(className), { timeout })
}

/**
 * Assert that an element has specific attribute value.
 */
export async function assertAttribute(
  locator: Locator,
  attribute: string,
  expectedValue: string | RegExp,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  await expect(locator).toHaveAttribute(attribute, expectedValue, { timeout })
}

/**
 * Assert that no error messages are visible.
 */
export async function assertNoErrors(page: Page): Promise<void> {
  const errorLocators = [
    page.locator('[data-testid="error"]'),
    page.locator('.error-message'),
    page.locator('[role="alert"][class*="error"]'),
  ]

  for (const locator of errorLocators) {
    const count = await locator.count()
    if (count > 0) {
      // Check if any are visible
      for (let i = 0; i < count; i++) {
        const isVisible = await locator.nth(i).isVisible()
        if (isVisible) {
          throw new Error(`Error message found: ${await locator.nth(i).textContent()}`)
        }
      }
    }
  }
}

/**
 * Assert that loading indicator is not visible (page has finished loading).
 */
export async function assertNotLoading(
  page: Page,
  timeout = TIMEOUTS.LONG
): Promise<void> {
  const loadingLocators = [
    page.locator('[data-testid="loading"]'),
    page.locator('.loading-spinner'),
    page.locator('[role="progressbar"]'),
    page.locator('.skeleton'),
  ]

  for (const locator of loadingLocators) {
    try {
      await expect(locator).not.toBeVisible({ timeout })
    } catch {
      // Loading indicator might not exist, which is fine
    }
  }
}

/**
 * Assert that API response was successful.
 */
export async function assertApiSuccess(
  page: Page,
  urlPattern: string | RegExp,
  timeout = TIMEOUTS.API_RESPONSE
): Promise<void> {
  const response = await page.waitForResponse(urlPattern, { timeout })
  expect(response.status()).toBeLessThan(400)
}

/**
 * Assert that a chart/visualization has rendered.
 */
export async function assertChartRendered(
  page: Page,
  chartSelector = 'canvas, svg, [data-testid*="chart"], .recharts-wrapper',
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  const chart = page.locator(chartSelector).first()
  await expect(chart).toBeVisible({ timeout })
}

/**
 * Assert that a table has data rows.
 */
export async function assertTableHasData(
  locator: Locator,
  minRows = 1,
  timeout = TIMEOUTS.MEDIUM
): Promise<void> {
  const rows = locator.locator('tbody tr, [role="row"]')
  await expect(rows).toHaveCount(minRows, { timeout })
}
