import { test, expect } from '@playwright/test'
import { DocumentationPage } from '../pages/documentation.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNoErrors } from '../utils/assertions'

test.describe('Documentation Page', () => {
  let docPage: DocumentationPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    docPage = new DocumentationPage(page)
    await docPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(docPage.pageHeader).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should display the section nav', async () => {
      await expect(docPage.sectionNav).toBeVisible()
    })
  })

  test.describe('Interactivity', () => {
    test('expands a pipeline stage', async ({ page }) => {
      await docPage.refuteStage.click()
      await expect(page.getByText(/Attack the estimate before believing it/i)).toBeVisible()
    })

    test('capability index links to live pages', async () => {
      await expect(docPage.capabilityIndex).toBeVisible()
      await expect(
        docPage.capabilityIndex.getByRole('link', { name: /Segment Analysis/i })
      ).toHaveAttribute('href', '/segment-analysis')
    })
  })

  test.describe('Footer entry point', () => {
    test('footer Documentation link navigates here', async ({ page }) => {
      await page.goto('/')
      await page.getByRole('contentinfo').getByRole('link', { name: /^Documentation$/i }).click()
      await expect(docPage.pageHeader).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })
  })
})
