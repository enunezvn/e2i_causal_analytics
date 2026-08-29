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

    test('causal impact section: variable-type key and a DAG path spotlight', async () => {
      await docPage.causalImpactNavLink.click()
      await expect(docPage.variableTypes).toBeVisible()
      for (const term of ['Treatment', 'Mediator', 'Outcome', 'Confounder']) {
        await expect(docPage.variableTypes.getByText(term, { exact: true })).toBeVisible()
      }
      await expect(docPage.causalDag).toBeVisible()
      await expect(docPage.causalDag.getByText(/illustrative example/i)).toBeVisible()
      await expect(docPage.dagPathButton(/^All paths$/i)).toHaveAttribute('aria-pressed', 'true')
      await expect(docPage.dagSelectedEdges).toHaveCount(0)

      await docPage.dagPathButton(/Backdoor confounders/i).click()
      await expect(docPage.dagPathButton(/Backdoor confounders/i)).toHaveAttribute(
        'aria-pressed',
        'true'
      )
      await expect(docPage.dagSelectedEdges).toHaveCount(4)
      await expect(docPage.causalDag.getByText(/back-door paths the estimator must close/i)).toBeVisible()
    })

    test('quality gate section: five refutation tests and the fail-state toggle', async () => {
      await docPage.qualityGateNavLink.click()
      await expect(docPage.refutationGate).toBeVisible()
      for (const name of [
        'Placebo Treatment',
        'Random Common Cause',
        'Data Subset',
        'Bootstrap',
        'Sensitivity (E-value)',
      ]) {
        await expect(docPage.refutationGate.getByRole('heading', { name })).toBeVisible()
      }
      await expect(docPage.refutationGate.getByText(/illustrative example/i)).toBeVisible()
      await expect(docPage.refutationOutcomeButton(/estimate survives/i)).toHaveAttribute(
        'aria-pressed',
        'true'
      )
      await expect(docPage.activeGateBand).toHaveAttribute('data-gate', 'proceed')

      await docPage.refutationOutcomeButton(/estimate fails/i).click()
      await expect(docPage.refutationOutcomeButton(/estimate fails/i)).toHaveAttribute(
        'aria-pressed',
        'true'
      )
      await expect(docPage.activeGateBand).toHaveAttribute('data-gate', 'block')
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
