import { test, expect } from '@playwright/test'
import { MemoryArchitecturePage } from '../pages/memory-architecture.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Memory Architecture Page', () => {
  let memoryPage: MemoryArchitecturePage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    memoryPage = new MemoryArchitecturePage(page)
    await memoryPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(memoryPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(memoryPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(memoryPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(memoryPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Architecture Diagram', () => {
    test('should display architecture diagram', async () => {
      const hasDiagram = await memoryPage.verifyArchitectureDiagramDisplayed()
      expect(hasDiagram).toBeTruthy()
    })
  })

  test.describe('Memory Cards', () => {
    test('should display memory cards', async () => {
      const hasCards = await memoryPage.verifyMemoryCardsDisplayed()
      expect(hasCards).toBeTruthy()
    })

    test('should show Working Memory card', async () => {
      await expect(memoryPage.workingMemoryCard).toBeVisible()
    })

    test('should show Episodic Memory card', async () => {
      await expect(memoryPage.episodicMemoryCard).toBeVisible()
    })

    test('should show Semantic Memory card', async () => {
      await expect(memoryPage.semanticMemoryCard).toBeVisible()
    })

    test('should show Procedural Memory card', async () => {
      await expect(memoryPage.proceduralMemoryCard).toBeVisible()
    })
  })

  test.describe('Backend Technology Labels', () => {
    test('should display backend labels', async () => {
      const hasBackends = await memoryPage.verifyBackendsDisplayed()
      expect(hasBackends).toBeTruthy()
    })

    test('should show Redis backend', async () => {
      await expect(memoryPage.redisBackend).toBeVisible()
    })

    test('should show Supabase backend', async () => {
      await expect(memoryPage.supabaseBackend).toBeVisible()
    })

    test('should show FalkorDB backend', async () => {
      await expect(memoryPage.falkordbBackend).toBeVisible()
    })
  })

  test.describe('Recent Memories', () => {
    test('should display recent memories section', async () => {
      const hasRecent = await memoryPage.verifyRecentMemoriesDisplayed()
      expect(hasRecent).toBeTruthy()
    })
  })

  test.describe('About Section', () => {
    test('should display about section', async () => {
      const hasAbout = await memoryPage.verifyAboutSectionDisplayed()
      expect(hasAbout).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(memoryPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await memoryPage.clickRefresh()
      await memoryPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await memoryPage.goto()
      await expect(memoryPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await memoryPage.goto()
      await expect(memoryPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await memoryPage.goto()
      await expect(memoryPage.mainContent).toBeVisible()
    })
  })
})
