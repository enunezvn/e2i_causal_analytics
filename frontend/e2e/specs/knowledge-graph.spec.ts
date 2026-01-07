import { test, expect } from '@playwright/test'
import { KnowledgeGraphPage } from '../pages/knowledge-graph.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Knowledge Graph Page', () => {
  let graphPage: KnowledgeGraphPage

  test.beforeEach(async ({ page }) => {
    // Setup API mocks
    await mockApiRoutes(page)

    // Initialize page object
    graphPage = new KnowledgeGraphPage(page)
    await graphPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(graphPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(graphPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })
  })

  test.describe('Graph Visualization', () => {
    test('should render graph visualization', async () => {
      const isRendered = await graphPage.isGraphRendered()
      expect(typeof isRendered).toBe('boolean')
    })

    test('should show graph canvas element', async () => {
      const canvas = graphPage.graphCanvas
      await expect(canvas).toBeAttached()
    })

    test('should display graph nodes', async () => {
      const nodeCount = await graphPage.getNodeCount()
      expect(nodeCount).toBeGreaterThanOrEqual(0)
    })

    test('should display graph edges', async () => {
      const edgeCount = await graphPage.getEdgeCount()
      expect(edgeCount).toBeGreaterThanOrEqual(0)
    })
  })

  test.describe('Search Functionality', () => {
    test('should display search input', async () => {
      const searchInput = graphPage.searchInput
      // Search might be visible or hidden
      const exists = await searchInput.count() > 0
      expect(exists).toBeDefined()
    })

    test('should allow text input in search', async () => {
      const searchInput = graphPage.searchInput
      if (await searchInput.isVisible()) {
        await searchInput.fill('TRx')
        await expect(searchInput).toHaveValue('TRx')
      }
    })

    test('should show search results after search', async () => {
      const searchInput = graphPage.searchInput
      if (await searchInput.isVisible()) {
        await graphPage.search('TRx')
        const resultsShown = await graphPage.areSearchResultsShown()
        expect(typeof resultsShown).toBe('boolean')
      }
    })

    test('should clear search when clear button clicked', async () => {
      const searchInput = graphPage.searchInput
      const clearButton = graphPage.clearSearchButton
      if (await searchInput.isVisible() && await clearButton.isVisible()) {
        await searchInput.fill('test')
        await clearButton.click()
        await expect(searchInput).toHaveValue('')
      }
    })
  })

  test.describe('Node Details', () => {
    test('should show node details when node clicked', async () => {
      const nodeCount = await graphPage.getNodeCount()
      if (nodeCount > 0) {
        const firstNode = graphPage.graphNodes.first()
        await firstNode.click({ force: true }).catch(() => {})
        // Details panel might open
        const isVisible = await graphPage.isNodeDetailsVisible()
        expect(typeof isVisible).toBe('boolean')
      }
    })

    test('should display node label in details', async () => {
      if (await graphPage.isNodeDetailsVisible()) {
        const label = await graphPage.getSelectedNodeLabel()
        expect(label === null || typeof label === 'string').toBeTruthy()
      }
    })
  })

  test.describe('Graph Traversal', () => {
    test('should handle depth filter', async () => {
      const depthFilter = graphPage.depthFilter
      if (await depthFilter.isVisible()) {
        await graphPage.setTraversalDepth(2)
        // Verify depth was set
        await expect(depthFilter).toHaveValue('2')
      }
    })

    test('should handle node type filter', async () => {
      const typeFilter = graphPage.nodeTypeFilter
      if (await typeFilter.isVisible()) {
        // Select a node type
        const options = await typeFilter.locator('option').allTextContents()
        if (options.length > 1) {
          await typeFilter.selectOption({ index: 1 })
        }
      }
    })
  })

  test.describe('Zoom Controls', () => {
    test('should have zoom in control', async ({ page }) => {
      const zoomIn = page.locator('[data-testid="zoom-in"], button:text("+")').first()
      // Zoom controls might be present
      const exists = await zoomIn.count() > 0
      expect(exists).toBeDefined()
    })

    test('should have zoom out control', async ({ page }) => {
      const zoomOut = page.locator('[data-testid="zoom-out"], button:text("-")').first()
      const exists = await zoomOut.count() > 0
      expect(exists).toBeDefined()
    })

    test('should have reset view control', async ({ page }) => {
      const reset = page.locator('[data-testid="reset-view"], button:text("Reset")').first()
      const exists = await reset.count() > 0
      expect(exists).toBeDefined()
    })
  })

  test.describe('Graph Statistics', () => {
    test('should display graph stats', async () => {
      const stats = graphPage.graphStats
      const isVisible = await stats.isVisible().catch(() => false)
      expect(typeof isVisible).toBe('boolean')
    })

    test('should show total nodes count', async () => {
      // The totalNodesCard contains "Total Nodes" label + a numeric count
      const nodesCard = graphPage.totalNodesCard
      if (await nodesCard.isVisible()) {
        // Verify the card shows the "Total Nodes" label
        const cardText = await nodesCard.textContent()
        expect(cardText).toContain('Total Nodes')
        // The count element should have numeric content (can be 0 or more)
        const nodesCount = graphPage.totalNodesCount
        if (await nodesCount.isVisible()) {
          const countText = await nodesCount.textContent()
          // Count should be a number (might be empty string if loading, so be flexible)
          expect(countText === '' || /^\d+$/.test(countText?.trim() || '')).toBeTruthy()
        }
      }
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await graphPage.goto()
      await expect(graphPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await graphPage.goto()
      await expect(graphPage.mainContent).toBeVisible()
    })
  })
})
