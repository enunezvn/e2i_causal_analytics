import { test, expect } from '@playwright/test'
import { KnowledgeGraphPage } from '../pages/knowledge-graph.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// Spec-local typed stubs for the Knowledge Graph endpoints.
//
// The page consumes the live API contract via TanStack Query hooks:
//   * useNodes()         -> GET /api/graph/nodes        -> ListNodesResponse
//   * useRelationships() -> GET /api/graph/relationships -> ListRelationshipsResponse
//   * useGraphStats()    -> GET /api/graph/stats        -> GraphStatsResponse
//
// The shared `mockApiRoutes` registers a single `**/api/graph/**` handler
// that returns one legacy payload for ALL three endpoints. That payload
// lacks `nodes_by_type`, so the page crashed at
// `Object.entries(stats.nodesByType)` with "Cannot convert undefined or
// null to object", and the AppErrorBoundary replaced the entire page
// (search input included) with the error fallback. That detached the
// `getByPlaceholder('Search nodes by name or type...')` input mid-action,
// which is the exact failure mode the (long-running) baseline e2e was
// hitting on `should allow text input in search` /
// `should show search results after search`.
//
// We register endpoint-specific stubs AFTER the shared mock so Playwright's
// LIFO route matcher picks ours first. The shared `**/api/graph/**` route
// remains the catch-all for other graph endpoints (e.g. /traverse, /search)
// that this spec doesn't exercise. We do NOT mutate `api-mocks.ts`: it is
// shared by 15+ specs and out of scope for this Cat-B baseline repair.
// The KG page now loads each brand's synthetic gold-standard CAUSAL graph: it
// fetches Variable nodes + brand-tagged CAUSES edges and derives the selected
// brand's subgraph client-side (`causalSubgraphForBrand`), dropping any node not
// on a brand CAUSES chain and pruning components < 3 nodes. The page defaults to
// the first brand, Kisqali, so the fixture below is a single Kisqali causal
// component of 3 Variable nodes — exactly one of which ("TRx Volume") matches
// the case-insensitive search substring "TRx", anchoring the "Found 1 nodes"
// assertion to the real filter. (The old Brand/KPI/HCP/Patient mock predated the
// per-brand causal rework and produced an empty subgraph: no CAUSES edges.)
const mockNodesResponse = {
  nodes: [
    { id: 'var-trx', type: 'Variable', name: 'TRx Volume', properties: {}, created_at: '2026-01-01T00:00:00Z' },
    { id: 'var-eng', type: 'Variable', name: 'HCP Engagement', properties: {}, created_at: '2026-01-01T00:00:00Z' },
    { id: 'var-adh', type: 'Variable', name: 'Patient Adherence', properties: {}, created_at: '2026-01-01T00:00:00Z' },
  ],
  total: 3,
  limit: 100,
  offset: 0,
  has_more: false,
  query_latency_ms: 5,
  timestamp: '2026-05-18T00:00:00Z',
}

const mockRelationshipsResponse = {
  relationships: [
    { id: 'rel-1', type: 'CAUSES', source_id: 'var-eng', target_id: 'var-trx', properties: { brand: 'Kisqali' }, confidence: 0.85 },
    { id: 'rel-2', type: 'CAUSES', source_id: 'var-adh', target_id: 'var-trx', properties: { brand: 'Kisqali' }, confidence: 0.78 },
  ],
  total: 2,
  limit: 200,
  offset: 0,
  has_more: false,
  query_latency_ms: 4,
  timestamp: '2026-05-18T00:00:00Z',
}

const mockGraphStatsResponse = {
  total_nodes: 3,
  total_relationships: 2,
  nodes_by_type: { Variable: 3 },
  relationships_by_type: { CAUSES: 2 },
  total_episodes: 0,
  total_communities: 0,
  timestamp: '2026-05-18T00:00:00Z',
}

test.describe('Knowledge Graph Page', () => {
  let graphPage: KnowledgeGraphPage

  test.beforeEach(async ({ page }) => {
    // Setup shared API mocks (broad **/api/** catch-alls).
    await mockApiRoutes(page)

    // Endpoint-specific stubs registered AFTER the broad mock so they win
    // LIFO route matching. Shapes must match `src/types/graph.ts`.
    await page.route('**/api/graph/stats', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockGraphStatsResponse),
      })
    })
    await page.route('**/api/graph/nodes**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockNodesResponse),
      })
    })
    await page.route('**/api/graph/relationships**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRelationshipsResponse),
      })
    })

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

    test('should show search results after search', async ({ page }) => {
      // Within the default brand's (Kisqali) causal subgraph, exactly one
      // Variable node's `name` matches the case-insensitive substring "TRx"
      // (the "TRx Volume" node). Once the page filters, the results banner
      // must render and report exactly 1 node found. Anchored to the page's
      // literal output so a regression that breaks the brand filter, the
      // search, or the banner trips this test instead of passing on a typeof
      // check.
      const searchInput = graphPage.searchInput
      await expect(searchInput).toBeVisible()
      await graphPage.search('TRx')
      await expect(page.getByText(/Found 1 nodes/)).toBeVisible()
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
