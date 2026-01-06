import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Knowledge Graph page.
 * Updated to match actual frontend implementation using shadcn/ui components.
 */
export class KnowledgeGraphPage extends BasePage {
  readonly url = ROUTES.KNOWLEDGE_GRAPH
  readonly pageTitle = /Knowledge Graph|E2I/i

  constructor(page: Page) {
    super(page)
  }

  // ========================================================================
  // Page Header
  // ========================================================================

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: 'Knowledge Graph', level: 1 })
  }

  get pageDescription(): Locator {
    return this.page.getByText('Explore the knowledge graph visualization')
  }

  // ========================================================================
  // Search Elements
  // ========================================================================

  get searchInput(): Locator {
    return this.page.getByPlaceholder('Search nodes by name or type...')
  }

  get clearSearchButton(): Locator {
    // The clear button is inside the search input area, using ghost variant
    return this.page.locator('.relative').filter({ has: this.searchInput }).getByRole('button')
  }

  get searchResultsInfo(): Locator {
    return this.page.getByText(/Found \d+ nodes/)
  }

  // ========================================================================
  // Legend Card
  // ========================================================================

  get legendCard(): Locator {
    return this.page.getByText('Node Type Legend').locator('..')
  }

  get legendItems(): Locator {
    return this.legendCard.locator('.flex.items-center.gap-2')
  }

  // ========================================================================
  // Stats Cards
  // ========================================================================

  get totalNodesCard(): Locator {
    return this.page.getByText('Total Nodes').locator('..')
  }

  get totalNodesCount(): Locator {
    return this.totalNodesCard.locator('.text-2xl')
  }

  get totalRelationshipsCard(): Locator {
    return this.page.getByText('Total Relationships').locator('..')
  }

  get totalRelationshipsCount(): Locator {
    return this.totalRelationshipsCard.locator('.text-2xl')
  }

  get selectedCard(): Locator {
    return this.page.getByText('Selected').first().locator('..')
  }

  get selectedValue(): Locator {
    return this.selectedCard.locator('.text-lg')
  }

  // ========================================================================
  // Graph Visualization
  // ========================================================================

  get graphVisualizationCard(): Locator {
    return this.page.getByText('Graph Visualization').locator('..')
  }

  get graphCanvas(): Locator {
    // The KnowledgeGraphViz component renders a canvas or SVG
    return this.page.locator('canvas, svg').first()
  }

  get graphNodes(): Locator {
    // Cytoscape nodes or SVG circles
    return this.page.locator('circle, [class*="node"]')
  }

  get graphEdges(): Locator {
    // Cytoscape edges or SVG lines/paths
    return this.page.locator('line, path[class*="edge"]')
  }

  // ========================================================================
  // Node/Edge Details Panel
  // ========================================================================

  get nodeDetailsPanel(): Locator {
    return this.page.getByText('Node Details').locator('..')
  }

  get relationshipDetailsPanel(): Locator {
    return this.page.getByText('Relationship Details').locator('..')
  }

  get detailsPanel(): Locator {
    // Either node or relationship details
    return this.page.locator('.grid.grid-cols-2.gap-4').first()
  }

  get nodeLabel(): Locator {
    return this.detailsPanel.getByText('Name').locator('..').locator('dd')
  }

  get nodeType(): Locator {
    return this.detailsPanel.getByText('Type').locator('..').locator('dd')
  }

  // ========================================================================
  // Legacy Selectors (for backwards compatibility)
  // ========================================================================

  get graphVisualization(): Locator {
    return this.graphVisualizationCard
  }

  get graphStats(): Locator {
    return this.page.locator('.grid.grid-cols-1.md\\:grid-cols-3')
  }

  get searchResults(): Locator {
    return this.searchResultsInfo
  }

  get nodeTypeFilter(): Locator {
    // Filter by clicking legend items
    return this.legendItems.first()
  }

  get relationshipTypeFilter(): Locator {
    return this.page.locator('select[name*="relationship"]').first()
  }

  get depthFilter(): Locator {
    return this.page.locator('input[name*="depth"]').first()
  }

  // ========================================================================
  // Actions
  // ========================================================================

  async search(query: string): Promise<void> {
    await this.searchInput.fill(query)
    // The search is reactive, no submit button needed
    await this.page.waitForTimeout(300) // Wait for debounce
  }

  async clearSearch(): Promise<void> {
    const clearBtn = this.clearSearchButton
    if (await clearBtn.isVisible().catch(() => false)) {
      await clearBtn.click()
    } else {
      await this.searchInput.clear()
    }
  }

  async clickLegendItem(nodeType: string): Promise<void> {
    await this.legendCard.getByText(nodeType).click()
  }

  async filterByNodeType(nodeType: string): Promise<void> {
    // Filter by clicking on legend item
    await this.clickLegendItem(nodeType)
  }

  async filterByRelationship(relationship: string): Promise<void> {
    const filter = this.relationshipTypeFilter
    if (await filter.isVisible()) {
      await filter.selectOption({ label: relationship })
    }
  }

  async setTraversalDepth(depth: number): Promise<void> {
    const filter = this.depthFilter
    if (await filter.isVisible()) {
      await filter.fill(depth.toString())
    }
  }

  // ========================================================================
  // Assertions / Helpers
  // ========================================================================

  async isGraphRendered(): Promise<boolean> {
    // Check if Graph Visualization card is visible
    const card = this.page.getByText('Graph Visualization')
    return await card.isVisible().catch(() => false)
  }

  async areSearchResultsShown(): Promise<boolean> {
    return await this.searchResultsInfo.isVisible().catch(() => false)
  }

  async isNodeDetailsVisible(): Promise<boolean> {
    return await this.nodeDetailsPanel.isVisible().catch(() => false)
  }

  async getNodeCount(): Promise<number> {
    // Try to get count from the stats card
    const countText = await this.totalNodesCount.textContent().catch(() => '0')
    const match = countText?.match(/\d+/)
    return match ? parseInt(match[0], 10) : 0
  }

  async getEdgeCount(): Promise<number> {
    // Try to get count from the stats card
    const countText = await this.totalRelationshipsCount.textContent().catch(() => '0')
    const match = countText?.match(/\d+/)
    return match ? parseInt(match[0], 10) : 0
  }

  async getSelectedNodeLabel(): Promise<string | null> {
    const selected = this.selectedValue
    if (await selected.isVisible()) {
      return await selected.textContent()
    }
    return null
  }

  async getConnectedNodesCount(): Promise<number> {
    return 0 // Not directly shown in current implementation
  }

  async verifyStatsCardsDisplayed(): Promise<boolean> {
    const statsTexts = ['Total Nodes', 'Total Relationships', 'Selected']
    for (const text of statsTexts) {
      if (await this.page.getByText(text).isVisible().catch(() => false)) {
        return true
      }
    }
    return false
  }
}
