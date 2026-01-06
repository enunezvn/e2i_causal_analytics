import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Agent Orchestration page.
 * Displays the 18-agent tiered orchestration system dashboard.
 */
export class AgentOrchestrationPage extends BasePage {
  readonly url = ROUTES.AGENT_ORCHESTRATION
  readonly pageTitle = /Agent Orchestration|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Agent Orchestration/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/18-agent|tiered|orchestration/i).first()
  }

  // Action Buttons
  get pauseAllButton(): Locator {
    return this.page.getByRole('button', { name: /pause all/i })
  }

  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i })
  }

  // Stats Cards (matches AgentOrchestration.tsx)
  get totalAgentsCard(): Locator {
    return this.page.getByText('Total Agents').first()
  }

  get tasksTodayCard(): Locator {
    return this.page.getByText('Tasks Today').first()
  }

  get avgResponseTimeCard(): Locator {
    return this.page.getByText('Avg Response Time').first()
  }

  get successRateCard(): Locator {
    return this.page.getByText('Success Rate').first()
  }

  // Tier Cards (for testing tiers display)
  get tierCards(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: /Tier \d/i })
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get overviewTab(): Locator {
    return this.page.getByRole('tab', { name: /overview/i })
  }

  get activityTab(): Locator {
    return this.page.getByRole('tab', { name: /activity/i })
  }

  get tiersTab(): Locator {
    return this.page.getByRole('tab', { name: /tier/i })
  }

  get agentsTab(): Locator {
    return this.page.getByRole('tab', { name: /all agents/i })
  }

  // Tier Overview
  get tierOverview(): Locator {
    return this.page.getByText('Tier Architecture').locator('..')
  }

  // Agent Status Panel
  get agentStatusPanel(): Locator {
    return this.page.getByText('Agent Status').locator('..')
  }

  // Activity Feed
  get activityFeed(): Locator {
    return this.page.getByText('Recent Activity').locator('..')
  }

  get activityItems(): Locator {
    return this.page.locator('[class*="activity"], .rounded-lg').filter({ hasText: /completed|in_progress|failed/i })
  }

  // Actions
  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async clickPauseAll(): Promise<void> {
    await this.pauseAllButton.click()
  }

  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  // Verification methods
  async verifyStatsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Total Agents').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const stats = ['Tasks Today', 'Avg Response Time', 'Success Rate']
      for (const stat of stats) {
        if (await this.page.getByText(stat).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    try {
      await this.tabsList.waitFor({ state: 'visible', timeout: 5000 })
      return await this.tabsList.isVisible()
    } catch {
      return false
    }
  }

  async verifyTierOverviewDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Tier Architecture').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyAgentTiersDisplayed(): Promise<boolean> {
    try {
      // Wait for page content to load
      await this.page.waitForTimeout(1000)

      // Look for "Tier Architecture" section which is always visible on overview tab
      const hasTierArchitecture = await this.page.getByText('Tier Architecture').first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasTierArchitecture) return true

      // Fallback: look for tier buttons like "Tier 0 Foundation 7 agents"
      const hasTierButton = await this.page.getByRole('button', { name: /Tier \d/i }).first().isVisible({ timeout: 2000 }).catch(() => false)
      if (hasTierButton) return true

      // Fallback: look for tier text content
      const hasTierText = await this.page.getByText(/Tier \d/i).first().isVisible({ timeout: 2000 }).catch(() => false)
      if (hasTierText) return true

      // Fallback: check for tier metric cards
      const cardCount = await this.tierCards.count()
      return cardCount > 0
    } catch {
      return false
    }
  }

  async verifyPipelineVisualizationDisplayed(): Promise<boolean> {
    try {
      // The page has a TierOverview component which serves as the "pipeline" visualization
      const hasTierOverview = await this.page.getByText('Tier Architecture').first().isVisible().catch(() => false)
      if (hasTierOverview) return true
      // Fallback: look for SVG elements that might be part of visualization
      const hasGraphic = await this.page.locator('svg').first().isVisible().catch(() => false)
      return hasGraphic
    } catch {
      return false
    }
  }
}
