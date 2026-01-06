import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for System Health page.
 * Displays service status grid, model health cards, and active alerts.
 */
export class SystemHealthPage extends BasePage {
  readonly url = ROUTES.SYSTEM_HEALTH
  readonly pageTitle = /System Health|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /System Health/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Service status|model health|alerts/i).first()
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i })
  }

  get lastUpdatedText(): Locator {
    return this.page.getByText(/Last updated/i).first()
  }

  // Overview Stats Cards
  get servicesCard(): Locator {
    return this.page.getByText('Services').first()
  }

  get modelHealthCard(): Locator {
    return this.page.getByText('Model Health').first()
  }

  get activeAlertsCard(): Locator {
    return this.page.getByText('Active Alerts').first()
  }

  get recentRunsCard(): Locator {
    return this.page.getByText('Recent Runs').first()
  }

  // Service Status Section
  get serviceStatusSection(): Locator {
    return this.page.getByText('Service Status').first()
  }

  get apiGatewayService(): Locator {
    return this.page.getByText('API Gateway').first()
  }

  get postgresService(): Locator {
    return this.page.getByText('PostgreSQL').first()
  }

  get redisService(): Locator {
    return this.page.getByText('Redis Cache').first()
  }

  get falkordbService(): Locator {
    return this.page.getByText('FalkorDB').first()
  }

  get bentomlService(): Locator {
    return this.page.getByText('BentoML').first()
  }

  // Model Health Section
  get modelHealthSection(): Locator {
    return this.page.getByText('Model Health').locator('..').first()
  }

  get modelCards(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: /Health|Drift|Trend/i })
  }

  get propensityModel(): Locator {
    return this.page.getByText('Propensity Model').first()
  }

  get churnModel(): Locator {
    return this.page.getByText('Churn Prediction').first()
  }

  get conversionModel(): Locator {
    return this.page.getByText('Conversion Model').first()
  }

  // Active Alerts Section
  get activeAlertsSection(): Locator {
    return this.page.getByText('Active Alerts').locator('..').first()
  }

  get alertItems(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /critical|warning|info/i })
  }

  // Status Badges
  get healthyBadges(): Locator {
    return this.page.locator('.rounded-full').filter({ hasText: /healthy/i })
  }

  get warningBadges(): Locator {
    return this.page.locator('.rounded-full').filter({ hasText: /warning/i })
  }

  get criticalBadges(): Locator {
    return this.page.locator('.rounded-full').filter({ hasText: /critical/i })
  }

  // Actions
  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  // Verification methods
  async verifyOverviewStatsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Services').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const stats = ['Model Health', 'Active Alerts', 'Recent Runs']
      for (const stat of stats) {
        if (await this.page.getByText(stat).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyServiceStatusDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Service Status').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyModelHealthDisplayed(): Promise<boolean> {
    try {
      // Look for model names
      await this.page.getByText('Propensity Model').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const models = ['Churn Prediction', 'Conversion Model']
      for (const model of models) {
        if (await this.page.getByText(model).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyAlertsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Active Alerts').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }
}
