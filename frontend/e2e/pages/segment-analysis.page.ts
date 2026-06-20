import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Segment Analysis page (`/segment-analysis`).
 *
 * Heterogeneous treatment-effect dashboard (Tier-2 Heterogeneous Optimizer).
 * The page (F-002 rewrite) renders ONLY real API results — no fabricated
 * sample fallback. Honest states this POM exposes:
 *  - health badge: "Agents Ready" / "Agents Unavailable" / "Checking..."
 *  - empty: EmptyState "No segment analysis available" (before any run)
 *  - error: QueryErrorState "Failed to load segment health"
 *  - loaded: KPI summary cards once a real analysis result is returned
 */
export class SegmentAnalysisPage extends BasePage {
  readonly url = '/segment-analysis'
  readonly pageTitle = /Segment Analysis|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Segment Analysis/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Segment Analysis/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Agent estimates CATE across all clinical segments/i).first()
  }

  // Health badge reflects the stubbed /api/segments/health response.
  get agentsReadyBadge(): Locator {
    return this.page.getByText('Agents Ready', { exact: true }).first()
  }

  get agentsUnavailableBadge(): Locator {
    return this.page.getByText('Agents Unavailable', { exact: true }).first()
  }

  get analysesTodayBadge(): Locator {
    return this.page.getByText(/analyses today/i).first()
  }

  // Honest error state for a failed /api/segments/health.
  get healthErrorState(): Locator {
    return this.page.getByText(/Failed to load segment health/i).first()
  }

  // Honest empty state before any analysis is run.
  get emptyState(): Locator {
    return this.page.getByText('No segment analysis available', { exact: true }).first()
  }

  get configurationCard(): Locator {
    return this.page.getByText('Analysis Configuration', { exact: true }).first()
  }

  get runAnalysisButton(): Locator {
    return this.page.getByRole('button', { name: /Run Analysis/i }).first()
  }

  // KPI summary — rendered ONLY when a real analysis result is loaded.
  get overallAteCard(): Locator {
    return this.page.getByText('Overall ATE', { exact: true }).first()
  }

  get resultTabs(): Locator {
    return this.page.getByRole('tab', { name: /CATE by Segment/i }).first()
  }

  async clickRunAnalysis(): Promise<void> {
    await this.runAnalysisButton.click()
  }
}
