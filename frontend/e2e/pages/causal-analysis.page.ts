import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the unified Causal Analysis page (`/causal-analysis`).
 *
 * ONE agent-led page (the former /causal-discovery is now a redirect here). The
 * LANDING is the validated-effects leaderboard: the analyst clicks "Discover
 * causal effects" and the causal_impact agent validates + ranks each candidate
 * question. A secondary "Pose your own question" panel keeps the manual
 * treatment/outcome path. Honest states this POM exposes:
 *  - header + "Agent-driven" badge
 *  - healthy banner: "Causal Engine Healthy" / degraded: "Service Issue"
 *  - the leaderboard "Discover causal effects" run control + grain/brand facets
 *  - empty: EmptyState "No discovery run yet" before a run
 *  - the "Pose your own question" manual panel trigger
 */
export class CausalAnalysisPage extends BasePage {
  readonly url = '/causal-analysis'
  readonly pageTitle = /Causal Analysis|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Causal Analysis/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/ranks them by confidence and impact/i).first()
  }

  get agentDrivenBadge(): Locator {
    return this.page.getByText('Agent-driven', { exact: false }).first()
  }

  get healthyBanner(): Locator {
    return this.page.getByText('Causal Engine Healthy', { exact: true }).first()
  }

  get serviceIssueBanner(): Locator {
    return this.page.getByText('Service Issue', { exact: true }).first()
  }

  get librariesCard(): Locator {
    return this.page.getByText('Libraries', { exact: true }).first()
  }

  get estimatorsCard(): Locator {
    return this.page.getByText('Estimators', { exact: true }).first()
  }

  // The leaderboard run control (landing).
  get discoverButton(): Locator {
    return this.page.getByRole('button', { name: /Discover causal effects/i }).first()
  }

  get grainSelect(): Locator {
    return this.page.getByLabel('Grain')
  }

  get brandSelect(): Locator {
    return this.page.getByLabel('Brand')
  }

  // Honest empty state on the leaderboard before a run.
  get emptyState(): Locator {
    return this.page.getByText('No discovery run yet', { exact: true }).first()
  }

  // The secondary manual path trigger.
  get poseYourOwnQuestion(): Locator {
    return this.page.getByRole('button', { name: /Pose your own question/i }).first()
  }
}
