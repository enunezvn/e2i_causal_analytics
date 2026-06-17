import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Causal Discovery page (`/causal-discovery`).
 *
 * The page is a VALIDATED-EFFECTS LEADERBOARD: the analyst clicks "Discover
 * causal effects" and the causal_impact agent validates each candidate question
 * (guided DAG discovery + data-driven estimator + refutation gate), then ranks
 * the effects by confidence and impact. Honest states this POM exposes:
 *  - header + "Agent-driven" badge
 *  - the "Discover causal effects" run control
 *  - empty: EmptyState "No discovery run yet" before a run
 *
 * The previous manual workbench (library routing / parallel pipeline / KG
 * buttons) and the one-click question form were removed.
 */
export class CausalDiscoveryPage extends BasePage {
  readonly url = ROUTES.CAUSAL_DISCOVERY
  readonly pageTitle = /Causal Discovery|E2I/i

  constructor(page: Page) {
    super(page)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: 'Causal Discovery', level: 1 })
  }

  get pageDescription(): Locator {
    return this.page.getByText(/ranks them by confidence and impact/i)
  }

  get agentDrivenBadge(): Locator {
    return this.page.getByText('Agent-driven', { exact: false }).first()
  }

  // The single run control: kicks off the validated-effects discovery job.
  get discoverButton(): Locator {
    return this.page.getByRole('button', { name: /Discover causal effects/i }).first()
  }

  // Honest empty state before any run.
  get emptyState(): Locator {
    return this.page.getByText('No discovery run yet', { exact: true }).first()
  }
}
