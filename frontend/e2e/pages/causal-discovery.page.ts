import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the RETIRED Causal Discovery route.
 *
 * `/causal-discovery` was unified into the agent-led `/causal-analysis` page and
 * is now a client-side redirect. This POM only asserts the redirect behavior;
 * the leaderboard + manual panel live on `CausalAnalysisPage`.
 */
export class CausalDiscoveryPage extends BasePage {
  readonly url = ROUTES.CAUSAL_DISCOVERY
  readonly pageTitle = /Causal Analysis|Causal Discovery|E2I/i

  constructor(page: Page) {
    super(page)
  }

  // After the redirect, the unified page's header renders.
  get redirectedHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }
}
