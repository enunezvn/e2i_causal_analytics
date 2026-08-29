import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the "How E2I Works" page (route /documentation).
 */
export class DocumentationPage extends BasePage {
  readonly url = ROUTES.DOCUMENTATION
  readonly pageTitle = /Documentation|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await this.page.goto(this.url)
    await this.page.waitForLoadState('domcontentloaded')
    await this.pageHeader.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {})
    await this.page.waitForTimeout(300)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /^How E2I Works$/i }).first()
  }

  get sectionNav(): Locator {
    return this.page.getByRole('navigation', { name: /on this page/i })
  }

  get refuteStage(): Locator {
    return this.page.getByRole('button', { name: /^Refute/i })
  }

  get capabilityIndex(): Locator {
    return this.page.getByRole('region', { name: /where to go for each question/i })
  }

  get causalImpactNavLink(): Locator {
    return this.sectionNav.getByRole('button', { name: /^Causal Impact$/i })
  }

  get variableTypes(): Locator {
    return this.page.getByRole('region', { name: /four types of causal variables/i })
  }

  get causalDag(): Locator {
    return this.page.getByRole('figure', { name: /Multi-path Revenue Impact/i })
  }

  dagPathButton(name: RegExp): Locator {
    return this.causalDag.getByRole('button', { name })
  }

  get dagSelectedEdges(): Locator {
    return this.causalDag.locator('[data-edge][data-selected="true"]')
  }

  get qualityGateNavLink(): Locator {
    return this.sectionNav.getByRole('button', { name: /^Quality Gate$/i })
  }

  get refutationGate(): Locator {
    return this.page.getByRole('region', { name: /five refutation tests/i })
  }

  refutationOutcomeButton(name: RegExp): Locator {
    return this.refutationGate.getByRole('button', { name })
  }

  get activeGateBand(): Locator {
    return this.refutationGate.locator('[data-gate-active="true"]')
  }

  /** Purpose section — the cohort / channel explainer. */
  get purposeRegion(): Locator {
    return this.page.getByRole('region', { name: /^purpose/i })
  }

  get cohortItems(): Locator {
    return this.purposeRegion.locator('[data-cohort]')
  }

  get channelItems(): Locator {
    return this.purposeRegion.locator('[data-channel]')
  }

  /** Sidebar links in render order (Home first). */
  get sidebarLinks(): Locator {
    return this.page.getByRole('complementary', { name: /main navigation/i }).getByRole('link')
  }
}
