import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Documentation page ("Understanding E2I").
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
    return this.page.getByRole('heading', { name: /Understanding E2I/i }).first()
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
}
