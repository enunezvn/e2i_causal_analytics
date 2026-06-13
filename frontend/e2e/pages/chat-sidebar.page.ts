import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the CopilotKit chat sidebar (E2IChatSidebar).
 *
 * The chat lives in the Layout (present on every protected page); we exercise
 * it from the Home dashboard. It only renders when CopilotKit is enabled
 * (VITE_COPILOT_ENABLED=true in the dev `.env`). Honest states this POM
 * exposes:
 *  - closed -> a floating toggle button (MessageSquare icon)
 *  - opened -> the "E2I Assistant" header + the CopilotChat textarea with the
 *    real placeholder
 */
export class ChatSidebarPage extends BasePage {
  readonly url = '/'
  readonly pageTitle = /E2I|Causal Analytics|Dashboard/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /E2I Executive Dashboard/i)
  }

  get dashboardHeading(): Locator {
    return this.page.getByRole('heading', { name: /E2I Executive Dashboard/i }).first()
  }

  // Floating toggle button (round, h-14 w-14) shown when the chat is closed.
  get toggleButton(): Locator {
    return this.page.locator('button.rounded-full.h-14.w-14').first()
  }

  get assistantHeader(): Locator {
    return this.page.getByRole('heading', { name: /E2I Assistant/i }).first()
  }

  // CopilotChat renders a <textarea> with the labels.placeholder we configured.
  get chatInput(): Locator {
    return this.page.getByPlaceholder(/Ask about KPIs, agents, or insights/i)
  }

  async openChat(): Promise<void> {
    await this.toggleButton.click()
  }
}
