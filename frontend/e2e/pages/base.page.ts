import { Page, Locator, expect } from '@playwright/test'

/**
 * Base Page Object Model class for E2E tests.
 * Provides common navigation and interaction methods.
 */
export abstract class BasePage {
  readonly page: Page
  abstract readonly url: string
  abstract readonly pageTitle: string | RegExp

  constructor(page: Page) {
    this.page = page
  }

  async goto(): Promise<void> {
    await this.page.goto(this.url)
    // Wait for the main content to be visible and network to settle
    await this.page.waitForLoadState('domcontentloaded')
    await this.mainContent.waitFor({ state: 'visible', timeout: 10000 }).catch(() => {})
    // Give React time to hydrate and render dynamic content
    await this.page.waitForTimeout(300)
  }

  async waitForPageLoad(): Promise<void> {
    await this.page.waitForLoadState('networkidle')
  }

  async verifyTitle(): Promise<void> {
    await expect(this.page).toHaveTitle(this.pageTitle)
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.waitForPageLoad()
      return true
    } catch {
      return false
    }
  }

  // Common UI element getters
  getByTestId(testId: string): Locator {
    return this.page.getByTestId(testId)
  }

  getByRole(role: Parameters<Page['getByRole']>[0], options?: Parameters<Page['getByRole']>[1]): Locator {
    return this.page.getByRole(role, options)
  }

  getByText(text: string | RegExp): Locator {
    return this.page.getByText(text)
  }

  getByLabel(label: string | RegExp): Locator {
    return this.page.getByLabel(label)
  }

  // Common navigation elements (sidebar, header)
  get sidebar(): Locator {
    return this.page.locator('[data-testid="sidebar"], nav[class*="sidebar"], aside')
  }

  get header(): Locator {
    return this.page.locator('header, [data-testid="header"]')
  }

  get mainContent(): Locator {
    // Pages use div containers with various class patterns, not semantic <main> elements
    // Look for common container patterns: container class, p-6, space-y-6, or flex layouts
    return this.page.locator('main, [data-testid="main-content"], [role="main"], .container, div.p-6, div.space-y-6').first()
  }

  // Wait utilities
  async waitForElement(selector: string, timeout = 10000): Promise<void> {
    await this.page.waitForSelector(selector, { timeout })
  }

  async waitForText(text: string | RegExp, timeout = 10000): Promise<void> {
    await this.page.getByText(text).waitFor({ timeout })
  }

  async waitForResponse(urlPattern: string | RegExp, timeout = 30000): Promise<void> {
    await this.page.waitForResponse(urlPattern, { timeout })
  }

  // Screenshot utility
  async screenshot(name: string): Promise<void> {
    await this.page.screenshot({ path: `e2e/screenshots/${name}.png`, fullPage: true })
  }

  // Common interactions
  async clickNavLink(linkText: string): Promise<void> {
    await this.sidebar.getByText(linkText).click()
  }

  async selectBrand(brand: string): Promise<void> {
    const brandSelector = this.page.locator('[data-testid="brand-selector"], select[name*="brand"]')
    if (await brandSelector.isVisible()) {
      await brandSelector.selectOption({ label: brand })
    }
  }

  async selectDateRange(start: string, end: string): Promise<void> {
    const startInput = this.page.locator('[data-testid="date-start"], input[name*="start"]')
    const endInput = this.page.locator('[data-testid="date-end"], input[name*="end"]')
    if (await startInput.isVisible()) {
      await startInput.fill(start)
      await endInput.fill(end)
    }
  }
}
