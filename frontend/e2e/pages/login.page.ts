import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'

/**
 * Page Object Model for the Login page (`/login`).
 *
 * Public (un-gated) auth page. The login flow is honest about its states:
 *  - unauthenticated -> the email/password form renders
 *  - invalid email -> client-side Zod validation error
 *  - failed sign-in -> the Supabase error is mapped + surfaced (labeled error)
 *  - already authenticated -> redirect to the dashboard ("/")
 *
 * No robust-heading nav helper here: /login is not lazy-gated and the chunk is
 * tiny, so a plain goto + form-field wait is reliable.
 */
export class LoginPage extends BasePage {
  readonly url = '/login'
  readonly pageTitle = /E2I|Causal Analytics|Sign in/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await this.page.goto(this.url)
    await this.page.waitForLoadState('domcontentloaded')
    await this.emailInput.waitFor({ state: 'visible', timeout: 10000 }).catch(() => {})
  }

  get brandHeading(): Locator {
    return this.page.getByRole('heading', { name: /E2I Analytics/i }).first()
  }

  get signInTitle(): Locator {
    return this.page.getByText('Sign in', { exact: true }).first()
  }

  get emailInput(): Locator {
    return this.page.locator('#email')
  }

  get passwordInput(): Locator {
    return this.page.locator('#password')
  }

  // Scope to the form's submit button. The page chrome also renders a "Sign in"
  // button (in the banner landmark), so match the type="submit" control inside
  // the login card to avoid a strict-mode collision.
  get submitButton(): Locator {
    return this.page.locator('button[type="submit"]').filter({ hasText: /Sign in/i })
  }

  get signupLink(): Locator {
    return this.page.getByRole('link', { name: /Sign up/i })
  }

  get forgotPasswordLink(): Locator {
    return this.page.getByRole('link', { name: /Forgot password/i })
  }

  // Auth-failure error block (mapped from the Supabase error).
  authError(messageRe: RegExp): Locator {
    return this.page.getByText(messageRe).first()
  }

  async fillAndSubmit(email: string, password: string): Promise<void> {
    await this.emailInput.fill(email)
    await this.passwordInput.fill(password)
    await this.submitButton.click()
  }
}
