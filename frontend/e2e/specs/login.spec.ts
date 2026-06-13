/**
 * Login Flow E2E Tests (#19 coverage gap)
 * =======================================
 *
 * The auth login flow had NO e2e coverage. These specs drive the real
 * `/login` form and assert HONEST states by stubbing the real Supabase/GoTrue
 * auth endpoints (route interception on `**\/auth/v1/**`):
 *   - unauthenticated -> the email/password form renders
 *   - invalid email -> client-side Zod validation error (honest)
 *   - failed sign-in -> the GoTrue 400 is mapped to a labeled error message
 *     ("Invalid email or password") — NOT a silent success
 *   - already authenticated -> redirect away from /login to the dashboard
 *
 * The dev `.env` configures VITE_SUPABASE_ANON_KEY so isSupabaseConfigured()
 * is true and the login form is live (when unconfigured the app fails CLOSED
 * with an AuthConfigurationError on protected routes instead).
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { LoginPage } from '../pages/login.page'
import { seedAuthSession, stubCopilotRuntime } from '../fixtures/page-harness'

/**
 * Stub the GoTrue password-grant endpoint to FAIL with invalid_credentials.
 * supabase-js posts to /auth/v1/token?grant_type=password; a 400 with an
 * error code surfaces through signInWithPassword().error -> mapAuthError.
 */
async function stubAuthFailure(page: Page): Promise<void> {
  await page.route('**/auth/v1/token**', async (route: Route) => {
    await route.fulfill({
      status: 400,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 'invalid_credentials',
        error_code: 'invalid_credentials',
        msg: 'Invalid login credentials',
        error: 'invalid_grant',
        error_description: 'Invalid login credentials',
      }),
    })
  })
}

test.describe('Login Flow', () => {
  let loginPage: LoginPage

  test.describe('Form rendering', () => {
    test.beforeEach(async ({ page }) => {
      loginPage = new LoginPage(page)
      await loginPage.goto()
    })

    test('loads at /login', async ({ page }) => {
      await expect(page).toHaveURL(/login/)
    })

    test('renders the brand heading and sign-in card', async () => {
      await expect(loginPage.brandHeading).toBeVisible()
      await expect(loginPage.signInTitle).toBeVisible()
    })

    test('renders the email and password fields and submit button', async () => {
      await expect(loginPage.emailInput).toBeVisible()
      await expect(loginPage.passwordInput).toBeVisible()
      await expect(loginPage.submitButton).toBeVisible()
    })

    test('renders the signup and forgot-password links', async () => {
      await expect(loginPage.signupLink).toBeVisible()
      await expect(loginPage.forgotPasswordLink).toBeVisible()
    })
  })

  test.describe('Honest validation + error states', () => {
    test.beforeEach(async ({ page }) => {
      loginPage = new LoginPage(page)
      await loginPage.goto()
    })

    test('blocks submission of an invalid email (no backend auth call)', async ({ page }) => {
      // The email field is type="email"; a malformed value is natively invalid
      // and the client refuses to submit it to the backend. This is the honest
      // contract — an invalid email NEVER reaches the auth endpoint and the
      // user stays on /login. (A valid-format-but-wrong credential is covered
      // by the "sign-in fails" test below.) We track the GoTrue token endpoint
      // to prove no auth attempt was made.
      let tokenCalled = false
      await page.route('**/auth/v1/token**', async (route: Route) => {
        tokenCalled = true
        await route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({ code: 'invalid_credentials', msg: 'x' }),
        })
      })

      await loginPage.emailInput.fill('not-an-email')
      await loginPage.passwordInput.fill('somepassword')
      await loginPage.submitButton.click()

      // Native form validation marks the field invalid and blocks submit.
      // The validity check is synchronous after the click — no hard wait
      // needed; expect.poll retries the evaluate until it settles.
      await expect
        .poll(() =>
          loginPage.emailInput.evaluate((el: HTMLInputElement) => !el.validity.valid),
        )
        .toBe(true)
      // No auth attempt reached the backend, and we never left /login.
      expect(tokenCalled).toBe(false)
      await expect(page).toHaveURL(/login/)
    })

    test('surfaces a labeled error when sign-in fails', async ({ page }) => {
      await stubAuthFailure(page)
      await loginPage.fillAndSubmit('user@example.com', 'wrong-password')
      // The GoTrue 400 maps to a user-facing labeled error — we stay on /login
      // and do NOT silently appear signed-in.
      await expect(page.getByText(/Invalid email or password/i)).toBeVisible({
        timeout: 10000,
      })
      await expect(page).toHaveURL(/login/)
    })
  })

  test.describe('Authenticated redirect', () => {
    test('redirects an already-authenticated user away from /login', async ({ page }) => {
      // Seed a valid session before the SPA boots (+ the CopilotKit runtime
      // stub so the dashboard subtree it redirects to does not crash). The
      // Login page's effect redirects an authenticated user to the dashboard
      // ("/"). We do NOT register a broad `**\/api/**` stub here — it would
      // shadow the CopilotKit runtime stub and break the auth/init flow; the
      // URL assertion fires on the redirect itself, before the dashboard's
      // data calls matter.
      await seedAuthSession(page)
      await stubCopilotRuntime(page)

      await page.goto('/login')
      // We must leave /login (redirected to the dashboard root).
      await expect(page).not.toHaveURL(/login/, { timeout: 10000 })
      await expect(page).toHaveURL(/\/$|\/\?/, { timeout: 10000 })
    })
  })
})
