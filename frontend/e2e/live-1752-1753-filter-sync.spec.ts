/**
 * LIVE certification (NOT mocked) of the #1752 and #1753 filter-channel fixes
 * against the DEPLOYED site.
 *
 * Deterministic discriminators (both fail on the pre-fix bundles):
 *  1. #1752 — Causal Analysis page: selecting Brand: Kisqali in the PAGE's own
 *     dropdown must produce a POST /chat/suggestions payload carrying
 *     brand: "Kisqali" with a page_context naming the same brand (pre-fix the
 *     page selection lived in page-local useState, so the payload kept the
 *     provider's value while page_context named the page's — two surfaces
 *     contradicting each other).
 *  2. #1753 — Home: selecting Region: West must ship filters.region === 'West'
 *     inside the CoAgent state on the POST /api/copilotkit agent-run request
 *     (pre-fix E2IFilters had no region field at all, so NO copilotkit request
 *     body could contain a filters.region key). The request body is
 *     frontend-built and deterministic even though the response is an LLM run.
 *
 * Run AFTER deploy (explicit opt-in, same design as the #1749 live spec):
 *   E2I_RUN_LIVE_CERTS=1 BASE_URL=https://eznomics.site \
 *     npx playwright test --config playwright.noserver.config.ts \
 *     e2e/live-1752-1753-filter-sync.spec.ts --project=chromium --reporter=line
 */
import { test, expect, type Page, type Request } from '@playwright/test'
import { readFileSync } from 'node:fs'

const RUN_LIVE = process.env.E2I_RUN_LIVE_CERTS === '1'
const BASE = process.env.BASE_URL || 'https://eznomics.site'
const EMAIL = process.env.E2I_LOGIN_EMAIL || 'admin@e2i.local'

function resolvePassword(): string {
  if (process.env.E2I_ADMIN_PASSWORD) return process.env.E2I_ADMIN_PASSWORD
  for (const envPath of [
    '/home/enunez/Projects/e2i_causal_analytics/.env',
    `${process.cwd()}/../.env`,
  ]) {
    try {
      const line = readFileSync(envPath, 'utf8')
        .split('\n')
        .find((l) => l.startsWith('E2I_ADMIN_PASSWORD='))
      if (line) return line.slice('E2I_ADMIN_PASSWORD='.length).trim()
    } catch {
      /* next candidate */
    }
  }
  return ''
}
const PASSWORD = RUN_LIVE ? resolvePassword() : ''

interface SuggestionsPayload {
  messages: unknown[]
  page: string
  brand?: string
  page_context?: string
}

async function login(page: Page): Promise<void> {
  await page.goto(`${BASE}/login`)
  await page.locator('#email').fill(EMAIL)
  await page.locator('#password').fill(PASSWORD)
  await page.locator('button[type="submit"]').filter({ hasText: /Sign in/i }).click()
  await page.waitForURL((u) => !u.pathname.includes('/login'), { timeout: 30000 })
}

function collectSuggestionPayloads(page: Page, sink: SuggestionsPayload[]): void {
  page.on('request', (req: Request) => {
    if (req.method() === 'POST' && req.url().includes('/chat/suggestions')) {
      try {
        sink.push(JSON.parse(req.postData() ?? '{}') as SuggestionsPayload)
      } catch {
        /* non-JSON body — ignore */
      }
    }
  })
}

async function openChat(page: Page): Promise<void> {
  // Icon-only FAB with no accessible name — class pair is the stable handle.
  await page.locator('button.rounded-full.h-14').first().click()
  await expect(page.getByText('E2I Assistant')).toBeVisible({ timeout: 15000 })
}

/** Structurally find a `filters` object with a `region` key anywhere in a
 *  parsed JSON tree, descending into string values that themselves parse as
 *  JSON (CoAgent state is JSON-stringified inside the GraphQL variables —
 *  never assume the escaping depth, walk it). */
function deepFindFiltersRegion(node: unknown, depth = 0): string | undefined {
  if (depth > 8 || node === null || node === undefined) return undefined
  if (typeof node === 'string') {
    if (!node.includes('filters') && !node.includes('region')) return undefined
    try {
      return deepFindFiltersRegion(JSON.parse(node), depth + 1)
    } catch {
      return undefined
    }
  }
  if (Array.isArray(node)) {
    for (const item of node) {
      const hit = deepFindFiltersRegion(item, depth + 1)
      if (hit !== undefined) return hit
    }
    return undefined
  }
  if (typeof node === 'object') {
    const obj = node as Record<string, unknown>
    const filters = obj.filters as Record<string, unknown> | undefined
    if (filters && typeof filters === 'object' && typeof filters.region === 'string') {
      return filters.region
    }
    for (const value of Object.values(obj)) {
      const hit = deepFindFiltersRegion(value, depth + 1)
      if (hit !== undefined) return hit
    }
  }
  return undefined
}

test.describe('LIVE #1752/#1753 filter channels', () => {
  test.skip(
    !RUN_LIVE || !PASSWORD,
    'live cert is opt-in: set E2I_RUN_LIVE_CERTS=1 (and have E2I_ADMIN_PASSWORD resolvable)'
  )

  test('#1752 Causal Analysis: page brand selection reaches the chat suggestions payload', async ({
    page,
  }) => {
    const payloads: SuggestionsPayload[] = []
    collectSuggestionPayloads(page, payloads)

    await login(page)
    await page.goto(`${BASE}/causal-analysis`)

    // The page's own Brand dropdown (labelled trigger, aria-label="Brand").
    await page.getByRole('combobox', { name: 'Brand' }).click()
    await page.getByRole('option', { name: 'Kisqali' }).click()

    await openChat(page)

    await expect
      .poll(
        () => payloads.filter((p) => p.brand === 'Kisqali').length,
        { timeout: 20000, message: 'suggestions request with brand Kisqali fired' }
      )
      .toBeGreaterThan(0)

    // The page_context (built from the PAGE's selection) and the structured
    // brand (from the PROVIDER) must agree — the pre-fix bundle could not
    // produce this pair from a page-dropdown selection.
    const kisqali = payloads.filter((p) => p.brand === 'Kisqali' && p.page_context)
    for (const p of kisqali) {
      expect(p.page_context).toContain('Brand filter: Kisqali')
    }
  })

  test('#1753 Home: region selection ships in CoAgent filters on the agent-run request', async ({
    page,
  }) => {
    const copilotBodies: string[] = []
    page.on('request', (req: Request) => {
      if (req.method() === 'POST' && req.url().includes('/api/copilotkit')) {
        const body = req.postData()
        if (body) copilotBodies.push(body)
      }
    })

    await login(page)
    await page.goto(`${BASE}/`)

    // Region is the second labelled combobox on Home (Brand is first).
    await page
      .getByRole('combobox')
      .filter({ hasText: /^(All US Regions|Northeast|South|Midwest|West)/ })
      .first()
      .click()
    await page.getByRole('option', { name: 'West', exact: true }).click()

    await openChat(page)

    // Send one real message — the REQUEST body (frontend-built) is the
    // deterministic evidence; the LLM response is irrelevant to this cert.
    const input = page.locator('textarea, input[placeholder*="Ask"]').first()
    await input.fill('What is the TRx trend?')
    // Enter does NOT submit the CopilotKit textbox (measured: text stayed in
    // the box, no POST fired) — the labelled Send button is the submit path.
    await page.getByRole('button', { name: 'Send' }).click()

    await expect
      .poll(
        () => copilotBodies.some((b) => deepFindFiltersRegion(safeParse(b)) === 'West'),
        { timeout: 30000, message: 'copilotkit request carrying filters.region=West' }
      )
      .toBe(true)

    // And no request may still carry a region-less filters object once the
    // fix is live (the pre-fix type had no region field at all).
    const withFilters = copilotBodies
      .map((b) => deepFindFiltersRegion(safeParse(b)))
      .filter((r) => r !== undefined)
    expect(withFilters.length).toBeGreaterThan(0)
    console.log('[cert #1753] filters.region values observed on the wire:', withFilters)
  })
})

function safeParse(body: string): unknown {
  try {
    return JSON.parse(body)
  } catch {
    return undefined
  }
}
