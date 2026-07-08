/**
 * LIVE rendered-content verification (NOT mocked) of PR #1171 on the DEPLOYED
 * site: the /digital-twin intervention dropdown must render all 8 identified
 * interventions (identification gate + revision-2 multi-channel DGP substrate)
 * and the Strategic Interpretation card must be present.
 *
 * Run AFTER deploy + the `backfill_segment_engagement.py --execute` data-op:
 *
 *   BASE_URL=https://eznomics.site E2I_LOGIN_EMAIL=... E2I_LOGIN_PASSWORD=... \
 *     npx playwright test --config playwright.noserver.config.ts \
 *     e2e/live-digital-twin-8-interventions.spec.ts --project=chromium --reporter=line
 *
 * Intentionally NOT under e2e/specs (which auto-applies api-mocks); raw
 * @playwright/test so the page talks to the REAL backend.
 */
import { test, expect } from "@playwright/test";

const BASE = process.env.BASE_URL || "https://eznomics.site";
const EMAIL = process.env.E2I_LOGIN_EMAIL || "";
const PASSWORD = process.env.E2I_LOGIN_PASSWORD || "";

const EXPECTED_LABELS = [
  "Email Campaign",
  "Increased Call Frequency",
  "Speaker Program Invitation",
  "Sample Distribution",
  "Peer Influence Activation",
  "Digital Engagement",
  "Patient Support Program",
  "Rep Training Quality",
];

test("live /digital-twin renders all 8 identified interventions + insight card", async ({
  page,
}) => {
  test.skip(
    !PASSWORD || !EMAIL,
    "E2I_LOGIN_EMAIL / E2I_LOGIN_PASSWORD not set",
  );

  await page.goto(`${BASE}/login`);
  await page.locator("#email").fill(EMAIL);
  await page.locator("#password").fill(PASSWORD);
  await page
    .locator('button[type="submit"]')
    .filter({ hasText: /Sign in/i })
    .click();
  await page.waitForURL((u) => !u.pathname.includes("/login"), {
    timeout: 30000,
  });

  await page.goto(`${BASE}/digital-twin`);

  // Native <select> directly after the "Intervention Type" label; enabled only
  // once /intervention-types resolves with at least one identified intervention.
  const select = page.locator('label:has-text("Intervention Type") + select');
  await expect(select).toBeEnabled({ timeout: 30000 });
  const options = select.locator("option");
  await expect(options).toHaveCount(8, { timeout: 30000 });
  const labels = await options.allTextContents();
  for (const label of EXPECTED_LABELS) {
    expect(labels).toContain(label);
  }

  // Strategic Interpretation card mounted (server-derived insight, PR #1110 pattern).
  await expect(page.getByText(/twin simulation program/i).first()).toBeVisible({
    timeout: 15000,
  });
});
