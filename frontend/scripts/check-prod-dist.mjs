#!/usr/bin/env node
/**
 * Production-bundle hygiene guard.
 *
 * Asserts that a production `dist/` contains no dev-only / mock artifacts.
 * This mirrors the guard RUN baked into docker/frontend/Dockerfile (added in
 * PR #908 / commit bd657c0f) so the same invariant is enforced in CI from the
 * `npm run build` job — not only when the production Docker image is built.
 *
 * Checks (all must hold for a clean prod bundle):
 *   1. dist/mockServiceWorker.js is absent (MSW dev service worker).
 *   2. No `initMSW` / `setupWorker` MSW bootstrap code in dist/assets/*.js.
 *   3. No baked dev Supabase URL (localhost:8443) in dist/assets/*.js.
 *
 * Exit 0 = clean, exit 1 = a dev-flavored artifact leaked into prod.
 *
 * Usage: node scripts/check-prod-dist.mjs [distDir]   (default: ./dist)
 *
 * NOTE: run this ONLY against a production build (NODE_ENV=production or the
 * default `vite build` mode). A deliberate dev-flavored build
 * (NODE_ENV=development vite build) legitimately ships MSW and will fail this
 * guard by design.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

const distDir = resolve(process.argv[2] ?? 'dist')
const assetsDir = join(distDir, 'assets')

const failures = []

if (!existsSync(distDir)) {
  console.error(`[check-prod-dist] dist dir not found: ${distDir}`)
  console.error('[check-prod-dist] run `npm run build` first.')
  process.exit(1)
}

// 1. mockServiceWorker.js must not be emitted into prod dist.
if (existsSync(join(distDir, 'mockServiceWorker.js'))) {
  failures.push('dist/mockServiceWorker.js is present (dev-only MSW worker leaked into prod)')
}

// 2 + 3. Scan emitted JS for MSW bootstrap code and a baked dev Supabase URL.
const mswPattern = /initMSW|setupWorker/
const devSupabasePattern = /localhost:8443/

// Walk dist/assets recursively (Vite can nest emitted chunks, e.g. workers),
// matching the Dockerfile guard's `grep -r` over dist/assets.
function collectJs(dir) {
  if (!existsSync(dir) || !statSync(dir).isDirectory()) return []
  const out = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...collectJs(full))
    else if (entry.isFile() && entry.name.endsWith('.js')) out.push(full)
  }
  return out
}

const jsFiles = collectJs(assetsDir)

for (const file of jsFiles) {
  const contents = readFileSync(file, 'utf8')
  if (mswPattern.test(contents)) {
    failures.push(`MSW bootstrap code (initMSW/setupWorker) found in ${file}`)
  }
  if (devSupabasePattern.test(contents)) {
    failures.push(`dev Supabase URL (localhost:8443) baked into ${file}`)
  }
}

if (failures.length > 0) {
  console.error('[check-prod-dist] FAIL: dev-flavored artifacts in production bundle:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}

console.log(`[check-prod-dist] OK: production bundle is clean (${jsFiles.length} JS chunks scanned).`)
