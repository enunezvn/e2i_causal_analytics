import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

const frontendRoot = path.resolve(__dirname, '../..')
const specsDir = path.join(frontendRoot, 'e2e', 'specs')

type Manifest = { budget: unknown; specs: unknown }

function loadManifest(): Manifest {
  const raw = fs.readFileSync(
    path.join(frontendRoot, 'e2e', '.quarantine.json'),
    'utf-8',
  )
  return JSON.parse(raw) as Manifest
}

describe('quarantine ratchet contract', () => {
  it('e2e/.quarantine.json exists, parses, and has {budget: number, specs: string[]}', () => {
    const m = loadManifest()
    expect(typeof m.budget).toBe('number')
    expect(Array.isArray(m.specs)).toBe(true)
    expect((m.specs as unknown[]).every((s) => typeof s === 'string')).toBe(true)
  })

  it('budget equals specs.length (manifest internally consistent)', () => {
    const m = loadManifest() as { budget: number; specs: string[] }
    expect(m.budget).toBe(m.specs.length)
  })

  it('every spec listed in the manifest exists in e2e/specs/', () => {
    const m = loadManifest() as { budget: number; specs: string[] }
    const missing = m.specs.filter(
      (s) => !fs.existsSync(path.join(specsDir, s)),
    )
    expect(missing).toEqual([])
  })

  it('_smoke.spec.ts is never quarantined (floor signal)', () => {
    const m = loadManifest() as { budget: number; specs: string[] }
    expect(m.specs).not.toContain('_smoke.spec.ts')
  })

  it('_smoke.spec.ts file exists in e2e/specs/ (CI shards need a non-empty collection)', () => {
    expect(fs.existsSync(path.join(specsDir, '_smoke.spec.ts'))).toBe(true)
  })

  it('playwright.config.ts integrates the manifest via testIgnore + E2E_QUARANTINE_OFF bypass', () => {
    const cfg = fs.readFileSync(
      path.join(frontendRoot, 'playwright.config.ts'),
      'utf-8',
    )
    expect(cfg).toMatch(/['"]\.\/e2e\/\.quarantine\.json['"]|e2e\/\.quarantine\.json/)
    expect(cfg).toMatch(/testIgnore\s*:/)
    expect(cfg).toMatch(/E2E_QUARANTINE_OFF/)
  })

  it('budget is non-negative and does not exceed the total number of e2e specs', () => {
    const m = loadManifest() as { budget: number; specs: string[] }
    expect(m.budget).toBeGreaterThanOrEqual(0)
    const allSpecs = fs
      .readdirSync(specsDir)
      .filter((f) => f.endsWith('.spec.ts'))
    expect(m.budget).toBeLessThanOrEqual(allSpecs.length)
  })

  it('manifest specs are unique (no duplicate entries)', () => {
    const m = loadManifest() as { budget: number; specs: string[] }
    expect(new Set(m.specs).size).toBe(m.specs.length)
  })
})
