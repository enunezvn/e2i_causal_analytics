import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

const frontendRoot = path.resolve(__dirname, '../..')

describe('e2e noserver-config wire-up contract', () => {
  it('frontend/package.json declares test:e2e:noserver script that targets playwright.noserver.config.ts', () => {
    const pkg = JSON.parse(
      fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf-8'),
    )
    expect(pkg.scripts).toBeDefined()
    expect(pkg.scripts['test:e2e:noserver']).toBeDefined()
    expect(pkg.scripts['test:e2e:noserver']).toMatch(/playwright\.noserver\.config\.ts/)
    expect(pkg.scripts['test:e2e:noserver']).toMatch(/^playwright test\b/)
  })

  it('frontend/playwright.noserver.config.ts exists, imports the base config, and overrides webServer to undefined', () => {
    const cfgPath = path.join(frontendRoot, 'playwright.noserver.config.ts')
    expect(fs.existsSync(cfgPath)).toBe(true)
    const content = fs.readFileSync(cfgPath, 'utf-8')
    expect(content).toMatch(/from\s+['"]\.\/playwright\.config['"]/)
    expect(content).toMatch(/webServer\s*:\s*undefined/)
    expect(content).toMatch(/defineConfig/)
  })

  it('frontend/e2e/README.md exists and documents both default and noserver modes', () => {
    const readmePath = path.join(frontendRoot, 'e2e', 'README.md')
    expect(fs.existsSync(readmePath)).toBe(true)
    const content = fs.readFileSync(readmePath, 'utf-8')
    expect(content).toMatch(/test:e2e(?!:)/)
    expect(content).toMatch(/test:e2e:noserver/)
    expect(content).toMatch(/playwright\.noserver\.config\.ts/)
  })

  it('package.json declares no orphan test:e2e:local script (canonical name is :noserver)', () => {
    const pkg = JSON.parse(
      fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf-8'),
    )
    expect(pkg.scripts['test:e2e:local']).toBeUndefined()
  })
})
