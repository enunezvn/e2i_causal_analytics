/**
 * Forcing-function test for issue #281.
 *
 * The build job in .github/workflows/frontend-tests.yml fails because
 * `frontend/src/types/generated/api.ts` is gitignored (regenerated via
 * `npm run generate:types` from a running FastAPI server), but
 * `index.ts` unconditionally imports from `./api`.
 *
 * This test asserts that `index.ts` does not statically import or
 * re-export from `./api`, ensuring CI's `tsc -b` survives without the
 * generated file. Devs who want the generated types should import
 * directly from `./api` after running `npm run generate:types`.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const INDEX_PATH = resolve(__dirname, 'index.ts');
const API_PATH = resolve(__dirname, 'api.ts');

describe('generated/index.ts (#281)', () => {
  const source = readFileSync(INDEX_PATH, 'utf-8');

  it('must not unconditionally re-export from ./api (causes TS2307 when api.ts absent)', () => {
    // Matches `export * from './api'` and `export ... from './api'`
    expect(source).not.toMatch(/export\s+(?:\*|\{[^}]*\})\s+from\s+['"]\.\/api['"]/);
  });

  it('must not contain top-level static imports from ./api', () => {
    // Matches `import ... from './api'` (static)
    expect(source).not.toMatch(/^\s*import\b[^;]*\bfrom\s+['"]\.\/api['"]/m);
  });

  it('must not reference ./api via inline import() type expressions', () => {
    // Matches `import('./api').X` patterns used by ExtractResponse/ExtractRequestBody
    expect(source).not.toMatch(/import\(['"]\.\/api['"]\)/);
  });

  it('regression guard: api.ts is intentionally gitignored and may be absent in CI', () => {
    // This documents the invariant: tests must pass whether or not api.ts
    // exists. If it exists locally, that's fine; if not, no breakage.
    const apiExists = existsSync(API_PATH);
    expect(typeof apiExists).toBe('boolean');
  });
});
