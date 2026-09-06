/**
 * Contract pin for passThroughText against CopilotKit react-core's runtime.
 *
 * react-core 1.51.2 types `convert(description, value)` but its bundled code
 * calls `convert(value)` with ONE argument, so a positional converter sent
 * the agent the string "undefined" (found in review, 2026-09-06). The real
 * hook cannot be imported under vitest here: its module graph pulls katex CSS
 * through Node's ESM loader, which is why src/test/setup.ts stubs the whole
 * package. This file therefore pins the contract two ways: the converter's
 * behaviour under both call shapes, and the SDK bundle's call site as text.
 * If a CopilotKit bump fails the fence, re-verify the runtime arity and
 * update passThroughText and this test together.
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import { passThroughText } from './copilotReadableConverters';

const SUMMARY = 'Home dashboard. Brand filter: Kisqali; region: All US.';

// vitest runs from frontend/ (locally and in CI); tolerate a repo-root cwd too.
// frontend/node_modules is a symlink in worktrees; readFileSync follows it.
const PKG_SEGMENTS = ['node_modules', '@copilotkit', 'react-core'];
const pkgRoot = [process.cwd(), path.join(process.cwd(), 'frontend')]
  .map((base) => path.join(base, ...PKG_SEGMENTS))
  .find((candidate) => existsSync(candidate));

const sdkFile = (rel: string): string => {
  if (!pkgRoot) throw new Error('@copilotkit/react-core not found under node_modules');
  return readFileSync(path.join(pkgRoot, rel), 'utf8');
};

describe('passThroughText', () => {
  it('returns the value under the runtime 1-arg call and the typed 2-arg call', () => {
    expect(passThroughText(SUMMARY)).toBe(SUMMARY);
    expect(passThroughText('description', SUMMARY)).toBe(SUMMARY);
  });

  it('does not JSON-quote or escape the prose', () => {
    const multi = 'Line one.\nLine "two" with quotes.';
    expect(passThroughText(multi)).toBe(multi);
    expect(passThroughText(multi)).not.toBe(JSON.stringify(multi));
  });
});

describe('CopilotKit react-core convert contract (SDK fence)', () => {
  it('runtime calls convert with the value as its only argument', () => {
    const bundle = sdkFile('dist/index.js');
    expect(bundle).toMatch(/\(convert\s*(?:!=\s*null\s*\?\s*convert\s*:|\?\?)\s*JSON\.stringify\)\(value\)/);
  });

  it('typings still declare the two-argument signature the runtime ignores', () => {
    const dts = sdkFile('dist/hooks/use-copilot-readable.d.ts');
    expect(dts).toMatch(/convert\?:\s*\(description:\s*string,\s*value:\s*any\)\s*=>\s*string/);
  });
});
