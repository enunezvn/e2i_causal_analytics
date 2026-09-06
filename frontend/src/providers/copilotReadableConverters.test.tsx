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
// The global setup (src/test/setup.ts) stubs @copilotkit/react-core for every
// test file; this contract test needs the REAL hook. Its module graph imports
// katex CSS through @copilotkitnext/react, which Node's loader rejects, so that
// package is replaced wholesale with a stub providing only what the hook uses.
vi.unmock('@copilotkit/react-core');
vi.mock('@copilotkitnext/react', () => ({
  useCopilotKit: () => ({ copilotkit: fakeCore }),
}));

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import * as React from 'react';
import { render } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useCopilotReadable } from '@copilotkit/react-core';
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

type ContextItem = { description: string; value: string };
const store: { context: Record<string, ContextItem>; n: number } = { context: {}, n: 0 };
const fakeCore = {
  get context() {
    return store.context;
  },
  addContext(item: ContextItem): string {
    const id = String(++store.n);
    store.context[id] = item;
    return id;
  },
  removeContext(id: string): void {
    delete store.context[id];
  },
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

const CALL_SITE_RE = /\(convert\s*(?:!=\s*null\s*\?\s*convert\s*:|\?\?)\s*JSON\.stringify\)\(value\)/;

describe('CopilotKit react-core convert contract (SDK fence)', () => {
  it('runtime calls convert with the value as its only argument', () => {
    const bundle = sdkFile('dist/index.js');
    expect(bundle).toMatch(CALL_SITE_RE);
  });

  it('typings still declare the two-argument signature the runtime ignores', () => {
    const dts = sdkFile('dist/hooks/use-copilot-readable.d.ts');
    expect(dts).toMatch(/convert\?:\s*\(description:\s*string,\s*value:\s*any\)\s*=>\s*string/);
  });

  it('the ESM chunk that Vite bundles has the same single-argument call', () => {
    if (!pkgRoot) throw new Error('@copilotkit/react-core not found under node_modules');
    const dist = path.join(pkgRoot, 'dist');
    const chunks = readdirSync(dist).filter((f) => /^chunk-.*\.mjs$/.test(f));
    expect(chunks.length).toBeGreaterThan(0);
    const hits = chunks.filter((f) => CALL_SITE_RE.test(readFileSync(path.join(dist, f), 'utf8')));
    expect(hits.length).toBeGreaterThan(0);
  });
});

describe('real useCopilotReadable with passThroughText (contract)', () => {
  beforeEach(() => {
    store.context = {};
    store.n = 0;
  });

  const Readable: React.FC<{ summary: string }> = ({ summary }) => {
    useCopilotReadable({
      description: 'D',
      value: summary,
      convert: passThroughText,
      available: summary ? 'enabled' : 'disabled',
    });
    return null;
  };

  it('hands the runtime the prose itself, and removes it when cleared', () => {
    const { rerender, unmount } = render(<Readable summary={SUMMARY} />);
    expect(Object.values(store.context)).toEqual([{ description: 'D', value: SUMMARY }]);
    rerender(<Readable summary="" />);
    expect(Object.values(store.context)).toEqual([]);
    unmount();
  });
});
