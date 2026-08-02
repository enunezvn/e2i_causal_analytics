/**
 * CopilotKit Markdown Table Styling
 * =================================
 *
 * Regression guard for the chat table-layout defect seen in
 * session_1785687617856_q47mlml, where a region x metric table rendered as a
 * run-on wall of text with no visible column boundaries.
 *
 * The assistant's markdown was never at fault — it emitted a well-formed GFM
 * table, and remark-gfm (enabled inside CopilotKit's <Markdown>) turned it into
 * a real <table>. The defect was that nothing styled it:
 *
 *   - CopilotKit's component map classes only a, code, h1-h6, p, pre,
 *     blockquote, ul and li, so table elements reach the DOM bare.
 *   - Its stylesheet carries no table rules at all.
 *   - @tailwindcss/typography is not installed, so there is no `prose` fallback.
 *   - Tailwind v4's Preflight then removes what the UA stylesheet would have
 *     supplied: `*, ::before, ::after { border: 0 solid }` zeroes every border
 *     width and `table { border-collapse: collapse }` drops the default
 *     border-spacing.
 *
 * These assertions are stylesheet-level on purpose. The defect is purely
 * visual, and jsdom performs no layout, so no render assertion can observe
 * column separation — the meaningful invariant is that the rules restoring the
 * grid exist and stay scoped to the chat.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const CSS = readFileSync(
  // vitest runs with the frontend package root as cwd.
  resolve(process.cwd(), 'src/index.css'),
  'utf8',
  // Comments routinely mention `table` and carry no declarations; strip them so
  // they can neither satisfy nor break a selector match.
).replace(/\/\*[\s\S]*?\*\//g, '');

/** Declaration body of the first rule whose selector matches `pattern`. */
function declarationsFor(pattern: RegExp): string | null {
  for (const rule of CSS.split('}')) {
    const [selector, body] = rule.split('{');
    if (selector && body && pattern.test(selector)) return body;
  }
  return null;
}

describe('CopilotKit markdown table styles', () => {
  it('gives header and body cells a visible border', () => {
    const cell = declarationsFor(/\.copilotKitMarkdown\s+th\b|\.copilotKitMarkdown\s+td\b/);

    expect(cell).not.toBeNull();
    // A non-zero width is the whole point: Preflight sets `border: 0 solid`, so
    // a bare `border-style` would still render nothing.
    expect(cell).toMatch(/border:\s*[1-9]\d*px\s+solid/);
  });

  it('gives header and body cells padding so text does not touch the rules', () => {
    const cell = declarationsFor(/\.copilotKitMarkdown\s+th\b|\.copilotKitMarkdown\s+td\b/);

    expect(cell).toMatch(/padding:\s*[^;]*[1-9]/);
  });

  it('tints the header row so it reads as a header', () => {
    const head = declarationsFor(/\.copilotKitMarkdown\s+thead\s+th\b/);

    expect(head).not.toBeNull();
    expect(head).toMatch(/background(-color)?:/);
  });

  it('lets a wide table scroll inside the narrow chat pane', () => {
    const table = declarationsFor(/\.copilotKitMarkdown\s+table\b/);

    expect(table).not.toBeNull();
    expect(table).toMatch(/overflow-x:\s*auto/);
    // Without a max-width the table would widen the whole message column
    // instead of scrolling within it.
    expect(table).toMatch(/max-width:\s*100%/);
  });

  it('keeps every table rule scoped to the chat so app tables are untouched', () => {
    const selectors = CSS.split('}')
      .map((rule) => rule.split('{')[0])
      .filter((selector) => /(^|[\s,>])(table|thead|tbody|tr|th|td)\b/.test(selector ?? ''));

    expect(selectors.length).toBeGreaterThan(0);
    for (const selector of selectors) {
      expect(selector).toMatch(/\.copilotKitMarkdown/);
    }
  });
});
