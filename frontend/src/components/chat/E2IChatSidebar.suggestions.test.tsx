/**
 * E2IChatSidebar — static suggestion pill tests (#1749)
 * =====================================================
 *
 * The static fallback pills are the guaranteed floor of the suggestion
 * surface (shown instantly and whenever POST /chat/suggestions fails), so
 * they must respect the dashboard's brand selection: name the brand the
 * user picked, and never invent a brand when none is selected ('All').
 *
 * #1749: with Kisqali selected on the page, the pills said "Remibrutinib
 * market share" — the provider's hardcoded default leaked through.
 */

import { describe, it, expect } from 'vitest';
import { buildChatSuggestions } from './E2IChatSidebar';

describe('buildChatSuggestions (#1749)', () => {
  it('names the SELECTED brand in the market-share pill', () => {
    const pills = buildChatSuggestions('/', 'Kisqali');

    const brandPill = pills.find((p) => p.title.toLowerCase().includes('market share'));
    expect(brandPill).toBeDefined();
    expect(brandPill!.title).toContain('Kisqali');
    expect(brandPill!.message).toContain('Kisqali');

    // No pill names a brand the user did not select ('/' page pill and the
    // generic pills are brand-free; only the market-share pill is branded).
    for (const p of pills) {
      expect(p.title).not.toContain('Remibrutinib');
      expect(p.message).not.toContain('Remibrutinib');
      expect(p.title).not.toContain('Fabhalta');
      expect(p.message).not.toContain('Fabhalta');
    }
  });

  it("offers a cross-brand comparison — never an 'All market share' pill — when no brand is selected", () => {
    const pills = buildChatSuggestions('/', 'All');

    // A comparison across the portfolio replaces the single-brand pill.
    expect(pills.some((p) => /compare/i.test(p.title))).toBe(true);

    // The 'All' sentinel must never be interpolated as if it were a brand.
    for (const p of pills) {
      expect(p.title).not.toContain('All market share');
      expect(p.message).not.toMatch(/for All\b/);
    }
  });
});
