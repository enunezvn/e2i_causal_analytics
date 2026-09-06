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
import { buildChatSuggestions, topUpChatSuggestions } from './E2IChatSidebar';

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

describe('topUpChatSuggestions (2026-09-05 validator top-up)', () => {
  it('returns the static pills when no adaptive pills exist', () => {
    expect(topUpChatSuggestions(null, '/', 'Kisqali')).toEqual(buildChatSuggestions('/', 'Kisqali'));
    expect(topUpChatSuggestions([], '/', 'Kisqali')).toEqual(buildChatSuggestions('/', 'Kisqali'));
  });

  it('fills up to four with static pills, adaptive first, no duplicate titles', () => {
    const adaptive = [
      { title: '📈 Chart the TRx trend', message: 'Chart the TRx trend for Kisqali' },
      { title: 'Persistence drivers', message: 'What drives persistent_180d for Kisqali?' },
    ];
    const pills = topUpChatSuggestions(adaptive, '/', 'Kisqali');

    expect(pills).toHaveLength(4);
    expect(pills.slice(0, 2)).toEqual(adaptive);
    const titles = pills.map((p) => p.title.toLowerCase());
    expect(new Set(titles).size).toBe(4);
    // the duplicated static "Chart the TRx trend" was skipped, the others filled in
    expect(pills.some((p) => p.title.includes('Kisqali market share'))).toBe(true);
  });

  it('never exceeds four and keeps a full adaptive set untouched', () => {
    const adaptive = [1, 2, 3, 4, 5].map((i) => ({ title: `Pill ${i}`, message: `Question ${i}?` }));
    expect(topUpChatSuggestions(adaptive, '/', 'All')).toEqual(adaptive.slice(0, 4));
  });
});
