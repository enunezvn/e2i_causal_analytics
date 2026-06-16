import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Sidebar } from './Sidebar';

function renderSidebar() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Sidebar />
    </MemoryRouter>,
  );
}

describe('Sidebar navigation (rendered IA)', () => {
  it('renders the five section headers in workflow order', () => {
    renderSidebar();
    const headers = screen
      .getAllByRole('heading', { level: 3 })
      .map((h) => h.textContent);
    expect(headers).toEqual([
      'Causal Analytics',
      'Predictive Modeling',
      'Decisions & Optimization',
      'Data & Reference',
      'System & Platform',
    ]);
  });

  it('renders all 25 nav links with causal pages ahead of system pages', () => {
    renderSidebar();
    const links = screen.getAllByRole('link').map((a) => a.textContent?.trim());
    expect(links).toHaveLength(25);
    expect(links).toContain('Causal Discovery');
    expect(links).toContain('System Health');
    // a causal page must come before a system page (the old order had it reversed)
    expect(links.indexOf('Causal Discovery')).toBeLessThan(links.indexOf('System Health'));
    // analytics pages that USED to be mislabeled "System" now lead the list
    expect(links.indexOf('Gap Analysis')).toBeLessThan(links.indexOf('Agent Orchestration'));
  });

  it('renders a real, distinct icon for every nav item (none silently fall back to Home)', () => {
    renderSidebar();
    // Build title -> rendered <svg> content signature from the actual DOM.
    const sigByTitle = new Map<string, string>();
    for (const a of screen.getAllByRole('link')) {
      const title = a.textContent?.trim() ?? '';
      sigByTitle.set(title, a.querySelector('svg')?.innerHTML ?? '');
    }

    const homeGlyph = sigByTitle.get('Home');
    expect(homeGlyph).toBeTruthy();

    // The 4 items that used to render the Home glyph (unmapped icon names) must
    // now render their own icon — this is exactly the content I could not see
    // below the fold in the screenshot.
    for (const title of ['Expert Reviews', 'Audit Chain', 'Feedback Learning', 'Analytics']) {
      expect(sigByTitle.get(title), `${title} should not use the Home fallback icon`).not.toBe(
        homeGlyph,
      );
    }

    // Strongest content check: ONLY Home renders the Home glyph — proves no nav
    // item anywhere silently falls back.
    const itemsUsingHomeGlyph = [...sigByTitle.entries()]
      .filter(([, sig]) => sig === homeGlyph)
      .map(([title]) => title);
    expect(itemsUsingHomeGlyph).toEqual(['Home']);
  });
});
