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
});
