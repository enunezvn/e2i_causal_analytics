import { describe, it, expect } from 'vitest';
import { getNavigationRoutes, getNavigationSections } from './routes';

/**
 * Navigation information-architecture grouping.
 *
 * Sections must be defined by an explicit `route.section`, NOT by positional
 * array slicing (the old `routes.slice(1, 8)` / `slice(8)` approach silently
 * dumped every late-added analytics page into "System").
 */
describe('navigation sections (IA grouping)', () => {
  it('groups nav routes into the 6 ordered sections with the right labels', () => {
    const sections = getNavigationSections();
    expect(sections.map((s) => s.key)).toEqual([
      'main',
      'causal',
      'predictive',
      'decisions',
      'data',
      'system',
    ]);
    expect(sections.map((s) => s.label)).toEqual([
      null,
      'Causal Analytics',
      'Predictive Modeling',
      'Decisions & Optimization',
      'Data & Reference',
      'System & Platform',
    ]);
  });

  it('puts Home alone at the top (main, no header)', () => {
    const main = getNavigationSections().find((s) => s.key === 'main');
    expect(main?.label).toBeNull();
    expect(main?.routes.map((r) => r.title)).toEqual(['Home']);
  });

  it('orders Causal Analytics by the analytical workflow', () => {
    const causal = getNavigationSections().find((s) => s.key === 'causal');
    expect(causal?.routes.map((r) => r.title)).toEqual([
      'Causal Discovery',
      'Knowledge Graph',
      'Causal Analysis',
      'Intervention Impact',
      'Segment Analysis',
      'Expert Reviews',
    ]);
  });

  it('keeps ONLY genuine platform internals under System & Platform', () => {
    const system = getNavigationSections().find((s) => s.key === 'system');
    const titles = system?.routes.map((r) => r.title) ?? [];
    // regression: analytics pages must no longer be mislabeled as "System"
    for (const t of [
      'Causal Analysis',
      'Gap Analysis',
      'Experiments',
      'Segment Analysis',
      'Digital Twin',
      'Expert Reviews',
    ]) {
      expect(titles).not.toContain(t);
    }
    expect(titles).toEqual([
      'System Health',
      'Monitoring',
      'Analytics',
      'Agent Orchestration',
      'Memory Architecture',
      'Audit Chain',
      'Feedback Learning',
    ]);
  });

  it('assigns every nav route to exactly one section (no orphans, no loss)', () => {
    const navRoutes = getNavigationRoutes();
    const grouped = getNavigationSections().flatMap((s) => s.routes);
    expect(grouped).toHaveLength(navRoutes.length);
    for (const r of navRoutes) {
      expect(r.section).toBeDefined();
    }
  });
});
