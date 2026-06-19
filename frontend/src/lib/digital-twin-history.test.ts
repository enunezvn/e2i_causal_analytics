/**
 * Tests for digital-twin history grouping (dedup look-alike runs).
 */

import { describe, it, expect } from 'vitest';
import { groupSimulationsByInterventionBrand } from './digital-twin-history';

interface Row {
  simulation_id: string;
  brand: string;
  intervention_type: string;
  created_at: string;
  ate_estimate: number;
}

// Mirrors the live prod state: 5 Remibrutinib/digital_engagement + 1 Fabhalta/email.
const ROWS: Row[] = [
  { simulation_id: 'a', brand: 'Remibrutinib', intervention_type: 'digital_engagement', created_at: '2026-06-19T00:48:55Z', ate_estimate: 0.4 },
  { simulation_id: 'b', brand: 'Remibrutinib', intervention_type: 'digital_engagement', created_at: '2026-06-18T01:11:18Z', ate_estimate: 0.397 },
  { simulation_id: 'c', brand: 'Remibrutinib', intervention_type: 'digital_engagement', created_at: '2026-06-18T01:10:32Z', ate_estimate: 0.392 },
  { simulation_id: 'd', brand: 'Fabhalta', intervention_type: 'email_campaign', created_at: '2026-06-16T17:21:22Z', ate_estimate: 0.149 },
  { simulation_id: 'e', brand: 'Remibrutinib', intervention_type: 'digital_engagement', created_at: '2026-06-16T14:46:52Z', ate_estimate: 0.159 },
  { simulation_id: 'f', brand: 'Remibrutinib', intervention_type: 'digital_engagement', created_at: '2026-06-16T14:45:07Z', ate_estimate: 0.147 },
];

describe('groupSimulationsByInterventionBrand', () => {
  it('collapses repeated (brand, intervention) runs into one group each', () => {
    const groups = groupSimulationsByInterventionBrand(ROWS);
    expect(groups).toHaveLength(2);
    const keys = groups.map((g) => g.key).sort();
    expect(keys).toEqual(['Fabhalta|email_campaign', 'Remibrutinib|digital_engagement']);
  });

  it('counts every run and keeps them all (no data dropped)', () => {
    const groups = groupSimulationsByInterventionBrand(ROWS);
    const digital = groups.find((g) => g.key === 'Remibrutinib|digital_engagement')!;
    expect(digital.count).toBe(5);
    expect(digital.runs).toHaveLength(5);
    expect(groups.find((g) => g.key === 'Fabhalta|email_campaign')!.count).toBe(1);
  });

  it('picks the most recent run as latest', () => {
    const groups = groupSimulationsByInterventionBrand(ROWS);
    const digital = groups.find((g) => g.key === 'Remibrutinib|digital_engagement')!;
    expect(digital.latest.simulation_id).toBe('a'); // 2026-06-19, newest
    expect(digital.latest.ate_estimate).toBe(0.4);
  });

  it('orders runs within a group newest-first', () => {
    const groups = groupSimulationsByInterventionBrand(ROWS);
    const digital = groups.find((g) => g.key === 'Remibrutinib|digital_engagement')!;
    expect(digital.runs.map((r) => r.simulation_id)).toEqual(['a', 'b', 'c', 'e', 'f']);
  });

  it('orders groups by their latest run (most recent group first)', () => {
    const groups = groupSimulationsByInterventionBrand(ROWS);
    expect(groups[0].key).toBe('Remibrutinib|digital_engagement'); // latest 06-19
    expect(groups[1].key).toBe('Fabhalta|email_campaign'); // latest 06-16
  });

  it('handles a single run and an empty list', () => {
    const one = groupSimulationsByInterventionBrand([ROWS[3]]);
    expect(one).toHaveLength(1);
    expect(one[0].count).toBe(1);
    expect(groupSimulationsByInterventionBrand([])).toEqual([]);
  });
});
