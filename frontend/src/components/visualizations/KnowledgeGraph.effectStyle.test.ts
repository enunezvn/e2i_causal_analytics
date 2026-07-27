/**
 * computeEffectStyle Tests
 * ========================
 *
 * Unit tests for the data-driven edge styling used by the brand-scoped causal
 * view: width tracks |ATE|, color tracks the effect's sign, opacity tracks
 * confidence. The mapping constants are asserted here so a styling change is a
 * deliberate, reviewed act rather than a silent drive-by.
 */

import { describe, it, expect } from 'vitest';
import { computeEffectStyle } from './KnowledgeGraph';

describe('computeEffectStyle', () => {
  it('colors positive effects emerald and negative effects rose', () => {
    expect(computeEffectStyle(0.3, 0.8).effectColor).toBe('#059669');
    expect(computeEffectStyle(-0.2, 0.8).effectColor).toBe('#e11d48');
  });

  it('scales width with |ATE| and clamps at the 0.5 ceiling', () => {
    const small = computeEffectStyle(0.05, 0.8);
    const large = computeEffectStyle(0.45, 0.8);
    expect(small.effectWidth).toBeLessThan(large.effectWidth);
    // Negative effects are as wide as equal-magnitude positive ones.
    expect(computeEffectStyle(-0.45, 0.8).effectWidth).toBe(large.effectWidth);
    // |ATE| ≥ 0.5 clamps to the max width; an outlier can't flatten the rest.
    expect(computeEffectStyle(2.0, 0.8).effectWidth).toBe(6);
    expect(computeEffectStyle(0, 0.8).effectWidth).toBe(1.5);
  });

  it('maps confidence into [0.35, 1] opacity, clamped', () => {
    expect(computeEffectStyle(0.3, 1).effectOpacity).toBe(1);
    expect(computeEffectStyle(0.3, 0.5).effectOpacity).toBe(0.35);
    // Below the 0.5 floor still renders faint, never invisible.
    expect(computeEffectStyle(0.3, 0.1).effectOpacity).toBe(0.35);
    const mid = computeEffectStyle(0.3, 0.75).effectOpacity;
    expect(mid).toBeGreaterThan(0.35);
    expect(mid).toBeLessThan(1);
  });

  it('treats a missing confidence as the faint floor', () => {
    expect(computeEffectStyle(0.3, undefined).effectOpacity).toBe(0.35);
  });
});
