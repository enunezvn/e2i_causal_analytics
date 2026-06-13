/**
 * Severity Mapper Tests
 * =====================
 *
 * #26: the backend emits several severity vocabularies that the UI must
 * collapse into ONE three-level visual vocabulary (critical / warning / info).
 *
 * Verified backend vocabularies (grep of the Python API + generated OpenAPI
 * types — see report):
 *   - AlertSeverity  (experiments.py):  "critical" | "warning" | "info"
 *   - DriftSeverity  (monitoring.py):   "none" | "low" | "medium" | "high" | "critical"
 *   - PatternSeverity(feedback.py):     "low" | "medium" | "high" | "critical"
 *
 * Mapping rule (backend -> UI):
 *   critical          -> critical
 *   high              -> warning   (drift/pattern "high" must NOT render as info)
 *   warning | medium  -> warning
 *   info | low | none -> info
 *   anything unknown  -> info      (safe default; never throws)
 */

import { describe, it, expect } from 'vitest';
import {
  mapSeverity,
  toUiSeverity,
  type UiSeverity,
} from './severity';

describe('toUiSeverity', () => {
  const cases: Array<[string, UiSeverity]> = [
    // AlertSeverity (already UI vocabulary)
    ['critical', 'critical'],
    ['warning', 'warning'],
    ['info', 'info'],
    // DriftSeverity / PatternSeverity (high/medium/low/none)
    ['high', 'warning'],
    ['medium', 'warning'],
    ['low', 'info'],
    ['none', 'info'],
  ];

  it.each(cases)('maps backend "%s" -> UI "%s"', (backend, ui) => {
    expect(toUiSeverity(backend)).toBe(ui);
  });

  it('is case-insensitive (backend enums may arrive upper-cased)', () => {
    expect(toUiSeverity('CRITICAL')).toBe('critical');
    expect(toUiSeverity('High')).toBe('warning');
  });

  it('falls back to "info" for unknown / empty / undefined values (never throws)', () => {
    expect(toUiSeverity('chartreuse')).toBe('info');
    expect(toUiSeverity('')).toBe('info');
    expect(toUiSeverity(undefined)).toBe('info');
    expect(toUiSeverity(null)).toBe('info');
  });
});

describe('mapSeverity (full descriptor)', () => {
  it('returns UI severity + human label + variant token for critical', () => {
    const d = mapSeverity('critical');
    expect(d.severity).toBe('critical');
    expect(d.label).toBe('Critical');
    expect(d.variant).toBe('destructive');
  });

  it('maps backend "high" to a warning descriptor', () => {
    const d = mapSeverity('high');
    expect(d.severity).toBe('warning');
    expect(d.label).toBe('Warning');
    expect(d.variant).toBe('warning');
  });

  it('maps "low" / "none" to an info descriptor', () => {
    expect(mapSeverity('low').severity).toBe('info');
    expect(mapSeverity('low').variant).toBe('secondary');
    expect(mapSeverity('none').severity).toBe('info');
  });

  it('uses a safe info descriptor for unknown values', () => {
    const d = mapSeverity('totally-unknown');
    expect(d.severity).toBe('info');
    expect(d.label).toBe('Info');
    expect(d.variant).toBe('secondary');
  });
});
