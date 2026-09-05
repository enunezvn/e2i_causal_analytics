import { describe, expect, it } from 'vitest';

import { columnLabel } from './column-labels';

// One display-label helper for every page that prints a gold-standard column
// (treatment / outcome) — the backend serves the curated label map
// (causal._COLUMN_LABELS) on GET /causal/variables and GET /segments/datasets;
// the pages must never re-derive a label from the raw column name when the
// backend has one. 2026-09-05: /segment-analysis rendered the labels (#1893)
// while /causal-analysis still printed `sample_dropped`.
describe('columnLabel', () => {
  it('returns the backend curated label when the map has the column', () => {
    const labels = { sample_dropped: 'Product samples provided (rep sample drop)' };
    expect(columnLabel(labels, 'sample_dropped')).toBe(
      'Product samples provided (rep sample drop)'
    );
  });

  it('falls back to the backend auto-label shape (spaces, first letter capitalised) when the map is absent or lacks the column', () => {
    // Same fallback the backend applies for an uncurated column, so a column
    // that is labelled on one page and unlabelled on another still reads alike.
    expect(columnLabel(undefined, 'control_group_flag')).toBe('Control group flag');
    expect(columnLabel(null, 'persistent_180d')).toBe('Persistent 180d');
    expect(columnLabel({}, 'conversion_flag')).toBe('Conversion flag');
  });

  it('matches the backend fallback byte-for-byte on mixed-case tokens (Python str.capitalize lowercases the rest)', () => {
    // Parity contract with causal._column_label — pinned by the same inputs in
    // tests/unit/test_api/test_causal_discover_effects.py. A one-hot dummy or an
    // acronym-bearing name must not read one way in the backend summary prose
    // and another way in the leaderboard cell beside it.
    expect(columnLabel(undefined, 'geographic_region=West')).toBe('Geographic region=west');
    expect(columnLabel({}, 'uas7_HIGH')).toBe('Uas7 high');
  });

  it('never returns an empty label for a named column, and never crashes on a missing one', () => {
    expect(columnLabel({ x: '' }, 'x')).toBe('X');
    expect(columnLabel({}, '')).toBe('');
    // A partial API payload (no treatment name) must render as nothing, not throw.
    expect(columnLabel({ a: 'A' }, undefined as unknown as string)).toBe('');
  });
});
