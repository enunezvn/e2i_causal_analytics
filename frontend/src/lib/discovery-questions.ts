/**
 * Discovery question identity
 * ===========================
 *
 * The SSOT keys a discover-effects candidate by (treatment, outcome, brand);
 * the selector, the page's selection state and the run's `questions` subset
 * all agree on this one string form of that triple.
 *
 * @module lib/discovery-questions
 */

import type { DiscoverQuestion } from '@/types/causal';

/** Stable identity of a candidate row: the SSOT keys a question by this triple. */
export function questionKey(q: Pick<DiscoverQuestion, 'treatment' | 'outcome' | 'brand'>): string {
  return `${q.treatment}->${q.outcome}->${q.brand ?? 'all'}`;
}
