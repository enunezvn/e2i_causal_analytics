/**
 * The minimal reviewer checklist (the migration-010 SYSTEM_TEMPLATE required
 * items) and the advisory-verdict chip styling shared by the queue page and
 * the linked-review card. Ids MUST stay in sync with
 * src/insights/expert_review_assessment.py CHECKLIST_QUESTIONS.
 */
import type { AssessmentVerdict } from '@/types/expert-review';

export const CHECKLIST_ITEMS: { id: string; question: string }[] = [
  { id: 'conf_complete', question: 'Are all known confounders included?' },
  { id: 'edge_plausible', question: 'Do causal arrows reflect domain knowledge?' },
  { id: 'no_forbidden', question: 'Are there no forbidden edges (future→past)?' },
  { id: 'mediators_correct', question: 'Are intermediate variables correctly positioned?' },
  { id: 'sutva_plausible', question: 'Is the no-interference assumption reasonable?' },
  { id: 'positivity', question: 'Is there sufficient overlap in treatment groups?' },
];

/** Concern is the only destructive signal; the other verdicts stay visually calm. */
export const VERDICT_VARIANT: Record<AssessmentVerdict, 'secondary' | 'destructive' | 'outline'> = {
  supports: 'secondary',
  concern: 'destructive',
  unclear: 'outline',
  no_evidence: 'outline',
};

export function shortHash(hash?: string | null): string {
  if (!hash) return '—';
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash;
}

/** approved / rejected / pending / anything else → badge variant. */
export function statusVariant(status?: string | null): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'approved') return 'default';
  if (status === 'rejected') return 'destructive';
  if (status === 'pending') return 'secondary';
  return 'outline';
}
