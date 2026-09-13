/**
 * gateBadge — the robustness-gate decision badge (#1991 debt 4).
 * ================================================================
 *
 * Split out of `CausalAnalysisDetail.tsx` into its own module so this file
 * exports ONLY non-component values (a constant + a function), clearing the
 * `react-refresh/only-export-components` warning a mixed component/helper
 * file would otherwise carry.
 *
 * @module components/causal/gateBadge
 */

import { Badge } from '@/components/ui/badge';
import type { RefutationGate } from '@/types/causal';

// #1991 debt 4: Record<RefutationGate, ...> makes an out-of-union decision a
// compile error, not a silently-missing badge.
export const GATE_BADGE: Record<
  RefutationGate,
  { label: string; variant: 'default' | 'secondary' | 'destructive' }
> = {
  proceed: { label: 'Proceed', variant: 'default' },
  review: { label: 'Review', variant: 'secondary' },
  block: { label: 'Blocked', variant: 'destructive' },
};

export function gateBadge(decision?: RefutationGate | null) {
  if (!decision) return <Badge variant="outline">—</Badge>;
  const b = GATE_BADGE[decision];
  return <Badge variant={b.variant}>{b.label}</Badge>;
}
