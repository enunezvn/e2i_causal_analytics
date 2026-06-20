/**
 * LabelGateBadge — surfaces the label-segmentation gater's verdict on a
 * recommendation (segment policy or gap opportunity).
 *
 * The backend already DE-PRIORITIZES off-label items (they sink to the bottom of
 * the returned ranking); this badge only makes that visible so an off-label
 * segment/bet is plainly flagged and seen to be down-ranked. It renders nothing
 * when `label_verdict` is absent (the gater was off) — an honest empty state.
 *
 * Source-chip styling mirrors `components/causal/ClinicalContextPanel.tsx`
 * (outline / secondary Badge, text-xs) for visual consistency.
 *
 * @module components/insights/LabelGateBadge
 */

import { AlertTriangle, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { LabelVerdict } from '@/types/segments';

export interface LabelGateBadgeProps {
  /** Structured verdict from the gater; when absent the badge renders nothing. */
  label_verdict?: LabelVerdict;
  /** Human-readable reason the item was judged off-label (shown as tooltip/subtext). */
  off_label_reason?: string;
  /** Redundant boolean flag from the backend (verdict is the source of truth). */
  off_label?: boolean;
  /** True when the verdict was confirmed against the FDA drug label. */
  label_evidence_confirmed?: boolean;
  className?: string;
}

/**
 * "confirmed by FDA label" provenance chip — mirrors the outline source-chip in
 * ClinicalContextPanel. Only shown when the verdict was label-confirmed.
 */
function ConfirmedChip() {
  return (
    <Badge variant="outline" className="ml-2 align-middle text-xs font-normal">
      <ShieldCheck className="mr-1 h-3 w-3" />
      confirmed by FDA label
    </Badge>
  );
}

export function LabelGateBadge({
  label_verdict,
  off_label_reason,
  label_evidence_confirmed,
  className,
}: LabelGateBadgeProps) {
  // Honest empty state: gater off -> render nothing.
  if (!label_verdict) {
    return null;
  }

  // indeterminate -> muted "Label: review" with no de-prioritization claim.
  if (label_verdict === 'indeterminate') {
    return (
      <span className={cn('inline-flex items-center', className)}>
        <Badge variant="secondary" className="text-xs font-normal text-muted-foreground">
          Label: review
        </Badge>
        {label_evidence_confirmed && <ConfirmedChip />}
      </span>
    );
  }

  // on_label -> subtle confirmation; conveys it is NOT de-prioritized.
  if (label_verdict === 'on_label') {
    return (
      <span className={cn('inline-flex items-center', className)}>
        <Badge variant="outline" className="text-xs font-normal text-emerald-600">
          <ShieldCheck className="mr-1 h-3 w-3" />
          On-label
        </Badge>
        {label_evidence_confirmed && <ConfirmedChip />}
      </span>
    );
  }

  // off_label / mixed -> warning badge + de-prioritization notice + reason.
  const isMixed = label_verdict === 'mixed';
  const label = isMixed ? 'Partly off-label' : 'Off-label';
  const deprioritizedNote = isMixed
    ? 'Partly off-label — de-prioritized in ranking.'
    : 'Off-label use — de-prioritized to the bottom of the ranking.';
  // Combine the standing reason (if any) with the de-prioritization note for the
  // hover title, so the flag and its consequence are both discoverable.
  const title = off_label_reason ? `${off_label_reason} — ${deprioritizedNote}` : deprioritizedNote;

  return (
    <span className={cn('inline-flex flex-wrap items-center gap-1', className)}>
      <Badge variant="warning" className="text-xs" title={title}>
        <AlertTriangle className="mr-1 h-3 w-3" />
        {label}
      </Badge>
      <span className="text-xs text-muted-foreground">{deprioritizedNote}</span>
      {off_label_reason && (
        <span className="text-xs italic text-muted-foreground">{off_label_reason}</span>
      )}
      {label_evidence_confirmed && <ConfirmedChip />}
    </span>
  );
}

export default LabelGateBadge;
