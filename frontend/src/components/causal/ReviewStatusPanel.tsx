/**
 * ReviewStatusPanel — the expert-review state of one agent run's DAG structure.
 * =============================================================================
 *
 * Renders ONLY what the API returned (spec §4.4): the structural verdict from
 * `refutation.expert_review_decision`, a link to the review row when the run
 * touched one, ONE reason line, and the durable discovered-DAG record id in
 * full with a copy affordance. Absent fields render nothing; an unknown
 * decision renders verbatim rather than a guessed label.
 *
 * The reason line: the expert-review halt message from `warnings` when the
 * run carries one (a rejection, the approval-enforcement switch, or the
 * route's fallback — the run's warnings are rendered nowhere else in the
 * drill-down, so the halt shows for any decision that carries one); otherwise
 * the agent's caveat from `refutation.review_caveat` (#1995) — the sentence
 * naming the approval (reviewer, validity window), the rejection (reviewer,
 * reason) or the queued / blocked / unavailable state. A BLOCK-band run never
 * halts (the statistical gate already withheld the estimate), so the caveat
 * is its only adjudication prose. The halt line embeds the caveat verbatim,
 * which is why the two are never rendered together.
 *
 * @module components/causal/ReviewStatusPanel
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';

type Variant = 'default' | 'secondary' | 'destructive' | 'outline';

const DECISION_COPY: Record<string, { label: string; meaning: string; variant: Variant }> = {
  proceed: {
    label: 'Structure approved',
    meaning:
      'A reviewer approved this DAG structure. Approval covers the structure only; the estimate still stands or falls on its own robustness checks.',
    variant: 'default',
  },
  renewal_required: {
    label: 'Approval expiring',
    meaning:
      'The structural approval is inside its renewal window. A reviewer should renew it before it lapses.',
    variant: 'secondary',
  },
  pending_review: {
    label: 'Pending expert review',
    meaning:
      'This DAG structure is queued for a reviewer. Re-running the analysis does not change that; resolving the review does.',
    variant: 'secondary',
  },
  rejected: {
    label: 'Structure rejected',
    meaning:
      'A reviewer rejected this DAG structure. The estimate is withheld on every band until a reviewer reopens the structure.',
    variant: 'destructive',
  },
  blocked: {
    label: 'No review possible',
    meaning: 'The structure holds no approval and no review could be queued for it.',
    variant: 'outline',
  },
  unavailable: {
    label: 'Review gate unavailable',
    meaning:
      'The review store could not be consulted for this run; nothing was checked or queued.',
    variant: 'outline',
  },
};

export interface ReviewStatusPanelProps {
  decision?: string | null;
  reviewId?: string | null;
  discoveredDagId?: string | null;
  /**
   * The run's warnings; the expert-review halt message — a rejection, the
   * approval-enforcement switch, or the route's fallback — lives there.
   */
  warnings?: string[];
  /**
   * `refutation.review_caveat` (#1995): the agent's band + expert-review
   * sentence. Rendered as the reason line only when `warnings` carries no halt
   * (the halt embeds it verbatim).
   */
  reviewCaveat?: string | null;
}

export function ReviewStatusPanel({
  decision,
  reviewId,
  discoveredDagId,
  warnings,
  reviewCaveat,
}: ReviewStatusPanelProps) {
  const [copied, setCopied] = useState(false);
  if (!decision && !discoveredDagId) return null;
  // Own-property lookup: a plain object resolves inherited members ("toString",
  // "constructor"), which would render an empty badge instead of the verbatim string.
  const copy =
    decision && Object.prototype.hasOwnProperty.call(DECISION_COPY, decision)
      ? DECISION_COPY[decision]
      : undefined;
  // At most one halt per run, and every producer of it starts with this prefix.
  const halt = (warnings ?? []).find((w) => w.startsWith('Estimate withheld'));
  // One reason line: the halt (which embeds the caveat) wins; else the caveat.
  const reason = halt ?? (reviewCaveat?.trim() ? reviewCaveat : undefined);

  return (
    <div
      className="space-y-1 rounded-md border border-[var(--color-border)] p-3 text-sm"
      data-testid="review-status"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">Review status</span>
        {copy ? (
          <Badge variant={copy.variant}>{copy.label}</Badge>
        ) : decision ? (
          <Badge variant="outline">{decision}</Badge>
        ) : null}
        {reviewId && (
          <Link
            to={`/expert-reviews?review=${encodeURIComponent(reviewId)}`}
            className="text-xs underline"
          >
            Open review
          </Link>
        )}
      </div>
      {copy && <p className="text-xs text-muted-foreground">{copy.meaning}</p>}
      {reason && <p className="text-xs text-muted-foreground">{reason}</p>}
      {discoveredDagId && (
        <p className="text-xs text-muted-foreground">
          Durable discovery record:{' '}
          <code className="font-mono text-[11px]">{discoveredDagId}</code>{' '}
          <button
            type="button"
            aria-label="Copy discovery record id"
            className="text-xs underline"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(discoveredDagId);
                setCopied(true);
              } catch {
                setCopied(false);
              }
            }}
          >
            {copied ? 'Copied' : 'Copy id'}
          </button>
        </p>
      )}
    </div>
  );
}
