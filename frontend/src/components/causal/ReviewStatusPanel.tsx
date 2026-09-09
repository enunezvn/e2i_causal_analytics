/**
 * ReviewStatusPanel — the expert-review state of one agent run's DAG structure.
 * =============================================================================
 *
 * Renders ONLY what the API returned (spec §4.4): the structural verdict from
 * `refutation.expert_review_decision`, a link to the review row when the run
 * touched one, the rejection halt message from `warnings`, and the durable
 * discovered-DAG record id. Absent fields render nothing; an unknown decision
 * renders verbatim rather than a guessed label.
 *
 * @module components/causal/ReviewStatusPanel
 */

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
  /** The run's warnings; the rejection halt message (reviewer + reason) lives there. */
  warnings?: string[];
}

export function ReviewStatusPanel({
  decision,
  reviewId,
  discoveredDagId,
  warnings,
}: ReviewStatusPanelProps) {
  if (!decision && !discoveredDagId) return null;
  const copy = decision ? DECISION_COPY[decision] : undefined;
  const halt =
    decision === 'rejected'
      ? (warnings ?? []).find((w) => w.startsWith('Estimate withheld'))
      : undefined;

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
      {halt && <p className="text-xs text-muted-foreground">{halt}</p>}
      {discoveredDagId && (
        <p className="text-xs text-muted-foreground">
          Durable discovery record:{' '}
          <code className="font-mono text-[11px]">{discoveredDagId}</code>
        </p>
      )}
    </div>
  );
}
