/**
 * ClinicalContextPanel — brand-faithful, sourced clinical narrative for a
 * discovered causal effect.
 *
 * Renders the drug + mechanism of action (ChEMBL), the disease's real pivotal
 * endpoints (ClinicalTrials.gov), and a real-world-evidence citation (PubMed) —
 * each with a source chip so a curated static fallback is disclosed, never
 * hidden. Always shows the synthetic-estimate / real-context honesty label.
 *
 * Additive presentation ONLY — it does not change the causal estimate.
 *
 * @module components/causal/ClinicalContextPanel
 */

import { FlaskConical, BookText, ExternalLink, Stethoscope } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import type { ClinicalContext } from '@/types/causal';

function sourceChip(source: string) {
  const live = source === 'chembl' || source === 'clinicaltrials.gov' || source === 'pubmed';
  const seed = source === 'pubmed_seed';
  if (live) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        {source}
      </Badge>
    );
  }
  if (seed) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        pubmed (curated seed)
      </Badge>
    );
  }
  return (
    <Badge variant="secondary" className="ml-2 align-middle text-xs">
      curated fallback
    </Badge>
  );
}

export function ClinicalContextPanel({ context }: { context: ClinicalContext }) {
  const { mechanism, pivotal_endpoints, real_world_evidence } = context;
  return (
    <div className="space-y-4 rounded-md border p-4">
      <div className="flex items-center gap-2">
        <Stethoscope className="h-4 w-4 text-muted-foreground" />
        <p className="text-sm font-medium">Clinical context</p>
      </div>

      {/* Drug + mechanism of action */}
      <div className="text-sm">
        <span className="font-medium capitalize">{context.drug_name}</span>{' '}
        <span className="text-muted-foreground">— {context.disease}</span>
        <div className="mt-1">
          <span className="text-muted-foreground">Mechanism of action: </span>
          <span className="font-medium">{mechanism.mechanism_of_action}</span>
          {sourceChip(mechanism.source)}
        </div>
      </div>

      {/* Our synthetic outcome -> the real pivotal endpoint framing */}
      {context.mapped_endpoint && (
        <div className="text-sm">
          <span className="text-muted-foreground">Our outcome </span>
          <code className="rounded bg-muted px-1 py-0.5 text-xs">{context.our_outcome}</code>
          <span className="text-muted-foreground"> maps to: </span>
          <span className="font-medium">{context.mapped_endpoint}</span>
        </div>
      )}

      {/* The disease's real pivotal endpoints */}
      {pivotal_endpoints.endpoints.length > 0 && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <FlaskConical className="h-3.5 w-3.5" />
            Real pivotal endpoints
            {sourceChip(pivotal_endpoints.source)}
          </div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {pivotal_endpoints.endpoints.map((ep) => (
              <li key={ep}>{ep}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Real-world-evidence citation (only when real) */}
      {real_world_evidence ? (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <BookText className="h-3.5 w-3.5" />
            Real-world evidence
            {sourceChip(real_world_evidence.source)}
          </div>
          <a
            href={real_world_evidence.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-start gap-1 text-primary hover:underline"
          >
            <span>
              {real_world_evidence.title}
              {real_world_evidence.journal ? ` — ${real_world_evidence.journal}` : ''}
              {real_world_evidence.pubdate ? ` (${real_world_evidence.pubdate})` : ''}
              {` · PMID ${real_world_evidence.pmid}`}
            </span>
            <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          </a>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          No real-world-evidence citation found for this brand.
        </p>
      )}

      {/* The synthetic/real honesty boundary — always shown */}
      <p className="border-t pt-3 text-xs text-muted-foreground">{context.honesty_label}</p>
    </div>
  );
}
