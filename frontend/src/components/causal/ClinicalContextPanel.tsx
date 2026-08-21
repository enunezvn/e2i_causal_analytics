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

import {
  BookText,
  Building2,
  ClipboardList,
  ExternalLink,
  FlaskConical,
  Microscope,
  Stethoscope,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import type { ClinicalContext } from '@/types/causal';

// The endpoint list grounds outcome definitions; it is NOT a data table. Cap how
// many measures we surface so a brand with many registered trial endpoints
// (Fabhalta has 13) reads as light clinical grounding, not a parameter dump.
const MAX_ENDPOINTS_SHOWN = 5;

function sourceChip(source: string) {
  const live =
    source === 'chembl' ||
    source === 'clinicaltrials.gov' ||
    source === 'pubmed' ||
    source === 'openfda';
  const seed = source === 'pubmed_seed';
  const brandLevel = source === 'pubmed_brand';
  const curated = source === 'curated';
  if (live) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        {source === 'openfda' ? 'live FDA label' : source}
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
  // The analysis-specific query found nothing and the brand-level query answered
  // instead — say so, rather than let a brand-level paper read as analysis-level.
  if (brandLevel) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        pubmed (brand-level)
      </Badge>
    );
  }
  if (curated) {
    return (
      <Badge variant="secondary" className="ml-2 align-middle text-xs">
        curated
      </Badge>
    );
  }
  if (source === 'static_fallback') {
    return (
      <Badge variant="secondary" className="ml-2 align-middle text-xs">
        curated fallback
      </Badge>
    );
  }
  // An unrecognised source is shown verbatim. Calling it "curated fallback" would
  // assert a provenance we do not know — the one thing this panel must never do.
  return (
    <Badge variant="secondary" className="ml-2 align-middle text-xs">
      {source}
    </Badge>
  );
}

// What the clinical sources can honestly say about each kind of treatment. A
// commercial lever has no biomedical literature about the lever itself, and the panel
// must say so instead of letting the drug's evidence read as evidence for the lever.
const TREATMENT_KIND_NOTE: Record<string, string> = {
  drug_therapy:
    'a therapy contrast — the mechanism, endpoints and label below describe this treatment directly.',
  clinical_covariate:
    'a patient-state variable used as an observational treatment — the sources below describe the disease context, not a therapy contrast.',
  commercial:
    'an access and promotion lever — the mechanism, endpoints and label below describe the therapy, not this lever. The real-world-evidence search does carry this lever\u2019s own health-services theme when one exists.',
};

export function ClinicalContextPanel({ context }: { context: ClinicalContext }) {
  const {
    mechanism,
    pivotal_endpoints,
    real_world_evidence,
    seminal_real_world_evidence,
    approved_indications,
    competitor_landscape,
    treatment_context,
    analysis_framing,
    causal_evidence,
    analysis_grounding,
  } = context;
  // Provenance-aware copy: the endpoint section renders for BOTH the live CT.gov
  // result and the curated static fallback, so the explanatory text must not claim
  // ClinicalTrials.gov / "verbatim" provenance when the source is the fallback.
  const endpointsFromCtgov = pivotal_endpoints.source === 'clinicaltrials.gov';
  return (
    <div className="space-y-4 rounded-md border p-4">
      <div>
        <div className="flex items-center gap-2">
          <Stethoscope className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm font-medium">Clinical context</p>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          The brand&rsquo;s clinical reality alongside this analysis — its mechanism, the real
          trial endpoints our outcomes stand in for, and the approved labeling. The label
          describes the therapy and its indication; it makes no claim about a commercial
          lever, and nothing below reads a commercial result as on-label.
        </p>
      </div>

      {/* The analysis this context is grounding. Absent on the brand-level view,
          where there is no single treatment -> outcome pair in scope. */}
      {analysis_framing && (
        <div className="rounded-md bg-muted/50 p-3">
          <p className="text-sm font-medium">{analysis_framing}</p>
          {treatment_context && (
            <p className="mt-1 text-xs text-muted-foreground">
              Treatment{' '}
              <code className="rounded bg-muted px-1 py-0.5">{treatment_context.column}</code> ={' '}
              {treatment_context.label} — {TREATMENT_KIND_NOTE[treatment_context.kind] ?? ''}
            </p>
          )}
        </div>
      )}

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

      {/* The disease's real trial endpoints — grounding for the outcome definition,
          NOT a data table. We surface the measures (capped) so they read as clinical
          ground truth; the per-endpoint time frame / NCT id / analysis-scenario
          parameters are intentionally NOT surfaced here (they belong to the raw
          trial record, not to a commercial read). */}
      {pivotal_endpoints.endpoints.length > 0 && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <FlaskConical className="h-3.5 w-3.5" />
            Real trial endpoints
            {sourceChip(pivotal_endpoints.source)}
          </div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {pivotal_endpoints.endpoints.slice(0, MAX_ENDPOINTS_SHOWN).map((ep) => (
              <li key={ep.measure}>{ep.measure}</li>
            ))}
          </ul>
          {pivotal_endpoints.endpoints.length > MAX_ENDPOINTS_SHOWN && (
            <p className="mt-1 pl-5 text-xs text-muted-foreground">
              + {pivotal_endpoints.endpoints.length - MAX_ENDPOINTS_SHOWN} more registered
              trial endpoints
            </p>
          )}
          <p className="mt-1 text-xs text-muted-foreground">
            {endpointsFromCtgov
              ? 'What this brand’s pivotal trials actually measured — the clinical ground truth our synthetic outcome stands in for.'
              : 'The disease’s established pivotal efficacy measures (curated reference) — the clinical ground truth our synthetic outcome stands in for.'}
          </p>
        </div>
      )}

      {/* #1775 — clinical grounding for THIS scenario. Verbatim label factors chosen
          by the OUTCOME (what drives stopping, or what gates starting) plus the
          competitive framing. Shown for commercial levers too: the panel used to
          answer a copay-support question with a refusal and stop there, which is the
          "accurate but isolated" complaint #1763 was filed about, still live for half
          the treatment space. The note carries the boundary — the label says nothing
          about the lever — so nothing here reads as a regulatory claim about it.

          The `note` alone is enough to render this block. It is the only thing that
          separates "the label was read and carries nothing bearing on this outcome"
          from "we could not read the label" (#1767), so gating it behind
          considerations-or-competitors removed the disclosure in exactly the case it
          exists for. Every curated brand currently has competitors, which made that
          honesty accidental rather than structural — one brand added without a
          competitor map would have silently dropped it. */}
      {analysis_grounding &&
        (analysis_grounding.label_considerations.length > 0 ||
          analysis_grounding.competitive_context ||
          analysis_grounding.note) && (
          <div className="text-sm">
            <div className="flex items-center gap-1 text-muted-foreground">
              <ClipboardList className="h-3.5 w-3.5" />
              What bears on this analysis
            </div>
            {analysis_grounding.label_considerations.length > 0 && (
              <ul className="mt-1 space-y-1">
                {analysis_grounding.label_considerations.map((consideration) => (
                  <li key={`${consideration.section}-${consideration.references}-${consideration.title}`}>
                    <span className="font-medium">{consideration.title}</span>
                    <span className="text-muted-foreground"> — {consideration.detail}</span>
                    <Badge variant="outline" className="ml-2 align-middle text-xs">
                      label {consideration.references}
                    </Badge>
                  </li>
                ))}
              </ul>
            )}
            {analysis_grounding.competitive_context && (
              <p className="mt-1">{analysis_grounding.competitive_context}</p>
            )}
            {analysis_grounding.note && (
              <p className="mt-1 text-xs text-muted-foreground">{analysis_grounding.note}</p>
            )}
          </div>
        )}

      {/* Public-KG evidence for THIS analysis: the Open Targets indication edge and
          literature whose abstracts were verified to name both entities. A commercial
          treatment lever gets the honest "these sources do not describe this lever"
          state instead of the drug's evidence under an analysis heading. */}
      {causal_evidence && causal_evidence.status !== 'not_requested' && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Microscope className="h-3.5 w-3.5" />
            Evidence for this analysis
          </div>
          {causal_evidence.indication_edge && (
            /* Badge renders a <div>, so this wrapper must not be a <p>. */
            <div className="mt-1">
              {/* The molecule Open Targets actually answered about — verified against
                  the brand's INN — so the sentence names what matched. */}
              <span className="capitalize">{causal_evidence.indication_edge.drug_name}</span>{' '}
              {causal_evidence.indication_edge.predicate === 'treats'
                ? 'is recorded as an approved therapy for'
                : 'is recorded in development for'}{' '}
              <span className="font-medium">{causal_evidence.indication_edge.disease_name}</span>
              <Badge variant="outline" className="ml-2 align-middle text-xs">
                {causal_evidence.indication_edge.max_clinical_stage} · open targets
              </Badge>
            </div>
          )}
          {causal_evidence.citations.length > 0 && (
            <ul className="mt-1 space-y-1">
              {causal_evidence.citations.map((citation) => (
                <li key={citation.pmid}>
                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-start gap-1 text-primary hover:underline"
                  >
                    <span>
                      {citation.title}
                      {citation.journal ? ` — ${citation.journal}` : ''}
                      {citation.pubdate ? ` (${citation.pubdate})` : ''}
                      {` · PMID ${citation.pmid}`}
                    </span>
                    <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  </a>
                  <span className="ml-1 text-xs text-muted-foreground">
                    abstract names {citation.entities_found.join(' + ')}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {causal_evidence.note && (
            <p className="mt-1 text-xs text-muted-foreground">{causal_evidence.note}</p>
          )}
        </div>
      )}

      {/* Seminal real-world evidence — curated, brand-SPECIFIC. Shown first so the
          brand of interest always has a brand-faithful reference, independent of the
          live relevance search below (which can surface a class-comparison paper). */}
      {seminal_real_world_evidence && (
        <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <BookText className="h-3.5 w-3.5" />
            Seminal real-world evidence — {context.drug_name}
            <Badge variant="secondary" className="ml-1 align-middle text-xs">
              curated · brand-specific
            </Badge>
          </div>
          <a
            href={seminal_real_world_evidence.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-start gap-1 text-primary hover:underline"
          >
            <span>
              {seminal_real_world_evidence.title}
              {seminal_real_world_evidence.journal ? ` — ${seminal_real_world_evidence.journal}` : ''}
              {seminal_real_world_evidence.pubdate ? ` (${seminal_real_world_evidence.pubdate})` : ''}
              {` · PMID ${seminal_real_world_evidence.pmid}`}
            </span>
            <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          </a>
        </div>
      )}

      {/* Live real-world-evidence citation from the PubMed relevance search. When a
          curated seminal RWE is shown above, this is labelled "Additional" so a
          broader class-comparison hit reads as supplementary, not the brand's own. */}
      {real_world_evidence ? (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <BookText className="h-3.5 w-3.5" />
            {seminal_real_world_evidence ? 'Additional real-world evidence' : 'Real-world evidence'}
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
          {real_world_evidence.search_term && (
            <p className="mt-1 text-xs text-muted-foreground">
              PubMed query:{' '}
              <code className="rounded bg-muted px-1 py-0.5">
                {real_world_evidence.search_term}
              </code>
            </p>
          )}
        </div>
      ) : (
        !seminal_real_world_evidence && (
          <p className="text-xs text-muted-foreground">
            No real-world evidence yet. RWE comes from post-marketing, observational
            data (registries, claims, routine-care cohorts) — distinct from the
            randomized trial endpoints above — and typically lags a brand&rsquo;s
            approval by years, so recently approved drugs often have none.
          </p>
        )
      )}

      {/* Approved labeling — the on-label gate. The FDA-approved indications keep any
          commercial read inside approved use. Safety detail (boxed warning, warnings)
          is intentionally NOT surfaced here — this panel grounds the commercial signal,
          it is not a safety-labeling reproduction. */}
      {approved_indications && approved_indications.indications.length > 0 && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Stethoscope className="h-3.5 w-3.5" />
            Approved use — the on-label gate
            {sourceChip(approved_indications.source)}
          </div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {approved_indications.indications.map((ind) => (
              <li key={ind}>{ind}</li>
            ))}
          </ul>
          {approved_indications.limitations_of_use && (
            <p className="mt-1 text-xs text-muted-foreground">
              <span className="font-medium">Limitations of use: </span>
              {approved_indications.limitations_of_use}
            </p>
          )}
        </div>
      )}

      {/* Market landscape — competitor products (curated, not FDA-sourced) */}
      {competitor_landscape && competitor_landscape.count > 0 && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Building2 className="h-3.5 w-3.5" />
            Market landscape ({competitor_landscape.count} rival
            {competitor_landscape.count === 1 ? '' : 's'})
            {sourceChip(competitor_landscape.source)}
          </div>
          <div className="mt-1 flex flex-wrap gap-1">
            {competitor_landscape.competitors.map((c) => (
              <Badge key={c} variant="secondary" className="text-xs font-normal">
                {c}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* The synthetic/real honesty boundary — always shown */}
      <p className="border-t pt-3 text-xs text-muted-foreground">{context.honesty_label}</p>
    </div>
  );
}
