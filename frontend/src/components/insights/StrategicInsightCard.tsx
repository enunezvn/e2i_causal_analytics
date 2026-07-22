import { ChevronDown, RefreshCw, Sparkles } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import type { GroundingChip, MitigationPlaybook } from '@/types/insights';

interface StrategicInsightCardProps {
  title?: string;
  description?: string;
  insight?: string;
  keyTakeaways?: string[];
  grounding?: GroundingChip[];
  isLoading?: boolean;
  error?: string | null;
  onGenerate?: () => void;
  isFallback?: boolean;
  provenance?: string;
  generatedAt?: string;
  /** Channel 2 of the constraint-aware triage: structural constraints —
   * escalation/investment considerations for data-strategy/platform owners,
   * rendered as a distinct block so recommendations are not diluted. */
  structuralConsiderations?: string | null;
  /** Authored claims-lag mitigation playbook (deterministic, never
   * LM-generated) — rendered beneath the structural channel inside the same
   * collapsible so the constraint statement is actionable (2026-07-22). */
  mitigationPlaybook?: MitigationPlaybook | null;
}

export function StrategicInsightCard({
  title = 'Strategic Interpretation',
  description = 'Agentic read of this view, grounded in the underlying numbers',
  insight,
  keyTakeaways = [],
  grounding = [],
  isLoading,
  error,
  onGenerate,
  isFallback,
  provenance,
  generatedAt,
  structuralConsiderations,
  mitigationPlaybook,
}: StrategicInsightCardProps) {
  return (
    <Card className="border-primary/40">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <CardTitle>{title}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && (
          <div className="space-y-2" aria-label="Generating insight">
            <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
          </div>
        )}
        {!isLoading && error && (
          <div className="space-y-2">
            <p className="text-sm text-destructive">{error}</p>
            {onGenerate && (
              <Button variant="outline" onClick={onGenerate}>
                <Sparkles className="mr-2 h-4 w-4" /> Try again
              </Button>
            )}
          </div>
        )}
        {!isLoading && !error && !insight && onGenerate && (
          <Button variant="outline" onClick={onGenerate}>
            <Sparkles className="mr-2 h-4 w-4" /> Generate strategic insight
          </Button>
        )}
        {!isLoading && !error && insight && (
          <>
            <p className="whitespace-pre-line leading-relaxed">{insight}</p>
            {keyTakeaways.length > 0 && (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {keyTakeaways.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            )}
            {(structuralConsiderations || mitigationPlaybook) && (
              /* Supplementary channel: collapsed by default (frontend review
                 2026-07-22) — escalation/investment context for data-strategy
                 owners, expandable on demand so it never crowds the
                 recommendations above. The authored mitigation playbook
                 renders beneath the LM channel — deterministic, so it shows
                 even when the LM omits channel 2. */
              <Collapsible className="rounded-md border border-amber-500/30 bg-amber-500/5 text-sm">
                <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 p-3 text-left font-medium text-amber-700 dark:text-amber-400">
                  <span>Structural constraints — escalation &amp; investment considerations</span>
                  <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
                </CollapsibleTrigger>
                <CollapsibleContent>
                  {structuralConsiderations && (
                    <p className="whitespace-pre-line px-3 pb-3 text-muted-foreground">
                      {structuralConsiderations}
                    </p>
                  )}
                  {mitigationPlaybook && (
                    <div className="space-y-2 border-t border-amber-500/20 px-3 py-3">
                      <p className="text-xs font-medium uppercase tracking-wide text-amber-700 dark:text-amber-400">
                        Mitigation playbook — faster signal for the claims lag
                      </p>
                      <p className="text-muted-foreground">{mitigationPlaybook.preamble}</p>
                      <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                        {mitigationPlaybook.source_classes.map((sc) => (
                          <li key={sc.name}>
                            <span className="font-medium text-foreground">{sc.name}</span>{' '}
                            <span className="whitespace-nowrap">({sc.latency})</span> — {sc.coverage}
                            {sc.illustrative_vendors.length > 0 && (
                              <>
                                {'. '}Illustrative vendors: {sc.illustrative_vendors.join(', ')}
                              </>
                            )}
                            {sc.status && (
                              <span className="ml-1.5 rounded bg-emerald-500/10 px-1.5 py-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                                {sc.status}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                      <p className="text-xs italic text-muted-foreground">
                        {mitigationPlaybook.vendor_note}
                      </p>
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>
            )}
            {grounding.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {grounding.map((c, i) => (
                  <span
                    key={i}
                    className="rounded-full border px-2 py-0.5 text-xs text-muted-foreground"
                  >
                    <span className="font-medium">{c.label}</span>: {c.value}
                  </span>
                ))}
              </div>
            )}
            <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-muted-foreground">
              {isFallback && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  factual summary — LLM unavailable
                </span>
              )}
              {provenance && <span>{provenance}</span>}
              {generatedAt && <span>· {new Date(generatedAt).toLocaleString()}</span>}
              {onGenerate && (
                <Button variant="ghost" size="sm" className="ml-auto h-7" onClick={onGenerate}>
                  <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> Regenerate
                </Button>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
