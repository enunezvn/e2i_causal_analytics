/**
 * CausalPipeline — the 5-stage causal workflow (Frame → Identify → Estimate →
 * Refute → Act), visually kin to visualizations/QueryProcessingFlow. Clicking
 * a stage expands a two-layer panel: plain language + "For analysts"
 * (Radix Collapsible).
 */
import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { PIPELINE_STAGES } from './content';

export function CausalPipeline() {
  const [openId, setOpenId] = useState<string | null>(null);
  const openStage = PIPELINE_STAGES.find((s) => s.id === openId);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        {PIPELINE_STAGES.map((stage, i) => (
          <div key={stage.id} className="flex items-center gap-2">
            {i > 0 && (
              <ChevronRight className="h-4 w-4 text-[var(--color-muted-foreground)]" aria-hidden="true" />
            )}
            <button
              type="button"
              onClick={() => setOpenId(openId === stage.id ? null : stage.id)}
              aria-expanded={openId === stage.id}
              className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors motion-reduce:transition-none ${
                openId === stage.id
                  ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                  : 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-foreground)] hover:border-[var(--color-primary)]/50'
              }`}
            >
              {stage.name}
            </button>
          </div>
        ))}
      </div>

      {openStage && (
        <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
          <p className="text-sm leading-6 text-[var(--color-foreground)]">{openStage.plain}</p>
          <Collapsible className="mt-3">
            <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-[var(--color-primary)] hover:underline">
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              For analysts
            </CollapsibleTrigger>
            <CollapsibleContent>
              <p className="mt-2 border-l-2 border-[var(--color-border)] pl-3 text-xs leading-5 text-[var(--color-muted-foreground)]">
                {openStage.analyst}
              </p>
            </CollapsibleContent>
          </Collapsible>
        </div>
      )}
    </div>
  );
}
