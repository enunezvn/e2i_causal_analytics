/**
 * DiscoveryQuestionSelect
 * =======================
 *
 * Multi-select for the SSOT candidate questions a discovery run would
 * validate. Every question costs minutes of agent time, so the analyst can
 * restrict the run to the treatment → outcome pairs they care about before
 * clicking "Discover causal effects".
 *
 * The options come from `GET /causal/discover-effects/questions` (already
 * labelled with the curated column labels); selection is fully controlled via
 * {@link DiscoveryQuestionSelectProps.selected} / `onChange`, keyed by
 * `questionKey` (lib/discovery-questions) so the page can send exactly the
 * checked rows back as the run's `questions` subset.
 *
 * @module components/causal/DiscoveryQuestionSelect
 */

import { ChevronsUpDown } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { questionKey } from '@/lib/discovery-questions';
import type { DiscoverQuestion } from '@/types/causal';

export interface DiscoveryQuestionSelectProps {
  /** Candidate questions for the active (grain, brand) scope. */
  questions: DiscoverQuestion[];
  /** Keys (see {@link questionKey}) of the questions that will run. */
  selected: string[];
  /** Called with the next selection whenever a checkbox / bulk action fires. */
  onChange: (next: string[]) => void;
  /** Disable the control (a run is in flight, or the scope is switching). */
  disabled?: boolean;
  /** The candidate list is still loading. */
  isLoading?: boolean;
  /** The candidate list could not be loaded (the run then covers every candidate). */
  loadError?: boolean;
  /** Show the brand of each row (all-brands scope lists the same pair per brand). */
  showBrand?: boolean;
}

/**
 * Popover-backed checkbox list of candidate questions with select-all / clear.
 */
export function DiscoveryQuestionSelect({
  questions,
  selected,
  onChange,
  disabled = false,
  isLoading = false,
  loadError = false,
  showBrand = false,
}: DiscoveryQuestionSelectProps) {
  const total = questions.length;
  const selectedSet = new Set(selected);
  const count = questions.filter((q) => selectedSet.has(questionKey(q))).length;

  const toggle = (key: string) => {
    onChange(selectedSet.has(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  };

  let triggerLabel: string;
  if (isLoading) triggerLabel = 'Loading questions…';
  else if (loadError) triggerLabel = 'All questions (list unavailable)';
  else if (total === 0) triggerLabel = 'No candidate questions';
  else if (count === total) triggerLabel = `All ${total} question${total === 1 ? '' : 's'}`;
  else if (count === 0) triggerLabel = 'No questions selected';
  else triggerLabel = `${count} of ${total} questions`;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-label="Questions to discover"
          disabled={disabled || isLoading || loadError || total === 0}
          className="w-64 justify-between font-normal"
        >
          <span className={count === 0 && total > 0 ? 'text-muted-foreground' : ''}>
            {triggerLabel}
          </span>
          <ChevronsUpDown className="h-4 w-4 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-96 p-2">
        <div className="flex items-center justify-between px-2 pb-2 text-xs text-muted-foreground">
          <span>
            {count} of {total} selected
          </span>
          <span className="flex gap-2">
            <button
              type="button"
              className="underline-offset-2 hover:underline disabled:opacity-50"
              disabled={disabled || count === total}
              onClick={() => onChange(questions.map(questionKey))}
            >
              Select all
            </button>
            <button
              type="button"
              className="underline-offset-2 hover:underline disabled:opacity-50"
              disabled={disabled || count === 0}
              onClick={() => onChange([])}
            >
              Clear
            </button>
          </span>
        </div>
        <div className="max-h-72 space-y-1 overflow-y-auto">
          {questions.map((q) => {
            const key = questionKey(q);
            const checkboxId = `discovery-question-${key}`;
            return (
              <div
                key={key}
                className="flex items-center gap-2 rounded-sm px-2 py-1.5 hover:bg-[var(--color-muted)]/50"
              >
                <Checkbox
                  id={checkboxId}
                  checked={selectedSet.has(key)}
                  disabled={disabled}
                  onCheckedChange={() => toggle(key)}
                />
                <Label htmlFor={checkboxId} className="flex-1 cursor-pointer font-normal">
                  <span>{q.treatment_label}</span>{' '}
                  <span className="text-muted-foreground">&rarr;</span>{' '}
                  <span>{q.outcome_label}</span>
                  {showBrand && q.brand && (
                    <span className="ml-2 text-xs text-muted-foreground">({q.brand})</span>
                  )}
                </Label>
              </div>
            );
          })}
        </div>
        <p className="border-t px-2 pt-2 text-xs text-muted-foreground">
          Each question takes the agent a few minutes. Only the checked questions run.
        </p>
      </PopoverContent>
    </Popover>
  );
}

export default DiscoveryQuestionSelect;
