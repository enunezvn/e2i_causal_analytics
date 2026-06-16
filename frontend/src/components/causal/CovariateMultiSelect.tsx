/**
 * CovariateMultiSelect
 * ====================
 *
 * Multi-select control for choosing covariates (controls) on the Causal
 * Discovery page. Renders a popover-triggered checkbox list so several
 * covariates can be toggled at once. Keeps the page lean by encapsulating the
 * popover + checkbox plumbing here.
 *
 * The list of options is supplied by the caller (already filtered to exclude
 * the currently-selected treatment / outcome variables). Selection state is
 * fully controlled via {@link CovariateMultiSelectProps.selected} /
 * {@link CovariateMultiSelectProps.onChange}.
 *
 * @module components/causal/CovariateMultiSelect
 */

import { ChevronsUpDown } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';

export interface CovariateMultiSelectProps {
  /** Selectable covariate names (already excludes treatment / outcome). */
  options: string[];
  /** Currently-selected covariate names. */
  selected: string[];
  /** Called with the next selection whenever a checkbox is toggled. */
  onChange: (next: string[]) => void;
  /** Disable the control (e.g. while candidate variables are loading). */
  disabled?: boolean;
}

/**
 * Popover-backed multi-select for covariates.
 */
export function CovariateMultiSelect({
  options,
  selected,
  onChange,
  disabled = false,
}: CovariateMultiSelectProps) {
  const toggle = (option: string) => {
    const next = selected.includes(option)
      ? selected.filter((value) => value !== option)
      : [...selected, option];
    onChange(next);
  };

  const triggerLabel =
    selected.length > 0
      ? `${selected.length} selected`
      : 'Select covariates';

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-label="Select covariates"
          disabled={disabled}
          className="w-full justify-between font-normal"
        >
          <span
            className={
              selected.length > 0 ? '' : 'text-muted-foreground'
            }
          >
            {triggerLabel}
          </span>
          <ChevronsUpDown className="h-4 w-4 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 p-2">
        {options.length === 0 ? (
          <p className="px-2 py-1.5 text-sm text-muted-foreground italic">
            No covariates available.
          </p>
        ) : (
          <div className="max-h-60 space-y-1 overflow-y-auto">
            {options.map((option) => {
              const checkboxId = `covariate-${option}`;
              const isChecked = selected.includes(option);
              return (
                <div
                  key={option}
                  className="flex items-center gap-2 rounded-sm px-2 py-1.5 hover:bg-[var(--color-muted)]/50"
                >
                  <Checkbox
                    id={checkboxId}
                    checked={isChecked}
                    disabled={disabled}
                    onCheckedChange={() => toggle(option)}
                  />
                  <Label
                    htmlFor={checkboxId}
                    className="flex-1 cursor-pointer font-normal"
                  >
                    {option}
                  </Label>
                </div>
              );
            })}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

export default CovariateMultiSelect;
