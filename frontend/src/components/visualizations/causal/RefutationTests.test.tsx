/**
 * RefutationTests — inverted-p-value labeling (Fix A)
 * ===================================================
 *
 * Focused regression tests for the review finding that the refutation card read
 * like an ordinary significance test: a non-significant p-value (e.g. 0.28) and a
 * red "Change" cell on a PASSING placebo row both looked like failures.
 *
 * Refutation p-values are INVERTED vs significance — a HIGHER (non-significant) p
 * means the estimate SURVIVED the challenge = PASS. The "Change" magnitude is
 * expected to be large for a passing placebo (the effect could not be reproduced
 * from noise), so it must never be colored as a failure while the row passes.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { RefutationTests, type RefutationResult } from './RefutationTests';

// A passing placebo with a large change + a non-significant p (mirrors the live
// Remibrutinib card: p≈0.28, big drop to ~0, yet PASS), a passing random-common-
// cause with a small change, and a genuinely failing test.
const results: RefutationResult[] = [
  {
    id: 'rcc',
    method: 'random_common_cause',
    originalEstimate: 0.15,
    refutedEstimate: 0.14,
    pValue: 0.72,
    passed: true,
  },
  {
    id: 'plc',
    method: 'placebo_treatment',
    originalEstimate: 0.15,
    refutedEstimate: 0.01, // -93.3% change — expected & GOOD for a placebo
    pValue: 0.28, // non-significant → placebo could not reproduce the effect → PASS
    passed: true,
  },
  {
    id: 'boot',
    method: 'bootstrap',
    originalEstimate: 0.15,
    refutedEstimate: 0.05, // -66.7% change
    pValue: 0.02,
    passed: false,
  },
];

describe('RefutationTests — inverted refutation p-value semantics (Fix A)', () => {
  it('labels the p column as robustness, not an ordinary significance p-value', () => {
    render(<RefutationTests results={results} />);
    expect(screen.getByText('Robustness p')).toBeInTheDocument();
    // The bare "P-value" significance header must be gone from this card.
    expect(screen.queryByText('P-value')).toBeNull();
  });

  it('frames the card as robustness checks and, per-check accurately, states a non-significant p is not a failure', () => {
    render(<RefutationTests results={results} />);
    expect(screen.getByText(/robustness checks, not significance tests/i)).toBeInTheDocument();
    // Accuracy guard: the description must NOT imply the p-rule is universal — it must
    // acknowledge the delta-based random-common-cause check (which ignores the p-value).
    expect(screen.getByText(/random-common-cause\s+passes when the estimate barely moves/i)).toBeInTheDocument();
    expect(screen.getByText(/non-significant p-value is\s+therefore not a failure/i)).toBeInTheDocument();
    // The misleading significance-threshold framing must be gone.
    expect(screen.queryByText(/threshold:/i)).toBeNull();
  });

  it('exposes an accessible affordance describing how to read refutation p-values', () => {
    render(<RefutationTests results={results} />);
    expect(
      screen.getByLabelText(/how to read refutation p-values/i)
    ).toBeInTheDocument();
  });

  it('does NOT color a passing test change cell as a failure', () => {
    render(<RefutationTests results={results} />);
    const passingPlaceboChange = screen.getByText('-93.3%');
    expect(passingPlaceboChange).not.toHaveClass('text-[var(--color-destructive)]');
    expect(passingPlaceboChange).toHaveClass('text-[var(--color-muted-foreground)]');
  });

  it('colors a FAILING test change cell as destructive', () => {
    render(<RefutationTests results={results} />);
    const failingChange = screen.getByText('-66.7%');
    expect(failingChange).toHaveClass('text-[var(--color-destructive)]');
  });

  it('notes the comparison chart shows only refutation tests that actually ran', () => {
    render(<RefutationTests results={results} />);
    expect(
      screen.getByText(/across all refutation tests that ran/i)
    ).toBeInTheDocument();
  });
});
