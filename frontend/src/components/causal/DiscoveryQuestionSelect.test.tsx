/**
 * DiscoveryQuestionSelect — the discovery run's question multi-select.
 *
 * Locks: curated labels (never raw columns) on every row, the trigger summary
 * ("All N" / "k of N" / "No questions selected"), toggling a row, select-all /
 * clear, and the disabled states (loading, load error, running).
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DiscoveryQuestionSelect } from './DiscoveryQuestionSelect';
import { questionKey } from '@/lib/discovery-questions';
import type { DiscoverQuestion } from '@/types/causal';

const QUESTIONS: DiscoverQuestion[] = [
  {
    treatment: 'treatment_arm',
    outcome: 'persistent_180d',
    brand: 'Remibrutinib',
    treatment_label: 'Treatment arm',
    outcome_label: 'Persistent at 180d',
    adjustment_set: ['disease_severity'],
  },
  {
    treatment: 'sample_dropped',
    outcome: 'treatment_initiated',
    brand: 'Remibrutinib',
    treatment_label: 'Product samples provided (rep sample drop)',
    outcome_label: 'Treatment initiated',
    adjustment_set: [],
  },
  {
    treatment: 'copay_card_used',
    outcome: 'persistent_180d',
    brand: 'Remibrutinib',
    treatment_label: 'Copay card used',
    outcome_label: 'Persistent at 180d',
    adjustment_set: [],
  },
];
const ALL_KEYS = QUESTIONS.map(questionKey);

describe('DiscoveryQuestionSelect', () => {
  it('summarises an all-selected list and lists every question by its curated label', async () => {
    const user = userEvent.setup();
    render(<DiscoveryQuestionSelect questions={QUESTIONS} selected={ALL_KEYS} onChange={vi.fn()} />);
    const trigger = screen.getByRole('combobox', { name: 'Questions to discover' });
    expect(trigger).toHaveTextContent('All 3 questions');
    await user.click(trigger);
    expect(
      await screen.findByText('Product samples provided (rep sample drop)')
    ).toBeInTheDocument();
    expect(screen.getAllByText('Persistent at 180d')).toHaveLength(2);
    expect(screen.queryByText('sample_dropped')).not.toBeInTheDocument();
    expect(screen.getAllByRole('checkbox')).toHaveLength(3);
    expect(screen.getAllByRole('checkbox').every((c) => c.getAttribute('aria-checked') === 'true')).toBe(
      true
    );
  });

  it('toggles a row off and reports the remaining selection', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<DiscoveryQuestionSelect questions={QUESTIONS} selected={ALL_KEYS} onChange={onChange} />);
    await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
    await user.click(await screen.findByLabelText(/Product samples provided/));
    expect(onChange).toHaveBeenCalledWith([ALL_KEYS[0], ALL_KEYS[2]]);
  });

  it('summarises a partial selection and toggles a row back on', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <DiscoveryQuestionSelect questions={QUESTIONS} selected={[ALL_KEYS[0]]} onChange={onChange} />
    );
    expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
      '1 of 3 questions'
    );
    await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
    await user.click(await screen.findByLabelText(/Copay card used/));
    expect(onChange).toHaveBeenCalledWith([ALL_KEYS[0], ALL_KEYS[2]]);
  });

  it('offers select-all and clear', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <DiscoveryQuestionSelect questions={QUESTIONS} selected={[ALL_KEYS[1]]} onChange={onChange} />
    );
    await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
    await user.click(await screen.findByRole('button', { name: 'Select all' }));
    expect(onChange).toHaveBeenLastCalledWith(ALL_KEYS);
    await user.click(screen.getByRole('button', { name: 'Clear' }));
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it('says so when nothing is selected', () => {
    render(<DiscoveryQuestionSelect questions={QUESTIONS} selected={[]} onChange={vi.fn()} />);
    expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
      'No questions selected'
    );
  });

  it('is disabled while loading, on a load error, and while a run is in flight', () => {
    const { rerender } = render(
      <DiscoveryQuestionSelect questions={[]} selected={[]} onChange={vi.fn()} isLoading />
    );
    const trigger = () => screen.getByRole('combobox', { name: 'Questions to discover' });
    expect(trigger()).toBeDisabled();
    expect(trigger()).toHaveTextContent('Loading questions…');
    rerender(<DiscoveryQuestionSelect questions={[]} selected={[]} onChange={vi.fn()} loadError />);
    expect(trigger()).toBeDisabled();
    expect(trigger()).toHaveTextContent('All questions (list unavailable)');
    rerender(
      <DiscoveryQuestionSelect questions={QUESTIONS} selected={ALL_KEYS} onChange={vi.fn()} disabled />
    );
    expect(trigger()).toBeDisabled();
  });

  it('shows the brand per row only when asked (all-brands scope)', async () => {
    const user = userEvent.setup();
    render(
      <DiscoveryQuestionSelect questions={QUESTIONS} selected={ALL_KEYS} onChange={vi.fn()} showBrand />
    );
    await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
    expect((await screen.findAllByText('(Remibrutinib)')).length).toBe(3);
  });
});
