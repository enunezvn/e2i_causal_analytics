import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StrategicInsightCard } from './StrategicInsightCard';

describe('StrategicInsightCard', () => {
  it('shows a Generate button when empty and onGenerate is provided', async () => {
    const onGenerate = vi.fn();
    render(<StrategicInsightCard onGenerate={onGenerate} />);
    const btn = screen.getByRole('button', { name: /generate strategic insight/i });
    await userEvent.click(btn);
    expect(onGenerate).toHaveBeenCalledOnce();
  });

  it('renders the narrative, grounding chips, and a fallback badge', () => {
    render(
      <StrategicInsightCard
        insight="Adherence drives NRx."
        isFallback
        grounding={[{ label: 'Nodes', value: '10' }]}
      />
    );
    expect(screen.getByText(/adherence drives nrx/i)).toBeInTheDocument();
    expect(screen.getByText('Nodes')).toBeInTheDocument();
    expect(screen.getByText(/factual summary/i)).toBeInTheDocument();
  });

  it('shows a loading skeleton', () => {
    const { container } = render(<StrategicInsightCard isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeTruthy();
  });

  it('shows an error message', () => {
    render(<StrategicInsightCard error="boom" />);
    expect(screen.getByText(/boom/i)).toBeInTheDocument();
  });

  it('shows a Regenerate button when an insight is present and onGenerate is provided', async () => {
    const onGenerate = vi.fn();
    render(<StrategicInsightCard insight="Adherence drives NRx." onGenerate={onGenerate} />);
    const btn = screen.getByRole('button', { name: /regenerate/i });
    await userEvent.click(btn);
    expect(onGenerate).toHaveBeenCalledOnce();
  });

  it('hides the Regenerate button when onGenerate is not provided', () => {
    render(<StrategicInsightCard insight="Adherence drives NRx." />);
    expect(screen.queryByRole('button', { name: /regenerate/i })).not.toBeInTheDocument();
  });

  it('shows a Try again button on error when onGenerate is provided', async () => {
    const onGenerate = vi.fn();
    render(<StrategicInsightCard error="boom" onGenerate={onGenerate} />);
    const btn = screen.getByRole('button', { name: /try again/i });
    await userEvent.click(btn);
    expect(onGenerate).toHaveBeenCalledOnce();
  });

  // Structural constraints are a supplementary channel — collapsed by default,
  // expandable on click (frontend review 2026-07-22).
  describe('structural considerations collapsible', () => {
    it('renders the header collapsed by default — the content is hidden until expanded', async () => {
      render(
        <StrategicInsightCard
          insight="Adherence drives NRx."
          structuralConsiderations="Claims lag gates outcome metrics."
        />
      );
      const trigger = screen.getByRole('button', {
        name: /structural constraints — escalation & investment considerations/i,
      });
      expect(trigger).toHaveAttribute('data-state', 'closed');
      expect(screen.queryByText(/claims lag gates outcome metrics/i)).not.toBeInTheDocument();

      await userEvent.click(trigger);
      expect(screen.getByText(/claims lag gates outcome metrics/i)).toBeInTheDocument();

      await userEvent.click(trigger);
      expect(screen.queryByText(/claims lag gates outcome metrics/i)).not.toBeInTheDocument();
    });

    it('renders no structural block at all when the channel is empty', () => {
      render(<StrategicInsightCard insight="Adherence drives NRx." structuralConsiderations="" />);
      expect(screen.queryByText(/structural constraints/i)).not.toBeInTheDocument();
    });
  });
});
