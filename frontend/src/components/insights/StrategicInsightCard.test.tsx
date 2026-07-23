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

  // Gating (2026-07-23): the generate/regenerate action must be disabled until a
  // discovery run has produced effects to interpret — otherwise a user can
  // synthesize an interpretation of an empty leaderboard.
  it('disables the Generate button and shows the hint when disabled', async () => {
    const onGenerate = vi.fn();
    render(
      <StrategicInsightCard
        onGenerate={onGenerate}
        disabled
        disabledHint="Run Discover Causal Effects first."
      />
    );
    const btn = screen.getByRole('button', { name: /generate strategic insight/i });
    expect(btn).toBeDisabled();
    expect(screen.getByText(/run discover causal effects first/i)).toBeInTheDocument();
    await userEvent.click(btn);
    expect(onGenerate).not.toHaveBeenCalled();
  });

  it('disables the Regenerate button when disabled', () => {
    render(
      <StrategicInsightCard insight="Adherence drives NRx." onGenerate={vi.fn()} disabled />
    );
    expect(screen.getByRole('button', { name: /regenerate/i })).toBeDisabled();
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

  // Item 2b (2026-07-22): the authored claims-lag mitigation playbook renders
  // inside the same collapsible, beneath the LM channel — deterministic and
  // therefore present even when the LM omits channel 2.
  describe('mitigation playbook', () => {
    const playbook = {
      preamble: 'Faster adjudicated (closed) claims are not achievable.',
      vendor_note: 'Illustrative examples — not vetted or contracted suppliers.',
      source_classes: [
        {
          name: 'Open (pre-adjudicated) claims',
          latency: '1-7 days from service',
          coverage: 'partial capture — trend, not levels',
          illustrative_vendors: ['IQVIA', 'Symphony Health'],
          status: null,
        },
        {
          name: 'Completion-factor nowcast on closed claims',
          latency: 'immediate (modeled)',
          coverage: 'models the as-of-today under-count',
          illustrative_vendors: [],
          status: 'already live in this platform',
        },
      ],
    };

    it('renders the playbook inside the collapsible after expanding', async () => {
      render(
        <StrategicInsightCard
          insight="Adherence drives NRx."
          structuralConsiderations="Claims lag gates outcome metrics."
          mitigationPlaybook={playbook}
        />
      );
      expect(screen.queryByText(/faster adjudicated/i)).not.toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: /structural constraints/i }));
      expect(screen.getByText(/mitigation playbook — faster signal/i)).toBeInTheDocument();
      expect(screen.getByText(/faster adjudicated \(closed\) claims are not achievable/i)).toBeInTheDocument();
      expect(screen.getByText(/open \(pre-adjudicated\) claims/i)).toBeInTheDocument();
      expect(screen.getByText(/IQVIA, Symphony Health/)).toBeInTheDocument();
      expect(screen.getByText(/already live in this platform/i)).toBeInTheDocument();
      expect(screen.getByText(/not vetted or contracted suppliers/i)).toBeInTheDocument();
    });

    it('renders the collapsible for the playbook even when the LM structural channel is empty', async () => {
      render(
        <StrategicInsightCard
          insight="Adherence drives NRx."
          structuralConsiderations=""
          mitigationPlaybook={playbook}
        />
      );
      const trigger = screen.getByRole('button', { name: /structural constraints/i });
      expect(trigger).toHaveAttribute('data-state', 'closed');
      await userEvent.click(trigger);
      expect(screen.getByText(/mitigation playbook — faster signal/i)).toBeInTheDocument();
    });
  });
});
