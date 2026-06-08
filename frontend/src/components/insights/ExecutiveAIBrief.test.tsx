import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ExecutiveAIBrief } from './ExecutiveAIBrief';
import * as useExec from '@/hooks/api/use-executive-insights';
import * as useCog from '@/hooks/api/use-cognitive';

vi.mock('@/hooks/api/use-executive-insights');
vi.mock('@/hooks/api/use-cognitive');

describe('ExecutiveAIBrief — real crystallized insights', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useCog.useCognitiveRAG).mockReturnValue({
      mutate: vi.fn(),
      data: undefined,
      isPending: false,
    } as unknown as ReturnType<typeof useCog.useCognitiveRAG>);
  });

  it('renders crystallized insight narratives when the brand has them', () => {
    vi.mocked(useExec.useExecutiveInsights).mockReturnValue({
      data: [
        {
          insight_id: 'ei_1',
          title: 'Detailing drives TRx',
          narrative: 'Detailing frequency is the strongest driver (b=0.42).',
          brand: 'Remibrutinib',
          crystallized_at: '2026-06-08T00:00:00Z',
          source_count: 3,
          effect_size: 0.42,
          effect_direction: 'positive',
        },
      ],
      isLoading: false,
      isError: false,
      isSuccess: true,
    } as unknown as ReturnType<typeof useExec.useExecutiveInsights>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    expect(screen.getByText('Detailing drives TRx')).toBeInTheDocument();
    expect(
      screen.getByText(/Detailing frequency is the strongest driver/)
    ).toBeInTheDocument();
  });

  it('falls back to the cognitive-RAG path when no crystallized insights exist', () => {
    vi.mocked(useExec.useExecutiveInsights).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      isSuccess: true,
    } as unknown as ReturnType<typeof useExec.useExecutiveInsights>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    // The static fallback heading from SAMPLE_BRIEF is still present.
    expect(screen.getByText('Key Performance Trend')).toBeInTheDocument();
  });
});
