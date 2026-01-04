/**
 * ExperimentCard Component Tests
 * ==============================
 *
 * Tests for the ExperimentCard and ExperimentListItem components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExperimentCard, ExperimentListItem, type Experiment, type ExperimentStatus } from './ExperimentCard';

// Sample experiment data
const createMockExperiment = (overrides: Partial<Experiment> = {}): Experiment => ({
  id: 'exp-001',
  name: 'HCP Engagement Test',
  description: 'Testing new HCP engagement strategy for Remibrutinib',
  status: 'running',
  start_date: '2026-01-01T00:00:00Z',
  end_date: '2026-01-31T00:00:00Z',
  treatment: {
    name: 'Treatment A',
    sample_size: 5000,
    conversion_rate: 0.12,
    mean_outcome: 45.5,
  },
  control: {
    name: 'Control',
    sample_size: 5000,
    conversion_rate: 0.10,
    mean_outcome: 42.0,
  },
  primary_metric: 'TRx Conversion Rate',
  result: {
    lift_percentage: 20.0,
    confidence_level: 95,
    p_value: 0.02,
    is_significant: true,
    winner: 'treatment',
  },
  brand: 'Remibrutinib',
  tags: ['Q1-2026', 'HCP', 'Priority'],
  ...overrides,
});

describe('ExperimentCard', () => {
  describe('Rendering', () => {
    it('renders experiment name and description', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('HCP Engagement Test')).toBeInTheDocument();
      expect(screen.getByText(/Testing new HCP engagement strategy/)).toBeInTheDocument();
    });

    it('renders status badge with correct label', () => {
      const experiment = createMockExperiment({ status: 'running' });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('Running')).toBeInTheDocument();
    });

    it('renders all status types correctly', () => {
      const statuses: ExperimentStatus[] = ['draft', 'running', 'paused', 'completed', 'failed', 'cancelled'];
      const labels = ['Draft', 'Running', 'Paused', 'Completed', 'Failed', 'Cancelled'];

      statuses.forEach((status, index) => {
        const { unmount } = render(<ExperimentCard experiment={createMockExperiment({ status })} />);
        expect(screen.getByText(labels[index])).toBeInTheDocument();
        unmount();
      });
    });

    it('displays total sample size', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      // 5000 + 5000 = 10,000
      expect(screen.getByText('10,000')).toBeInTheDocument();
    });

    it('displays primary metric', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('TRx Conversion Rate')).toBeInTheDocument();
    });

    it('renders brand tag when provided', () => {
      const experiment = createMockExperiment({ brand: 'Fabhalta' });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('Fabhalta')).toBeInTheDocument();
    });

    it('renders experiment tags', () => {
      const experiment = createMockExperiment({ tags: ['Q1-2026', 'HCP', 'Priority'] });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('Q1-2026')).toBeInTheDocument();
      expect(screen.getByText('HCP')).toBeInTheDocument();
      expect(screen.getByText('Priority')).toBeInTheDocument();
    });

    it('renders date range', () => {
      const experiment = createMockExperiment({
        start_date: '2026-01-01T00:00:00Z',
        end_date: '2026-01-31T00:00:00Z',
      });
      render(<ExperimentCard experiment={experiment} />);

      // Check for date formatting - dates will be locale-dependent, so just verify structure
      // There should be two date spans with a hyphen between them
      expect(screen.getByText('-')).toBeInTheDocument();
      // The container should have calendar icon
      const dateContainer = screen.getByText('-').parentElement;
      expect(dateContainer).toBeInTheDocument();
    });
  });

  describe('Variant Comparison', () => {
    it('displays treatment and control sections', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('Treatment')).toBeInTheDocument();
      expect(screen.getByText('Control')).toBeInTheDocument();
    });

    it('shows sample sizes for both variants', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      // Both variants have sample_size: 5000
      const sampleSizes = screen.getAllByText('5,000');
      expect(sampleSizes.length).toBeGreaterThanOrEqual(2);
    });

    it('shows conversion rates when available', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} />);

      // Treatment: 12%, Control: 10%
      expect(screen.getByText('12.0% conv.')).toBeInTheDocument();
      expect(screen.getByText('10.0% conv.')).toBeInTheDocument();
    });
  });

  describe('Results Display', () => {
    it('displays lift percentage with sign', () => {
      const experiment = createMockExperiment({
        result: {
          lift_percentage: 20.0,
          confidence_level: 95,
          p_value: 0.02,
          is_significant: true,
          winner: 'treatment',
        },
      });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('+20.0%')).toBeInTheDocument();
    });

    it('displays negative lift correctly', () => {
      const experiment = createMockExperiment({
        result: {
          lift_percentage: -15.5,
          confidence_level: 95,
          p_value: 0.03,
          is_significant: true,
          winner: 'control',
        },
      });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('-15.5%')).toBeInTheDocument();
    });

    it('shows significance badge when result is significant', () => {
      const experiment = createMockExperiment({
        result: {
          lift_percentage: 20.0,
          confidence_level: 95,
          p_value: 0.02,
          is_significant: true,
          winner: 'treatment',
        },
      });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('Significant')).toBeInTheDocument();
    });

    it('does not show significance badge when not significant', () => {
      const experiment = createMockExperiment({
        result: {
          lift_percentage: 2.0,
          confidence_level: 80,
          p_value: 0.15,
          is_significant: false,
          winner: 'none',
        },
      });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.queryByText('Significant')).not.toBeInTheDocument();
    });

    it('displays confidence level', () => {
      const experiment = createMockExperiment({
        result: {
          lift_percentage: 20.0,
          confidence_level: 95,
          p_value: 0.02,
          is_significant: true,
          winner: 'treatment',
        },
      });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.getByText('95% confidence')).toBeInTheDocument();
    });

    it('does not show results section when no result', () => {
      const experiment = createMockExperiment({ result: undefined });
      render(<ExperimentCard experiment={experiment} />);

      expect(screen.queryByText(/confidence$/)).not.toBeInTheDocument();
    });
  });

  describe('Compact Mode', () => {
    it('hides description in compact mode', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} compact />);

      expect(screen.queryByText(/Testing new HCP engagement strategy/)).not.toBeInTheDocument();
    });

    it('hides tags in compact mode', () => {
      const experiment = createMockExperiment({ tags: ['Q1-2026', 'Priority'] });
      render(<ExperimentCard experiment={experiment} compact />);

      expect(screen.queryByText('Q1-2026')).not.toBeInTheDocument();
      expect(screen.queryByText('Priority')).not.toBeInTheDocument();
    });

    it('hides variant comparison in compact mode', () => {
      const experiment = createMockExperiment();
      render(<ExperimentCard experiment={experiment} compact />);

      expect(screen.queryByText('Treatment')).not.toBeInTheDocument();
      expect(screen.queryByText('Control')).not.toBeInTheDocument();
    });
  });

  describe('Interactions', () => {
    it('calls onClick when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      const experiment = createMockExperiment();

      render(<ExperimentCard experiment={experiment} onClick={handleClick} />);

      await user.click(screen.getByText('HCP Engagement Test'));
      expect(handleClick).toHaveBeenCalledWith(experiment);
    });

    it('has cursor-pointer when onClick is provided', () => {
      const experiment = createMockExperiment();
      const { container } = render(<ExperimentCard experiment={experiment} onClick={() => {}} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-pointer');
    });

    it('does not have cursor-pointer when onClick is not provided', () => {
      const experiment = createMockExperiment();
      const { container } = render(<ExperimentCard experiment={experiment} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).not.toContain('cursor-pointer');
    });
  });

  describe('Custom className', () => {
    it('applies custom className', () => {
      const experiment = createMockExperiment();
      const { container } = render(<ExperimentCard experiment={experiment} className="custom-class" />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('custom-class');
    });
  });
});

describe('ExperimentListItem', () => {
  it('renders experiment name', () => {
    const experiment = createMockExperiment();
    render(<ExperimentListItem experiment={experiment} />);

    expect(screen.getByText('HCP Engagement Test')).toBeInTheDocument();
  });

  it('renders primary metric', () => {
    const experiment = createMockExperiment();
    render(<ExperimentListItem experiment={experiment} />);

    expect(screen.getByText('TRx Conversion Rate')).toBeInTheDocument();
  });

  it('renders status badge', () => {
    const experiment = createMockExperiment({ status: 'completed' });
    render(<ExperimentListItem experiment={experiment} />);

    expect(screen.getByText('Completed')).toBeInTheDocument();
  });

  it('shows result indicator when result exists', () => {
    const experiment = createMockExperiment({
      result: {
        lift_percentage: 15.5,
        confidence_level: 95,
        p_value: 0.02,
        is_significant: true,
        winner: 'treatment',
      },
    });
    render(<ExperimentListItem experiment={experiment} />);

    expect(screen.getByText('+15.5%')).toBeInTheDocument();
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();
    const experiment = createMockExperiment();

    render(<ExperimentListItem experiment={experiment} onClick={handleClick} />);

    await user.click(screen.getByText('HCP Engagement Test'));
    expect(handleClick).toHaveBeenCalledWith(experiment);
  });

  it('has cursor-pointer when onClick is provided', () => {
    const experiment = createMockExperiment();
    const { container } = render(<ExperimentListItem experiment={experiment} onClick={() => {}} />);

    const item = container.firstChild as HTMLElement;
    expect(item.className).toContain('cursor-pointer');
  });
});
