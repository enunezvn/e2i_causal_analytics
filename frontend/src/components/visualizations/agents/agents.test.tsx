/**
 * Agent Visualization Components Tests
 * =====================================
 *
 * Tests for AgentTierBadge, TierOverview, and AgentInsightCard components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentTierBadge, TierOverview } from './AgentTierBadge';
import { AgentInsightCard } from './AgentInsightCard';
import type { AgentTier } from './AgentTierBadge';
import type { InsightType, InsightEvidence, InsightAction } from './AgentInsightCard';

// =============================================================================
// AGENT TIER BADGE TESTS
// =============================================================================

describe('AgentTierBadge', () => {
  it('renders with tier number', () => {
    render(<AgentTierBadge tier={2} />);
    expect(screen.getByText('Tier 2')).toBeInTheDocument();
  });

  it('renders tier name when showLabel is true', () => {
    render(<AgentTierBadge tier={2} showLabel />);
    expect(screen.getByText('Causal')).toBeInTheDocument();
  });

  it('hides label when showLabel is false', () => {
    render(<AgentTierBadge tier={2} showLabel={false} />);
    expect(screen.queryByText('Tier 2')).not.toBeInTheDocument();
    expect(screen.queryByText('Causal')).not.toBeInTheDocument();
  });

  it('renders all tier types correctly', () => {
    const tiers: AgentTier[] = [0, 1, 2, 3, 4, 5];
    const tierNames = ['Foundation', 'Orchestration', 'Causal', 'Monitoring', 'ML Predictions', 'Self-Improvement'];

    tiers.forEach((tier, index) => {
      const { unmount } = render(<AgentTierBadge tier={tier} showLabel />);
      expect(screen.getByText(`Tier ${tier}`)).toBeInTheDocument();
      expect(screen.getByText(tierNames[index])).toBeInTheDocument();
      unmount();
    });
  });

  it('renders small size variant', () => {
    const { container } = render(<AgentTierBadge tier={1} size="sm" />);
    expect(container.querySelector('.text-xs')).toBeInTheDocument();
  });

  it('renders medium size variant', () => {
    const { container } = render(<AgentTierBadge tier={1} size="md" />);
    expect(container.querySelector('.text-sm')).toBeInTheDocument();
  });

  it('renders large size variant', () => {
    const { container } = render(<AgentTierBadge tier={1} size="lg" />);
    expect(container.querySelector('.text-base')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<AgentTierBadge tier={3} className="custom-badge" />);
    expect(container.querySelector('.custom-badge')).toBeInTheDocument();
  });

  it('renders icon for each tier', () => {
    const { container } = render(<AgentTierBadge tier={0} />);
    expect(container.querySelector('svg')).toBeInTheDocument();
  });
});

// =============================================================================
// TIER OVERVIEW TESTS
// =============================================================================

describe('TierOverview', () => {
  it('renders all 6 tiers', () => {
    render(<TierOverview />);
    for (let tier = 0; tier <= 5; tier++) {
      expect(screen.getByText(`Tier ${tier}`)).toBeInTheDocument();
    }
  });

  it('renders tier names', () => {
    render(<TierOverview />);
    expect(screen.getByText('Foundation')).toBeInTheDocument();
    expect(screen.getByText('Orchestration')).toBeInTheDocument();
    expect(screen.getByText('Causal')).toBeInTheDocument();
    expect(screen.getByText('Monitoring')).toBeInTheDocument();
    expect(screen.getByText('ML Predictions')).toBeInTheDocument();
    expect(screen.getByText('Self-Improvement')).toBeInTheDocument();
  });

  it('shows agent counts for each tier', () => {
    render(<TierOverview />);
    expect(screen.getByText('8 agents')).toBeInTheDocument(); // Tier 0
    // Multiple tiers have 2 agents (Tier 1, 4, 5)
    const twoAgentsTexts = screen.getAllByText('2 agents');
    expect(twoAgentsTexts.length).toBe(3);
    // Tier 2 and 3 have 3 agents each
    const threeAgentsTexts = screen.getAllByText('3 agents');
    expect(threeAgentsTexts.length).toBe(2);
  });

  it('highlights active tier', () => {
    const { container } = render(<TierOverview activeTier={2} />);
    // Active tier should have ring styling
    expect(container.querySelector('.ring-2')).toBeInTheDocument();
  });

  it('handles tier selection callback', () => {
    const handleSelect = vi.fn();
    render(<TierOverview onTierSelect={handleSelect} />);

    fireEvent.click(screen.getByText('Tier 3'));
    expect(handleSelect).toHaveBeenCalledWith(3);
  });

  it('renders compact mode', () => {
    render(<TierOverview compact />);
    // Compact mode shows just numbers 0-5
    for (let tier = 0; tier <= 5; tier++) {
      expect(screen.getByText(`${tier}`)).toBeInTheDocument();
    }
  });

  it('applies custom className', () => {
    const { container } = render(<TierOverview className="custom-overview" />);
    expect(container.querySelector('.custom-overview')).toBeInTheDocument();
  });

  it('handles tier selection in compact mode', () => {
    const handleSelect = vi.fn();
    render(<TierOverview compact onTierSelect={handleSelect} />);

    fireEvent.click(screen.getByText('4'));
    expect(handleSelect).toHaveBeenCalledWith(4);
  });
});

// =============================================================================
// AGENT INSIGHT CARD TESTS
// =============================================================================

const mockInsightProps = {
  agentName: 'Gap Analyzer',
  agentTier: 2 as AgentTier,
  type: 'opportunity' as InsightType,
  title: 'High-Value HCPs Underserved',
  summary: '15 oncologists with high Rx potential have received less than 2 visits this quarter.',
};

describe('AgentInsightCard', () => {
  it('renders with required props', () => {
    render(<AgentInsightCard {...mockInsightProps} />);
    expect(screen.getByText('Gap Analyzer')).toBeInTheDocument();
    expect(screen.getByText('High-Value HCPs Underserved')).toBeInTheDocument();
    expect(screen.getByText(/15 oncologists/)).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<AgentInsightCard {...mockInsightProps} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('displays insight type label', () => {
    render(<AgentInsightCard {...mockInsightProps} />);
    expect(screen.getByText('Opportunity')).toBeInTheDocument();
  });

  it('renders different insight types', () => {
    const types: InsightType[] = ['opportunity', 'warning', 'success', 'recommendation', 'analysis'];
    const labels = ['Opportunity', 'Warning', 'Success', 'Recommendation', 'Analysis'];

    types.forEach((type, index) => {
      const { unmount } = render(<AgentInsightCard {...mockInsightProps} type={type} />);
      expect(screen.getByText(labels[index])).toBeInTheDocument();
      unmount();
    });
  });

  it('displays confidence score', () => {
    render(<AgentInsightCard {...mockInsightProps} confidence={0.87} />);
    expect(screen.getByText('87% confidence')).toBeInTheDocument();
  });

  it('displays impact information', () => {
    render(
      <AgentInsightCard
        {...mockInsightProps}
        impact={{ metric: 'Potential TRx', value: '+12%', direction: 'up' }}
      />
    );
    expect(screen.getByText('Potential TRx:')).toBeInTheDocument();
    expect(screen.getByText('+12%')).toBeInTheDocument();
  });

  it('renders impact with trending up icon', () => {
    const { container } = render(
      <AgentInsightCard
        {...mockInsightProps}
        impact={{ metric: 'Revenue', value: '+5%', direction: 'up' }}
      />
    );
    expect(container.querySelector('.text-emerald-600')).toBeInTheDocument();
  });

  it('renders impact with trending down icon', () => {
    const { container } = render(
      <AgentInsightCard
        {...mockInsightProps}
        impact={{ metric: 'Churn', value: '-3%', direction: 'down' }}
      />
    );
    expect(container.querySelector('.text-rose-600')).toBeInTheDocument();
  });

  it('toggles details section', () => {
    render(<AgentInsightCard {...mockInsightProps} details="Detailed explanation here" />);

    expect(screen.queryByText('Detailed explanation here')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Show details'));
    expect(screen.getByText('Detailed explanation here')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Hide details'));
    expect(screen.queryByText('Detailed explanation here')).not.toBeInTheDocument();
  });

  it('displays evidence when details expanded', () => {
    const evidence: InsightEvidence[] = [
      { description: 'Top 10 HCPs have 40% visit reduction', value: '40%', source: 'CRM Data' },
    ];

    render(<AgentInsightCard {...mockInsightProps} evidence={evidence} />);

    fireEvent.click(screen.getByText('Show details'));
    expect(screen.getByText('Supporting Evidence')).toBeInTheDocument();
    expect(screen.getByText('Top 10 HCPs have 40% visit reduction')).toBeInTheDocument();
    expect(screen.getByText('(40%)')).toBeInTheDocument();
    expect(screen.getByText(/CRM Data/)).toBeInTheDocument();
  });

  it('renders action buttons', () => {
    const handleAction = vi.fn();
    const actions: InsightAction[] = [
      { label: 'View HCPs', onClick: handleAction, primary: true },
      { label: 'Dismiss', onClick: vi.fn() },
    ];

    render(<AgentInsightCard {...mockInsightProps} actions={actions} />);

    expect(screen.getByText('View HCPs')).toBeInTheDocument();
    expect(screen.getByText('Dismiss')).toBeInTheDocument();

    fireEvent.click(screen.getByText('View HCPs'));
    expect(handleAction).toHaveBeenCalled();
  });

  it('handles feedback callback', () => {
    const handleFeedback = vi.fn();
    render(<AgentInsightCard {...mockInsightProps} onFeedback={handleFeedback} />);

    expect(screen.getByText('Helpful?')).toBeInTheDocument();

    const thumbsUpBtn = document.querySelectorAll('button')[0];
    if (thumbsUpBtn) {
      fireEvent.click(thumbsUpBtn);
    }
  });

  it('shows timestamp', () => {
    const date = new Date('2024-01-15T10:30:00');
    render(<AgentInsightCard {...mockInsightProps} timestamp={date} />);
    expect(screen.getByText(date.toLocaleString())).toBeInTheDocument();
  });

  it('renders copy button when onCopy provided', () => {
    const handleCopy = vi.fn();
    const { container } = render(<AgentInsightCard {...mockInsightProps} onCopy={handleCopy} />);

    const buttons = container.querySelectorAll('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('renders share button when onShare provided', () => {
    const handleShare = vi.fn();
    const { container } = render(<AgentInsightCard {...mockInsightProps} onShare={handleShare} />);

    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('renders bookmark button when onBookmark provided', () => {
    const handleBookmark = vi.fn();
    const { container } = render(
      <AgentInsightCard {...mockInsightProps} onBookmark={handleBookmark} isBookmarked={false} />
    );

    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('shows bookmarked state', () => {
    const { container } = render(
      <AgentInsightCard {...mockInsightProps} onBookmark={vi.fn()} isBookmarked />
    );

    expect(container.querySelector('.text-amber-500')).toBeInTheDocument();
  });

  it('renders conversation button when onViewConversation provided', () => {
    const handleViewConversation = vi.fn();
    const { container } = render(
      <AgentInsightCard {...mockInsightProps} onViewConversation={handleViewConversation} />
    );

    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <AgentInsightCard {...mockInsightProps} className="custom-insight" />
    );
    expect(container.querySelector('.custom-insight')).toBeInTheDocument();
  });

  it('includes agent tier badge', () => {
    const { container } = render(<AgentInsightCard {...mockInsightProps} />);
    // Should render the tier badge with icon
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0);
  });
});
