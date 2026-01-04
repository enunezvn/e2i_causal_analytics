/**
 * Dashboard Visualization Components Tests
 * ========================================
 *
 * Tests for KPICard, StatusBadge, ProgressRing, and AlertCard components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { KPICard } from './KPICard';
import { StatusBadge, StatusDot } from './StatusBadge';
import { ProgressRing, ProgressRingGroup } from './ProgressRing';
import { AlertCard, AlertList } from './AlertCard';

// =============================================================================
// KPI CARD TESTS
// =============================================================================

describe('KPICard', () => {
  it('renders with title and value', () => {
    render(<KPICard title="Total Revenue" value={125000} />);
    expect(screen.getByText('Total Revenue')).toBeInTheDocument();
    expect(screen.getByText('125.0K')).toBeInTheDocument();
  });

  it('formats large numbers correctly', () => {
    render(<KPICard title="Users" value={1500000} />);
    expect(screen.getByText('1.5M')).toBeInTheDocument();
  });

  it('displays prefix and unit', () => {
    render(<KPICard title="Revenue" value={50000} prefix="$" unit="K" />);
    expect(screen.getByText('$50.0KK')).toBeInTheDocument();
  });

  it('shows trend with previousValue', () => {
    render(<KPICard title="Sales" value={110} previousValue={100} />);
    // Should show +10% trend
    expect(screen.getByText('+10.0%')).toBeInTheDocument();
  });

  it('shows negative trend correctly', () => {
    render(<KPICard title="Sales" value={90} previousValue={100} />);
    expect(screen.getByText('-10.0%')).toBeInTheDocument();
  });

  it('shows target progress when target provided', () => {
    render(<KPICard title="Goal" value={75} target={100} showTarget />);
    expect(screen.getByText(/Target:/)).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('handles loading state', () => {
    const { container } = render(<KPICard title="Loading" value={0} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn();
    render(<KPICard title="Clickable" value={100} onClick={handleClick} />);
    fireEvent.click(screen.getByText('Clickable').closest('div')!);
    expect(handleClick).toHaveBeenCalled();
  });

  it('renders different size variants', () => {
    const { rerender, container } = render(<KPICard title="Small" value={100} size="sm" />);
    expect(container.querySelector('.p-3')).toBeInTheDocument();

    rerender(<KPICard title="Large" value={100} size="lg" />);
    expect(container.querySelector('.p-5')).toBeInTheDocument();
  });

  it('shows description tooltip icon', () => {
    render(<KPICard title="With Info" value={100} description="Some description" />);
    // Info icon should be rendered (SVG element)
    expect(document.querySelector('svg')).toBeInTheDocument();
  });

  it('renders string values correctly', () => {
    render(<KPICard title="Status" value="Active" />);
    expect(screen.getByText('Active')).toBeInTheDocument();
  });

  it('renders with warning status color', () => {
    const { container } = render(<KPICard title="Warning KPI" value={50} status="warning" />);
    // Should have amber-500 border-left color
    expect(container.querySelector('.border-l-amber-500')).toBeInTheDocument();
  });

  it('renders with critical status color', () => {
    const { container } = render(<KPICard title="Critical KPI" value={10} status="critical" />);
    // Should have rose-500 border-left color
    expect(container.querySelector('.border-l-rose-500')).toBeInTheDocument();
  });

  it('renders with healthy status color', () => {
    const { container } = render(<KPICard title="Healthy KPI" value={95} status="healthy" />);
    // Should have emerald-500 border-left color
    expect(container.querySelector('.border-l-emerald-500')).toBeInTheDocument();
  });

  it('formats decimal values correctly', () => {
    render(<KPICard title="Rate" value={0.75} />);
    // Decimal value should show 0.75
    expect(screen.getByText('0.75')).toBeInTheDocument();
  });

  it('formats small decimal values correctly', () => {
    render(<KPICard title="Precision" value={0.8523} />);
    // Should show 0.85 (2 decimal places)
    expect(screen.getByText('0.85')).toBeInTheDocument();
  });
});

// =============================================================================
// STATUS BADGE TESTS
// =============================================================================

describe('StatusBadge', () => {
  it('renders with correct status label', () => {
    render(<StatusBadge status="healthy" />);
    expect(screen.getByText('Healthy')).toBeInTheDocument();
  });

  it('uses custom label when provided', () => {
    render(<StatusBadge status="warning" label="Drift Detected" />);
    expect(screen.getByText('Drift Detected')).toBeInTheDocument();
    expect(screen.queryByText('Warning')).not.toBeInTheDocument();
  });

  it('renders different statuses correctly', () => {
    const statuses = ['healthy', 'success', 'warning', 'error', 'critical', 'pending', 'loading', 'unknown'] as const;

    statuses.forEach((status) => {
      const { unmount } = render(<StatusBadge status={status} />);
      expect(document.body.querySelector('span')).toBeInTheDocument();
      unmount();
    });
  });

  it('hides icon when showIcon is false', () => {
    const { container } = render(<StatusBadge status="healthy" showIcon={false} />);
    expect(container.querySelectorAll('svg').length).toBe(0);
  });

  it('renders different size variants', () => {
    const { rerender, container } = render(<StatusBadge status="healthy" size="sm" />);
    expect(container.querySelector('.text-xs')).toBeInTheDocument();

    rerender(<StatusBadge status="healthy" size="lg" />);
    expect(container.querySelector('.text-base')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<StatusBadge status="healthy" className="custom-class" />);
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });

  it('renders loading status with animation', () => {
    const { container } = render(<StatusBadge status="loading" />);
    expect(container.querySelector('.animate-spin')).toBeInTheDocument();
  });
});

describe('StatusDot', () => {
  it('renders as a simple dot', () => {
    const { container } = render(<StatusDot status="healthy" />);
    expect(container.querySelector('.rounded-full')).toBeInTheDocument();
  });

  it('shows pulse animation when specified', () => {
    const { container } = render(<StatusDot status="healthy" pulse />);
    expect(container.querySelector('.animate-ping')).toBeInTheDocument();
  });

  it('renders different sizes', () => {
    const { rerender, container } = render(<StatusDot status="healthy" size="sm" />);
    expect(container.querySelector('.h-1\\.5')).toBeInTheDocument();

    rerender(<StatusDot status="healthy" size="lg" />);
    expect(container.querySelector('.h-3')).toBeInTheDocument();
  });
});

// =============================================================================
// PROGRESS RING TESTS
// =============================================================================

describe('ProgressRing', () => {
  it('renders with value', () => {
    render(<ProgressRing value={75} />);
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('clamps values to max', () => {
    render(<ProgressRing value={150} max={100} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('handles custom max values', () => {
    render(<ProgressRing value={50} max={200} />);
    expect(screen.getByText('25%')).toBeInTheDocument();
  });

  it('shows custom label', () => {
    render(<ProgressRing value={75} label="3/4" />);
    expect(screen.getByText('3/4')).toBeInTheDocument();
  });

  it('hides label when showLabel is false', () => {
    render(<ProgressRing value={75} showLabel={false} />);
    expect(screen.queryByText('75%')).not.toBeInTheDocument();
  });

  it('renders children as center content', () => {
    render(
      <ProgressRing value={50}>
        <span>Custom Content</span>
      </ProgressRing>
    );
    expect(screen.getByText('Custom Content')).toBeInTheDocument();
  });

  it('renders loading skeleton', () => {
    const { container } = render(<ProgressRing value={50} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('renders SVG elements', () => {
    const { container } = render(<ProgressRing value={50} />);
    expect(container.querySelector('svg')).toBeInTheDocument();
    expect(container.querySelectorAll('circle').length).toBe(2); // track + progress
  });

  it('applies custom size', () => {
    const { container } = render(<ProgressRing value={50} size={120} />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('width')).toBe('120');
    expect(svg?.getAttribute('height')).toBe('120');
  });
});

describe('ProgressRingGroup', () => {
  it('renders multiple rings', () => {
    const items = [
      { label: 'Accuracy', value: 92 },
      { label: 'Precision', value: 88 },
      { label: 'Recall', value: 75 },
    ];
    render(<ProgressRingGroup items={items} />);

    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Precision')).toBeInTheDocument();
    expect(screen.getByText('Recall')).toBeInTheDocument();
  });

  it('shows values for each ring', () => {
    const items = [
      { label: 'A', value: 50 },
      { label: 'B', value: 75 },
    ];
    render(<ProgressRingGroup items={items} />);

    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
  });
});

// =============================================================================
// ALERT CARD TESTS
// =============================================================================

describe('AlertCard', () => {
  it('renders with title and severity', () => {
    render(<AlertCard severity="warning" title="Drift Detected" />);
    expect(screen.getByText('Drift Detected')).toBeInTheDocument();
  });

  it('renders message when provided', () => {
    render(
      <AlertCard
        severity="error"
        title="Error"
        message="Something went wrong with the model."
      />
    );
    expect(screen.getByText('Something went wrong with the model.')).toBeInTheDocument();
  });

  it('renders source badge', () => {
    render(
      <AlertCard
        severity="info"
        title="Info"
        source="Drift Monitor"
      />
    );
    expect(screen.getByText('Drift Monitor')).toBeInTheDocument();
  });

  it('renders timestamp', () => {
    const recentDate = new Date(Date.now() - 5 * 60 * 1000); // 5 minutes ago
    render(<AlertCard severity="info" title="Recent" timestamp={recentDate} />);
    expect(screen.getByText('5m ago')).toBeInTheDocument();
  });

  it('renders action buttons', () => {
    const handleAction = vi.fn();
    render(
      <AlertCard
        severity="warning"
        title="Action Required"
        actions={[
          { label: 'Retrain', onClick: handleAction, primary: true },
          { label: 'Dismiss', onClick: vi.fn() },
        ]}
      />
    );

    expect(screen.getByText('Retrain')).toBeInTheDocument();
    expect(screen.getByText('Dismiss')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Retrain'));
    expect(handleAction).toHaveBeenCalled();
  });

  it('calls onDismiss when dismissible', () => {
    const handleDismiss = vi.fn();
    render(
      <AlertCard
        severity="info"
        title="Dismissible"
        dismissible
        onDismiss={handleDismiss}
      />
    );

    const dismissButton = document.querySelector('button');
    fireEvent.click(dismissButton!);
    expect(handleDismiss).toHaveBeenCalled();
  });

  it('shows new indicator when isNew', () => {
    const { container } = render(
      <AlertCard severity="info" title="New Alert" isNew />
    );
    expect(container.querySelector('.animate-ping')).toBeInTheDocument();
  });

  it('renders compact variant', () => {
    const { container } = render(
      <AlertCard severity="warning" title="Compact" compact />
    );
    expect(container.querySelector('.py-2')).toBeInTheDocument();
  });

  it('renders all severity types', () => {
    const severities = ['info', 'success', 'warning', 'error', 'critical'] as const;

    severities.forEach((severity) => {
      const { unmount } = render(<AlertCard severity={severity} title={`${severity} alert`} />);
      expect(screen.getByText(`${severity} alert`)).toBeInTheDocument();
      unmount();
    });
  });

  it('renders timestamp in hours format', () => {
    const hoursAgo = new Date(Date.now() - 3 * 60 * 60 * 1000); // 3 hours ago
    render(<AlertCard severity="info" title="Hours ago" timestamp={hoursAgo} />);
    expect(screen.getByText('3h ago')).toBeInTheDocument();
  });

  it('renders timestamp in days format', () => {
    const daysAgo = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000); // 2 days ago
    render(<AlertCard severity="info" title="Days ago" timestamp={daysAgo} />);
    expect(screen.getByText('2d ago')).toBeInTheDocument();
  });

  it('renders timestamp as date for older alerts', () => {
    const oldDate = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000); // 10 days ago
    render(<AlertCard severity="info" title="Old alert" timestamp={oldDate} />);
    // Should show localized date format
    expect(screen.getByText(oldDate.toLocaleDateString())).toBeInTheDocument();
  });

  it('renders timestamp as Just now for very recent alerts', () => {
    const justNow = new Date(Date.now() - 30 * 1000); // 30 seconds ago
    render(<AlertCard severity="info" title="Recent Alert" timestamp={justNow} />);
    expect(screen.getByText('Just now')).toBeInTheDocument();
  });
});

describe('AlertList', () => {
  const mockAlerts = [
    { severity: 'warning' as const, title: 'Alert 1', message: 'Message 1' },
    { severity: 'error' as const, title: 'Alert 2', message: 'Message 2' },
    { severity: 'info' as const, title: 'Alert 3', message: 'Message 3' },
  ];

  it('renders list of alerts', () => {
    render(<AlertList alerts={mockAlerts} />);

    expect(screen.getByText('Alert 1')).toBeInTheDocument();
    expect(screen.getByText('Alert 2')).toBeInTheDocument();
    expect(screen.getByText('Alert 3')).toBeInTheDocument();
  });

  it('limits displayed alerts with maxItems', () => {
    render(<AlertList alerts={mockAlerts} maxItems={2} />);

    expect(screen.getByText('Alert 1')).toBeInTheDocument();
    expect(screen.getByText('Alert 2')).toBeInTheDocument();
    expect(screen.queryByText('Alert 3')).not.toBeInTheDocument();
    expect(screen.getByText('+1 more alerts')).toBeInTheDocument();
  });

  it('shows empty message when no alerts', () => {
    render(<AlertList alerts={[]} />);
    expect(screen.getByText('No alerts')).toBeInTheDocument();
  });

  it('uses custom empty message', () => {
    render(<AlertList alerts={[]} emptyMessage="All clear!" />);
    expect(screen.getByText('All clear!')).toBeInTheDocument();
  });

  it('renders loading skeleton', () => {
    const { container } = render(<AlertList alerts={[]} isLoading />);
    expect(container.querySelectorAll('.animate-pulse').length).toBe(3);
  });

  it('renders compact alerts', () => {
    const { container } = render(<AlertList alerts={mockAlerts} compact />);
    expect(container.querySelectorAll('.py-2').length).toBe(3);
  });
});
