/**
 * PriorityBadge Component Tests
 * =============================
 *
 * Tests for the PriorityBadge and PriorityDot components.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PriorityBadge, PriorityDot, type Priority } from './PriorityBadge';

describe('PriorityBadge', () => {
  describe('Priority Levels', () => {
    const priorities: Priority[] = ['critical', 'high', 'medium', 'low', 'info'];
    const expectedLabels: Record<Priority, string> = {
      critical: 'Critical',
      high: 'High',
      medium: 'Medium',
      low: 'Low',
      info: 'Info',
    };

    it.each(priorities)('renders %s priority with correct label', (priority) => {
      render(<PriorityBadge priority={priority} />);
      expect(screen.getByText(expectedLabels[priority])).toBeInTheDocument();
    });

    it('renders critical priority with red styling', () => {
      const { container } = render(<PriorityBadge priority="critical" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('bg-red');
      expect(badge.className).toContain('text-red');
    });

    it('renders high priority with orange styling', () => {
      const { container } = render(<PriorityBadge priority="high" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('bg-orange');
      expect(badge.className).toContain('text-orange');
    });

    it('renders medium priority with yellow styling', () => {
      const { container } = render(<PriorityBadge priority="medium" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('bg-yellow');
      expect(badge.className).toContain('text-yellow');
    });

    it('renders low priority with blue styling', () => {
      const { container } = render(<PriorityBadge priority="low" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('bg-blue');
      expect(badge.className).toContain('text-blue');
    });

    it('renders info priority with gray styling', () => {
      const { container } = render(<PriorityBadge priority="info" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('bg-gray');
      expect(badge.className).toContain('text-gray');
    });
  });

  describe('Icon Display', () => {
    it('shows icon by default', () => {
      const { container } = render(<PriorityBadge priority="critical" />);
      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });

    it('hides icon when showIcon is false', () => {
      const { container } = render(<PriorityBadge priority="critical" showIcon={false} />);
      const svg = container.querySelector('svg');
      expect(svg).not.toBeInTheDocument();
    });

    it('shows icon when showIcon is true', () => {
      const { container } = render(<PriorityBadge priority="high" showIcon={true} />);
      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });

  describe('Label Display', () => {
    it('shows label by default', () => {
      render(<PriorityBadge priority="critical" />);
      expect(screen.getByText('Critical')).toBeInTheDocument();
    });

    it('hides label when showLabel is false', () => {
      render(<PriorityBadge priority="critical" showLabel={false} />);
      expect(screen.queryByText('Critical')).not.toBeInTheDocument();
    });

    it('shows label when showLabel is true', () => {
      render(<PriorityBadge priority="high" showLabel={true} />);
      expect(screen.getByText('High')).toBeInTheDocument();
    });
  });

  describe('Size Variants', () => {
    it('uses medium size by default', () => {
      const { container } = render(<PriorityBadge priority="critical" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('px-2');
      expect(badge.className).toContain('py-1');
    });

    it('renders small size correctly', () => {
      const { container } = render(<PriorityBadge priority="critical" size="sm" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('px-1.5');
      expect(badge.className).toContain('py-0.5');
    });

    it('renders medium size correctly', () => {
      const { container } = render(<PriorityBadge priority="critical" size="md" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('px-2');
      expect(badge.className).toContain('py-1');
    });

    it('renders large size correctly', () => {
      const { container } = render(<PriorityBadge priority="critical" size="lg" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('px-2.5');
      expect(badge.className).toContain('py-1.5');
    });

    it('applies correct icon size for small', () => {
      const { container } = render(<PriorityBadge priority="critical" size="sm" />);
      const svg = container.querySelector('svg');
      // Small size uses h-3 w-3 - use getAttribute since SVG className returns SVGAnimatedString
      const svgClass = svg?.getAttribute('class') || '';
      expect(svgClass).toMatch(/h-3\b/);
      expect(svgClass).toMatch(/w-3\b/);
    });

    it('applies correct icon size for large', () => {
      const { container } = render(<PriorityBadge priority="critical" size="lg" />);
      const svg = container.querySelector('svg');
      // Large size uses h-4 w-4 - use getAttribute since SVG className returns SVGAnimatedString
      const svgClass = svg?.getAttribute('class') || '';
      expect(svgClass).toMatch(/h-4\b/);
      expect(svgClass).toMatch(/w-4\b/);
    });
  });

  describe('Pulsing Animation', () => {
    it('does not pulse by default', () => {
      const { container } = render(<PriorityBadge priority="critical" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).not.toContain('animate-pulse');
    });

    it('pulses critical priority when pulsing is true', () => {
      const { container } = render(<PriorityBadge priority="critical" pulsing />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('animate-pulse');
    });

    it('does not pulse non-critical priorities when pulsing is true', () => {
      const { container } = render(<PriorityBadge priority="high" pulsing />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).not.toContain('animate-pulse');
    });
  });

  describe('Custom Props', () => {
    it('applies custom className', () => {
      const { container } = render(<PriorityBadge priority="critical" className="custom-class" />);
      const badge = container.firstChild as HTMLElement;
      expect(badge.className).toContain('custom-class');
    });

    it('passes through HTML attributes', () => {
      render(<PriorityBadge priority="critical" data-testid="priority-badge" />);
      expect(screen.getByTestId('priority-badge')).toBeInTheDocument();
    });
  });
});

describe('PriorityDot', () => {
  describe('Priority Colors', () => {
    it('renders critical priority with red dot', () => {
      const { container } = render(<PriorityDot priority="critical" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('bg-red-500');
    });

    it('renders high priority with orange dot', () => {
      const { container } = render(<PriorityDot priority="high" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('bg-orange-500');
    });

    it('renders medium priority with yellow dot', () => {
      const { container } = render(<PriorityDot priority="medium" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('bg-yellow-500');
    });

    it('renders low priority with blue dot', () => {
      const { container } = render(<PriorityDot priority="low" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('bg-blue-500');
    });

    it('renders info priority with gray dot', () => {
      const { container } = render(<PriorityDot priority="info" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('bg-gray-500');
    });
  });

  describe('Dot Styling', () => {
    it('has correct base styling', () => {
      const { container } = render(<PriorityDot priority="critical" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('h-2');
      expect(dot.className).toContain('w-2');
      expect(dot.className).toContain('rounded-full');
    });

    it('pulses for critical priority', () => {
      const { container } = render(<PriorityDot priority="critical" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('animate-pulse');
    });

    it('does not pulse for non-critical priorities', () => {
      const { container } = render(<PriorityDot priority="high" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).not.toContain('animate-pulse');
    });
  });

  describe('Accessibility', () => {
    it('has aria-label for screen readers', () => {
      const { container } = render(<PriorityDot priority="critical" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.getAttribute('aria-label')).toBe('Priority: critical');
    });

    it.each(['critical', 'high', 'medium', 'low', 'info'] as Priority[])(
      'has correct aria-label for %s priority',
      (priority) => {
        const { container } = render(<PriorityDot priority={priority} />);
        const dot = container.firstChild as HTMLElement;
        expect(dot.getAttribute('aria-label')).toBe(`Priority: ${priority}`);
      }
    );
  });

  describe('Custom className', () => {
    it('applies custom className', () => {
      const { container } = render(<PriorityDot priority="critical" className="custom-class" />);
      const dot = container.firstChild as HTMLElement;
      expect(dot.className).toContain('custom-class');
    });
  });
});
