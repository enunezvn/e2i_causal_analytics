/**
 * ValidationBadge Component Tests
 * ===============================
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  ValidationBadge,
  ProceedBadge,
  ReviewBadge,
  BlockBadge,
} from './ValidationBadge';

describe('ValidationBadge', () => {
  describe('status variants', () => {
    it('should render proceed status', () => {
      render(<ValidationBadge status="proceed" />);

      expect(screen.getByText('PROCEED')).toBeInTheDocument();
    });

    it('should render review status', () => {
      render(<ValidationBadge status="review" />);

      expect(screen.getByText('REVIEW')).toBeInTheDocument();
    });

    it('should render block status', () => {
      render(<ValidationBadge status="block" />);

      expect(screen.getByText('BLOCK')).toBeInTheDocument();
    });
  });

  describe('confidence display', () => {
    it('should show confidence percentage when provided', () => {
      render(<ValidationBadge status="proceed" confidence={95} />);

      expect(screen.getByText('(95%)')).toBeInTheDocument();
    });

    it('should hide confidence when showConfidence is false', () => {
      render(
        <ValidationBadge status="proceed" confidence={95} showConfidence={false} />
      );

      expect(screen.queryByText('(95%)')).not.toBeInTheDocument();
    });
  });

  describe('custom label', () => {
    it('should display custom label when provided', () => {
      render(<ValidationBadge status="proceed" label="APPROVED" />);

      expect(screen.getByText('APPROVED')).toBeInTheDocument();
      expect(screen.queryByText('PROCEED')).not.toBeInTheDocument();
    });
  });

  describe('size variants', () => {
    it('should render sm size', () => {
      const { container } = render(<ValidationBadge status="proceed" size="sm" />);

      expect(container.firstChild).toBeInTheDocument();
    });

    it('should render md size (default)', () => {
      const { container } = render(<ValidationBadge status="proceed" size="md" />);

      expect(container.firstChild).toBeInTheDocument();
    });

    it('should render lg size', () => {
      const { container } = render(<ValidationBadge status="proceed" size="lg" />);

      expect(container.firstChild).toBeInTheDocument();
    });
  });
});

describe('Preset Badges', () => {
  it('should render ProceedBadge', () => {
    render(<ProceedBadge confidence={95} />);

    expect(screen.getByText('PROCEED')).toBeInTheDocument();
  });

  it('should render ReviewBadge', () => {
    render(<ReviewBadge confidence={75} />);

    expect(screen.getByText('REVIEW')).toBeInTheDocument();
  });

  it('should render BlockBadge', () => {
    render(<BlockBadge confidence={30} />);

    expect(screen.getByText('BLOCK')).toBeInTheDocument();
  });
});
