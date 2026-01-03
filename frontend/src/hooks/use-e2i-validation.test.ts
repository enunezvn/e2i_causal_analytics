/**
 * useE2IValidation Hook Tests
 * ===========================
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useE2IValidation } from './use-e2i-validation';

describe('useE2IValidation', () => {
  it('should return empty validations initially', () => {
    const { result } = renderHook(() => useE2IValidation());

    expect(result.current.validations).toEqual([]);
    expect(result.current.pendingReviews).toEqual([]);
  });

  it('should compute proceed status for high confidence', () => {
    const { result } = renderHook(() => useE2IValidation());

    const status = result.current.computeStatus(95, 'analyze');

    expect(status).toBe('proceed');
  });

  it('should compute review status for medium confidence', () => {
    const { result } = renderHook(() => useE2IValidation());

    const status = result.current.computeStatus(75, 'analyze');

    expect(status).toBe('review');
  });

  it('should compute block status for low confidence', () => {
    const { result } = renderHook(() => useE2IValidation());

    const status = result.current.computeStatus(40, 'analyze');

    expect(status).toBe('block');
  });

  it('should require review for always-review actions', () => {
    const { result } = renderHook(() => useE2IValidation());

    const status = result.current.computeStatus(99, 'delete');

    expect(status).toBe('review');
  });

  it('should add a validation result', () => {
    const { result } = renderHook(() => useE2IValidation());

    let validationId: string;
    act(() => {
      validationId = result.current.addValidation({
        status: 'proceed',
        confidence: 95,
        explanation: 'Test validation',
        actionType: 'analyze',
        requiresReview: false,
      });
    });

    expect(result.current.validations).toHaveLength(1);
    expect(result.current.getValidation(validationId!)).toBeDefined();
  });

  it('should track pending reviews', () => {
    const { result } = renderHook(() => useE2IValidation());

    act(() => {
      result.current.addValidation({
        status: 'review',
        confidence: 75,
        explanation: 'Needs review',
        actionType: 'modify',
        requiresReview: true,
      });
    });

    expect(result.current.pendingReviews).toHaveLength(1);
  });

  it('should approve a validation', () => {
    const { result } = renderHook(() => useE2IValidation());

    let validationId: string;
    act(() => {
      validationId = result.current.addValidation({
        status: 'review',
        confidence: 75,
        explanation: 'Needs review',
        actionType: 'modify',
        requiresReview: true,
      });
    });

    act(() => {
      result.current.approveValidation(validationId!);
    });

    const validation = result.current.getValidation(validationId!);
    expect(validation?.status).toBe('proceed');
    expect(validation?.requiresReview).toBe(false);
  });

  it('should reject a validation', () => {
    const { result } = renderHook(() => useE2IValidation());

    let validationId: string;
    act(() => {
      validationId = result.current.addValidation({
        status: 'review',
        confidence: 75,
        explanation: 'Needs review',
        actionType: 'modify',
        requiresReview: true,
      });
    });

    act(() => {
      result.current.rejectValidation(validationId!);
    });

    const validation = result.current.getValidation(validationId!);
    expect(validation?.status).toBe('block');
  });

  it('should clear all validations', () => {
    const { result } = renderHook(() => useE2IValidation());

    act(() => {
      result.current.addValidation({
        status: 'proceed',
        confidence: 95,
        explanation: 'Test',
        actionType: 'analyze',
        requiresReview: false,
      });
    });

    act(() => {
      result.current.clearValidations();
    });

    expect(result.current.validations).toEqual([]);
  });

  it('should update validation config', () => {
    const { result } = renderHook(() => useE2IValidation());

    act(() => {
      result.current.setConfig({ proceedThreshold: 80 });
    });

    expect(result.current.config.proceedThreshold).toBe(80);
  });
});
