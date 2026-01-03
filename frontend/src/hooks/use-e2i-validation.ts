/**
 * E2I Validation Hook
 * ===================
 *
 * Provides validation state management for AI responses and actions.
 * Tracks PROCEED/REVIEW/BLOCK status for AI interactions.
 *
 * @module hooks/use-e2i-validation
 */

import * as React from 'react';
import type { ValidationStatus } from '@/components/chat/ValidationBadge';

// =============================================================================
// TYPES
// =============================================================================

export interface ValidationResult {
  id: string;
  status: ValidationStatus;
  confidence: number;
  explanation: string;
  actionType: string;
  timestamp: Date;
  requiresReview: boolean;
}

export interface ValidationConfig {
  /** Minimum confidence for PROCEED status */
  proceedThreshold: number;
  /** Minimum confidence for REVIEW status (below this = BLOCK) */
  reviewThreshold: number;
  /** Auto-approve low-risk actions */
  autoApprove: boolean;
  /** List of action types that always require review */
  alwaysReviewActions: string[];
}

export interface UseE2IValidationReturn {
  /** Current validation results */
  validations: ValidationResult[];
  /** Current validation config */
  config: ValidationConfig;
  /** Add a validation result */
  addValidation: (result: Omit<ValidationResult, 'id' | 'timestamp'>) => string;
  /** Get validation by ID */
  getValidation: (id: string) => ValidationResult | undefined;
  /** Approve a validation (mark as reviewed) */
  approveValidation: (id: string) => void;
  /** Reject a validation */
  rejectValidation: (id: string) => void;
  /** Clear all validations */
  clearValidations: () => void;
  /** Get pending reviews */
  pendingReviews: ValidationResult[];
  /** Update validation config */
  setConfig: (config: Partial<ValidationConfig>) => void;
  /** Compute status from confidence score */
  computeStatus: (confidence: number, actionType: string) => ValidationStatus;
}

// =============================================================================
// DEFAULT VALUES
// =============================================================================

const DEFAULT_CONFIG: ValidationConfig = {
  proceedThreshold: 90,
  reviewThreshold: 60,
  autoApprove: false,
  alwaysReviewActions: ['delete', 'modify_schema', 'export_data'],
};

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook for managing AI action validation state.
 *
 * @example
 * ```tsx
 * const { addValidation, pendingReviews, computeStatus } = useE2IValidation();
 *
 * // Add a validation result
 * const validationId = addValidation({
 *   status: computeStatus(85, 'analyze'),
 *   confidence: 85,
 *   explanation: 'Analysis request validated',
 *   actionType: 'analyze',
 *   requiresReview: false,
 * });
 *
 * // Check pending reviews
 * if (pendingReviews.length > 0) {
 *   // Show review modal
 * }
 * ```
 */
export function useE2IValidation(): UseE2IValidationReturn {
  const [validations, setValidations] = React.useState<ValidationResult[]>([]);
  const [config, setConfigState] = React.useState<ValidationConfig>(DEFAULT_CONFIG);

  const generateId = (): string => {
    return `val-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  };

  const computeStatus = React.useCallback(
    (confidence: number, actionType: string): ValidationStatus => {
      // Actions that always require review
      if (config.alwaysReviewActions.includes(actionType)) {
        return 'review';
      }

      if (confidence >= config.proceedThreshold) {
        return 'proceed';
      }

      if (confidence >= config.reviewThreshold) {
        return 'review';
      }

      return 'block';
    },
    [config]
  );

  const addValidation = React.useCallback(
    (result: Omit<ValidationResult, 'id' | 'timestamp'>): string => {
      const id = generateId();
      const validation: ValidationResult = {
        ...result,
        id,
        timestamp: new Date(),
      };

      setValidations((prev) => [...prev, validation]);
      return id;
    },
    []
  );

  const getValidation = React.useCallback(
    (id: string): ValidationResult | undefined => {
      return validations.find((v) => v.id === id);
    },
    [validations]
  );

  const approveValidation = React.useCallback((id: string): void => {
    setValidations((prev) =>
      prev.map((v) =>
        v.id === id
          ? { ...v, status: 'proceed' as ValidationStatus, requiresReview: false }
          : v
      )
    );
  }, []);

  const rejectValidation = React.useCallback((id: string): void => {
    setValidations((prev) =>
      prev.map((v) =>
        v.id === id
          ? { ...v, status: 'block' as ValidationStatus, requiresReview: false }
          : v
      )
    );
  }, []);

  const clearValidations = React.useCallback((): void => {
    setValidations([]);
  }, []);

  const setConfig = React.useCallback((newConfig: Partial<ValidationConfig>): void => {
    setConfigState((prev) => ({ ...prev, ...newConfig }));
  }, []);

  const pendingReviews = React.useMemo(
    () => validations.filter((v) => v.requiresReview && v.status === 'review'),
    [validations]
  );

  return {
    validations,
    config,
    addValidation,
    getValidation,
    approveValidation,
    rejectValidation,
    clearValidations,
    pendingReviews,
    setConfig,
    computeStatus,
  };
}

export default useE2IValidation;
