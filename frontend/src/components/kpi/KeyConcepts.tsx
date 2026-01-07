/**
 * KeyConcepts Component
 * =====================
 *
 * Displays key concepts and assumptions for the KPI Dictionary:
 * - Causal Inference Assumptions
 * - Data Quality Dimensions
 * - Model Performance Considerations
 *
 * @module components/kpi/KeyConcepts
 */

import React from 'react';

// ============================================================================
// Types
// ============================================================================

interface ConceptItem {
  term: string;
  description: string;
}

export interface KeyConceptsProps {
  /** Optional className for custom styling */
  className?: string;
}

// ============================================================================
// Data Constants
// ============================================================================

const CAUSAL_INFERENCE_ASSUMPTIONS: ConceptItem[] = [
  {
    term: 'SUTVA (Stable Unit Treatment Value Assumption)',
    description: "One unit's treatment doesn't affect another unit's outcome (no interference between units)",
  },
  {
    term: 'Ignorability/Unconfoundedness',
    description: 'Treatment assignment is independent of potential outcomes given observed covariates',
  },
  {
    term: 'Positivity/Common Support',
    description: 'Every unit has a non-zero probability of receiving each treatment level',
  },
  {
    term: 'Consistency',
    description: 'The potential outcome under treatment equals the observed outcome when that treatment is received',
  },
  {
    term: 'Parallel Trends (DiD)',
    description: 'Treatment and control groups would have evolved similarly in absence of treatment',
  },
];

const DATA_QUALITY_DIMENSIONS: ConceptItem[] = [
  {
    term: 'Completeness',
    description: 'Degree to which required data is present',
  },
  {
    term: 'Validity',
    description: 'Data conforms to defined formats and business rules',
  },
  {
    term: 'Uniqueness',
    description: 'No inappropriate duplicates exist',
  },
  {
    term: 'Consistency',
    description: 'Data is uniform across systems and time',
  },
  {
    term: 'Timeliness',
    description: 'Data is available when needed',
  },
  {
    term: 'Accuracy',
    description: 'Data correctly represents real-world entities',
  },
];

const MODEL_PERFORMANCE_CONSIDERATIONS: ConceptItem[] = [
  {
    term: 'Class Imbalance',
    description: 'When positive cases are rare, PR-AUC is more informative than ROC-AUC',
  },
  {
    term: 'Calibration',
    description: 'Well-calibrated models have predicted probabilities that match observed frequencies',
  },
  {
    term: 'Drift',
    description: 'Feature distributions can shift over time (data drift) affecting model performance (concept drift)',
  },
  {
    term: 'Fairness',
    description: 'Models should perform equitably across relevant subgroups',
  },
  {
    term: 'Interpretability',
    description: 'SHAP values help explain individual predictions for trust and debugging',
  },
];

// ============================================================================
// Sub-components
// ============================================================================

interface ConceptSectionProps {
  title: string;
  items: ConceptItem[];
}

const ConceptSection: React.FC<ConceptSectionProps> = ({ title, items }) => (
  <div className="mb-6">
    <h4 className="text-base font-semibold text-gray-800 mb-3">{title}</h4>
    <ul className="space-y-3">
      {items.map((item, index) => (
        <li key={index} className="flex">
          <span className="text-indigo-500 mr-2 mt-1">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
          </span>
          <div>
            <strong className="text-gray-700">{item.term}:</strong>{' '}
            <span className="text-gray-600">{item.description}</span>
          </div>
        </li>
      ))}
    </ul>
  </div>
);

// ============================================================================
// Main Component
// ============================================================================

export const KeyConcepts: React.FC<KeyConceptsProps> = ({ className = '' }) => {
  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center mb-6">
        <h2 className="text-xl font-bold text-gray-800 flex items-center justify-center gap-2">
          <span>Key Concepts & Assumptions</span>
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Foundational concepts for understanding KPIs and causal inference
        </p>
      </div>

      {/* Content Card */}
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
        <ConceptSection
          title="Causal Inference Assumptions"
          items={CAUSAL_INFERENCE_ASSUMPTIONS}
        />

        <div className="border-t border-gray-200 my-6" />

        <ConceptSection
          title="Data Quality Dimensions"
          items={DATA_QUALITY_DIMENSIONS}
        />

        <div className="border-t border-gray-200 my-6" />

        <ConceptSection
          title="Model Performance Considerations"
          items={MODEL_PERFORMANCE_CONSIDERATIONS}
        />
      </div>
    </div>
  );
};

export default KeyConcepts;
