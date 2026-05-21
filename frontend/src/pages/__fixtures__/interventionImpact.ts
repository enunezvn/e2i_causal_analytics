/**
 * InterventionImpact Test/Storybook Fixtures
 * ===========================================
 *
 * Hardcoded fixture data formerly inlined in InterventionImpact.tsx as
 * SAMPLE_* constants. Moved here so production rendering paths cannot
 * reach them (F-002). Tests and Storybook stories may import from here.
 *
 * @module pages/__fixtures__/interventionImpact
 */

// Minimal local types — keep stand-alone from the page module so the
// fixture file is consumable from test contexts without page-side imports.

interface ImpactDataPointFixture {
  date: string;
  actual: number;
  counterfactual: number;
  upperBound: number;
  lowerBound: number;
}

interface TreatmentEffectFixture {
  id: string;
  intervention: string;
  metric: string;
  ate: number;
  ci: [number, number];
  pValue: number;
  isSignificant: boolean;
  sampleSize: number;
  effectSize: 'small' | 'medium' | 'large';
}

interface BeforeAfterFixture {
  metric: string;
  beforeMean: number;
  afterMean: number;
  change: number;
  changePercent: number;
  isPositive: boolean;
}

interface SegmentEffectFixture {
  segment: string;
  sampleSize: number;
  effect: number;
  ci: [number, number];
  isSignificant: boolean;
}

const generateImpactData = (startDate: Date): ImpactDataPointFixture[] => {
  const data: ImpactDataPointFixture[] = [];
  for (let i = 0; i < 90; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);

    const isPostIntervention = i >= 30;
    const baseValue = 1000 + i * 0.5 + Math.sin(i / 7) * 20;
    const interventionEffect = isPostIntervention ? 80 + (i - 30) * 0.5 : 0;
    const noise = (Math.random() - 0.5) * 40;

    const actual = baseValue + interventionEffect + noise;
    const counterfactual = baseValue + noise * 0.8;
    const uncertainty = 25 + (isPostIntervention ? (i - 30) * 0.3 : 0);

    data.push({
      date: date.toISOString().split('T')[0],
      actual: Math.round(actual),
      counterfactual: Math.round(counterfactual),
      upperBound: Math.round(counterfactual + uncertainty),
      lowerBound: Math.round(counterfactual - uncertainty),
    });
  }
  return data;
};

export const FIXTURE_IMPACT_DATA = generateImpactData(new Date('2024-01-15'));

export const FIXTURE_TREATMENT_EFFECTS: TreatmentEffectFixture[] = [
  {
    id: 'te-001',
    intervention: 'Q1 2024 HCP Engagement Campaign',
    metric: 'TRx Volume',
    ate: 85.3,
    ci: [62.1, 108.5],
    pValue: 0.0012,
    isSignificant: true,
    sampleSize: 1250,
    effectSize: 'large',
  },
  {
    id: 'te-002',
    intervention: 'Q1 2024 HCP Engagement Campaign',
    metric: 'NRx Volume',
    ate: 24.7,
    ci: [18.2, 31.2],
    pValue: 0.0034,
    isSignificant: true,
    sampleSize: 1250,
    effectSize: 'medium',
  },
  {
    id: 'te-003',
    intervention: 'Digital Rep Training Program',
    metric: 'Conversion Rate',
    ate: 3.2,
    ci: [1.8, 4.6],
    pValue: 0.0078,
    isSignificant: true,
    sampleSize: 450,
    effectSize: 'medium',
  },
  {
    id: 'te-004',
    intervention: 'Digital Rep Training Program',
    metric: 'HCP Satisfaction',
    ate: 0.8,
    ci: [-0.2, 1.8],
    pValue: 0.1245,
    isSignificant: false,
    sampleSize: 450,
    effectSize: 'small',
  },
];

export const FIXTURE_BEFORE_AFTER: BeforeAfterFixture[] = [
  { metric: 'TRx Volume', beforeMean: 1024, afterMean: 1109, change: 85, changePercent: 8.3, isPositive: true },
  { metric: 'NRx Volume', beforeMean: 312, afterMean: 337, change: 25, changePercent: 8.0, isPositive: true },
  { metric: 'Market Share', beforeMean: 23.4, afterMean: 24.8, change: 1.4, changePercent: 6.0, isPositive: true },
  { metric: 'HCP Reach', beforeMean: 856, afterMean: 912, change: 56, changePercent: 6.5, isPositive: true },
  { metric: 'Cost per TRx', beforeMean: 42.5, afterMean: 38.2, change: -4.3, changePercent: -10.1, isPositive: true },
];

export const FIXTURE_SEGMENT_EFFECTS: SegmentEffectFixture[] = [
  { segment: 'High-Volume HCPs', sampleSize: 245, effect: 112.5, ci: [78.3, 146.7], isSignificant: true },
  { segment: 'Medium-Volume HCPs', sampleSize: 520, effect: 78.2, ci: [52.1, 104.3], isSignificant: true },
  { segment: 'Low-Volume HCPs', sampleSize: 485, effect: 45.8, ci: [18.9, 72.7], isSignificant: true },
  { segment: 'Northeast Region', sampleSize: 380, effect: 95.3, ci: [68.2, 122.4], isSignificant: true },
  { segment: 'Southeast Region', sampleSize: 350, effect: 72.1, ci: [42.5, 101.7], isSignificant: true },
  { segment: 'Midwest Region', sampleSize: 290, effect: 68.4, ci: [35.2, 101.6], isSignificant: true },
  { segment: 'West Region', sampleSize: 230, effect: 52.6, ci: [12.8, 92.4], isSignificant: true },
];
