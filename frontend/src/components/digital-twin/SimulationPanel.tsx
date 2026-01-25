/**
 * Simulation Panel Component
 * ==========================
 *
 * Controls panel for running digital twin simulations.
 * Allows users to configure intervention parameters and launch simulations.
 *
 * @module components/digital-twin/SimulationPanel
 */

import { useState } from 'react';
import { Play, Loader2, FlaskConical, Settings, AlertCircle } from 'lucide-react';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import {
  InterventionType,
  type SimulationRequest,
  type SimulationFormValues,
} from '@/types/digital-twin';

// =============================================================================
// VALIDATION SCHEMA
// =============================================================================

/**
 * Budget validation constants
 */
const BUDGET_MIN = 0;
const BUDGET_MAX = 999_999_999;

/**
 * Zod schema for simulation form validation
 */
const SimulationFormSchema = z.object({
  interventionType: z.nativeEnum(InterventionType),
  brand: z.string().min(1, 'Brand is required'),
  sampleSize: z.number().int().min(100).max(10000),
  durationDays: z.number().int().min(30).max(365),
  targetRegions: z.array(z.string()),
  targetSegments: z.array(z.string()),
  budget: z.number()
    .min(BUDGET_MIN, `Budget must be at least $${BUDGET_MIN.toLocaleString()}`)
    .max(BUDGET_MAX, `Budget cannot exceed $${BUDGET_MAX.toLocaleString()}`)
    .optional(),
});

type ValidationErrors = Partial<Record<keyof SimulationFormValues, string>>;

// =============================================================================
// TYPES
// =============================================================================

export interface SimulationPanelProps {
  /** Callback when simulation is requested */
  onSimulate: (request: SimulationRequest) => void;
  /** Whether a simulation is currently running */
  isSimulating?: boolean;
  /** Initial brand selection */
  initialBrand?: string;
  /** Available brands */
  brands?: string[];
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const INTERVENTION_TYPE_LABELS: Record<InterventionType, string> = {
  [InterventionType.HCP_ENGAGEMENT]: 'HCP Engagement Campaign',
  [InterventionType.PATIENT_SUPPORT]: 'Patient Support Program',
  [InterventionType.PRICING]: 'Pricing Change',
  [InterventionType.REP_TRAINING]: 'Rep Training Program',
  [InterventionType.DIGITAL_MARKETING]: 'Digital Marketing',
  [InterventionType.FORMULARY_ACCESS]: 'Formulary Access Initiative',
};

const DEFAULT_BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali'];

const REGIONS = [
  'Northeast',
  'Southeast',
  'Midwest',
  'West',
  'Southwest',
  'Pacific Northwest',
];

const HCP_SEGMENTS = [
  'High-Volume HCPs',
  'Medium-Volume HCPs',
  'Low-Volume HCPs',
  'Early Adopters',
  'Key Opinion Leaders',
  'Academic Centers',
];

// =============================================================================
// COMPONENT
// =============================================================================

export function SimulationPanel({
  onSimulate,
  isSimulating = false,
  initialBrand = 'Remibrutinib',
  brands = DEFAULT_BRANDS,
  className = '',
}: SimulationPanelProps) {
  const [formValues, setFormValues] = useState<SimulationFormValues>({
    interventionType: InterventionType.HCP_ENGAGEMENT,
    brand: initialBrand,
    sampleSize: 1000,
    durationDays: 90,
    targetRegions: [],
    targetSegments: [],
    budget: undefined,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});

  /**
   * Validate form values before submission
   * Returns true if valid, false otherwise
   */
  const validateForm = (): boolean => {
    const result = SimulationFormSchema.safeParse(formValues);

    if (!result.success) {
      const errors: ValidationErrors = {};
      result.error.issues.forEach((issue) => {
        const field = issue.path[0] as keyof SimulationFormValues;
        if (!errors[field]) {
          errors[field] = issue.message;
        }
      });
      setValidationErrors(errors);
      return false;
    }

    setValidationErrors({});
    return true;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate form before submission
    if (!validateForm()) {
      return;
    }

    const request: SimulationRequest = {
      intervention_type: formValues.interventionType,
      brand: formValues.brand,
      sample_size: formValues.sampleSize,
      duration_days: formValues.durationDays,
      target_regions: formValues.targetRegions.length > 0 ? formValues.targetRegions : undefined,
      target_segments: formValues.targetSegments.length > 0 ? formValues.targetSegments : undefined,
      budget: formValues.budget,
    };

    onSimulate(request);
  };

  const toggleRegion = (region: string) => {
    setFormValues((prev) => ({
      ...prev,
      targetRegions: prev.targetRegions.includes(region)
        ? prev.targetRegions.filter((r) => r !== region)
        : [...prev.targetRegions, region],
    }));
  };

  const toggleSegment = (segment: string) => {
    setFormValues((prev) => ({
      ...prev,
      targetSegments: prev.targetSegments.includes(segment)
        ? prev.targetSegments.filter((s) => s !== segment)
        : [...prev.targetSegments, segment],
    }));
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical className="h-5 w-5" />
              Simulation Configuration
            </CardTitle>
            <CardDescription>
              Configure and run digital twin simulations to pre-screen interventions
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Settings className="h-4 w-4 mr-2" />
            {showAdvanced ? 'Basic' : 'Advanced'}
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Intervention Type */}
            <div className="space-y-2">
              <Label htmlFor="intervention-type">Intervention Type</Label>
              <Select
                value={formValues.interventionType}
                onValueChange={(value: InterventionType) =>
                  setFormValues((prev) => ({ ...prev, interventionType: value }))
                }
              >
                <SelectTrigger id="intervention-type">
                  <SelectValue placeholder="Select intervention type" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(INTERVENTION_TYPE_LABELS).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Brand */}
            <div className="space-y-2">
              <Label htmlFor="brand">Target Brand</Label>
              <Select
                value={formValues.brand}
                onValueChange={(value) =>
                  setFormValues((prev) => ({ ...prev, brand: value }))
                }
              >
                <SelectTrigger id="brand">
                  <SelectValue placeholder="Select brand" />
                </SelectTrigger>
                <SelectContent>
                  {brands.map((brand) => (
                    <SelectItem key={brand} value={brand}>
                      {brand}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Sample Size and Duration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="sample-size">
                Sample Size: <span className="font-bold">{formValues.sampleSize.toLocaleString()}</span>
              </Label>
              <Slider
                id="sample-size"
                min={100}
                max={10000}
                step={100}
                value={[formValues.sampleSize]}
                onValueChange={([value]) =>
                  setFormValues((prev) => ({ ...prev, sampleSize: value }))
                }
              />
              <p className="text-xs text-muted-foreground">
                Number of HCPs in treatment group
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="duration">
                Duration: <span className="font-bold">{formValues.durationDays} days</span>
              </Label>
              <Slider
                id="duration"
                min={30}
                max={365}
                step={15}
                value={[formValues.durationDays]}
                onValueChange={([value]) =>
                  setFormValues((prev) => ({ ...prev, durationDays: value }))
                }
              />
              <p className="text-xs text-muted-foreground">
                Simulation time horizon
              </p>
            </div>
          </div>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-border">
              {/* Budget */}
              <div className="space-y-2">
                <Label htmlFor="budget">Budget (Optional)</Label>
                <Input
                  id="budget"
                  type="number"
                  placeholder="Enter budget in dollars"
                  min={BUDGET_MIN}
                  max={BUDGET_MAX}
                  step={1}
                  value={formValues.budget ?? ''}
                  onChange={(e) => {
                    const value = e.target.value;
                    setFormValues((prev) => ({
                      ...prev,
                      budget: value ? Number(value) : undefined,
                    }));
                    // Clear budget validation error on change
                    if (validationErrors.budget) {
                      setValidationErrors((prev) => ({ ...prev, budget: undefined }));
                    }
                  }}
                  className={validationErrors.budget ? 'border-red-500 focus:ring-red-500' : ''}
                  aria-invalid={!!validationErrors.budget}
                  aria-describedby={validationErrors.budget ? 'budget-error' : undefined}
                />
                {validationErrors.budget ? (
                  <p id="budget-error" className="text-xs text-red-500 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {validationErrors.budget}
                  </p>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Budget allocation for ROI calculations (max ${BUDGET_MAX.toLocaleString()})
                  </p>
                )}
              </div>

              {/* Target Regions */}
              <div className="space-y-2">
                <Label>Target Regions</Label>
                <div className="flex flex-wrap gap-2">
                  {REGIONS.map((region) => (
                    <Badge
                      key={region}
                      variant={formValues.targetRegions.includes(region) ? 'default' : 'outline'}
                      className="cursor-pointer"
                      onClick={() => toggleRegion(region)}
                    >
                      {region}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  Leave empty for all regions
                </p>
              </div>

              {/* HCP Segments */}
              <div className="space-y-2">
                <Label>HCP Segments</Label>
                <div className="flex flex-wrap gap-2">
                  {HCP_SEGMENTS.map((segment) => (
                    <Badge
                      key={segment}
                      variant={formValues.targetSegments.includes(segment) ? 'default' : 'outline'}
                      className="cursor-pointer"
                      onClick={() => toggleSegment(segment)}
                    >
                      {segment}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  Leave empty for all segments
                </p>
              </div>
            </div>
          )}

          {/* Form-level validation error summary */}
          {Object.keys(validationErrors).length > 0 && (
            <div className="p-3 rounded-md bg-red-50 border border-red-200 dark:bg-red-900/20 dark:border-red-800">
              <div className="flex items-center gap-2 text-red-700 dark:text-red-400">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <p className="text-sm font-medium">Please fix the following errors:</p>
              </div>
              <ul className="mt-2 ml-6 text-sm text-red-600 dark:text-red-400 list-disc">
                {Object.entries(validationErrors).map(([field, error]) => (
                  error && <li key={field}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full"
            disabled={isSimulating}
            size="lg"
          >
            {isSimulating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Simulation
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default SimulationPanel;
