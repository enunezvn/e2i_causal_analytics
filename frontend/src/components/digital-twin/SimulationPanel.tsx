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
import { Play, Loader2, FlaskConical, Settings } from 'lucide-react';
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

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
                  value={formValues.budget ?? ''}
                  onChange={(e) =>
                    setFormValues((prev) => ({
                      ...prev,
                      budget: e.target.value ? Number(e.target.value) : undefined,
                    }))
                  }
                />
                <p className="text-xs text-muted-foreground">
                  Budget allocation for ROI calculations
                </p>
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
