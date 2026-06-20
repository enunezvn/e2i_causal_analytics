/**
 * CompetitorDensityBadge — surfaces the curated market-landscape competitor
 * density the gap-analyzer attaches to each strategic bet (#1056).
 *
 * INFORMATIONAL / surface-only: the backend computes this per bet but it NEVER
 * affects the ROI value or the prioritizer ranking. Renders nothing when the
 * density is absent / zero / "unknown" (the backend fail-open state) — an honest
 * empty state, mirroring the causal `ClinicalContextPanel` "N rivals" treatment.
 *
 * @module components/insights/CompetitorDensityBadge
 */

import { Building2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

export interface CompetitorDensityBadgeProps {
  /** Curated count of competing products; absent / 0 -> render nothing. */
  competitor_products_count?: number;
  /** Market saturation: limited | moderate | crowded | unknown. */
  competitor_density_label?: string;
  /** Names of the competing products (curated, not FDA-sourced). */
  competitor_drug_names?: string[];
  className?: string;
}

export function CompetitorDensityBadge({
  competitor_products_count,
  competitor_density_label,
  competitor_drug_names,
  className,
}: CompetitorDensityBadgeProps) {
  const count = competitor_products_count ?? 0;
  // Honest empty state (issue #1056 acceptance): render nothing when density is
  // 0 / "unknown" / absent — the backend's fail-open state (it only ever pairs
  // the "unknown" label with count 0). Mirrors the causal panel's count > 0 gate.
  if (count <= 0 || competitor_density_label === 'unknown') {
    return null;
  }

  const names = competitor_drug_names ?? [];
  const hasSaturation = !!competitor_density_label;

  return (
    <div className={cn('text-sm', className)}>
      <div className="flex items-center gap-1 text-muted-foreground">
        <Building2 className="h-3.5 w-3.5" />
        Market landscape ({count} rival{count === 1 ? '' : 's'})
        {hasSaturation && (
          <Badge variant="outline" className="ml-1 text-xs font-normal capitalize">
            {competitor_density_label}
          </Badge>
        )}
      </div>
      {names.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {names.map((c) => (
            <Badge key={c} variant="secondary" className="text-xs font-normal">
              {c}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export default CompetitorDensityBadge;
