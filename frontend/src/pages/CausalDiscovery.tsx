/**
 * CausalDiscovery Page
 * ====================
 *
 * Page component for causal discovery analysis with DAG visualization,
 * effect estimates, refutation tests, and confidence intervals.
 *
 * Features:
 * - Interactive causal DAG visualization with D3.js
 * - Effect estimates with 95% confidence intervals
 * - Refutation test results for causal validation
 * - Export functionality (SVG)
 * - Zoom and pan controls
 *
 * @module pages/CausalDiscovery
 */

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import { Badge } from '@/components/ui/badge';
import { Brain, FlaskConical, GitBranch, Shield } from 'lucide-react';

function CausalDiscovery() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Causal Discovery</h1>
          <p className="text-muted-foreground">
            Causal analysis with DAG visualization, effect estimates, and refutation tests.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Brain className="h-3 w-3" />
            DoWhy
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <FlaskConical className="h-3 w-3" />
            EconML
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <GitBranch className="h-3 w-3" />
            DAG
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            Refutation
          </Badge>
        </div>
      </div>
      <CausalDiscoveryViz
        showControls
        showDetails
        showEffectsTable
        showRefutationTests
      />
    </div>
  );
}

export default CausalDiscovery;
