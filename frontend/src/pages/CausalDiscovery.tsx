/**
 * CausalDiscovery Page
 * ====================
 *
 * Page component for causal discovery analysis with DAG visualization,
 * effect estimates, and refutation tests.
 *
 * @module pages/CausalDiscovery
 */

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';

function CausalDiscovery() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">Causal Discovery</h1>
      <p className="text-muted-foreground mb-8">
        Causal analysis with DAG visualization, effect estimates, and refutation tests.
      </p>
      <CausalDiscoveryViz showControls showDetails />
    </div>
  );
}

export default CausalDiscovery;
