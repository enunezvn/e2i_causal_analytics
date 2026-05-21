"""Per-library executor package for the multi-library causal pipeline.

This package was extracted from `pipeline/orchestrator.py` in phase C-1 of GH
#354 (NetworkX → DoWhy → EconML → CausalML canonical-routing). Each executor
lives in its own file so Wave-1 (C-2..C-5) parallel dispatchers can edit
strictly-disjoint surfaces with zero rebase pressure.

Public surface (re-exported here for ergonomic imports + by `pipeline/__init__.py`
for backward compatibility with external callers):

- `LibraryExecutor` (ABC) — `executors/base.py`
- `NetworkXExecutor` — `executors/networkx.py`
- `DoWhyExecutor` — `executors/dowhy.py`
- `EconMLExecutor` — `executors/econml.py`
- `CausalMLExecutor` — `executors/causalml.py`

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §2.1, §2.2
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1-§5
- Real-library wrap points for Wave-1 (per dispatch plan §0 verification):
  - DoWhy: `causal_engine/refutation_runner.py:35` (V-03)
  - EconML: `causal_engine/energy_score/estimator_selector.py:252` (V-04)
  - CausalML: `causal_engine/uplift/{random_forest,gradient_boosting}.py` (V-05)
  - NetworkX: `causal_engine/discovery/{driver_ranker,base,gate,runner}.py` (V-06)
"""

from .base import LibraryExecutor
from .causalml import CausalMLExecutor
from .dowhy import DoWhyExecutor
from .econml import EconMLExecutor
from .networkx import NetworkXExecutor

__all__ = [
    "LibraryExecutor",
    "NetworkXExecutor",
    "DoWhyExecutor",
    "EconMLExecutor",
    "CausalMLExecutor",
]
