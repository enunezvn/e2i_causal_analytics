"""Synthetic learning_signals (Shard 09): real reward/dspy fuel for feedback_learner.

learning_signals holds only 6 rows on the faithful DB (F15 -- feedback_learner
starved). We emit is_training_example=true signals carrying a real reward and
dspy_metric_value. signal_type and rated_agent are enum-exact (22P02 landmine).
Positive signals land in the GEPA fuel band (reward >= 0.5).
"""

import uuid
from datetime import datetime, timezone

import pandas as pd

from .base import BaseGenerator

_SIGNAL_TYPES = ["thumbs_up", "thumbs_down", "rating", "implicit_positive"]
_RATED = ["gap_analyzer", "causal_impact", "tool_composer", "heterogeneous_optimizer"]
_REGIONS = ["northeast", "south", "midwest", "west"]
_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


class FeedbackGenerator(BaseGenerator[pd.DataFrame]):
    @property
    def entity_type(self) -> str:
        return "learning_signals"

    def generate(self) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(self.config.n_records):
            stype = _SIGNAL_TYPES[i % len(_SIGNAL_TYPES)]
            positive = stype in ("thumbs_up", "implicit_positive")
            reward = round(
                float(self._rng.uniform(0.5, 0.95) if positive else self._rng.uniform(0.0, 0.5)),
                3,
            )
            rows.append(
                {
                    "signal_id": str(uuid.uuid4()),
                    "signal_type": stype,
                    "signal_value": reward,
                    "signal_details": {"note": "synthetic feedback fuel"},
                    "applies_to_type": "agent",
                    "applies_to_id": _RATED[i % len(_RATED)],
                    "brand": _BRANDS[i % 3],
                    "region": _REGIONS[i % 4],
                    "rated_agent": _RATED[i % len(_RATED)],
                    "is_training_example": True,
                    "dspy_metric_name": "rubric_total",
                    "dspy_metric_value": reward,
                    "training_input": f"synthetic query {i}",
                    "training_output": f"synthetic response {i}",
                    "reward": reward,
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )
        return pd.DataFrame(rows)
