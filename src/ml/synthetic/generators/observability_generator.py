"""Synthetic observability span slice (Shard 09).

ml_observability_spans already holds recent real rows (the audit F3 defect is a
code kwarg bug, NOT substrate). We add a thin is_synthetic=true slice so Shard 07's
leakage test has a taggable row to assert exclusion on. agent_name/agent_tier are
enum-exact (22P02 landmine); started_at falls inside the last ~5 days.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from .base import BaseGenerator

# agent_name_enum x agent_tier_enum (faithful labels, paired sensibly)
_AGENT_TIER = [
    ("gap_analyzer", "causal_analytics"),
    ("causal_impact", "causal_analytics"),
    ("tool_composer", "coordination"),
    ("drift_monitor", "monitoring"),
    ("experiment_monitor", "monitoring"),
]


class ObservabilityGenerator(BaseGenerator[pd.DataFrame]):
    @property
    def entity_type(self) -> str:
        return "ml_observability_spans"

    def generate(self) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(self.config.n_records):
            agent, tier = _AGENT_TIER[i % len(_AGENT_TIER)]
            in_tok = int(self._rng.integers(50, 800))
            out_tok = int(self._rng.integers(20, 400))
            started = now - timedelta(minutes=int(self._rng.integers(0, 60 * 24 * 5)))
            dur = int(self._rng.integers(50, 4000))
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "trace_id": uuid.uuid4().hex,
                    "span_id": uuid.uuid4().hex,
                    "parent_span_id": None,
                    "agent_name": agent,
                    "agent_tier": tier,
                    "operation_type": "agent_run",
                    "started_at": started.isoformat(),
                    "ended_at": (started + timedelta(milliseconds=dur)).isoformat(),
                    "duration_ms": dur,
                    "model_name": "claude-haiku",
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    "status": "ok",
                    "fallback_used": False,
                    "attributes": {"source": "synthetic"},
                    "is_synthetic": True,
                }
            )
        return pd.DataFrame(rows)
