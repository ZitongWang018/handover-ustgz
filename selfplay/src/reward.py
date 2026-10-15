"""Per-episode counters shared by rollout and training."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrajectoryStats:
    won: bool = False
    n_steps: int = 0
    n_valid_steps: int = 0
    n_invalid: int = 0
    n_format_violation: int = 0
    n_repeat: int = 0
    truncated: bool = False
    task_type: str = "unknown"
    task_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
