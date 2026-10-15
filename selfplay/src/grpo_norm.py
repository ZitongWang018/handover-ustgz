"""Sequence-level reduction for the clipped policy objective."""
from __future__ import annotations

import torch

SEQUENCE_LEVEL = "seq"
TOKEN_LEVEL = "token"


def grpo_reduce(per_tok: torch.Tensor,
                mask: torch.Tensor,
                loss_norm: str = SEQUENCE_LEVEL,
                sequence_reduction: str = "mean") -> torch.Tensor:
    """Reduce masked per-token losses; the trainer uses sequence-level means."""
    if sequence_reduction not in ("mean", "sum"):
        raise ValueError("sequence_reduction must be 'mean' or 'sum'")
    if loss_norm == TOKEN_LEVEL:
        if sequence_reduction == "sum":
            raise ValueError("sequence_reduction='sum' is defined only for seq normalization")
        return (per_tok * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_norm != SEQUENCE_LEVEL:
        raise ValueError(f"loss_norm must be {SEQUENCE_LEVEL!r} or {TOKEN_LEVEL!r}")
    active = mask.sum(dim=1)
    sequence_means = (per_tok * mask).sum(dim=1) / active.clamp(min=1.0)
    sequence_means = sequence_means * (active > 0)
    if sequence_reduction == "sum":
        return sequence_means.sum()
    return sequence_means.sum() / (active > 0).sum().clamp(min=1)
