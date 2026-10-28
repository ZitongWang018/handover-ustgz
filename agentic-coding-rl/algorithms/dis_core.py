"""Direct double-sided IS policy-gradient terms, independent of Miles/CUDA.

SAO, https://arxiv.org/html/2607.07508v1, equations (1)--(3).
The ratio is current learner / the probability actually used to sample a token.

We minimize -1{low < r < high} * r * A. Its gradient is exactly the negative
score-function gradient -stop_gradient(1{low < r < high} * r * A) * grad(log p).
This avoids differentiating an IS weight a second time through a weight*log-p
expression. The interval test is independent of the sign of A, unlike PPO min.

Reduction is deliberately left to Miles: rejected tokens are zero terms, not
removed from the original action-token or sequence normalization denominator.
"""
from dataclasses import dataclass
import math

import torch

from algorithms.dis_config import DEFAULT_DIS_RATIO_HIGH, DEFAULT_DIS_RATIO_LOW, validate_dis_bounds


@dataclass(frozen=True)
class DISTerms:
    policy_loss: torch.Tensor
    kept: torch.Tensor
    low_rejected: torch.Tensor
    high_rejected: torch.Tensor
    ratio_weight: torch.Tensor
    log_ratio: torch.Tensor


def dis_policy_terms(
    learner_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    lower: float = DEFAULT_DIS_RATIO_LOW,
    upper: float = DEFAULT_DIS_RATIO_HIGH,
) -> DISTerms:
    """Return unreduced per-token terms; all tensors must be response-aligned.

    Active non-finite input is an error. Non-finite observation/padding entries
    are sanitized *before* subtraction/multiplication/exponentiation. Ratios
    outside the trust region are never exponentiated, so extreme lag cannot
    create inf*0 / NaN gradients. Endpoints are excluded, matching equation (3).
    """
    validate_dis_bounds(lower, upper)
    tensors = (learner_log_probs, rollout_log_probs, advantages, action_mask)
    if any(t.shape != learner_log_probs.shape for t in tensors) or learner_log_probs.ndim != 1:
        raise ValueError("DIS inputs must be 1-D tensors with identical shapes")
    if any(t.device != learner_log_probs.device for t in tensors):
        raise ValueError("DIS inputs must be on the same device")
    if not all(t.is_floating_point() for t in tensors[:3]):
        raise ValueError("Log probabilities and advantages must be floating-point tensors")
    if not torch.all((action_mask == 0) | (action_mask == 1)):
        raise ValueError("DIS action_mask must contain only 0 and 1")
    active = action_mask.bool()
    for name, value in zip(("learner log probabilities", "rollout log probabilities", "advantages"), tensors[:3]):
        if not torch.isfinite(value.detach()[active]).all():
            raise FloatingPointError(f"Non-finite {name} at an active action token")
    dtype = torch.float64 if learner_log_probs.dtype == torch.float64 else torch.float32
    learner = torch.where(active, learner_log_probs.to(dtype), 0.0)
    rollout = torch.where(active, rollout_log_probs.detach().to(dtype), 0.0)
    advantage = torch.where(active, advantages.detach().to(dtype), 0.0)
    log_ratio = learner - rollout
    if not torch.isfinite(log_ratio.detach()[active]).all():
        raise FloatingPointError("Non-finite learner/rollout log ratio")
    low_rejected = active & (log_ratio.detach() <= math.log(lower))
    high_rejected = active & (log_ratio.detach() >= math.log(upper))
    kept = active & ~low_rejected & ~high_rejected
    # Mask before exp, not after: exp(1000)*0 would have an invalid backward.
    safe_log_ratio = torch.where(kept, log_ratio, 0.0)
    ratio = safe_log_ratio.exp()
    weight = torch.where(kept, ratio, 0.0)
    loss = -weight * advantage
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError("Non-finite DIS policy loss")
    return DISTerms(loss, kept, low_rejected, high_rejected, weight.detach(), log_ratio.detach())
