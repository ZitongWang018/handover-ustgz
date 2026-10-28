"""Miles custom loss: GRPO advantages with SAO's direct two-sided IS mask."""
from collections.abc import Callable

import torch

from algorithms.dis_config import DEFAULT_DIS_RATIO_HIGH, DEFAULT_DIS_RATIO_LOW
from algorithms.dis_core import dis_policy_terms
from miles.utils.args.schema import A, Arg, BaseConfig


class DISConfig(BaseConfig):
    dis_ratio_low: A[float, Arg(help="Strict lower bound on learner/actual-rollout ratio")] = DEFAULT_DIS_RATIO_LOW
    dis_ratio_high: A[float, Arg(help="Strict upper bound on learner/actual-rollout ratio")] = DEFAULT_DIS_RATIO_HIGH


def _load_miles_ops():
    # The launcher/schema can import this hook without importing GPU kernels.
    # Real training uses the pinned Miles TP/CP response-alignment primitives.
    from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
    from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
    from miles.backends.training_utils.loss.hub.math_utils import compute_approx_kl
    return get_log_probs_and_entropy, get_local_response_loss_masks, compute_approx_kl


def loss_function(args, batch: dict, logits: torch.Tensor, sum_of_sample_mean: Callable):
    """Use Miles' standard four-argument custom-loss contract and scaling.

    DIS masks the PG term. Reference KL and entropy are independent regularizers,
    kept on all original action tokens, exactly as in the existing baseline.
    Thus an out-of-region token has zero *PG* gradient; with nonzero KL/entropy
    coefficients it can still receive a regularizer gradient.
    """
    if args.advantage_estimator != "grpo" or not args.use_rollout_logprobs:
        raise ValueError("GRPO+DIS requires GRPO advantages and recorded rollout log probabilities")
    if args.use_tis or args.custom_tis_function_path is not None or args.use_opsm:
        raise ValueError("DIS must not be stacked with TIS/MIS or OPSM")
    if args.custom_pg_loss_reducer_function_path is not None:
        raise ValueError("DIS retains the original Miles reducer; a custom PG reducer is unsupported")
    if args.use_sampling_support_replay or args.use_unbiased_kl or args.use_opd:
        raise ValueError("This DIS entry does not support filtered-support replay, unbiased KL, or OPD")
    if args.rollout_top_p != 1 or args.rollout_top_k != -1:
        raise ValueError("DIS entry requires unfiltered sampling (top_p=1, top_k=-1)")
    if batch.get("rollout_log_probs") is None:
        raise ValueError("Missing actual generation-time rollout_log_probs; recomputed old scores are not a substitute")
    extract, local_masks, approx_kl = _load_miles_ops()
    entropy_requested = args.entropy_coef != 0 or args.observe_training_entropy
    result = extract(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=entropy_requested, entropy_requires_grad=args.entropy_coef != 0,
        max_seq_lens=batch.get("max_seq_lens"),
    )
    masks = local_masks(batch["total_lengths"], batch["response_lengths"],
                        batch["loss_masks"], args.qkv_format, batch.get("max_seq_lens"))
    scores = result["log_probs"]
    for learner, rollout, advantage, mask in zip(scores, batch["rollout_log_probs"], batch["advantages"], masks, strict=True):
        if not (learner.shape == rollout.shape == advantage.shape == mask.shape):
            raise ValueError("DIS response rows are misaligned after TP/CP slicing")
    learner = torch.cat(scores)
    rollout = torch.cat(batch["rollout_log_probs"])
    advantage = torch.cat(batch["advantages"])
    action_mask = torch.cat(masks).to(device=learner.device)
    terms = dis_policy_terms(learner, rollout, advantage, action_mask,
                             lower=args.dis_ratio_low, upper=args.dis_ratio_high)
    # Crucially do not pass terms.kept as a new reducer mask.
    pg_loss = sum_of_sample_mean(terms.policy_loss)
    loss = pg_loss
    metrics = {
        "pg_loss": pg_loss.detach(),
        "dis_kept_fraction": sum_of_sample_mean(terms.kept.float()).detach(),
        "dis_low_mask_fraction": sum_of_sample_mean(terms.low_rejected.float()).detach(),
        "dis_high_mask_fraction": sum_of_sample_mean(terms.high_rejected.float()).detach(),
        "dis_ratio_weight": sum_of_sample_mean(terms.ratio_weight).detach(),
        "dis_log_ratio": sum_of_sample_mean(terms.log_ratio).detach(),
    }
    active = action_mask.bool()
    entropy_loss = pg_loss.new_zeros(())
    if entropy_requested:
        entropy = torch.cat(result["entropy"])
        if entropy.shape != learner.shape or not torch.isfinite(entropy.detach()[active]).all():
            raise FloatingPointError("Invalid action-token entropy")
        entropy = torch.where(active, entropy, 0.0)
        entropy_loss = sum_of_sample_mean(entropy)
        if args.entropy_coef:
            loss = loss - args.entropy_coef * entropy_loss
    metrics["entropy_loss"] = entropy_loss.detach()
    if args.use_kl_loss:
        if batch.get("ref_log_probs") is None:
            raise ValueError("Missing reference log probabilities for the KL regularizer")
        reference = torch.cat(batch["ref_log_probs"]).detach()
        if reference.shape != learner.shape or not torch.isfinite(reference[active]).all():
            raise FloatingPointError("Invalid action-token reference log probabilities")
        # Sanitize observations before the KL exponential as well.
        actor = torch.where(active, learner, 0.0)
        reference = torch.where(active, reference, 0.0)
        kl = approx_kl(actor, reference, kl_loss_type=args.kl_loss_type)
        if not torch.isfinite(kl.detach()[active]).all():
            raise FloatingPointError("Non-finite reference KL")
        kl_loss = sum_of_sample_mean(torch.where(active, kl, 0.0))
        if args.kl_loss_coef:
            loss = loss + args.kl_loss_coef * kl_loss
        metrics["kl_loss"] = kl_loss.detach()
    if learner.numel() == 0:
        # Empty CP rows still participate in autograd/parallel collectives.
        loss = loss + 0 * logits.sum(dtype=torch.float32)
    metrics["loss"] = loss.detach()
    return loss, metrics


loss_function.config_class = DISConfig
