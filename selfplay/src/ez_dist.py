"""Small, dependency-light distributed helpers used only by Explorer-Zero.

EZ uses raw PEFT models because the active adapter changes between phases.  These
helpers make the cross-rank algorithm explicit without wrapping the model in DDP.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, List, Sequence, Tuple


def local_group_size(global_group_size: int, world_size: int) -> int:
    if world_size < 1 or global_group_size < world_size or global_group_size % world_size:
        raise ValueError(f"group_size={global_group_size} must be >= and divisible by world_size={world_size}")
    return global_group_size // world_size


def global_group_advantages(rewards: Sequence[float], ddof: int = 1) -> List[float]:
    """Population/ sample-standardize one complete global GRPO group."""
    import numpy as np
    x = np.asarray(rewards, dtype=np.float64)
    if not len(x):
        return []
    std = float(x.std(ddof=ddof)) if len(x) > ddof else 0.0
    if std < 1e-8:
        return [0.0] * len(x)
    return ((x - x.mean()) / (std + 1e-8)).astype(float).tolist()


def shard_group(values: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    if len(values) % world_size:
        raise ValueError(f"global group length {len(values)} is not divisible by world_size={world_size}")
    return list(values[rank::world_size])


def gather_group_values(local_values: Sequence[Any], rank: int, world_size: int,
                        dist_module=None) -> List[Any]:
    """Gather rank-strided group members in their original global order."""
    if world_size == 1:
        return list(local_values)
    dist = dist_module
    if dist is None:
        import torch.distributed as dist
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, list(local_values))
    if any(len(part) != len(local_values) for part in gathered):
        raise ValueError("all ranks must contribute the same local group size")
    return [gathered[r][i] for i in range(len(local_values)) for r in range(world_size)]


def choose_pool_indices(pool_size: int, count: int, step: int, seed: int) -> List[int]:
    """Deterministic shuffled-cycle sampling with unique ids inside each step."""
    if pool_size <= 0 or count <= 0:
        return []
    if count > pool_size:
        raise ValueError(f"cannot select {count} distinct tasks from pool of {pool_size}")
    order = list(range(pool_size))
    random.Random(seed).shuffle(order)
    cursor = step * count
    offset = cursor % pool_size
    return (order[offset:] + order[:offset])[:count]


def broadcast_pool_indices(pool_size: int, count: int, step: int, seed: int,
                           rank: int, world_size: int, dist_module=None) -> List[int]:
    if rank == 0:
        selected = choose_pool_indices(pool_size, count, step, seed)
    else:
        selected = None
    if world_size > 1:
        dist = dist_module
        if dist is None:
            import torch.distributed as dist
        obj = [selected]
        dist.broadcast_object_list(obj, src=0)
        selected = obj[0]
    return selected or []


def active_lora_parameters(model, adapter: str) -> List[Tuple[str, Any]]:
    """Return active adapter parameters in a stable name order."""
    found = [(n, p) for n, p in model.named_parameters()
             if "lora_" in n and adapter in n]
    return sorted(found, key=lambda x: x[0])


def broadcast_adapter_parameters(model, world_size: int, dist_module=None) -> None:
    """Make both freshly-created adapters identical before any rank-local update."""
    if world_size <= 1:
        return
    dist = dist_module
    if dist is None:
        import torch.distributed as dist
    params = sorted(((n, p) for n, p in model.named_parameters() if "lora_" in n),
                    key=lambda x: x[0])
    if not params:
        raise RuntimeError("no LoRA parameters found to initialize across ranks")
    for _name, param in params:
        dist.broadcast(param.data, src=0)


def select_adapter(model, adapter: str) -> None:
    """Select one PEFT adapter and verify its trainability boundary."""
    model.set_adapter(adapter)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        should_train = adapter in name
        if bool(param.requires_grad) != should_train:
            param.requires_grad_(should_train)


def sync_active_adapter_gradients(model, adapter: str, world_size: int,
                                  bucket_mb: int = 25, dist_module=None) -> None:
    """Average gradients in fixed-order buckets; missing local grads are zeros.

    Inactive adapter gradients are rejected, which prevents AdamW momentum from
    carrying updates across proposer/solver phases.
    """
    import torch
    if world_size == 1:
        for name, p in model.named_parameters():
            if "lora_" in name and adapter not in name and p.grad is not None:
                raise RuntimeError(f"inactive adapter has gradient: {name}")
        return
    dist = dist_module
    if dist is None:
        import torch.distributed as dist
    params = active_lora_parameters(model, adapter)
    if not params:
        raise RuntimeError(f"no LoRA parameters found for adapter {adapter!r}")
    if any("lora_" in n and adapter not in n and p.grad is not None
           for n, p in model.named_parameters()):
        raise RuntimeError("inactive adapter has a gradient before optimizer step")
    max_elems = max(1, bucket_mb * 1024 * 1024 // 4)
    entries = []
    for name, p in params:
        if not p.requires_grad:
            raise RuntimeError(f"active adapter parameter unexpectedly frozen: {name}")
        grad = p.grad if p.grad is not None else torch.zeros_like(p)
        entries.append((name, p, grad))
    # Buckets are assembled from the same sorted parameter sequence on every rank.
    bucket: List[Any] = []
    size = 0
    def flush():
        nonlocal bucket, size
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for _, _, g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        pos = 0
        for _name, p, _g in bucket:
            n = p.numel()
            synced = flat[pos:pos+n].view_as(p)
            if p.grad is None:
                p.grad = synced.clone()
            else:
                p.grad.copy_(synced)
            pos += n
        bucket, size = [], 0
    for entry in entries:
        n = entry[1].numel()
        if bucket and size + n > max_elems:
            flush()
        bucket.append(entry)
        size += n
        if size >= max_elems:
            flush()
    flush()
    finite = torch.ones((), dtype=torch.int32, device=entries[0][1].device)
    for _name, p in params:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            finite.zero_()
            break
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f"non-finite synchronized gradient in adapter {adapter}")


def check_adapter_sync(model, adapter: str, world_size: int, dist_module=None,
                       atol: float = 0.0) -> float:
    """Return maximum true parameter deviation from rank zero; raises if unequal."""
    if world_size == 1:
        return 0.0
    import torch
    dist = dist_module
    if dist is None:
        import torch.distributed as dist
    maximum = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    nonfinite = torch.zeros((), device=maximum.device, dtype=torch.int32)
    for name, p in active_lora_parameters(model, adapter):
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        if not bool(torch.isfinite(p.detach()).all()):
            nonfinite.fill_(1)
        delta = (p.detach() - ref).abs().max() if p.numel() else maximum.new_zeros(())
        maximum = torch.maximum(maximum, delta.float())
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
    value = float(maximum.item())
    if bool(nonfinite.item()) or not torch.isfinite(maximum) or value > atol:
        raise RuntimeError(f"adapter {adapter} parameters diverged across ranks: max_abs={value}")
    return value


def require_all_ranks_finite(local_finite: bool, device, world_size: int,
                             dist_module=None, label: str = "training value") -> None:
    """Raise synchronously on every rank if any rank observed a non-finite value."""
    import torch
    finite = torch.tensor([1 if local_finite else 0], dtype=torch.int32, device=device)
    if world_size > 1:
        dist = dist_module
        if dist is None:
            import torch.distributed as dist
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f"non-finite {label} on at least one rank")


def require_adapter_finite(model, adapter: str, world_size: int,
                           dist_module=None, label: str = "updated adapter") -> None:
    """Check active adapter weights locally, then fail collectively."""
    import torch
    params = active_lora_parameters(model, adapter)
    finite = bool(params) and all(bool(torch.isfinite(p.detach()).all()) for _, p in params)
    require_all_ranks_finite(finite, next(model.parameters()).device, world_size,
                             dist_module, label=f"{label} ({adapter})")
