import math
import numbers
from collections import defaultdict
from collections.abc import Mapping

import torch

def _real(value, name, nonnegative):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value

def _values(group):
    for key in ("lr", "betas", "eps", "weight_decay"):
        if key not in group:
            raise ValueError(f"missing {key}")
    lr = _real(group["lr"], "lr", True)
    eps = _real(group["eps"], "eps", True)
    weight_decay = _real(group["weight_decay"], "weight_decay", True)
    betas = group["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("betas must have two values")
    beta1 = _real(betas[0], "beta1", False)
    beta2 = _real(betas[1], "beta2", False)
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("betas must be in [0, 1)")
    if group.get("maximize", False):
        raise ValueError("maximize is not supported")
    if group.get("amsgrad", False):
        raise ValueError("amsgrad is not supported")
    return lr, beta1, beta2, eps, weight_decay

def _check_array(value, name):
    if (value.layout != torch.strided or value.dtype != torch.float32
            or value.device.type == "meta"):
        raise ValueError(f"only dense float32 {name}s are supported")

def _cpu_moment(value, param):
    if (not torch.is_tensor(value) or value.layout != torch.strided
            or value.dtype != torch.float32 or value.device.type == "meta"):
        raise ValueError("invalid optimizer moment")
    if tuple(value.shape) != tuple(param.shape):
        raise ValueError("optimizer moment shape mismatch")
    # torch.load(map_location='cpu') already owns these tensors. Like ordinary
    # AdamW, reuse CPU state storage; cloning all moments during eight-rank
    # resume would temporarily double hundreds of GB of host memory.
    result = value.detach().to(device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("non-finite optimizer moment")
    return result

def _cpu_step(value):
    if torch.is_tensor(value):
        if (value.numel() != 1 or value.layout != torch.strided
                or value.dtype == torch.bool or value.is_complex()
                or value.device.type == "meta"):
            raise ValueError("invalid step")
        raw = float(value.detach().to(device="cpu", dtype=torch.float64).item())
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        raw = float(value)
    else: raise ValueError("invalid step")
    if not math.isfinite(raw) or raw < 0 or raw != math.floor(raw):
        raise ValueError("invalid step")
    if raw > torch.finfo(torch.float32).max:
        raise ValueError("step is too large")
    return torch.tensor(raw, dtype=torch.float32, device="cpu")

class CPUStateAdamW(torch.optim.Optimizer):
    """AdamW whose moments and step counters always live on the CPU."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _values(group)
            for param in group["params"]:
                _check_array(param, "parameter")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = []
        with torch.no_grad():
            for group in self.param_groups:
                values = _values(group)
                groups.append((group, *values))
                for param in group["params"]:
                    _check_array(param, "parameter")
                    if param.grad is not None:
                        _check_array(param.grad, "gradient")
            for group, lr, beta1, beta2, eps, weight_decay in groups:
                for param in group["params"]:
                    grad = param.grad
                    if grad is None: continue
                    state = self.state[param]
                    if not state:
                        state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                        state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                    step = state["step"]; step.add_(1)
                    param_cpu = param.detach().to(device="cpu", copy=True)
                    grad_cpu = grad.detach().to(device="cpu", copy=True)
                    param_cpu.mul_(1 - lr * weight_decay)
                    state["exp_avg"].lerp_(grad_cpu, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        grad_cpu, grad_cpu, value=1 - beta2)
                    step_number = float(step.item())
                    bias_correction1 = 1 - beta1 ** step_number
                    bias_correction2 = 1 - beta2 ** step_number
                    bias_correction2_sqrt = bias_correction2 ** 0.5
                    denom = (state["exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(eps)
                    param_cpu.addcdiv_(state["exp_avg"], denom, value=-(lr / bias_correction1))
                    param.copy_(param_cpu)
        return loss

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        if "state" not in state_dict or "param_groups" not in state_dict:
            raise ValueError("malformed state_dict")
        saved_state = state_dict["state"]
        saved_groups = state_dict["param_groups"]
        if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, (list, tuple)):
            raise ValueError("malformed state_dict")
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("param group count mismatch")
        id_to_param = {}
        for current, saved in zip(self.param_groups, saved_groups):
            if not isinstance(saved, Mapping):
                raise ValueError("malformed param group")
            ids = saved.get("params")
            if not isinstance(ids, (list, tuple)) or len(ids) != len(current["params"]):
                raise ValueError("param group size mismatch")
            for pid, param in zip(ids, current["params"]):
                if not isinstance(pid, int) or isinstance(pid, bool) or pid in id_to_param:
                    raise ValueError("invalid or duplicate parameter id")
                id_to_param[pid] = param

        staged_state = defaultdict(dict)
        for pid, entry in saved_state.items():
            if pid not in id_to_param or not isinstance(entry, Mapping):
                raise ValueError("state does not match parameters")
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("malformed optimizer state")
            param = id_to_param[pid]
            staged_state[param] = dict(
                step=_cpu_step(entry["step"]),
                exp_avg=_cpu_moment(entry["exp_avg"], param),
                exp_avg_sq=_cpu_moment(entry["exp_avg_sq"], param))

        staged_groups = []
        for saved in saved_groups:
            _values(saved)
            staged_groups.append({k: v for k, v in saved.items() if k != "params"})

        self.state.clear()
        self.state.update(staged_state)
        for current, options in zip(self.param_groups, staged_groups):
            params = current["params"]
            current.clear()
            current.update(options)
            current["params"] = params
