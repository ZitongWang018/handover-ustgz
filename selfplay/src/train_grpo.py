#!/usr/bin/env python
"""ALFWorld × Qwen3-4B-Instruct 的 GRPO 训练入口（自研实现，不依赖 TRL/vLLM）。

为什么自己写而不是直接用 TRL 的 `GRPOTrainer`：
  * TRL 0.26 的 `rollout_func` 只有在 `use_vllm=True` 时才会被调用，而本机
    vLLM 0.8.5 与 transformers 4.57 不匹配（TRL 要求 vLLM>=0.10.2）；
  * 多轮 + 环境插入观测的长轨迹，需要"只对模型自己产出的词元算 loss"
    这种细粒度控制，自己写反而更短、更可控。

算法（单次迭代的 vanilla GRPO，`beta=0` 不带参考模型）：
  1. 每个 rank 采 `n_tasks_per_iter` 个任务，每个任务采 `group_size` 条独立轨迹；
  2. 轨迹奖励按"同任务组内"标准化得到 advantage（组内无区分度则该组 advantage=0）；
  3. loss = -E[ min(ratio·A, clip(ratio,1±ε)·A) ]，只在 mask=1 的 assistant 词元上算；
  4. DDP 全归约 → clip → step。

用法（8 卡）：
    torchrun --nproc_per_node=8 src/train_grpo.py --config configs/train_qwen3_4b.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---- 必须在 import textworld / alfworld 之前装好兼容层 -------------------------
from alfworld_compat import install as install_alfworld_compat  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config, task_instance_id, task_type_of  # noqa: E402
from grpo_norm import grpo_reduce  # noqa: E402
from reward import RewardConfig, group_advantages, make_reward_cfg, summarize_grouping  # noqa: E402
from rollout import HFPolicy, RolloutConfig, Trajectory, generation_mode, run_rollouts  # noqa: E402


# --------------------------------------------------------------------------- 分布式


def dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def setup_dist() -> Tuple[int, int, int]:
    rank, world, local = dist_info()
    if world > 1:
        torch.cuda.set_device(local)
        # NCCL 默认 timeout 只有 10 分钟：某个 rank 卡在 rollout / 评测（其它 rank 在
        # barrier 干等）超过 10 分钟就会被 watchdog 判死直接 abort 整队。
        # ALFWorld 单步推理最长实测要好几分钟，统一放宽到 30 分钟。
        dist.init_process_group(backend="nccl", rank=rank, world_size=world,
                                timeout=timedelta(minutes=30))
    return rank, world, local


def is_main(rank: int) -> bool:
    return rank == 0


def log0(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


# --------------------------------------------------------------------------- 工具


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def build_task_pool(config: dict, cfg: dict, rank: int, world: int) -> List[Dict[str, Any]]:
    """每个 rank 拿一份互不重叠的训练任务池（按 split 收集后取模切分）。"""
    game_files = collect_game_files(config, cfg["env"]["split"])
    allowed = set(cfg["env"].get("task_types") or [])
    if allowed:
        ids = set()
        for tt in allowed:
            ids.add(tt)
        from alfworld_env import TASK_TYPE_IDS

        names = {TASK_TYPE_IDS[t] for t in ids if t in TASK_TYPE_IDS} | {t for t in allowed if isinstance(t, str)}
        game_files = [g for g in game_files if task_type_of(g) in names]
    game_files = sorted(game_files)
    rng = random.Random(cfg["run"].get("seed", 42))
    rng.shuffle(game_files)
    mine = game_files[rank::world]
    return [
        {"game_file": g, "task_id": task_instance_id(g), "task_type": task_type_of(g)}
        for g in mine
    ]


# --------------------------------------------------------------------------- 模型


def build_model_and_tokenizer(cfg: dict, local_rank: int, world: int, wrap_ddp: bool = True):
    """构建模型 + tokenizer。

    `wrap_ddp=False` 时**只**返回未包装的 HF 模型（但仍然搬到 cuda:local_rank）。
    Explorer-Zero 必须用 False：它要在模型上再挂**两个 LoRA adapter**，而 DDP 的
    all-reduce hook 是在**构造时刻**按当时 requires_grad 的参数集合注册的 ——
    先 wrap 再挂 adapter，adapter 的梯度永远不会被同步（静默错）。
    EZ 的流程是：建模型 → 挂双 adapter → 让两个 adapter 都 trainable → 才 DDP。
    """
    import torch

    # ⚠️ 单卡（world==1）时原来没有任何地方把模型搬到 GPU，
    #    模型会留在 CPU 上，第一次 forward 直接报
    #    "Expected all tensors to be on the same device"。这里统一处理。
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    mcfg = cfg["model"]
    model_path = os.path.expandvars(os.path.expanduser(str(mcfg["path"])))
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"模型目录不存在：{model_path}\n"
            f"  配置里的 model.path = {mcfg['path']!r}；请 export ZTWANG_MODEL_PATH=<本地模型目录>"
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        mcfg.get("dtype", "bfloat16")
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    attn = mcfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - flash-attn 缺失时自动退回 sdpa
        if attn == "flash_attention_2":
            print(f"[model] flash_attention_2 不可用（{exc}），退回 sdpa")
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            raise
    model.config.use_cache = False
    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    # ⚠️⚠️ DDP **不会**帮你把模型搬到 GPU：device_ids 只是声明各 rank 用哪张卡，
    #    module 本身必须先 .to(device)。否则 DDP 包到的是 CPU 模型，立刻报
    #        "DistributedDataParallel device_ids ... but got module parameters {device(type='cpu')}"
    #    （实测 job 12986001，2026-10-12，8 卡起跑 2 分钟即挂）。
    #    之前写成 elif 分支，导致**只有单卡路径会搬 GPU、多卡路径永远不搬**，
    #    而所有冒烟都是单卡跑的，所以这个雷一直藏到第一次真 8 卡才爆。
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda", local_rank))
    if world > 1 and wrap_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return model, tokenizer


# --------------------------------------------------------------------------- loss


#: `per_token_logps` 里 fp32 log-softmax 的分块大小。
#: 不分块时要物化 B×W×vocab×4B 的 fp32 张量并全部留给 backward ——
#: Qwen3 的 vocab=151936，W=8192、B=2 时就是 20GB，直接把 80G 卡打爆。
#: 分块后每块只有 chunk×vocab×4B，且中间结果逐块释放。
_LOGSOFTMAX_CHUNK = 512


def _autocast_bf16(model):
    """参数是 fp32 时开 bf16 autocast（省激活显存 + 提速）；bf16 时关掉（避免重复开销）。

    ⚠️ 为什么必须支持 fp32 参数：纯 bf16 参数配 AdamW 时，更新量会被**舍入抹掉**。
       bf16 相对精度约 2^-8，|w|≈2e-2 时 ulp ≈ 1.6e-4，而 lr=1e-6 的 AdamW 更新量
       只有 ~1e-6 —— 不到 ulp 的 1%，`p.addcdiv_()` 直接舍回原值。
       后果极隐蔽：loss 非零、梯度非零、日志一切正常，但**参数一步都没动**。
       标准解法是 fp32 master weights（这里直接用 fp32 参数）+ bf16 autocast 计算。
    """
    torch = sys.modules.get("torch")
    if torch is None:
        import torch  # noqa: F811
    try:
        dt = next(model.parameters()).dtype
    except StopIteration:
        return torch.autocast(device_type="cuda", enabled=False)
    # ⚠️ `cache_enabled=False`：autocast 默认会把 cast 过的权重缓存成 bf16 副本，
    #    4B 模型 = 额外 8GB 常驻显存。fp32 参数 16G + 梯度 16G + AdamW 32G 已经 64G，
    #    A800 80G 只剩 16G 给激活和 logits，这 8GB 不能省错地方。
    #    代价：每次 forward 重新 cast（~0.05s），相对 3-5 min/iter 可以忽略。
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=(dt == torch.float32), cache_enabled=False)


def per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    chunk: int = _LOGSOFTMAX_CHUNK) -> torch.Tensor:
    """返回**每个位置**的 log p(target_token)。

    数学上就是 `logits.log_softmax(-1).gather(targets)`，但把 softmax 沿序列维分块算，
    避免一次性物化 fp32 的 B×W×vocab 中间张量（这是长上下文 RL 最典型的显存杀手）。
    """
    with _autocast_bf16(model):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]          # 切片是 view，不额外占显存
    targets = input_ids[:, 1:]
    width = logits.size(1)
    if width <= chunk:
        return _chunk_logp(logits, targets)
    parts: List[torch.Tensor] = []
    for s in range(0, width, chunk):
        lc = logits[:, s:s + chunk, :]
        tc = targets[:, s:s + chunk]
        # ⚠️ 光"分块"其实**不省显存**：autograd 会把每块的 fp32 中间结果都留到 backward，
        #    累加起来 = 一次性物化 W×V×4。真正省的办法是对每块套 activation checkpoint，
        #    让 fp32 中间结果在反向时重算 —— 只保留 bf16 输入（本来就在图里）。
        #    W=8192 时峰值从 ~7.5G 降到 ~2.5G，对 fp32 参数（静态已 64G）很关键。
        if torch.is_grad_enabled():
            parts.append(torch.utils.checkpoint.checkpoint(
                _chunk_logp, lc, tc, use_reentrant=False))
        else:
            parts.append(_chunk_logp(lc, tc))
    return torch.cat(parts, dim=1)


def _chunk_logp(logits_chunk: "torch.Tensor", targets_chunk: "torch.Tensor"):
    """对一块 logits 算 log p(target)。输入保持 bf16，`.float()` 放在函数内部，
    这样套 checkpoint 时只需要保存 bf16 输入。"""
    lg = logits_chunk.float()
    return lg.log_softmax(dim=-1).gather(-1, targets_chunk.unsqueeze(-1)).squeeze(-1)


@dataclass
class BatchTensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    advantages: torch.Tensor
    n_active: int


def make_batch(trajs: Sequence[Trajectory], pad_id: int, device, max_len: Optional[int] = None) -> BatchTensors:
    lens = [len(t.input_ids) for t in trajs]
    width = max(lens) if max_len is None else min(max(lens), max_len)
    width = max(width, 2)
    n = len(trajs)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    lmask = torch.zeros((n, width), dtype=torch.float32)
    adv = torch.zeros(n, dtype=torch.float32)
    for i, t in enumerate(trajs):
        L = min(len(t.input_ids), width)
        ids[i, :L] = torch.tensor(t.input_ids[:L], dtype=torch.long)
        attn[i, :L] = 1
        lmask[i, :L] = torch.tensor(t.loss_mask[:L], dtype=torch.float32)
        adv[i] = t.advantage
    return BatchTensors(
        input_ids=ids.to(device),
        attention_mask=attn.to(device),
        loss_mask=lmask.to(device),
        advantages=adv.to(device),
        # ⚠️ 分母口径必须和 `grpo_loss` 里的 `mask = loss_mask[:, 1:]` 一致：
        #    logits[:, t] 预测 token t+1，所以最后一个位置的 loss_mask 不参与。
        #    用全量 lmask.sum() 会让 micro-batch 的权重 `weight = n_active/total_active`
        #    略微偏大（每条轨迹多算 1 个位置），虽然数值上几乎看不出来，
        #    但两个口径混用就是埋雷。
        n_active=int(lmask[:, 1:].sum().item()),
    )


def grpo_loss(
    model,
    batch: BatchTensors,
    old_logps: torch.Tensor,
    clip_eps: float,
    loss_norm: str = "seq",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """GRPO/PPO-clip  surrogate loss。

    ⚠️⚠️ `loss_norm`（2026-10-12 修，交叉 review 时由第二个模型独立指出、我再自己推一遍核实）

        `"seq"`（默认，**论文语义**，DeepSeekMath 公式 4）：
            J = (1/G) Σ_i [ (1/|o_i|) Σ_t ( ... ) ]
            先每条序列内部按 token 平均，再**组间等权**平均。

        `"token"`（旧行为，**有长度偏置**）：
            J = ( Σ_i Σ_t (...) ) / ( Σ_i |o_i| )
            把 G 条序列的 token 混在一起平均 = 按长度加权。

        为什么必须改：`ratio ≡ 1` 时 `per_tok = -adv_i` 在**序列内是常数**，于是

            token 级： loss = - Σ_i |o_i|·adv_i / Σ_i |o_i|    ← 长度加权
            序列级：   loss = - (1/G) Σ_i adv_i                ← 组内等权

        GRPO 归一化后 Σ_i adv_i = 0，所以序列级下"组内奖励对称"必然给出
        净信号 0（这是 GRPO *应有*的性质）；而 token 级下**长度本身变成信号**。
        实测（scripts/ez_repro_lossnorm.py，G=8、长度 20~300、奖励 4 好 4 坏、
        2000 组）：序列级净信号恒为 0，token 级 |净信号| 均值 0.149、最大 0.614
        —— 方向完全由长度分布决定。极端例子：G=2、长度 20 vs 200、
        一好一坏，token 级权重是 (0.091, 0.909)，长序列独占 91%。
        这就是 Dr.GRPO 点名的 length bias：模型会去学"写长/写短"，而不是"答对"。
    """
    new_logps = per_token_logps(model, batch.input_ids, batch.attention_mask)
    # 对齐 mask：logp[:, i] 预测的是 input_ids[:, i+1]
    mask = batch.loss_mask[:, 1:]
    ratio = torch.exp(new_logps - old_logps)
    adv = batch.advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_tok = -torch.min(surr1, surr2)
    denom = mask.sum().clamp(min=1.0)
    # 归约口径单独放在 `grpo_norm.grpo_reduce`（只依赖 torch），
    # 这样离线单测能直接测到**跑的这份代码**，而不是测一份复刻实现。
    loss = grpo_reduce(per_tok, mask, loss_norm=loss_norm)

    with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > clip_eps).float()
        stats = {
            # ★ 这个式子看着像随手写的残差，其实是标准的 **k3 KL 估计**
            #   （Schulman）：令 Δ = new - old，则
            #       E_old[ exp(Δ) - 1 - Δ ] = Σ_old·(new/old) - 1 - E_old[Δ]
            #                               = 1 - 1 - E_old[Δ] = KL(old ‖ new)
            #   一阶项 -Δ 就是 k1，k3 比 k1 多了个非负的二阶项，方差更低且无偏。
            #   别"顺手改成 (old - new).mean()" —— 那退化成 k1 了。
            "kl_approx": float(((ratio - 1.0) - (new_logps - old_logps)).mean().item()) if mask.sum() > 0 else 0.0,
            "clip_frac": float((clipped * mask).sum().item() / max(denom.item(), 1.0)),
            "ratio_mean": float(((ratio * mask).sum() / denom).item()),
        }
    return loss, stats


# --------------------------------------------------------------------------- checkpoint


def _save_model_manual(model, out_dir: Path) -> None:
    """自己写 config + state_dict，**绕开** `save_pretrained`。

    ⚠️⚠️ 为什么不能用 `save_pretrained`（这条是实测踩出来的，不是理论推测）：

        save_pretrained
          -> transformers.modeling_utils.unwrap_model
          -> accelerate.utils.other.extract_model_from_parallel
          -> `from deepspeed import DeepSpeedEngine`
          -> deepspeed 在 **import 时**就要 CUDA_HOME：
                 deepspeed/ops/op_builder/builder.py:51
                 raise MissingCUDAException("CUDA_HOME does not exist, ...")

    也就是说：**只要环境里装了 deepspeed 而 CUDA_HOME 没设，任何一次
    save_pretrained 都会把进程打死**，跟你用不用 deepspeed 毫无关系。
    实测（二期 A40，job 12978649，2026-10-12）：iter 0 跑完，iter 1 第一次
    存断点时整个作业 FAILED(exit 1)，300 卡时的整机作业会在 iter 25 死在这儿。

    这里手写落盘，彻底不碰 accelerate / deepspeed。
    """
    model.config.save_pretrained(str(out_dir))
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.save_pretrained(str(out_dir))
        except Exception:  # noqa: BLE001 - generation_config 写不出来不影响权重
            pass

    # tied weight（Qwen3 的 lm_head 与 embed_tokens 共享存储）只留一份，
    # 否则 safetensors 会以 "Some tensors share memory" 拒绝写入。
    # ⚠️ `data_ptr()` 必须在 `.to("cpu", copy=True)` **之前**取：
    #    copy 之后每个张量都是独立新分配，ptr 必然互不相同，去重就退化成死代码，
    #    tied weight 会被存两份（fp32 4B 白大 ~2GB）。
    state: Dict[str, Any] = {}
    _ptrs: Dict[int, str] = {}
    for k, v in model.state_dict().items():
        p = v.data_ptr()
        if p in _ptrs:
            continue
        _ptrs[p] = k
        state[k] = v.detach().to("cpu", copy=True)
    try:
        from safetensors.torch import save_file
        save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    except Exception:  # noqa: BLE001 - safetensors 不可用就退回 torch.save
        torch.save(state, str(out_dir / "pytorch_model.bin"))


def save_checkpoint(model, tokenizer, out_dir: Path, tag: str, cfg: dict, extra: dict) -> None:
    """只保留最新一个断点：写完新的，把旧的删掉（用户铁律二）。"""
    ckpt_dir = out_dir / "ckpt"
    tmp_dir = out_dir / "ckpt.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    _save_model_manual(unwrapped, tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    with open(tmp_dir / "train_state.json", "w") as fh:
        json.dump({"tag": tag, "config": cfg, **extra}, fh, indent=2, default=str)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp_dir.rename(ckpt_dir)


# --------------------------------------------------------------------------- 评测


@torch.no_grad()
def evaluate(env_eval, policy: HFPolicy, tasks: Sequence[Dict[str, Any]], cfg: dict, rollout_cfg: RolloutConfig,
             reward_cfg: RewardConfig, rank: int, world: int) -> Dict[str, float]:
    """贪心（temperature→0）跑一遍评测集。只在 rank 0 上跑，其它 rank 直接对齐。"""
    if world > 1 and rank != 0:
        return {}
    n = min(len(tasks), cfg["train"].get("eval_tasks", 16))
    if n == 0:
        return {}
    # 评测 batch 必须整除 env batch_size —— 评测环境按需临时建，batch_size = n（单条轨迹）
    eval_cfg = RolloutConfig(**{**rollout_cfg.__dict__, "group_size": 1, "temperature": 0.0})
    # ⚠️ tokenize=False：评测只用 stats（won/invalid/steps），不需要训练词元序列。
    #    默认 True 会给每条评测轨迹白跑 build_token_sequence（16 条 × W~6k 的渲染+tokenize），
    #    纯浪费 CPU/时间，还会在评测这个"本来就只发生在 rank0"的环节里拖长其它 rank 的 barrier 等待。
    with generation_mode(policy.model):
        res = run_rollouts(env_eval, policy, list(tasks[:n]), eval_cfg, reward_cfg, tokenize=False)
    won = sum(1 for t in res if t.stats.won)
    invalid = sum(t.stats.n_invalid for t in res)
    steps = sum(t.stats.n_steps for t in res)
    return {
        "eval/success_rate": won / n,
        "eval/invalid_action_rate": invalid / max(steps, 1),
        "eval/mean_steps": steps / n,
        "eval/n": float(n),
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只做一轮 rollout，不更新参数")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    rank, world, local_rank = setup_dist()
    set_seed(cfg["run"].get("seed", 42), rank)

    run_dir = (SRC_DIR.parent / cfg["run"]["output_dir"] / cfg["run"]["name"]).resolve()
    # ⚠️ **所有 rank 都要 mkdir**，不能只让 rank0 建：
    #    文件系统（Lustre/GPFS）上 mkdir 之后其它节点的 dentry 缓存可能还没同步，
    #    rank0 建完、rank7 立刻 open 会撞 No such file or directory。
    #    8 卡作业排十几个小时才起来，死在这种地方最亏。
    run_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        import torch.distributed as _dist
        _dist.barrier()
    log_fh = open(run_dir / f"train_rank{rank}.jsonl", "a")

    # ---- 兼容层 ----
    compat = install_alfworld_compat(verbose=is_main(rank))

    # ---- 配置 ----
    alf_cfg = load_alfworld_config(
        str((SRC_DIR.parent / cfg["env"]["config_path"]).resolve())
        if not os.path.isabs(cfg["env"]["config_path"])
        else cfg["env"]["config_path"]
    )
    rollout_cfg = RolloutConfig(**{**cfg["rollout"], "max_steps": cfg["env"]["max_episode_steps"]})
    reward_cfg = make_reward_cfg(cfg.get("reward"))
    # ⚠️ group_size 的**唯一权威来源是 `env.group_size`**。
    #    RolloutConfig 里那个同名默认值是 8，yaml 的 `rollout:` 段里又没有这一项，
    #    所以直接 `g = rollout_cfg.group_size` 会静默拿到 8 —— 配置里写 4 实际跑 8。
    #    两处都写了就要求一致，避免以后又埋一个"配置骗人"的坑。
    g = int(cfg["env"].get("group_size", rollout_cfg.group_size))
    if "group_size" in cfg.get("rollout", {}) and int(cfg["rollout"]["group_size"]) != g:
        raise SystemExit(
            f"group_size 冲突：env.group_size={g} vs rollout.group_size={cfg['rollout']['group_size']}；"
            "请只保留 env.group_size（rollout 段不要写这一项）"
        )
    rollout_cfg.group_size = g
    n_tasks = cfg["env"]["n_tasks_per_iter"]
    n_slots = n_tasks * g

    # ---- 任务池 ----
    pool = build_task_pool(alf_cfg, cfg, rank, world)
    log0(rank, f"[data] rank{rank}: {len(pool)} train tasks  ({pool[0]['task_type'] if pool else '-'})")
    if not pool:
        raise SystemExit("任务池为空")

    # ---- 环境 ----
    env = ALFWorldBatchEnv(alf_cfg, split=cfg["env"]["split"], batch_size=n_slots, seed=cfg["run"].get("seed", 42))
    log0(rank, f"[env] batch_size={n_slots} max_episode_steps={env.max_episode_steps} pool={len(env.game_files)}")

    # ---- 模型 ----
    model, tokenizer = build_model_and_tokenizer(cfg, local_rank, world)
    policy = HFPolicy(model.module if hasattr(model, "module") else model, tokenizer)
    log0(rank, f"[model] {cfg['model']['path']}")

    # ⚠️ 自检：把「参数dtype / attention实现 / GC / autocast」这四件事打进日志。
    #    这几个组合只要错一个，症状都是"能跑、但结果全废"或者"第一次forward就崩"，
    #    而日志里没有任何一行会告诉你。fp32 + flash_attention_2 尤其危险：
    #    transformers 只在 stderr 打一条 warning 就放过（实测 job 12978649 里它确实
    #    自己 cast 回 bf16 跑通了），所以必须显式确认到底用的哪条路。
    _m = model.module if hasattr(model, "module") else model
    log0(rank, "[model] dtype=%s attn=%s gradient_checkpointing=%s autocast=%s" % (
        next(_m.parameters()).dtype,
        getattr(getattr(_m, "config", None), "_attn_implementation", "?"),
        bool(getattr(_m, "is_gradient_checkpointing", False)),
        "bf16" if next(_m.parameters()).dtype == torch.float32 else "off",
    ))

    trainable = [p for p in model.parameters() if p.requires_grad]
    # 优化器可选：fp32 参数下 AdamW 的 m/v 要 32G，A800 80G 会很紧（静态 ~64G）。
    # 装不下时把 train.optimizer 改成 adafactor —— 它几乎不存二阶动量，能省掉这 32G。
    _opt_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    if _opt_name == "adafactor":
        try:
            from transformers import Adafactor
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"要用 adafactor 但 transformers.Adafactor 不可用：{exc}")
        optimizer = Adafactor(
            trainable,
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
            scale_parameter=False,
            relative_step=False,
        )
        log0(rank, "[opt] Adafactor（省 ~32G 优化器状态）")
    elif _opt_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                                      weight_decay=cfg["train"].get("weight_decay", 0.0))
    else:
        raise SystemExit(f"未知 optimizer={_opt_name!r}（可选 adamw / adafactor）")
    # 用 `is not None` 而不是 `or`：显式传 --max-iters 0（只跑构建/冒烟自检）时
    # `or` 会把 0 当 falsy 吃掉，退回 yaml 里的 total_iters，直接跑满全程。
    total_iters = args.max_iters if args.max_iters is not None else cfg["train"]["total_iters"]
    warmup = cfg["train"].get("warmup_iters", 20)

    def lr_at(it: int) -> float:
        if it < warmup:
            return cfg["train"]["lr"] * (it + 1) / max(1, warmup)
        prog = (it - warmup) / max(1, total_iters - warmup)
        return cfg["train"]["lr"] * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    micro = cfg["train"].get("micro_batch_size", 4)
    clip_eps = cfg["train"].get("clip_eps", 0.2)
    max_grad_norm = cfg["train"].get("max_grad_norm", 1.0)
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    # ---- 评测环境（只在 rank0 用）----
    env_eval = None
    eval_tasks: List[Dict[str, Any]] = []
    if is_main(rank) and cfg["train"].get("eval_every", 0):
        eval_cfg_alf = load_alfworld_config(
            cfg["env"]["config_path"] if os.path.isabs(cfg["env"]["config_path"])
            else str((SRC_DIR.parent / cfg["env"]["config_path"]).resolve())
        )
        games = sorted(collect_game_files(eval_cfg_alf, cfg["env"]["eval_split"]))
        rng_eval = random.Random(cfg["run"].get("seed", 42) + 777)
        rng_eval.shuffle(games)
        eval_tasks = [
            {"game_file": gg, "task_id": task_instance_id(gg), "task_type": task_type_of(gg)}
            for gg in games[: cfg["train"].get("eval_tasks", 16)]
        ]
        env_eval = ALFWorldBatchEnv(
            eval_cfg_alf, split=cfg["env"]["eval_split"], batch_size=len(eval_tasks), seed=1234
        )
        log0(rank, f"[eval] {len(eval_tasks)} held-out tasks from {cfg['env']['eval_split']}")

    history: List[Dict[str, Any]] = []
    t_start = time.time()
    for it in range(total_iters):
        it_t0 = time.time()
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(it)

        # ---- 采样任务 ----
        rng = random.Random(cfg["run"].get("seed", 42) * 1_000_003 + rank * 10_007 + it)
        tasks = [pool[rng.randrange(len(pool))] for _ in range(n_tasks)]

        # ---- rollout ----
        # ⚠️ `generation_mode` 在**整段 rollout 外面**一次性关掉 GC（generate 开着 GC
        #    会让 Qwen3 输出乱码 -> 动作 100% 非法 -> 奖励同档 -> advantage=0）。
        #    放在外面而不是每步 generate 里开关，是因为 enable() 每次都会往
        #    embeddings 挂一个新 forward hook，9000 次就是 9000 个 hook。
        with torch.no_grad(), generation_mode(model):
            trajs = run_rollouts(env, policy, tasks, rollout_cfg, reward_cfg,
                                 log_fn=(lambda m: log0(rank, f"[iter {it}] {m}")))

        # ---- advantage ----
        # ⚠️ 这里依赖 run_rollouts 的返回契约：**task-major / 组连续**。
        #    textworld 的 batch env 在 reset() 时会把 gamefiles 打乱
        #    (textworld/gym/envs/textworld_batch.py:96-117)，如果 run_rollouts 忘了
        #    按 slot_task 重排，下面的 [i // g] 就会把**不同 task** 混进同一组，
        #    GRPO 的组内归一化 advantage 会静默算错。所以这里显式断言守一下。
        if len(trajs) % g != 0:
            raise RuntimeError(f"轨迹条数 {len(trajs)} 不是 group_size={g} 的整数倍")
        for _i in range(0, len(trajs), g):
            _grp = trajs[_i:_i + g]
            if len({os.path.realpath(t.game_file) for t in _grp}) != 1:
                raise RuntimeError(
                    "GRPO 分组被污染：同一个 group 里混入了不同 task 的轨迹 "
                    f"(indices {_i}..{_i + g - 1}, task_ids={[t.task_id for t in _grp]})。"
                    "请检查 run_rollouts 是否按 slot_task 重排成 task-major 顺序。"
                )
        group_ids = [i // g for i in range(len(trajs))]
        advs = group_advantages([t.reward for t in trajs], group_ids,
                                use_std=cfg["train"].get("adv_use_std", True))
        # 课程权重 task_type_weights 作用在**组内归一化之后**的 advantage 上：
        # 同组所有轨迹归一化前 reward 乘同一个 w 是严格无效的（被 std 归一化约掉），
        # 只有乘在归一化后的 advantage 上才真的改变梯度权重。
        _ttw = getattr(reward_cfg, "task_type_weights", None) or {}
        for t, a in zip(trajs, advs):
            t.advantage = a * _ttw.get(t.task_type, 1.0)
        gstats = summarize_grouping([t.reward for t in trajs], group_ids)

        # ⚠️ 奖励符号自检：非法动作数与奖励的相关系数**必须为负**。
        #    若为正，说明 penalty 项符号写反了（`r += penalty` 而不是 `r -=`）——
        #    此时"输出非法动作"反而拿高分，模型会被直接训坏。
        #    这个错误极其隐蔽：advantage 非零、std 非零、loss 非零，
        #    所有基于数值的异常检测全部失效，只有这个相关性指标能抓到。
        try:
            _inv = [float(t.stats.n_invalid) for t in trajs]
            _rw = [float(t.reward) for t in trajs]
            if len(set(_inv)) > 1 and len(set(_rw)) > 1:
                invalid_reward_corr = float(np.corrcoef(_inv, _rw)[0, 1])
            else:
                invalid_reward_corr = 0.0
        except Exception:  # noqa: BLE001
            invalid_reward_corr = 0.0

        # ---- 训练 ----
        order = list(range(len(trajs)))
        n_micro = math.ceil(len(order) / micro)
        # 所有 rank 的轨迹条数相同 -> micro 数相同 -> DDP 同步次数一致
        adv_abs_mean = float(np.mean([abs(a) for a in advs]))
        # ⚠️ "这一步是否真的更新"必须各 rank 一致，否则 DDP 会挂死；用 MAX 归约统一决策
        step_flag = torch.tensor([1.0 if adv_abs_mean > 1e-8 else 0.0], device=next(model.parameters()).device)
        if world > 1:
            dist.all_reduce(step_flag, op=dist.ReduceOp.MAX)
        did_step = (args.dry_run is False) and step_flag.item() > 0.5
        stats_acc: Dict[str, float] = {}
        # 下面两个是"参数真的动了吗"的实测指标（见 optimizer.step() 处的说明）
        update_rel = 0.0
        param_norm = 0.0
        update_abs = 0.0
        loss_val = 0.0
        if did_step:
            optimizer.zero_grad(set_to_none=True)
            # 先算 old logps（当前策略，不带梯度）
            batches: List[Tuple[BatchTensors, torch.Tensor]] = []
            with torch.no_grad():
                for k in range(n_micro):
                    chunk = [trajs[i] for i in order[k * micro:(k + 1) * micro]]
                    bt = make_batch(chunk, pad_id, device)
                    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
                    batches.append((bt, old))
            total_active = sum(bt.n_active for bt, _ in batches) or 1
            # 除最后一个 micro-batch 外都用 no_sync() 挂起 all-reduce：每个 micro 的
            # backward 都会触发一次 DDP 通信，micro 数一多通信量翻倍。no_sync 上下文
            # 会在退出时自动恢复 require_backward_grad_sync，所以不需要手动再设回 True。
            for k, (bt, old) in enumerate(batches):
                _sync_ctx = (model.no_sync() if (world > 1 and k < len(batches) - 1)
                             else contextlib.nullcontext())
                with _sync_ctx:
                    loss, st = grpo_loss(model, bt, old, clip_eps)
                    weight = bt.n_active / total_active
                    (loss * weight).backward()
                loss_val += float(loss.item()) * weight
                for kk, vv in st.items():
                    stats_acc[kk] = stats_acc.get(kk, 0.0) + vv * weight
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            # ⚠️ 实测"参数真的动了吗"。
            #    纯 bf16 参数 + AdamW 小 lr 时，更新量远小于 bf16 的 ulp，会被舍入抹掉：
            #    loss 非零、grad_norm 非零、日志一切正常，但参数一步都没动。
            #    抽一个中等大小的参数张量，记下 step 前后的相对变化量 —— 这是唯一能
            #    直接抓到该问题的指标，所以务必写进日志。
            _probe = trainable[len(trainable) // 2]
            _before = _probe.detach().clone()
            optimizer.step()
            with torch.no_grad():
                update_abs = float((_probe.detach() - _before).norm().item())
                param_norm = float(_probe.detach().norm().item())
            update_rel = update_abs / max(param_norm, 1e-12)

        # ⚠️ 这里曾经写成 `"won_rate": gstats["reward_mean"]` —— 等于把奖励均值**复制一份**
        #    当胜率。后果是日志里出现 `won_rate=1.6` 这种 >1 的"胜率"，
        #    `[iter N] won=...` 那行也跟着骗人，真实胜率被完全掩盖。
        #    胜率必须从 `t.stats.won` 现场数。
        _n = max(1, len(trajs))
        _steps = sum(t.stats.n_steps for t in trajs)
        row = {
            "iter": it,
            "lr": lr_at(it),
            **gstats,
            "won_rate": sum(1 for t in trajs if t.stats.won) / _n,
            # 补三个"能直接看出是模型不行还是奖励不行"的诊断量：
            #   mean_steps 贴住 max_steps 且 invalid_rate=1  -> 模型根本没吐出可解析的动作
            #   mean_steps 贴住 max_steps 且 invalid_rate≈0 -> 模型动作合法但任务做不完
            "mean_steps": _steps / _n,
            "invalid_rate": (sum(t.stats.n_invalid for t in trajs) / _steps) if _steps else 0.0,
            "format_viol_rate": (sum(t.stats.n_format_violation for t in trajs) / _steps) if _steps else 0.0,
            "adv_abs_mean": adv_abs_mean,
            "loss": loss_val,
            # 奖励符号自检（正 = penalty 符号写反了，模型正在被训成输出非法动作）
            "invalid_reward_corr": invalid_reward_corr,
            # 参数更新有效性（抓"bf16 舍入把更新抹掉"这类静默失败）
            "update_rel": update_rel,
            "update_abs": update_abs,
            "param_norm": param_norm,
            **{f"train/{k}": v for k, v in stats_acc.items()},
            "iter_sec": time.time() - it_t0,
        }

        # ---- 评测 ----
        if env_eval is not None and cfg["train"].get("eval_every", 0) and (
            (it + 1) % cfg["train"]["eval_every"] == 0 or it == total_iters - 1
        ):
            ev = evaluate(env_eval, policy, eval_tasks, cfg, rollout_cfg, reward_cfg, rank, world)
            row.update(ev)
        if world > 1:
            dist.barrier()

        # ---- 保存 ----
        # save_every=0 表示"训练中不周期存"，但**最终模型必须落盘**（否则跑完 24h
        # 什么都没留下）。所以最后一轮无条件保存。
        _se = cfg["train"].get("save_every", 0)
        if is_main(rank) and ((_se and (it + 1) % _se == 0) or it == total_iters - 1):
            save_checkpoint(model, tokenizer, run_dir, f"iter{it+1}", cfg, {"iter": it + 1})
            log0(rank, f"[ckpt] saved {run_dir / 'ckpt'} (iter {it + 1})")

        history.append(row)
        log0(rank, "[iter %d] won=%.3f adv=%.3f loss=%.4f clip=%.3f lr=%.2e %.1fs" % (
            it, row["won_rate"], adv_abs_mean, loss_val, stats_acc.get("clip_frac", 0.0), row["lr"], row["iter_sec"]))
        if log_fh is not None:
            log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_fh.flush()

    if is_main(rank):
        with open(run_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)
        log0(rank, f"[done] {total_iters} iters in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")
    env.close()
    if env_eval is not None:
        env_eval.close()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
