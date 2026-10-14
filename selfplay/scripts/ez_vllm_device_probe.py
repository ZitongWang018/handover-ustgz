#!/usr/bin/env python
"""P2 前置检查（单进程版）：在 `CUDA_VISIBLE_DEVICES=<i>` 下建 vLLM 引擎，报告它落在哪。

背景：EZ 训练原来用 `torchrun --nproc_per_node=8`，8 个 rank **都看得到 8 张卡**，
靠 `torch.cuda.set_device(local_rank)` 各就各位。但 **vLLM 的 `LLM()` 没有 device 参数**，
它按 `CUDA_VISIBLE_DEVICES` 自己挑卡 —— 8 个 rank 的引擎会全部挤到物理卡 0（OOM）。

试过并**否决**的办法：先 `torch.cuda.set_device(local)` 再改 `os.environ["CUDA_VISIBLE_DEVICES"]`。
实测直接崩（job 13009409）：
    IndexError: list index out of range
    torch/_dynamo/device_interface.py: caching_worker_device_properties["cuda"][device]
因为 torch 的设备缓存已经按 8 张卡建好了，环境变量一改就对不上号。

采用的方案（本脚本验证的那一个）：**启动时就只暴露一张卡**。
    CUDA_VISIBLE_DEVICES=<i> RANK=<i> WORLD_SIZE=8 LOCAL_RANK=0 python src/ez_train.py ...
这样训练侧和 vLLM 子进程看到的都只有一张卡（= 物理卡 i），双方一致，零代码改动。
"""
from __future__ import annotations

import os
import subprocess
import sys
import torch

TOK = os.environ.get("ZTWANG_MODEL_PATH", "")


def main() -> int:
    tag = os.environ.get("PROBE_TAG", os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
    if not TOK:
        raise SystemExit("set ZTWANG_MODEL_PATH")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    # 模拟"训练侧模型已经在卡上"：先占 256MiB，制造父进程的 CUDA 上下文。
    torch.cuda.set_device(0)
    hold = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()

    from vllm import LLM, SamplingParams
    eng = LLM(model=TOK, tokenizer=TOK, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.35, max_model_len=2048,
              enable_lora=True, max_lora_rank=32, max_num_seqs=8,
              enforce_eager=True, disable_log_stats=True)
    out = eng.generate([{"prompt_token_ids": [9707, 1879, 374]}],
                       SamplingParams(max_tokens=8, temperature=1.0), use_tqdm=False)
    print(f"[probe tag={tag}] CUDA_VISIBLE_DEVICES={visible} | "
          f"torch_visible={torch.cuda.device_count()} | "
          f"engine_tokens={list(out[0].outputs[0].token_ids)} | OK", flush=True)
    del hold
    return 0


if __name__ == "__main__":
    sys.exit(main())
