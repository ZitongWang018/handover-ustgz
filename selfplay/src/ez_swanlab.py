#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SwanLab 上报的**防御式**薄封装。

# ===========================================================================
# ⛔ 唯一铁律：上报失败绝对不能影响训练。
# ===========================================================================

使用方 distinguished 三种失败都要能让训练照常跑完：

    1. 环境里没装 swanlab          → ImportError → 静默降级，全部 log 变 no-op
    2. 装了但网络不通（三期无外网） → mode="local"，写本地日志文件，事后 `swanlab sync`
    3. API key 失效 / 服务端报错    → 吞掉异常，只警告**一次**（否则每步刷屏）

另外两条必须遵守的细节：
    - **只在 rank0 上报**。8 个 rank 各自 log 同名指标会把云端曲线搞成锯齿。
    - **`step` 必须显式传**。不传的话 SwanLab 按调用次数自增，两个相位交替着
      log（先 P 再 S）就会把两条曲线画在同一个 x 上。

用法（在 ez_train.py 里）：

    from ez_swanlab import EzSwanLab
    rep = EzSwanLab(cfg, run_dir=run_dir, rank=rank,
                    phase_step=0, total_steps=N*(M_P+M_S))
    rep.log({"train/loss": loss_val, "train/grad_norm": gn}, step=global_step)
    rep.finish()

环境变量：
    SWANLAB_API_KEY  云端上传密钥（也可以在 swanlab 登录后本机 ~/.swanlab 里）
    EZ_SWANLAB_MODE  auto（默认，能连云端就云端，连不上自动转 local）
                     local（强制只写本地 logdir，事后 sync）
                     off（完全关闭）
    EZ_SWANLAB_PROJECT / EZ_SWANLAB_WORKSPACE   覆盖项目名/工作空间
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


class EzSwanLab:
    """SwanLab 上报包装：任何异常都被吞掉，最坏情况退化成 no-op。"""

    def __init__(self, cfg: Dict[str, Any], run_dir: Path, rank: int = 0,
                 extra_config: Optional[Dict[str, Any]] = None) -> None:
        self._rank = rank
        self._warned = False
        self._run = None
        self._swanlab = None
        self._step_cache: Dict[int, float] = {}
        self.mode = "off"

        # 只在 rank0 上报；其它 rank 直接变成 no-op 对象。
        if rank != 0:
            return

        if os.environ.get("EZ_SWANLAB_MODE", "auto").strip().lower() == "off":
            return

        try:
            import swanlab  # noqa: PLC0415 - 延迟导入：没装也不能让主程序挂
        except Exception:
            self._warn("未安装 swanlab（pip install swanlab），本次训练不上报指标")
            return

        self._swanlab = swanlab
        project = os.environ.get("EZ_SWANLAB_PROJECT", "self-play")
        workspace = os.environ.get("EZ_SWANLAB_WORKSPACE", "ZitongWang")

        tcfg = cfg.get("train", {}) or {}
        config = {
            "model": (cfg.get("model", {}) or {}).get("path", "?"),
            "lora_r": (cfg.get("model", {}) or {}).get("lora_r", "?"),
            "seed": (cfg.get("run", {}) or {}).get("seed", "?"),
            "group_size_G": tcfg.get("group_size", "?"),
            "group_size_G_P": tcfg.get("solve_rate_rollouts", "?"),
            "tasks_per_step": tcfg.get("tasks_per_step", "?"),
            "outer_loops": tcfg.get("outer_loops", "?"),
            "lr": tcfg.get("lr", "?"),
            "clip_eps": tcfg.get("clip_eps", "?"),
            "max_grad_norm": tcfg.get("max_grad_norm", "?"),
            # ★ 这两个是 2026-10-12 算法核查后改的口径，必须跟着曲线一起记录，
            #   否则以后回看实验分不清是哪次改动带来的差异。
            "loss_norm": tcfg.get("loss_norm", "seq"),
            "adv_ddof": tcfg.get("adv_ddof", 1),
        }
        if extra_config:
            config.update(extra_config)

        # mode=auto 时：先试云端；连不上自动降级到 local（离线 logdir），
        # 这样三期那种完全无外网的集群也能先把数据存下来，事后再 sync 上去。
        m = os.environ.get("EZ_SWANLAB_MODE", "auto").strip().lower()
        logdir = str(Path(run_dir) / "swanlog")
        if m == "local":
            modes = ["local"]
        else:
            modes = ["cloud", "local"]

        for cand in modes:
            try:
                self._run = swanlab.init(
                    project=project,
                    workspace=workspace,
                    experiment_name=Path(run_dir).name,
                    config=config,
                    mode=cand,
                    logdir=logdir,
                )
                self.mode = cand
                break
            except Exception as exc:                      # noqa: BLE001
                self._once(f"swanlab.init(mode={cand}) 失败：{type(exc).__name__}: {exc}")
                continue

        if self._run is None:
            self._once("SwanLab 初始化全部失败，本次训练不上传指标（不影响训练本身）")
        else:
            tail = "" if self.mode == "cloud" else f"（离线，事后 `swanlab sync {logdir}`）"
            print(f"[swanlab] mode={self.mode} project={project} "
                  f"run={Path(run_dir).name} {tail}", flush=True)

    # ------------------------------------------------------------------ 内部

    def _once(self, msg: str) -> None:
        """只警告一次：训练每步都要 log，异常刷屏会把真正的训练日志淹了。"""
        if not self._warned:
            self._warned = True
            print(f"[swanlab] {msg}", flush=True)

    def _warn(self, msg: str) -> None:
        self._once(msg)

    # ------------------------------------------------------------------ 对外

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """上报一组标量。step 必须显式给（见模块 docstring 的踩坑说明）。"""
        if self._run is None:
            return
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue                                  # 非标量直接丢，不报错
        if not clean:
            return
        try:
            self._swanlab.log(clean, step=step)
        except Exception as exc:                          # noqa: BLE001
            self._once(f"swanlab.log 失败（后续不再重试）：{type(exc).__name__}: {exc}")
            self._run = None                              # 失败一次就彻底关掉，避免每步都撞一次异常

    def alert(self, title: str, text: str) -> None:
        """把严重告警（比如「solver 全 0 奖励」）发到 SwanLab，方便手机端看到。"""
        if self._run is None:
            return
        try:
            self._swanlab.alert(title=title, text=text)
        except Exception:                                 # noqa: BLE001
            pass                                          # alert 失败无关紧要

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._swanlab.finish()
        except Exception:                                 # noqa: BLE001
            pass
        finally:
            self._run = None
