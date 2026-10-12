#!/usr/bin/env python
"""B：测试那个测试 —— 证明两个新守卫「错的时候真的会炸」。

思路：不重写守卫逻辑，而是**驱动真实 main()**，只把外部世界换成假的：
* 用 `FakeEnv` 模拟 textworld batch env：`reset(games)` 返回一个**非恒等置换**的 used
  （模拟 shuffle 打乱 slot）。
* 用**真实的** `rollout.run_rollouts`（所以真的会走 _align_slot_tasks 与重排代码）。
* 用 monkeypatch 把 `rollout._align_slot_tasks` 换成**恒等** -> 模拟"未修复"。

对照：
  ① 恒等 align（未修复）+ train_grpo.main() -> 期望抛 RuntimeError（分组断言）
  ② 真实 align（已修复）    + train_grpo.main() -> 期望正常跑完
  ③ 恒等 align（未修复）+ eval_alfworld.main() -> 期望抛 RuntimeError（对齐守卫）
  ④ 真实 align（已修复）    + eval_alfworld.main() -> 期望正常跑完

不需要 GPU、不需要 textworld、不需要集群。

运行：
    python scripts/verify_guards.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import rollout as R  # noqa: E402

FAILS: list[str] = []
REAL_ALIGN = R._align_slot_tasks


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name} :: {detail}")


def identity_align(games, used, group_size):
    """模拟"未修复"：假设 slot 顺序 == 请求顺序（slot_task[s] = s // g），完全不看 used。

    这就是打乱前的旧假设；返回值一定落在 [0, n_slots//g) 内，不会越界。
    """
    return [i // group_size for i in range(len(games))]


# --------------------------------------------------------------------------- 假环境


class FakeView:
    def __init__(self, env_desc="You are in a room.", task_desc="put apple in fridge"):
        self.raw = env_desc
        self.env_desc = env_desc
        self.task_desc = task_desc


class FakeStep:
    def __init__(self, n, won=True):
        self.obs = ["You see nothing special."] * n
        self.rewards = [0.0] * n
        self.dones = [True] * n          # 一步结束，跑得快
        self.infos = {}
        self.game_files = []
        self.won = [won] * n
        self.admissible = [["look", "go to shelf 1"]] * n
        self.expert_plan = [None] * n


class FakeEnv:
    """模拟 textworld batch env：reset 返回**非恒等置换**的 used（模拟 slot 被打乱）。

    用一个"左旋一位"的置换（而不是简单反转）：对 games=[A,A,B,B] 得到 used=[A,B,B,A]，
    这样在"未修复"的恒等对齐下，前两个 slot 恰好是 A,B —— 不同 task —— 正好触发分组断言；
    而反转 [B,B,A,A] 会侥幸凑成同组，验证不到东西。
    """

    def __init__(self, batch_size=1, max_episode_steps=6, **_kw):
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.game_files = ["/fake/task0/game.tw-pddl", "/fake/task1/game.tw-pddl"]
        self._used = []

    def reset(self, game_files=None, seed=None):
        gf = list(game_files or [])
        used = (gf[1:] + gf[:1]) if len(gf) > 1 else list(gf)   # 左旋一位
        self._used = used
        obs = ["welcome\n\nYou are in a room.\n\nYour task is to: put apple in fridge"] * len(gf)
        views = [FakeView() for _ in obs]
        adm = [["look", "go to shelf 1"]] * len(gf)
        return obs, views, adm, used, [None] * len(gf)

    def step(self, actions):
        return FakeStep(len(actions))

    def close(self):
        pass


# --------------------------------------------------------------------------- 假 tokenizer / 策略


class FakeTok:
    """只实现 build_token_sequence 需要的两个方法；逐字符编码保证前缀稳定。"""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 997 for c in text]}


class FakePolicy:
    """模拟 HFPolicy：render/tokenize_text 与 generate 返回的 token 必须自洽。"""

    def __init__(self, tok):
        self.tok = tok

    def render(self, messages, add_generation_prompt=False):
        return self.tok.apply_chat_template(list(messages))

    def tokenize_text(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def truncate_messages(self, messages, cfg):
        # 假对话很短，永不触发裁剪；只需满足 run_rollouts 的接口
        return list(messages)

    def generate(self, messages_list, cfg=None):
        content = "Thought: look\nAction: look"
        gens = [self.tokenize_text(content) for _ in messages_list]
        texts = [content + "\n" for _ in messages_list]
        return texts, gens, [len(g) for g in gens]


# --------------------------------------------------------------------------- ① / ②


def run_train_guard(broken: bool):
    import torch

    import train_grpo as T

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="guard_train_")).as_posix()
    cfg = {
        "run": {"name": "guardtest", "seed": 42, "output_dir": tmp},
        "model": {"path": "x", "dtype": "float32", "attn_implementation": "sdpa",
                  "gradient_checkpointing": False},
        "env": {"config_path": "configs/base_config.yaml", "split": "train",
                "eval_split": "eval_out_of_distribution", "max_episode_steps": 6,
                "n_tasks_per_iter": 2, "group_size": 2, "task_types": []},
        "rollout": {"group_size": 2, "max_new_tokens": 8, "temperature": 1.0, "top_p": 0.95,
                    "repetition_penalty": 1.05, "max_context_tokens": 8192,
                    "keep_recent_turns": 12, "invalid_action_mode": "raw", "fuzzy_threshold": 0.6},
        "reward": None,
        "train": {"total_iters": 1, "lr": 1e-6, "weight_decay": 0.0, "warmup_iters": 1,
                  "clip_eps": 0.2, "max_grad_norm": 1.0, "adv_use_std": True,
                  "micro_batch_size": 2, "save_every": 0, "eval_every": 0, "eval_tasks": 0},
    }

    saved = {}
    for name, val in {
        "load_cfg": lambda path: cfg,
        "setup_dist": lambda: (0, 1, 0),
        "install_alfworld_compat": lambda **kw: None,
        "load_alfworld_config": lambda path, overrides=None: {},
        "collect_game_files": lambda config, split="train": [f"/fake/task{i}/game.tw-pddl" for i in range(4)],
        "build_task_pool": lambda config, c, rank, world: [
            {"game_file": f"/fake/task{i}/game.tw-pddl", "task_id": f"t{i}", "task_type": "tt"}
            for i in range(4)
        ],
        "ALFWorldBatchEnv": lambda *a, **k: FakeEnv(batch_size=k.get("batch_size", 4),
                                                   max_episode_steps=6),
        "build_model_and_tokenizer": lambda c, lr, w: (torch.nn.Linear(3, 3), FakeTok()),
        "HFPolicy": lambda model, tok: FakePolicy(tok),
    }.items():
        saved[name] = getattr(T, name)
        setattr(T, name, val)

    old_argv = sys.argv
    sys.argv = ["train_grpo.py", "--config", "x", "--dry-run", "--max-iters", "1"]
    R._align_slot_tasks = identity_align if broken else REAL_ALIGN
    try:
        rc = T.main()
        err = None
    except BaseException as exc:  # noqa: BLE001
        rc, err = None, exc
    finally:
        sys.argv = old_argv
        R._align_slot_tasks = REAL_ALIGN
        for name, val in saved.items():
            setattr(T, name, val)
    return rc, err


# --------------------------------------------------------------------------- ③ / ④


def run_eval_guard(broken: bool):
    import eval_alfworld as E

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="guard_eval_")).as_posix()
    fake_gfs = ["/fake/task0/game.tw-pddl", "/fake/task1/game.tw-pddl"]

    saved = {}
    for name, val in {
        "install_alfworld_compat": lambda **kw: None,
        "load_alfworld_config": lambda path, overrides=None: {},
        "collect_game_files": lambda cfg, split: list(fake_gfs),
        "ALFWorldBatchEnv": lambda *a, **k: FakeEnv(batch_size=k.get("batch_size", 2), max_episode_steps=6),
    }.items():
        saved[name] = getattr(E, name)
        setattr(E, name, val)

    old_argv = sys.argv
    sys.argv = ["eval_alfworld.py", "--agent", "random", "--split", "train",
                "--n", "2", "--batch", "2", "--max-steps", "6", "--out-dir", tmp]
    R._align_slot_tasks = identity_align if broken else REAL_ALIGN
    try:
        rc = E.main()
        err = None
    except BaseException as exc:  # noqa: BLE001
        rc, err = None, exc
    finally:
        sys.argv = old_argv
        R._align_slot_tasks = REAL_ALIGN
        for name, val in saved.items():
            setattr(E, name, val)
    return rc, err


# --------------------------------------------------------------------------- main


def main() -> int:
    print("\n== B1. train_grpo 分组断言 ==")
    rc, err = run_train_guard(broken=True)
    check("① 恒等 align（未修复）-> 分组断言抛 RuntimeError",
          isinstance(err, RuntimeError) and "分组被污染" in str(err),
          f"rc={rc} err={err!r}")
    if isinstance(err, RuntimeError):
        print(f"       抛出：{str(err)[:120]}")

    rc, err = run_train_guard(broken=False)
    check("② 真实 align（已修复）-> 正常跑完（rc=0，无异常）",
          err is None and rc == 0, f"rc={rc} err={err!r}")

    print("\n== B2. eval_alfworld 对齐守卫 ==")
    rc, err = run_eval_guard(broken=True)
    check("③ 恒等 align（未修复）-> 对齐守卫抛 RuntimeError",
          isinstance(err, RuntimeError) and "对不上" in str(err),
          f"rc={rc} err={err!r}")
    if isinstance(err, RuntimeError):
        print(f"       抛出：{str(err)[:120]}")

    rc, err = run_eval_guard(broken=False)
    check("④ 真实 align（已修复）-> 正常跑完（rc=0，无异常）",
          err is None and rc == 0, f"rc={rc} err={err!r}")

    print("\n" + ("B 全部通过 ✅" if not FAILS else f"B 失败 {len(FAILS)} 项 ❌"))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)
