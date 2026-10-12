"""ALFWorld 文本环境的自研适配层（不依赖 AgentGym，不需要 HTTP 服务）。

设计要点
--------
* 直接用 `alfworld.agents.environment.alfred_tw_env.AlfredTWEnv` + `textworld.gym`
  的**批量**环境（`AsyncBatchEnv`，每个 slot 一个独立子进程 → 世界编译并行）。
* 每轮迭代通过 `env.gamefiles = [...]` + `env.seed(s)` 控制这一批到底玩哪些局，
  再用 `infos["extra.gamefile"]` 反查每个 slot 实际拿到的是哪一局
  （GRPO 要按"同一局"分组算 advantage，所以这个反查是必需的）。
* 动作空间**始终用 `admissible_commands` 约束**——ALFWorld 是稀疏奖励 + 长时域，
  不收窄动作空间基本训不动。
* 初始观测的第三段（`Your task is to: ...`）被单独解析成 `task_desc`，
  放进 prompt 的固定位置，避免每步重复。

用法
----
    from alfworld_compat import install
    install()                       # 必须在 import textworld 之前
    from alfworld_env import ALFWorldBatchEnv, load_alfworld_config, collect_game_files

    cfg = load_alfworld_config("configs/base_config.yaml")
    env = ALFWorldBatchEnv(cfg, split="train", batch_size=16)
    obs, views, admissible, used, expert_plan = env.reset([g] * 16)
    step = env.step(["look", ...])   # step.won / step.admissible / step.expert_plan
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SPLITS = {
    "train": "data_path",
    "eval_in_distribution": "eval_id_data_path",   # valid_seen
    "eval_out_of_distribution": "eval_ood_data_path",  # valid_unseen
}

TASK_TYPE_IDS = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# `Your task is to: put a hot apple in the fridge.` / `Your task is to: ...`
_TASK_RE = re.compile(r"Your task is to:\s*(.*)", re.IGNORECASE)


# --------------------------------------------------------------------------- config


def load_alfworld_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """读 ALFWorld 官方 config，把 `$ALFWORLD_DATA` 展开。

    `overrides` 支持点号路径，例如 {"general.training_method": "dagger"}。
    """
    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    data_root = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = data_root

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    cfg = _expand(cfg)

    # init_env() 会读 general.training_method 来决定 max_episode_steps
    cfg.setdefault("general", {})
    cfg["general"].setdefault("training_method", "dagger")
    cfg.setdefault("dagger", {}).setdefault("training", {}).setdefault("max_nb_steps_per_episode", 50)
    cfg.setdefault("rl", {}).setdefault("training", {}).setdefault("max_nb_steps_per_episode", 50)
    cfg.setdefault("env", {}).setdefault("expert_type", "handcoded")
    cfg.setdefault("env", {}).setdefault("domain_randomization", False)
    cfg.setdefault("env", {}).setdefault("goal_desc_human_anns_prob", 0.0)
    cfg.setdefault("env", {}).setdefault("task_types", list(TASK_TYPE_IDS))

    for dotted, value in (overrides or {}).items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def collect_game_files(config: dict, split: str = "train") -> List[str]:
    """收集某个 split 下所有 game.tw-pddl（走 ALFWorld 自己的过滤逻辑：可解 + task_type 白名单）。"""
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    env = AlfredTWEnv(config, train_eval=split)
    return list(env.game_files)


def task_type_of(game_file: str) -> str:
    """从 game 路径里取任务类型（目录名形如 `pick_and_place_simple-Mug-None-CoffeeTable-208`）。"""
    for part in reversed(game_file.split(os.sep)):
        if part.endswith(".tw-pddl"):
            continue
        head = part.split("-")[0]
        if head in TASK_TYPE_IDS.values():
            return head
    return "unknown"


def task_instance_id(game_file: str) -> str:
    """去掉 data root 前缀，得到一个短而唯一的实例 id。"""
    parts = game_file.rstrip("/").split(os.sep)
    # .../json_2.1.1/<split>/<task_type>-.../<trial_...>/game.tw-pddl
    return "/".join(parts[-3:-1])


# --------------------------------------------------------------------------- parsing


def parse_initial_obs(raw: str) -> Tuple[str, str]:
    """ALFWorld 的初始观测是 `welcome\\n\\nenv_desc\\n\\ntask_desc`，返回 (env_desc, task_desc)。"""
    parts = raw.split("\n\n")
    if len(parts) >= 3:
        return parts[1].strip(), parts[2].strip()
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return raw.strip(), ""


def extract_task_desc(raw: str) -> str:
    m = _TASK_RE.search(raw)
    return m.group(1).strip() if m else ""


def clean_step_obs(raw: str) -> str:
    """后续 step 的观测去掉 TextWorld 的欢迎头。"""
    if "=-" in raw.split("\n")[0]:
        raw = "\n".join(raw.split("\n")[2:])
    return raw.strip()


@dataclass
class ObsView:
    """一次 `reset()` 的结果拆解。"""

    raw: str
    env_desc: str
    task_desc: str


def parse_obs(raw: str) -> ObsView:
    env_desc, task_desc = parse_initial_obs(raw)
    if not task_desc:
        task_desc = extract_task_desc(raw)
    return ObsView(raw=raw, env_desc=env_desc, task_desc=task_desc)


# --------------------------------------------------------------------------- env


def _extract_expert_plan(infos: Dict[str, Any], n_slots: int) -> List[Optional[str]]:
    """从 batch infos 里取出官方 expert 的下一步动作计划。

    `AlfredExpert` 会把 `state["extra.expert_plan"]` 写成**单元素 list**（例如
    `["go to desk 1"]`；内部出错时回退 `["look"]`）。不启用 expert 时 infos 里没有这个
    键，返回全 None（向后兼容）。
    """
    plans = infos.get("extra.expert_plan")
    if not plans:
        return [None] * n_slots
    out: List[Optional[str]] = []
    for p in plans:
        if isinstance(p, (list, tuple)):
            p = p[0] if p else None
        out.append(str(p) if p else None)
    if len(out) < n_slots:
        out.extend([None] * (n_slots - len(out)))
    return out[:n_slots]


@dataclass
class BatchStep:
    obs: List[str]
    rewards: List[float]
    dones: List[bool]
    infos: Dict[str, List[Any]]
    game_files: List[str] = field(default_factory=list)
    won: List[bool] = field(default_factory=list)
    admissible: List[List[str]] = field(default_factory=list)
    expert_plan: List[Optional[str]] = field(default_factory=list)


class ALFWorldBatchEnv:
    """`textworld.gym` 批量环境的一层薄封装。

    注意：`init_env(batch_size=N)` 会拉起 N 个常驻子进程；之后每轮 `reset(games)`
    只是在子进程里"卸载旧局 + 装载新局"，不会重建进程。装载（PDDL 编译）是主要开销。
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        batch_size: int = 1,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        game_files: Optional[Sequence[str]] = None,
        with_expert: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split 必须是 {list(SPLITS)} 之一，收到 {split!r}")

        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        if max_episode_steps is not None:
            config = _shallow_copy_cfg(config)
            config["dagger"]["training"]["max_nb_steps_per_episode"] = max_episode_steps
            config["rl"]["training"]["max_nb_steps_per_episode"] = max_episode_steps

        self.split = split
        self.batch_size = batch_size
        self.seed_value = seed
        self.with_expert = with_expert

        self._alfred = AlfredTWEnv(config, train_eval=split)
        if game_files is not None:
            self._alfred.game_files = list(game_files)
            self._alfred.num_games = len(self._alfred.game_files)
        self.game_files: List[str] = list(self._alfred.game_files)
        if not self.game_files:
            raise RuntimeError(f"split={split} 下没有可用 game 文件，检查 ALFWORLD_DATA 是否正确")

        self.env = self._make_env(batch_size=batch_size)  # AsyncBatchEnv 包装
        self._max_steps = int(
            config["dagger"]["training"]["max_nb_steps_per_episode"]
            if config["general"]["training_method"] == "dagger"
            else config["rl"]["training"]["max_nb_steps_per_episode"]
        )

    def _make_env(self, batch_size: int):
        """注册并构造 `textworld.gym` 批量环境。

        这段逻辑照抄官方 `AlfredTWEnv.init_env`（`third_party/.../alfworld/agents/
        environment/alfred_tw_env.py`）—— 即"把官方逻辑复制到我们自己的文件里"，
        **不修改官方包**。唯一区别：`with_expert=True` 时**无条件**追加 `AlfredExpert`
        wrapper 并请求 `expert_plan`；官方只在 `train_eval == "train"` 时才加这两行，
        导致评测 split 上拿不到 expert 计划。
        """
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredExpert,
            AlfredInfos,
        )

        config = self._alfred.config
        domain_randomization = config["env"]["domain_randomization"]
        if self._alfred.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = config["env"]["expert_type"]
        training_method = config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = config["dagger"]["training"]["max_nb_steps_per_episode"]
        else:
            raise NotImplementedError(f"未知 training_method={training_method!r}")

        if self.with_expert:
            wrappers.append(AlfredExpert(expert_type))
            request_infos.extras.append("expert_plan")

        env_id = textworld.gym.register_games(
            self._alfred.game_files,
            request_infos,
            batch_size=batch_size,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    # ---------------------------------------------------------------- gym API

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    def reset(self, game_files: Optional[Sequence[str]] = None, seed: Optional[int] = None) -> Tuple[List[str], List[ObsView], List[List[str]], List[str], List[Optional[str]]]:
        """装载并 reset 一批局。

        Returns:
            (obs, obs_views, admissible_commands, game_files_used, expert_plan)

            `expert_plan[i]` 是官方 expert 给出的该 slot 下一步动作（`with_expert=False`
            时全为 None）。
        """
        if game_files is None:
            import random

            game_files = random.sample(self.game_files, self.batch_size)
        game_files = list(game_files)
        if len(game_files) != self.batch_size:
            raise ValueError(f"需要恰好 {self.batch_size} 个 game，收到 {len(game_files)}")

        # 这一批玩哪些局由我们说了算：重设 gamefiles + 重新 seed 迭代器
        self.env.gamefiles = game_files
        self.env.seed(self.seed_value if seed is None else seed)

        obs, infos = self.env.reset()
        used = list(infos["extra.gamefile"])
        admissible = [list(a) for a in infos["admissible_commands"]]
        views = [parse_obs(o) for o in obs]
        expert_plan = _extract_expert_plan(infos, self.batch_size)
        return list(obs), views, admissible, used, expert_plan

    def step(self, actions: Sequence[str]) -> BatchStep:
        obs, scores, dones, infos = self.env.step(list(actions))
        return BatchStep(
            obs=list(obs),
            rewards=[float(s) for s in scores],
            dones=[bool(d) for d in dones],
            infos=infos,
            game_files=list(infos.get("extra.gamefile", [])),
            won=[bool(w) for w in infos.get("won", [False] * self.batch_size)],
            admissible=[list(a) for a in infos.get("admissible_commands", [[]] * self.batch_size)],
            expert_plan=_extract_expert_plan(infos, self.batch_size),
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - 子进程已退出时不必再报错
            pass

    def __del__(self):
        self.close()


def _shallow_copy_cfg(cfg: dict) -> dict:
    import copy

    return copy.deepcopy(cfg)
