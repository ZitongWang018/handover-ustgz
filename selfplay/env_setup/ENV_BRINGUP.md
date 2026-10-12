# ENV_BRINGUP —— 让 `thinktool`(py3.10) 单进程跑起 ALFWorld 文本环境

> 负责人：env-bringer ｜ 目标集群：HKUST-GZ 二期 `hpc2login.hpc.hkust-gz.edu.cn`（登录节点实为 `mgmt-3`）
> 结论：**Plan A 成功**。`thinktool` 环境无需任何改动，靠 `PYTHONPATH=~/ztwang/selfplay/third_party/py310deps` + glibc 兼容层即可单进程跑通 ALFWorld 文本环境；
> 已完成 `install() → import → init_env → reset → step` 端到端实测（PASS）。

---

## 0. TL;DR

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thinktool
export ALFWORLD_DATA=$HOME/.cache/alfworld
python ~/ztwang/selfplay/env_setup/check_alfworld_thinktool.py --batch-size 2 --steps 5
# => RESULT: PASS
```

* **没有修改** `~/miniconda3/envs/*`、`~/thinktool/`、`models/`、他人目录。全部产物只落在 `~/ztwang/selfplay/`。
* `thinktool` 自带 `numpy 1.26.4` **不需要升级到 2.x**（实测见 §4.1）。
* 除 glibc 兼容层（已有）外，绕过的 3 个坑：`gym` 无 cp310 wheel、`ai2thor` 缺失、`textworld.gym` 的 batch `step()` 语义。

---

## 1. 环境事实（实测）

| 项 | 值 |
|---|---|
| 目标解释器 | `/hpc2hdd/home/xyu608/miniconda3/envs/thinktool/bin/python` = **Python 3.10.19** |
| 主机 | `mgmt-3`，64 vCPU（登录/管理节点） |
| `ALFWORLD_DATA` | `/hpc2hdd/home/xyu608/.cache/alfworld`（3.5G，`json_2.1.1/{train,valid_seen,valid_train,valid_unseen}`、`logic/`、`detectors/`） |
| `thinktool` 已有相关包 | numpy 1.26.4、torch 2.6.0+cu124、torchvision 0.21.0+cu124、cv2 4.12.0、pandas 2.3.3、PyYAML、tqdm、Flask 3.1.2、requests |
| 参考环境（只读） | `~/miniconda3/envs/agentenv-alfworld`（py3.9.25）里有 textworld 1.6.2 / alfworld 0.3.3 / fast_downward_textworld 20.6.4 / gym 0.26.2 / ai2thor 2.1.0 |
| 参考实现（只读） | `/hpc2hdd/home/xyu608/thinktool/rllm/rllm/environments/agentgym/AgentGym/agentenv-alfworld/` |

---

## 2. 从零复现的命令序列

> 全部步骤只写 `~/ztwang/`。所有 `.sh` 均为 LF 换行。
> 下面 5 个脚本就是本次实际执行的脚本，已放在 `~/ztwang/selfplay/env_setup/`，可直接 `bash` 重跑（幂等）。

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate thinktool
cd ~/ztwang/selfplay/env_setup

bash _install_step1.sh              # ① 核心包 + 纯 python 依赖 + 复制 gym
bash _install_step2.sh              # ② ai2thor（--no-deps）
bash _install_step4_sdists.sh       # ④ 3 个无 wheel 包的 sdist + wheel 清单
python check_alfworld_thinktool.py --batch-size 2 --steps 5   # ⑤ 端到端自检
python bench_alfworld_thinktool.py --batches 1 2 4 8 --steps 30   # ⑥ 吞吐基准
```

**① 核心包**（等价于 `_install_step1.sh`）

```bash
PY=$HOME/miniconda3/envs/thinktool/bin/python
T=$HOME/ztwang/selfplay/third_party/py310deps
mkdir -p "$T"

# 二进制/自带 wheel 的包：--no-deps，避免把 numpy/torch 拖进 target 遮蔽 thinktool
"$PY" -m pip install --target "$T" --no-cache-dir --no-deps \
    textworld==1.6.2 fast_downward_textworld==20.6.4 alfworld==0.3.3

# 纯 python / 轻量依赖
"$PY" -m pip install --target "$T" --no-cache-dir \
    "numpy==1.26.4" "tqdm>=4.17.1" cffi networkx more_itertools tatsu==5.8.3 \
    hashids jericho mementos termcolor prompt_toolkit pyyaml

# gym 0.26.2：PyPI 只有 sdist，且纯 python —— 直接从只读的 py3.9 环境复制
A=$HOME/miniconda3/envs/agentenv-alfworld/lib/python3.9/site-packages
cp -r "$A/gym" "$A/gym_notices" "$A"/gym-0.26.2.dist-info "$A"/gym_notices-0.1.0.dist-info "$T/"
```

**② ai2thor**

```bash
"$PY" -m pip install --target "$T" --no-cache-dir --no-deps ai2thor==2.1.0
"$PY" -m pip install --target "$T" --no-cache-dir progressbar2
```

**③ 使用方式（重要）**：`py310deps` 必须插在 `sys.path` **最前面**，且 glibc 兼容层必须在 `import textworld` **之前**：

```python
import os, sys
repo = os.path.expanduser("~/ztwang/selfplay")
sys.path.insert(0, os.path.join(repo, "src"))                       # alfworld_compat
sys.path.insert(0, os.path.join(repo, "third_party", "py310deps"))  # ← 必须在 thinktool site-packages 之前
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

from alfworld_compat import install
install()                                   # 必须在 import textworld 之前

import textworld, gym, alfworld             # noqa
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
```

> ⚠️ 更干净的做法（推荐给主 agent）：不要靠 `sys.path.insert`，而是
> `PYTHONPATH=$HOME/ztwang/selfplay/third_party/py310deps:$HOME/ztwang/selfplay/src`。
> 两者等价；但**不要**用 `pip install -e` 或写 `.pth` 到 `thinktool` 里（那会改动环境）。

---

## 3. 依赖清单（安装到 `py310deps/` 的包）

`py310deps` 体积 **864 MB**，134 个顶层条目，65 个 dist-info。完整 pin 清单见
`third_party/alfworld_py310deps_requirements.txt`（脚本生成）与 `third_party/WHEELS_MANIFEST.txt`。

关键包：

| 包 | 版本 | 来源 |
|---|---|---|
| textworld | 1.6.2 | PyPI wheel `cp310-manylinux_2_17_x86_64` |
| alfworld | 0.3.3 | PyPI wheel `py3-none-any` |
| fast_downward_textworld | 20.6.4 | PyPI **sdist** 现编（无 wheel） |
| gym / gym-notices | 0.26.2 / 0.1.0 | 从 `agentenv-alfworld`(py3.9) 复制（纯 python） |
| ai2thor | 2.1.0 | PyPI wheel `py2.py3-none-any`（`--no-deps`） |
| jericho | 3.3.1 | PyPI **sdist** 现编（无 cp310/cp311 wheel） |
| spacy / thinc / blis / … | 3.8.16 / 8.3.13 / 1.3.3 | `jericho` 的依赖链（体积主要来源） |
| mementos | 1.3.1 | `textworld.logic.__init__` 依赖 |
| tatsu | **5.8.3**（严格 pin） | textworld 要求 |
| numpy | 1.26.4（与 thinktool 同版本） | 见 §4.1 |

> `pip install --target` 在结尾打印的 `opentelemetry-sdk … conflicts` 是 pip 对 **当前环境(thinktool)** 的冲突提示，
> **不是** 我们装进去的东西。已核对 `thinktool/lib/python3.10/site-packages/opentelemetry*` 的 mtime 仍是 2026-01-04，
> 未被本次操作改动。

---

## 4. 关键坑与解决办法

### 4.1 numpy ABI —— **结论：不需要 numpy 2.x**

* 担心：textworld 的 cp310 wheel 可能是用 numpy 2.x 编的，而 thinktool 是 1.26.4。
* 实测：在 `py310deps` 里显式装 `numpy==1.26.4`，`import textworld` / `textworld.gym` / `alfworld` **全部正常**，
  且 `import torch` = 2.6.0+cu124 也正常（torch 与 numpy 1.26.4 本来就是 thinktool 的既有组合）。
* 结论：**无需把 numpy 2.x 装进 py310deps**。`py310deps/numpy` 与 thinktool 的 numpy 版本一致，不存在遮蔽冲突。
* 附带好处：`py310deps` 里也放了 numpy 1.26.4 的 cp310/cp311 wheel，三期复现时不会被解析成 2.x。

### 4.2 gym 0.26.2 —— 用「复制」而不是「编译」

* `pip install --target ... --only-binary=:all: gym==0.26.2` → `ERROR: No matching distribution found`（PyPI 上只有 sdist）。
* sdist 编译路径可行但会额外拉 `setuptools` 构建链；而 gym 0.26.2 是**纯 python**，
  直接从只读的 `~/miniconda3/envs/agentenv-alfworld`（py3.9）把 `gym/`、`gym_notices/` 及其 dist-info 复制到 `py310deps/` 即可，
  py3.9→py3.10 无 ABI 问题（实测 `import gym` → 0.26.2 成功）。
  这是「读外部环境、写自己目录」，不违反只读铁律。

### 4.3 `ai2thor` 必须装（虽然只跑文本环境）

* `alfworld/agents/environment/__init__.py` 会 `from ...alfred_thor_env import AlfredThorEnv`
  → `alfworld/env/thor_env.py` → `from ai2thor.controller import Controller`。
  也就是说 **只要 import `alfworld.agents.environment.*` 这个包，就会连带 import 到 THOR 分支**。
* 解决：装 `ai2thor==2.1.0`（wheel 仅 25 KB，`--no-deps`）。
  它声明的依赖（flask / numpy / pyyaml / requests / progressbar2）里，只有 `progressbar2` thinktool 没有，单独补装。
  THOR 分支需要的 `torchvision`(0.21.0+cu124) 和 `cv2`(4.12.0) **thinktool 本来就自带**，所以不用重复安装。
* 备选（未采用）：绕过包 `__init__` 用 `importlib` 直接按文件路径加载 `alfred_tw_env.py`。
  可行但脆弱，且主 agent 大概率会写 `from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv`，故采用装 ai2thor 的方案。

### 4.4 `jericho` → `spacy` 依赖链（体积）

* `textworld/logic/__init__.py` → `mementos`；`jericho` 依赖 `spacy>=2.1.0`，spacy 又拉 `thinc/blis/cymem/…`。
* 这是 `py310deps` 864 MB 的主要原因（spacy 家族约占 300+ MB）。**不是可选项**，`import textworld` 就会走到。

### 4.5 `textworld.gym` 的 batch `step()` 语义（写训练循环时必踩）

* `init_env(batch_size=N)` 返回的是「批环境」，`step()` 要求 **每个子环境一个 action**：
  `tw.step([a0, a1])`，长度必须等于 `N`，否则 `AssertionError: Expected one action per environment.`
* `reset()` 返回 `(obs_list, info_dict)`，`info_dict` 的每个 value 都是长度 N 的 list
  （`admissible_commands` / `won` / `extra.gamefile`）。
* `AlfredTWEnv.init_env()` 里：`general.training_method` 决定取 `rl.training.max_nb_steps_per_episode`(dqn)
  还是 `dagger.training.max_nb_steps_per_episode`(dagger)，**必须是 `dagger` 或 `dqn`**；
  `env.expert_type` 用 `handcoded`。注意：`train` split 才会挂 expert wrapper；`eval_*` split 不挂。
* config 里的 `$ALFWORLD_DATA` 由 `alfworld` 的 `os.path.expandvars(...)` 在运行时展开，
  所以 **必须先把 `ALFWORLD_DATA` 写进 `os.environ`**，再构造 `AlfredTWEnv`。AgentGym 那份 `utils.load_config()`
  只做 `yaml.safe_load`，不做展开。

### 4.6 glibc / `libdownward.so`（已由前置工作解决，此处只复述）

* TextWorld 的 `fast_downward/libdownward.so` 需要 glibc 2.32；二期 2.31、三期 2.28 都不满足。
* 已用 `env_setup/patch_libdownward.py` 生成降级副本 + 符号垫片到 `third_party/glibc232/`，
  运行时由 `src/alfworld_compat.py::install()` 接管 `fast_downward.interface.DOWNWARD_LIB_PATH`。
* 本次实测：`install()` 后 `fast_downward.load_lib()` → `True`（0.04–0.10 s）。

### 4.7 第二个隐藏的 glibc 坑：`libstdc++` 需要 `GLIBCXX_3.4.29`

在做 py3.11 离线验证时新发现（**对主 agent 也重要**）：

```
OSError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
         (required by .../libdownward.so)
```

* 实测：系统 `/lib/x86_64-linux-gnu/libstdc++.so.6`（Ubuntu 20.04 / GCC 9）**不含** `GLIBCXX_3.4.29`；
  `~/miniconda3/envs/thinktool/lib/libstdc++.so.6`（conda 自带，GCC 11）**含** 2 处。
* 所以在 conda 环境里跑没问题（conda 的 `lib/` 会在库搜索路径里）；
  **但如果直接用系统 python（不带 conda 环境）跑，就会挂**。
* 规避：`export LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib:$LD_LIBRARY_PATH`，
  或在 conda env 中运行。三期同理（需要一个新的 libstdc++）。

### 4.8 `opencv-python` 与 `numpy` 的元数据冲突（离线安装时会撞）

* `opencv-python 4.12.0.88` 的元数据声明 `numpy>=2; python_version>="3.9"`，
  而本方案要求 `numpy==1.26.4`（textworld / torch 2.6 都吃这个版本）。
* 用 pip 正常解析会报 `ResolutionImpossible`。
* 但 `thinktool` 现状就是 `numpy 1.26.4 + opencv-python 4.12.0.88` 共存，且 `import cv2` 实测正常
  —— 说明**实际运行没问题，只是元数据过严**。
* 解决办法：离线安装时对这套 wheel 用 `--no-deps` 全量安装（我们已提供完整闭包），
  或对 opencv 单独 `--no-deps`。`_install_step10_offline_install.sh` 用的就是这个方式。


---

## 5. 端到端自检结果（`check_alfworld_thinktool.py`）

`eval_out_of_distribution`（`valid_unseen`，共 134 个 game）：

```
[ OK ] alfworld_compat.install()
[ OK ] import numpy    -> 1.26.4
[ OK ] import torch    -> 2.6.0+cu124
[ OK ] import textworld-> 1.6.2
[ OK ] import gym      -> 0.26.2
[ OK ] import alfworld -> .../py310deps/alfworld/__init__.py
[ OK ] import alfworld.agents.environment.alfred_tw_env -> <class AlfredTWEnv>
[ OK ] fast_downward.load_lib() -> True
RESULT: PASS
```

第一次 `reset()` 的观测（514 字符）：

```
-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 6, ... a toaster 1.

Your task is to: put a clean mug in coffeemachine.
infos keys: ['admissible_commands', 'extra.gamefile', 'won']
admissible_commands[0] 长度 = 27
extra.gamefile[0] = .../json_2.1.1/valid_unseen/pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-10/trial_T20190907_221300_362511/game.tw-pddl
```

结果 JSON 存于 `env_setup/results/check_n1.json`、`check_n2.json`。

---

## 6. 实测耗时（吞吐设计依据）

主机 `mgmt-3`（64 vCPU，登录节点，**非独占**；因该节点 `srun/sbatch` 不可用，见 §8）。
每档 `reset` 4 次、`step` 30 次。原始数据：`env_setup/results/bench.json`。

| batch_size N | `init_env(N)` | `reset()` 冷启动 | `reset()` 稳态均值 | `step()` avg | `step()` p50 / p90 | **env-steps/s** | wall batch-steps/s |
|---|---|---|---|---|---|---|---|
| 1 | 0.005 s | 3.402 s | **2.747 s** | 0.0996 s | 0.0994 / 0.1002 | **10.0** | 10.0 |
| 2 | 0.073 s | 3.107 s | **3.102 s** | 0.1240 s | 0.1263 / 0.1342 | **16.1** | 8.06 |
| 4 | 0.091 s | 3.256 s | **3.196 s** | 0.1400 s | 0.1380 / 0.1568 | **28.6** | 7.14 |
| 8 | 0.162 s | 3.428 s | **3.255 s** | 0.1356 s | 0.1312 / 0.1487 | **59.0** | 7.37 |

**结论 / 对训练设计的含义**

1. `init_env` 几乎免费（< 0.2 s），真正的固定开销在 **`reset()` ≈ 3 s**。
   `reset()` 每次都要新载入一个 `game.tw-pddl`（解析 PDDL + 建状态机），与 `batch_size` 基本无关。
2. `step()` 延迟随 N 增长很慢（0.100 s @N=1 → 0.136 s @N=8），
   因为 textworld gym 用 `asynchronous=True` 的 worker 池并行；**env-steps/s 近似线性放大**（10 → 59）。
3. **reset 是主要瓶颈**：以 `max_nb_steps_per_episode=50` 估算，
   单环境一个 episode ≈ 3.1 s(reset) + 50×0.10 s(step) ≈ **8.1 s**，其中 reset 占 ~38%。
   → 建议：批量环境尽量大（N≥8）以摊薄 reset，或把 reset 与模型前向/生成重叠（异步 rollout）。
4. 因此「**一个进程 = 一个大 batch 的 TextWorld env**」比「一进程一环境」高效得多；
   跨节点扩展时优先加大 batch_size，而不是加进程数（进程数多了每个进程的 reset 开销会重复付出）。
5. 注意这是登录节点上非独占测量的保守值；独占 CPU 后 step 延迟应更低。

---

## 7. 离线 wheel 包（给三期 / 无外网）

位置：`~/ztwang/selfplay/third_party/`

| 目录 | 内容 | 体积 |
|---|---|---|
| `wheels/` | **94 个 wheel**：cp310 + cp311 + `py3-none-any` 全套闭包 | 319 MB |
| `wheels_sdist/` | 3 个源的 tarball（兜底/溯源） | 2.9 MB |
| `WHEELS_MANIFEST.txt` | 全部文件 + 版本 + **SHA256** 清单（104 行） | — |
| `requirements_py310deps.txt` | 从 `py310deps/*.dist-info` 反推出的 pin 清单（65 条） | — |
| `requirements_offline_extra.txt` | 宿主(thinktool)自带、因此没进上面清单、但依赖链需要的 12 个包 | — |

**PyPI 上没有 wheel 的 3 个包** —— 已用「`~/ztwang` 内的独立 CPython 3.11.17」现场编成 `py3-none-any` wheel：

| 包 | 产物 | 说明 |
|---|---|---|
| `fast_downward_textworld==20.6.4` | `fast_downward_textworld-20.6.4-py3-none-any.whl`（12.2 MB） | 内含预编译 `fast_downward/libdownward.so`（44 MB 解压后），**与 python 版本无关** |
| `gym==0.26.2` | `gym-0.26.2-py3-none-any.whl`（0.83 MB） | 纯 python |
| `jericho==3.3.1` | `jericho-3.3.1-py3-none-any.whl`（0.32 MB） | 内含预编译 `jericho/libfrotz.so`（上游打成了 none-any，事实上是 linux-x86_64） |

> 因为三者最后都拿到了「无版本标签」的 wheel，**三期不需要编译器、也不需要 cmake**，直接用 `wheels/` 就能装。

**`requirements_offline_extra.txt`（12 个）**：
`cloudpickle==3.1.2`（gym 依赖）、`opencv-python==4.12.0.88`、`opencv-python-headless==4.12.0.88`、`pandas==2.3.3`、
`werkzeug==3.1.4`、`python-dateutil==2.9.0.post0`、`pytz==2026.5`、`six==1.17.0`、`tzdata==2025.3`、
`flask==3.1.3`（ai2thor 依赖）、`blinker==1.9.0`、`itsdangerous==2.2.0`。
它们本来由 `thinktool` 提供，所以第一次 `pip install --target` 时没被收进 `py310deps`。

### 7.1 离线安装已验证（py3.11，干净 venv，`--no-index`）

`_install_step10_offline_install.sh` 做了实测：

```bash
$HOME/ztwang/selfplay/third_party/py311/bin/python3 -m venv /tmp/fresh
/tmp/fresh/bin/python -m pip install --no-cache-dir --no-index \
    --find-links ~/ztwang/selfplay/third_party/wheels --no-deps -r <全部 77 条 spec>
# rc=0  → 装了 80 个发行包

LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib /tmp/fresh/bin/python -c "..."
# numpy 1.26.4 / gym 0.26.2 / jericho 3.3.1 / cv2 4.12.0 / pandas 2.3.3 / flask 3.1.3
# textworld 1.6.2 / textworld.gym OK / fast_downward.load_lib() -> OK
# => ALL GOOD
```

**注意**：上面这次验证的是 `import textworld/gym/jericho/ai2thor/cv2/pandas/alfworld` 的 **torch-free 子集**。
`import alfworld.agents.environment` 会走到 `alfworld.agents.detector.mrcnn` → `torchvision` → `torch`，
而 **torch/torchvision 必须由目标环境提供**（CUDA 版本要匹配，不能随 wheels 分发）。
三期请把 `torch`/`torchvision` 视作前置条件（与 `thinktool` 一样：torch 2.6.0+cu124 / torchvision 0.21.0+cu124）。

### 7.2 三期离线安装命令（照抄）

```bash
B=~/ztwang/selfplay/third_party
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH     # 关键：拿到新的 libstdc++
pip install --no-index --find-links $B/wheels --no-deps \
    -r $B/requirements_py310deps.txt -r $B/requirements_offline_extra.txt
# torch / torchvision 自行按 CUDA 版本预装好
```


---

## 8. 风险与未解问题

| # | 风险 / 问题 | 影响 | 现状 / 缓解 |
|---|---|---|---|
| R1 | **登录节点 `mgmt-3` 上 SLURM 不可用**：`/etc/slurm-llnl/slurm.conf` 不存在，`srun/sbatch/sinfo/scontrol` 全部 fatal。 | 无法按原计划用 `srun -p debug` 跑测试/基准 | 基准改在登录节点直接跑（≤8 worker、约 2 min，登录节点 64 vCPU，规模很小）。**主 agent 提交训练作业时需先确认真正的提交节点/`SLURM_CONF`**，这是个独立阻塞项。 |
| R2 | 三期（Rocky 8 / glibc 2.28）尚未验证 | 三期能否跑通未知 | glibc 兼容层对 2.28 同样适用（垫片与降级副本与 glibc 版本无关）；`wheels/` 已同时备好 **cp311**，三期若是 py3.11 可直接离线装（§7.1 已在 py3.11 实测通过）。仍需一期现场验证。 |
| R3 | `libstdc++` 需要 `GLIBCXX_3.4.29`（见 §4.7） | 不用 conda 环境时 `load_lib()` 直接崩 | 在 conda env 内运行，或设 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`。三期若用系统 python 需自行提供新 libstdc++ |
| R4 | `py310deps` 864 MB，`jericho→spacy` 链很重 | 磁盘、复制成本 | 实测 `jericho-3.3.1-py3-none-any.whl` 里只有一个 `libfrotz.so`（Z-machine 用），ALFWorld 文本流程大概不需要它；但**未验证** textworld 是否在 `import` 期就用到 jericho，暂按必需处理 |
| R5 | `opencv-python 4.12` 元数据要求 `numpy>=2`，与 `numpy==1.26.4` 冲突（见 §4.8） | 正常 pip 解析会 `ResolutionImpossible` | 离线安装用 `--no-deps` 全量装（已提供完整闭包），实测 `import cv2` 正常 |
| R6 | 依赖 `sys.path` 顺序 / `PYTHONPATH` | 顺序错了会 import 到错的东西 | 已封装，见 §2 ③；自检脚本里显式断言 |
| R7 | 登录节点非独占，耗时数据有噪声 | 主要影响 step p90 | 同一轮内 `p50≈p90`（差异 <15%），噪声可接受；正式压测应在计算节点做 |
| R8 | `train` split 会挂 expert wrapper（`AlfredExpert` + `request_infos.extras += expert_plan`），比 eval 慢 | 训练吞吐 | 本次只实测了 `eval_out_of_distribution`；`train` split 的 reset 可能更慢，需主 agent 自行补测 |
| R9 | ALFWorld 的 `valid_train` split | `ALFWorld_Wrapper` 用到 | 数据已在 `json_2.1.1/valid_train`，本次未单独跑该 split |


---

## 9. 文件清单（本次新增，全部在 `~/ztwang/selfplay/`）

```
env_setup/
  ENV_BRINGUP.md                      # 本报告（本地同名一份）
  check_alfworld_thinktool.py         # 端到端自检脚本
  bench_alfworld_thinktool.py         # 吞吐基准脚本
  _install_step1.sh                   # ① 核心包 + gym 复制
  _install_step2.sh                   # ② ai2thor + progressbar2
  _install_step3_wheels.sh            # ③ 批量 wheel 下载（整批会因为 1 个包失败而全灭 → 见 3b）
  _install_step3b_wheels_loop.sh      # ③b 逐包 wheel 下载（容忍单包失败）
  _install_step4_sdists.sh            # ④ 3 个 sdist + 生成 WHEELS_MANIFEST.txt
  _install_step5_py311wheels.sh       # ⑤ 用 ztwang 内的独立 py3.11 编 cp310/311 无 wheel 包
  _install_step6_verify_wheels.sh     # ⑥ wheel 标签审计
  _install_step7/8/9_offline_verify*.sh  # ⑦⑧⑨ 离线安装验证（含失败记录）
  _install_step10_offline_install.sh  # ⑩ 全量 --no-deps 离线安装 + import 验证（最终可用版本）
  results/
    check_n1.json  check_n2.json      # 自检结果
    bench.json                        # 耗时基准原始数据
    wheels_download.log  wheels_loop.log  py311_build.log
configs/
  base_config.yaml                    # 从 AgentGym 只读目录拷来的 ALFWorld 官方 config
third_party/
  py310deps/                          # 864 MB，py3.10 的 ALFWorld 依赖（实际使用对象）
  wheels/                             # 94 个 wheel（cp310 + cp311 + py3-none-any）
  wheels_sdist/                       # 3 个源的 tarball
  wheels_missing/                     # 单包下载失败记录 + stderr
  py311/                              # 358 MB，独立 CPython 3.11.17（仅用于编 cp311 wheel）
  glibc232/                           # libdownward.so 降级副本 + 符号垫片
  WHEELS_MANIFEST.txt                 # 94+3 个文件的 SHA256 清单
  requirements_py310deps.txt          # 65 条 pin
  requirements_offline_extra.txt      # 12 条 host-provided 补齐
```

> **未改动**：`~/thinktool/**`、`~/miniconda3/envs/**`、`models/**`、`~/ztwang/selfplay/src/**`（主 agent 负责）。
> 已核对 `thinktool/.../site-packages/opentelemetry*` 的 mtime 仍为 2026-01-04，说明 `pip install --target` 没有污染 thinktool。


---

## 10. 给主 agent 的接入建议

1. 在训练/评测入口**最顶部**加：

   ```python
   import os, sys
   R = os.path.expanduser("~/ztwang/selfplay")
   sys.path.insert(0, R + "/src"); sys.path.insert(0, R + "/third_party/py310deps")
   os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
   from alfworld_compat import install; install()
   ```
   或设环境变量 `PYTHONPATH=$HOME/ztwang/selfplay/third_party/py310deps:$HOME/ztwang/selfplay/src`。

2. 训练用 `AlfredTWEnv(config, train_eval="train")`，评测用 `eval_out_of_distribution`；
   `config["general"]["training_method"]` 设 `'dagger'`（`init_env` 要读它取 `max_nb_steps_per_episode`）。
3. `step()` 记得传 **长度 = batch_size 的 action list**。
4. 吞吐按 **env-steps/s ≈ 7.4 × N**（N≥2，粗略）估算；reset ≈ 3 s/次 是主要开销，建议 budget 里单独算。


---

## 11. 三期 (ACD / Rocky 8.10, glibc 2.28) 离线部署 — Stage-1 已完成

**目标**：把 ALFWorld 文本环境在三期跑起来；三期**无外网**，只能离线部署。
**边界**：只写 `/data/user/xyu608/ztwang/**` 与 `/data/user/xyu608/.cache/alfworld/**`；
`/share/anaconda3/envs/**` 只读（`touch` → Permission denied），一律 `pip install --target` + `PYTHONPATH` 注入。

### 11.1 目标 env 选择

| env | python | torch | transformers | torchvision | numpy |
|---|---|---|---|---|---|
| **`vllm017`（选用）** | **3.11.15** | 2.10.0+cu128 | 4.57.6 | 0.25.0+cu128 ✅ | **2.2.6** |
| `vllm`（备） | 3.12.11 | 2.7.0+cu126 | 4.53.1 | 0.22.0+cu126 ✅ | 2.x |
| `pytorch271`（备） | 3.12.11 | 2.7.1+cu128 | **无** | 0.22.1+cu128 ✅ | 2.x |

选 `vllm017`：py3.11 最接近二期的 py3.10，且 torch/transformers/torchvision 齐全。

### 11.2 实测传输量（二期 → 三期，rsync 直连 `10.120.81.30`）

| 段 | 内容 | 文件数 | 字节 |
|---|---|---|---|
| A | env_setup + src + configs + glibc232 + wheels + wheels_sdist + requirements/manifest | 180 | 365,213,071 |
| B | alfworld: logic + json_2.1.1/{valid_seen,valid_train,valid_unseen} | 2,218 | 223,876,604 |
| **合计** | **Stage-1** | **2,398** | **589,089,675 (≈0.589 GB)** |

> `du -sh` 报的 360M 是 **4K 块占用**（大量小 json 文件放大）；真实传输量以 rsync `Total file size` 为准（224MB）。
> A 段**不含** `third_party/py310deps`、`py311/`、`train/`(2.01GB)、`detectors/`(170MB)。

三期落地回验（`du -sb`，含目录条目所以略大于文件字节）：
`wheels` 327,183,988 + `glibc232` 34,724,326 + `logic` 33,317 + `valid_seen` 80,496,742 + `valid_train` 64,074,402 + `valid_unseen` 84,478,159。

### 11.3 安装（离线）

```bash
# 二期
bash ~/ztwang/selfplay/env_setup/phase3_push_stage1.sh
# 三期
bash /data/user/xyu608/ztwang/selfplay/env_setup/phase3_install_stage1.sh \
     /share/anaconda3/envs/vllm017/bin/python
```
- `phase3_install_stage1.sh` 按目标 env 的 py 版本自动选 `wheels/`(cp310+cp311) 或 `wheels312/`，
  落到 `third_party/py311deps` / `py312deps`，全程 `--no-index --find-links`，**不碰任何 conda env**。
- 本次成功安装 **76 个包**（`pip rc=0`），`py311deps` 实测 677,521,134 B / 13,930 文件。
- `numpy / torch / torchvision` 故意不在清单内，由目标 env 提供。

### 11.4 自检（纯 import 级，登录节点，不构造环境 / 不 step / 不 srun）

脚本 `phase3_verify_imports.py`，**ALL 5 CHECKS PASSED**：

1. `python 3.11.15`；`PYTHONPATH` = `src:py311deps` 生效，`sys.path[:3]` 前两位正是 src / py311deps ✅
2. `torch 2.10.0+cu128` / `transformers 4.57.6` / `textworld 1.6.2` / `gym 0.26.2`；另 `alfworld` + `AlfredTWEnv` + `textworld.gym` 均可 import ✅
3. `fast_downward.interface.load_lib()` → **OK**（glibc 2.28 唯一硬骨头，靠 `glibc232/` 兼容层 + RTLD_GLOBAL 垫片）✅
4. `DOWNWARD_LIB_PATH` = `.../glibc232/libdownward.so`；`ldd` 全解析 —— `libstdc++.so.6 => /share/anaconda3/envs/vllm017/lib/libstdc++.so.6` ✅
5. `LD_LIBRARY_PATH` 含 `vllm017/lib`；该 libstdc++ 提供 **GLIBCXX_3.4.29 ×2**（系统 `/usr/lib64` 是 ×0）✅

日志：`.../env_setup/results/{phase3_push_stage1.log,phase3_install_stage1.log,phase3_verify_imports.log}`。

### 11.5 三期剩余风险

- **R-P3-1（新，重要）**：`vllm017` 的 numpy 是 **2.2.6**，而 `gym 0.26.2` 启动即打印
  “Gym has been unmaintained since 2022 and does not support NumPy 2.0”。import 级不炸，
  且二期 `agentenv-alfworld`（numpy 2.0.2 + textworld 1.6.2）实证可跑，故**大概率无碍**；
  但 ALFWorld 真正 `reset()`/`step()` 时若报 numpy ABI/类型错，第一处置是给 `py311deps` 里加一份
  numpy 1.26.x（本期 wheel 集已含 cp311 的 numpy 1.26.4），靠 `PYTHONPATH` 顺序压过 env 的 numpy。
- **R-P3-2**：`train` 切分(2.01GB) 与 `detectors/`(170MB) **尚未搬**——Stage-2 待 team-lead 放话。
- **R-P3-3**：Stage-1 只做了 import 级验证；端到端 `AlfredTWEnv → init_env → reset → step`
  在三期尚未跑（team-lead 要求先不跑计算）。二期同版本组合已 100% 通过，预期一致。
- **R-P3-4**：三期 ACD 纯 CPU 作业必须 `-p debug`（ACD 队列强制 `--gres=gpu:N`）。

