#!/bin/bash
# 全量 valid_unseen 评测（论文可比口径）：134 任务 × N 采样，8 卡分片并行。
#
# 用法：
#   sbatch --export=ALL,ZTWANG_MODEL_PATH=...,EZ_CKPT=<ckpt> scripts/ez_eval_full_8gpu.sbatch
#   可选：EZ_SAMPLES=4 EZ_BACKEND=vllm EZ_SHARDS=8
#
# 为什么分片：单个 eval_ez.py 是串行 batch，跑 536 条轨迹要很久；
#   切成 8 片各占一卡，8 倍并行，整节点一次跑完。
#
# ⚠️ 分区写死 i64m1tga800u8ka：1 卡作业会掉进 1483 个作业的长队里排队，
#    而 8 卡分区经常有空闲节点（本脚本只用到 1 卡/分片，但整节点申请才排得上）。
#SBATCH --job-name=ez-evalfull
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=0-02:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p logs eval_out

: "${EZ_CKPT:?Set EZ_CKPT}"
if [ ! -d "$EZ_CKPT/solver" ] && [ -d "$EZ_CKPT/ckpt/solver" ]; then
  EZ_CKPT="$EZ_CKPT/ckpt"
fi
[ -d "$EZ_CKPT/solver" ] || { echo "FATAL: $EZ_CKPT 下没有 solver/"; exit 2; }

SAMPLES="${EZ_SAMPLES:-4}"
BACKEND="${EZ_BACKEND:-vllm}"
NSHARD="${EZ_SHARDS:-8}"
TOTAL=$(( ${EZ_TOTAL:-134} ))
PER=$(( (TOTAL + NSHARD - 1) / NSHARD ))
TAG="$(basename "$(dirname "$EZ_CKPT")")-$(basename "$EZ_CKPT")"
# 基座对照：--base-only 时忽略 adapter；tag 加后缀避免与 adapter 结果互相覆盖
# ▲ 步数上限必须显式可调并写进 tag：专家 oracle 实测 30 步会给成绩硬造 85.1% 天花板
#   （20/134 个任务连官方专家都赢不了），所以不同 cap 的数字绝不能混在一起比。
MAXSTEPS="${EZ_MAX_STEPS:-30}"
# ▲ 采样参数必须显式可调：Qwen3-4B base 自己的 generation_config.json 推荐
#   temperature=0.6 / top_k=20，而我们一直用 1.0 / 禁用 top_k —— 噪音大得多。
TEMP="${EZ_TEMPERATURE:-1.0}"; TOPK="${EZ_TOP_K:--1}"
MAXNEW="${EZ_MAX_NEW:-96}"
EXTRA="--max-steps ${MAXSTEPS} --temperature ${TEMP} --top-k ${TOPK} --max-new-tokens ${MAXNEW}"
TAG="${TAG}-mn${MAXNEW}"
TAG="${TAG}-ms${MAXSTEPS}-t${TEMP}-k${TOPK}"
# ★ 所有会影响结果的开关都必须进 tag，否则不同配置会**静默互相覆盖**输出文件
[ "${EZ_SOLVER_ADMISSIBLE:-0}" = "1" ] && TAG="${TAG}-adm"
[ "${EZ_SOLVER_FEWSHOT_SCHEMA:-0}" = "1" ] && TAG="${TAG}-fs"
[ "${EZ_SOLVER_STYLE:-ours}" != "ours" ] && TAG="${TAG}-${EZ_SOLVER_STYLE}"
[ "${EZ_SOLVER_BARE_FALLBACK:-0}" = "1" ] && TAG="${TAG}-bare"
[ "${EZ_PARSER_STRICT:-0}" = "1" ] && TAG="${TAG}-strict"
[ -n "${EZ_TAGSUF:-}" ] && TAG="${TAG}-${EZ_TAGSUF}"
[ "${EZ_BASE_ONLY:-0}" = "1" ] && { EXTRA="$EXTRA --base-only"; TAG="${TAG}-baseonly"; }

echo "job=${SLURM_JOB_ID} ckpt=$EZ_CKPT backend=$BACKEND samples=$SAMPLES"
echo "全量 $TOTAL 任务 切成 $NSHARD 片，每片 $PER 个"
nvidia-smi -L

for k in $(seq 0 $((NSHARD - 1))); do
  OFF=$((k * PER))
  OUT="eval_out/full_${TAG}_shard${k}.json"
  ( CUDA_VISIBLE_DEVICES=$k python src/eval_ez.py \
        --backend "$BACKEND" $EXTRA --base "$ZTWANG_MODEL_PATH" --ckpt "$EZ_CKPT" \
        --select head --offset "$OFF" --n "$PER" \
        --samples "$SAMPLES" --batch "$PER" --seed 42 --out "$OUT" \
        > "logs/evalshard_${TAG}_${k}.log" 2>&1 ; \
    echo "shard$k (offset=$OFF) exit=$?" ) &
done
wait

echo "=================== 合并 ==================="
python3 - <<PYEOF
import json, glob, statistics as st
files = sorted(glob.glob("eval_out/full_${TAG}_shard*.json"))
rows, by_type, cfg = [], {}, None
for f in files:
    d = json.load(open(f))
    cfg = cfg or d
    rows.extend(d["samples"])
    for k, v in d.get("task_type", {}).items():
        by_type.setdefault(k, []).extend([v["mean"]] * v["n"])
if not rows:
    print("没有结果（所有分片都失败了，看 logs/evalshard_*.log）")
    raise SystemExit(1)
won = [1.0 if r["won"] else 0.0 for r in rows]
print(f"分片数 {len(files)}  轨迹数 {len(rows)}  任务数 {len({r['task_id'] for r in rows})}")
print(f"[FULL valid_unseen] sample SR = {st.mean(won):.4f} ± {st.pstdev(won):.4f}")
for k in sorted(by_type):
    v = by_type[k]
    print(f"   {k:28s} n={len(v):4d}  SR={sum(v)/len(v):.4f}")
json.dump({"n_samples": len(rows), "sr": st.mean(won), "std": st.pstdev(won),
           "backend": cfg.get("backend"), "samples_per_task": cfg.get("samples_per_task"),
           "decode": cfg.get("decode"), "decode_effective": cfg.get("decode_effective"),
           "per_type": {k: sum(v)/len(v) for k, v in by_type.items()},
           "samples": rows},
          open(f"eval_out/full_${TAG}_MERGED.json", "w"), ensure_ascii=False, indent=2)
print(f"-> eval_out/full_${TAG}_MERGED.json")
PYEOF
