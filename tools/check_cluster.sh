#!/bin/bash
# ============================================================================
# 集群一键巡检  —  HKUST-GZ HPC 二期 / 三期通用
#
# 用法:
#   bash ~/ztwang/tools/check_cluster.sh            # 自动识别当前集群
#   bash ~/ztwang/tools/check_cluster.sh --du       # 额外统计家目录体积(慢)
#
# 作用: 全局负载、我的队列、GPU 队列排队、预测起算时间、存储
# ============================================================================
U=xyu608

# ---- 自动识别集群 ----
if [ -d /opt/slurm/bin ]; then
  export PATH=/opt/slurm/bin:$PATH
fi
HOSTN=$(hostname)
case "$HOSTN" in
  ACD-*|acd*)  CLUSTER="三期 ACD (H100)";;
  mgmt*|*)     CLUSTER="二期 (A800/A40)";;
esac
if sinfo -o "%P" -h 2>/dev/null | grep -q "acd_u"; then CLUSTER="三期 ACD (H100)"; fi

echo "════════════════════════════════════════════════════════"
echo " 集群巡检  $CLUSTER   $(date '+%Y-%m-%d %H:%M:%S')"
echo " 登录节点: $HOSTN   Slurm: $(sbatch --version 2>/dev/null | awk '{print $2}')"
echo "════════════════════════════════════════════════════════"

if [ "$CLUSTER" = "三期 ACD (H100)" ]; then
  # ================= 三期 =================
  echo
  echo "── 1. 节点状态汇总（三期 squeue 不可信：PrivateData=jobs）──"
  sinfo -N -h -o "%T" 2>/dev/null | sort | uniq -c | sort -rn

  echo
  echo "── 2. 分区占用 (已分配/空闲/其他/总计) ──"
  sinfo -h -o "%22P %6a %16F %12c %10m %10G %12l" 2>/dev/null

  echo
  echo "── 3. GPU 占用统计 ──"
  tot=0; alloc=0
  for n in $(sinfo -h -N -o "%N" 2>/dev/null | sort -u); do
    c=$(scontrol show node "$n" 2>/dev/null | grep -oP "CfgTRES=.*?gres/gpu=\K[0-9]+" | head -1)
    a=$(scontrol show node "$n" 2>/dev/null | grep -oP "AllocTRES=.*?gres/gpu=\K[0-9]+" | head -1)
    tot=$((tot+${c:-0})); alloc=$((alloc+${a:-0}))
  done
  echo "  GPU 总数=$tot  已分配=$alloc  空闲=$((tot-alloc))"
  free_nodes=$(for n in $(sinfo -h -N -o "%N" | sort -u); do
    a=$(scontrol show node "$n" 2>/dev/null | grep -oP "AllocTRES=.*?gres/gpu=\K[0-9]+" | head -1)
    [ "${a:-0}" = "0" ] && echo "$n"
  done | wc -l)
  echo "  完全空闲的节点数=$free_nodes"

  echo
  echo "── 4. 我的作业 ──"
  squeue -u $U 2>/dev/null || echo "  无作业"

  echo
  echo "── 5. 预测起算时间 (不会真的提交) ──"
  echo -n "  H100 x8 (acd_u)      : "
  sbatch --test-only -p acd_u --gres=gpu:8 --ntasks=1 --cpus-per-task=32 \
         --mem=200G --time=1:00:00 --wrap "true" 2>&1 | head -1
  echo -n "  H100 x8 (emergency)  : "
  sbatch --test-only -p emergency_acd --gres=gpu:8 --ntasks=1 --cpus-per-task=32 \
         --mem=200G --time=1:00:00 --wrap "true" 2>&1 | head -1
  echo -n "  H100 x1 (acd_u)      : "
  sbatch --test-only -p acd_u --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
         --mem=64G --time=1:00:00 --wrap "true" 2>&1 | head -1

  echo
  echo "── 6. 存储 ──"
  df -h 2>/dev/null | grep -E "/data|Filesystem"

  echo
  echo "════════════════════════════════════════════════════════"
  echo " 提示: 三期无外网(sync 用 --host hpc3)；GPU 配额 16 张；时限 7 天"
  echo " 提示: 训练必须存断点，且只保留一个"
  echo "════════════════════════════════════════════════════════"
  exit 0
fi

# ================= 二期 =================
echo
echo "── 1. 全局作业数 ──"
echo "运行中: $(squeue -h -t R 2>/dev/null | wc -l)   排队中: $(squeue -h -t PD 2>/dev/null | wc -l)"

echo
echo "── 2. 节点状态汇总 ──"
sinfo -N -h -o "%T" 2>/dev/null | sort | uniq -c | sort -rn

echo
echo "── 3. 分区占用 (已分配/空闲/其他/总计) ──"
sinfo -h -o "%22P %6a %16F %22G %12l" 2>/dev/null | head -30

echo
echo "── 4. 我的作业 ──"
squeue -u $U 2>/dev/null || echo "无作业"

echo
echo "── 5. A800 队列排队情况 ──"
for p in i64m1tga800u long_gpu i64m1tga800u8ka emergency_gpu i64m1tga40u8ka; do
  r=$(squeue -h -p $p -t R 2>/dev/null | wc -l)
  pd_all=$(squeue -h -p $p -t PD 2>/dev/null | wc -l)
  held=$(squeue -h -p $p -t PD -o "%r" 2>/dev/null | grep -c "Held")
  dep=$(squeue -h -p $p -t PD -o "%r" 2>/dev/null | grep -c "Dependency")
  printf "  %-20s 运行=%-4s 排队=%-4s (挂起=%-3s 等依赖=%-3s 有效=%-3s)\n" \
    "$p" "$r" "$pd_all" "$held" "$dep" "$((pd_all-held-dep))"
done

echo
echo "── 5b. 二期 GPU 总卡数 ──"
tot=0; alloc=0
for n in $(sinfo -h -N -o "%N" 2>/dev/null | sort -u); do
  c=$(scontrol show node "$n" 2>/dev/null | grep -oP "CfgTRES=.*?gres/gpu=\K[0-9]+" | head -1)
  a=$(scontrol show node "$n" 2>/dev/null | grep -oP "AllocTRES=.*?gres/gpu=\K[0-9]+" | head -1)
  tot=$((tot+${c:-0})); alloc=$((alloc+${a:-0}))
done
echo "  GPU 总数=$tot  已分配=$alloc  空闲=$((tot-alloc))"

echo
echo "── 6. 预测起算时间 (不会真的提交) ──"
echo -n "  A800 x8  (8ka 队列): "
sbatch --test-only -p i64m1tga800u8ka --gres=gpu:a800:8 --ntasks=1 \
       --cpus-per-task=32 --mem=200G --time=1:00:00 --wrap "true" 2>&1 | head -1
echo -n "  A800 x1  (普通队列): "
sbatch --test-only -p i64m1tga800u --gres=gpu:a800:1 --ntasks=1 \
       --cpus-per-task=8 --mem=64G --time=1:00:00 --wrap "true" 2>&1 | head -1

echo
echo "── 7. 存储 ──"
df -h 2>/dev/null | grep -E "hpc2|Filesystem"

if [ "${1:-}" = "--du" ]; then
  echo "  (家目录体积统计，HDD 上较慢，请耐心等待)"
  du -sh ~/* 2>/dev/null | sort -rh | head -10
else
  echo "  家目录体积 Top10: 加 --du 参数统计（HDD 上较慢）"
fi

echo
echo "════════════════════════════════════════════════════════"
echo " 提示: 8 卡作业只能提交到 i64m1tga800u8ka / i64m1tga40u8ka"
echo " 提示: 训练必须存断点，且只保留一个"
echo "════════════════════════════════════════════════════════"
