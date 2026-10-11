#!/bin/bash
# ============================================================================
# 集群一键巡检  —  HKUST-GZ HPC 二期
# 用法:  bash ~/ztwang/tools/check_cluster.sh
# 作用:  看全局负载、我的队列、A800/8卡队列、磁盘、预测起算时间
# ============================================================================
export PATH=/opt/slurm/bin:$PATH
U=xyu608

echo "════════════════════════════════════════════════════════"
echo " 集群巡检  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"

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
echo "════════════════════════════════════════════════════════"
