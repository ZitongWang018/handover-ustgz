#!/bin/bash
# phase3_push_stage1.sh -- 三期(ACD) 第一步搬迁：二期 -> 三期，只搬 ALFWorld eval 最小集合。
#
# 在【二期登录节点】执行（cluster-3 无外网，必须走二期直连 rsync，不经本地 Windows）。
# 真实字节约 0.58 GB：代码+依赖+wheel 365 MB，eval 数据 229 MB。
# train 训练集(2.01 GB) 属于 Stage-2，本脚本不碰。
#
# 用法：  bash ~/ztwang/selfplay/env_setup/phase3_push_stage1.sh          # 真搬
#         DRYRUN=1 bash ~/ztwang/selfplay/env_setup/phase3_push_stage1.sh # 干跑核对
set -u

DST_USER=xyu608
DST_HOST=10.120.81.30                 # cluster-3 ACD-Login-2
DST=/data/user/xyu608/                # rsync -R 会在此下还原 ztwang/ 与 .cache/ 结构
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
DRY=""
[ "${DRYRUN:-0}" = "1" ] && DRY="-n"

cd "$HOME" || exit 1
echo "=== phase3 Stage-1 push  $(date -Is)  DRY=$DRY ==="

echo "--- A) 代码 + 依赖清单 + wheel 集 ---"
rsync -a -R $DRY --info=stats2 -e "$RSH" \
  ./ztwang/selfplay/env_setup \
  ./ztwang/selfplay/src \
  ./ztwang/selfplay/configs \
  ./ztwang/selfplay/third_party/glibc232 \
  ./ztwang/selfplay/third_party/wheels \
  ./ztwang/selfplay/third_party/wheels_sdist \
  ./ztwang/selfplay/third_party/requirements_env_common.txt \
  ./ztwang/selfplay/third_party/requirements_offline_extra.txt \
  ./ztwang/selfplay/third_party/WHEELS_MANIFEST.txt \
  "${DST_USER}@${DST_HOST}:${DST}" || { echo "FAIL A rc=$?"; exit 2; }

echo "--- B) ALFWorld eval 数据 (logic + valid_seen/train/unseen) ---"
rsync -a -R $DRY --info=stats2 -e "$RSH" \
  ./.cache/alfworld/logic \
  ./.cache/alfworld/json_2.1.1/valid_seen \
  ./.cache/alfworld/json_2.1.1/valid_train \
  ./.cache/alfworld/json_2.1.1/valid_unseen \
  "${DST_USER}@${DST_HOST}:${DST}" || { echo "FAIL B rc=$?"; exit 3; }

if [ -z "$DRY" ]; then
  echo "--- C) 三期落地校验 ---"
  ${RSH} "${DST_USER}@${DST_HOST}" \
    "du -sb /data/user/xyu608/ztwang/selfplay/third_party/wheels \
            /data/user/xyu608/ztwang/selfplay/third_party/glibc232 \
            /data/user/xyu608/.cache/alfworld/logic \
            /data/user/xyu608/.cache/alfworld/json_2.1.1/valid_seen \
            /data/user/xyu608/.cache/alfworld/json_2.1.1/valid_train \
            /data/user/xyu608/.cache/alfworld/json_2.1.1/valid_unseen; \
     echo '--- wheels 数:'; ls /data/user/xyu608/ztwang/selfplay/third_party/wheels | wc -l; \
     echo '--- 下一步: 三期执行 phase3_install_stage1.sh <env_python>'"
fi
echo "=== phase3 Stage-1 push DONE  $(date -Is) ==="
