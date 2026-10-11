#!/usr/bin/env bash
# =============================================================================
# xfer.sh — 二期(hpc2) <-> 三期(hpc3) 跨集群文件传输
#
# 两个集群没有共享文件系统，但内网互通（实测 120-190 MB/s）。
# 本脚本自动识别当前在哪个集群，并把另一个集群当作 peer。
#
# 用法（在任一集群的登录节点上执行）：
#   xfer.sh peer                                 显示对端集群
#   xfer.sh ls   <remote_path>                   列对端目录
#   xfer.sh du   <remote_path>                   看对端路径体积
#   xfer.sh pull <remote_path> [local_dest]      从对端拉过来
#   xfer.sh push <local_path>  <remote_dir>      推到对端
#   xfer.sh size <local_path>                    本地实际大小
#
# 例：
#   xfer.sh pull '~/thinktool/models/qwen3_4b_inst_2507' ~/models/
#   xfer.sh pull '~/output/ckpt/step_400' ~/ckpt/step_400
#   xfer.sh push ~/ztwang/ '~/ztwang/'
#
# 特性：断点续传（--partial --append-verify），显示进度，自动跳过已完成分片。
# 注意：请在**登录节点**执行；不要把大流量压到计算节点上。
# 注意：远端路径用单引号包住，避免本地 shell 提前展开 ~。
# =============================================================================
set -euo pipefail

HOST2="hpc2login.hpc.hkust-gz.edu.cn"
HOST3="hpc3login.hpc.hkust-gz.edu.cn"
USER_NAME="${USER:-xyu608}"

# --- 自动识别当前集群 ---
case "${HOME}" in
  /hpc2hdd/home/*)  ME="二期(hpc2)";  PEER="${USER_NAME}@${HOST3}"; PEER_NAME="三期(hpc3)" ;;
  /data/user/*)     ME="三期(hpc3)";  PEER="${USER_NAME}@${HOST2}"; PEER_NAME="二期(hpc2)" ;;
  *)
    echo "错误：无法识别当前集群（HOME=${HOME}）" >&2
    echo "本脚本只支持二期(/hpc2hdd/home/*)和三期(/data/user/*)。" >&2
    exit 1 ;;
esac

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ConnectTimeout=15"
RSYNC_OPTS=(-a --partial --append-verify --info=progress2 --human-readable)
RSYNC_SSH="ssh ${SSH_OPTS}"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

cmd="${1:-}"; shift || true

case "${cmd}" in
  peer)
    echo "当前: ${ME}"
    echo "对端: ${PEER_NAME}  ->  ${PEER}"
    ;;

  ls)
    [ $# -ge 1 ] || { echo "用法: xfer.sh ls <remote_path>" >&2; exit 2; }
    ssh ${SSH_OPTS} "${PEER}" "ls -la $1"
    ;;

  du)
    [ $# -ge 1 ] || { echo "用法: xfer.sh du <remote_path>" >&2; exit 2; }
    ssh ${SSH_OPTS} "${PEER}" "du -sh $1 2>/dev/null || echo '路径不存在或无权限'"
    ;;

  size)
    [ $# -ge 1 ] || { echo "用法: xfer.sh size <local_path>" >&2; exit 2; }
    du -sh "$1"
    ;;

  pull)
    src="${1:?用法: xfer.sh pull <remote_path> [local_dest]}"
    dst="${2:-.}"
    mkdir -p "${dst}"
    echo ">>> 从 ${PEER_NAME} 拉取: ${src}  ->  ${dst}"
    rsync "${RSYNC_OPTS[@]}" -e "${RSYNC_SSH}" "${PEER}:${src}" "${dst}"
    echo ">>> 完成"
    ;;

  push)
    src="${1:?用法: xfer.sh push <local_path> <remote_dir>}"
    dst="${2:?用法: xfer.sh push <local_path> <remote_dir>}"
    ssh ${SSH_OPTS} "${PEER}" "mkdir -p ${dst}"
    echo ">>> 推送到 ${PEER_NAME}: ${src}  ->  ${dst}"
    rsync "${RSYNC_OPTS[@]}" -e "${RSYNC_SSH}" "${src}" "${PEER}:${dst}"
    echo ">>> 完成"
    ;;

  ""|-h|--help|help)
    usage 0
    ;;

  *)
    echo "未知命令: ${cmd}" >&2
    usage 2
    ;;
esac
