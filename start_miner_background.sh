#!/usr/bin/env bash
# 在 Ubuntu 上用 nohup 后台启动矿工；实时统计见 rpow2_miner_status.py --watch
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

STATS="${RPOW_STATS_FILE:-$DIR/rpow2_miner_stats.json}"
PIDF="${RPOW_PID_FILE:-$DIR/rpow2_miner.pid}"
COOKIE="${RPOW_COOKIE_FILE:-$DIR/cookie.txt}"
LOG="${RPOW_MINER_LOG:-$DIR/rpow2_miner.log}"
PY="${RPOW_PYTHON:-python3}"

if [[ ! -f "$COOKIE" ]]; then
  echo "缺少 cookie 文件: $COOKIE（可设置环境变量 RPOW_COOKIE_FILE）" >&2
  exit 1
fi

export RPOW_STATS_FILE="$STATS"
nohup env PYTHONUNBUFFERED=1 \
  "$PY" "$DIR/rpow2_gpu_miner.py" \
  --cookie-file "$COOKIE" \
  --stats-file "$STATS" \
  --pid-file "$PIDF" \
  >>"$LOG" 2>&1 &

echo "已启动后台矿工（nohup）。"
echo "  日志: $LOG"
echo "  统计: $STATS"
echo "  PID 将由矿工写入: $PIDF"
echo "查看统计: $PY $DIR/rpow2_miner_status.py --stats-file \"$STATS\" --watch"
