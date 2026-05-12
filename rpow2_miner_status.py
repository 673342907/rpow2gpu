#!/usr/bin/env python3
"""读取 rpow2_gpu_miner.py 写入的统计 JSON，终端中查看或 --watch 实时监控。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _fmt_mh(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x / 1000:.3f} GH/s"
    return f"{x:.3f} MH/s"


def _alive(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    if os.name != "posix":
        return None
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def print_report(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
    except FileNotFoundError:
        print(f"未找到统计文件: {path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"统计文件 JSON 损坏: {e}", file=sys.stderr)
        return 1

    pid = s.get("miner_pid")
    alive = _alive(pid) if pid else None
    run_hint = ""
    if alive is True:
        run_hint = "（进程在运行）"
    elif alive is False:
        run_hint = "（进程已不在，可能已退出）"

    wall = s.get("session_wall_seconds")
    cum = s.get("cumulative_minted")
    sess = s.get("session_minted")
    total_hashes = s.get("total_hash_attempts")
    ts = s.get("total_solve_seconds")

    avg_c = s.get("avg_mh_s_compute")
    avg_w = s.get("avg_mh_s_wall")
    last = s.get("last_mh_s")
    recent = s.get("recent_avg_mh_s")

    print("── rpow2 挖矿统计 ──" + run_hint)
    print(f"  账户           {s.get('email', '—')}")
    print(f"  余额 / 链上已铸 {s.get('balance', '—')} / {s.get('minted_api', '—')}")
    print(f"  本轮已挖出      {sess if sess is not None else '—'}")
    print(f"  累计挖出(本地)  {cum if cum is not None else '—'}  （跨重启累加，见说明）")
    print(f"  阶段           {s.get('phase', '—')}")
    if s.get("phase_detail"):
        print(f"  阶段说明       {s['phase_detail']}")
    print(f"  失败次数       {s.get('failures', '—')}")
    print(f"- 算力 (MH/s) -")
    print(f"  上一题瞬时     {_fmt_mh(last)}")
    if recent is not None:
        print(f"  近 N 题平均     {_fmt_mh(recent)}  （最多 10 题滑动）")
    print(f"  平均(仅解题)   {_fmt_mh(avg_c)}  （总哈希 / 总解题秒数）")
    print(f"  平均(含等待)   {_fmt_mh(avg_w)}  （总哈希 / 会话墙钟时间）")
    print(f"- 体量 -")
    print(f"  累计哈希尝试   {total_hashes if total_hashes is not None else '—'}")
    if ts is not None and wall is not None:
        print(f"  解题总用时     {ts:.2f} s  |  会话墙钟 {wall:.1f} s")
    print(f"  最近难度       {s.get('last_difficulty_bits', '—')} bits")
    print(f"  更新于         {s.get('updated_at_iso', s.get('updated_at', '—'))}")
    if pid:
        print(f"  矿工 PID       {pid}")
    print("")
    print("说明: 「累计挖出」依赖同一 --stats-file；链上总数以网站「minted」为准。")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="查看 rpow2 GPU 矿工统计（--stats-file 对应 JSON）")
    p.add_argument(
        "--stats-file",
        default=os.environ.get("RPOW_STATS_FILE", "rpow2_miner_stats.json"),
        help="与矿工 --stats-file 一致的路径（可用环境变量 RPOW_STATS_FILE）",
    )
    p.add_argument(
        "--watch",
        type=float,
        metavar="SEC",
        nargs="?",
        const=1.0,
        help="每隔 SEC 秒刷新（默认 1）；Ctrl+C 退出",
    )
    args = p.parse_args()

    if args.watch is not None:
        interval = max(0.2, float(args.watch))
        try:
            while True:
                if os.name == "posix":
                    os.system("clear")
                else:
                    os.system("cls")
                print_report(args.stats_file)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n已停止监控。")
            return 0

    return print_report(args.stats_file)


if __name__ == "__main__":
    raise SystemExit(main())
