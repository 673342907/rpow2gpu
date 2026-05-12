#!/usr/bin/env python3
"""rpow2 GPU miner.

Upstream reference: https://github.com/ImMike/rpow2-gpu-miner (MIT)

Mines rpow2 SHA-256 proof-of-work on any Vulkan-capable GPU (AMD, NVIDIA,
Intel, integrated). Uses Taichi to JIT-compile a SPIR-V compute shader from
the Python source — no separate driver toolchain required.

Ubuntu server (NVIDIA example): install proprietary driver + Vulkan (e.g.
nvidia-driver-XXX and venv with `pip install -r requirements.txt`). Cookie
must include the full `rpow_session=...` string copied from logged-in browser.

Quick start:

    pip install -r requirements.txt
    export RPOW_COOKIE='rpow_session=...'   # DevTools → Network → request cookie
    python rpow2_gpu_miner.py

Or on the server: put the same string in a file and use `--cookie-file`.

Statistics (MH/s, mint counts) default to `--stats-file rpow2_miner_stats.json`;
view with `python rpow2_miner_status.py --watch`. Background: `bash start_miner_background.sh`
(after `chmod +x` and setting `cookie.txt` or env vars).

Run forever, or stop after N tokens with `--rounds N`. See `--help` for all
flags.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import taichi as ti

# --------------------------------------------------------------------------
# rpow2 API constants. Override via env if you're testing against a fork.
# --------------------------------------------------------------------------
API_BASE = os.environ.get("RPOW_API_BASE", "https://api.rpow2.com")
ORIGIN = os.environ.get("RPOW_ORIGIN", "https://rpow2.com")
# 站点可能拒绝明显「矿工」UA；默认模拟桌面 Chrome（Linux）。可在 DevTools → Network
# 里复制任一条 api 请求的 User-Agent 到环境变量 RPOW_USER_AGENT。
_DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
USER_AGENT = os.environ.get("RPOW_USER_AGENT", _DEFAULT_BROWSER_UA)


# --------------------------------------------------------------------------
# HTTP helper. Stdlib only, no `requests` dependency.
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"http {status}: {body}")
        self.status = status
        self.body = body


def http(method: str, path: str, cookie: str, body=None, timeout: float = 60.0):
    ref = f"{ORIGIN}/"
    headers = {
        "cookie": cookie,
        "origin": ORIGIN,
        "referer": ref,
        "user-agent": USER_AGENT,
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
    }
    data = None
    if body is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        raise ApiError(e.code, parsed) from None


def normalize_rpow_cookie(raw: str) -> str:
    """Parse browser DevTools cookie copy: full header, or only ``rpow_session=`` pair.

    Strips BOM/whitespace, optional ``Cookie:`` prefix, and picks the
    ``rpow_session=...`` segment from ``;``-separated cookie strings.
    """
    if not raw:
        return ""
    s = raw.lstrip("\ufeff").strip()
    line = s.splitlines()[0].strip()
    if len(line) >= 2 and line[0] == line[-1] and line[0] in "'\"":
        line = line[1:-1].strip()
    if line.lower().startswith("cookie:"):
        line = line.split(":", 1)[1].strip()
    for part in line.split(";"):
        p = part.strip()
        if "=" not in p:
            continue
        name, value = p.split("=", 1)
        if name.strip().lower() == "rpow_session":
            return f"rpow_session={value}"
    if "=" in line:
        name, value = line.split("=", 1)
        if name.strip().lower() == "rpow_session":
            return f"rpow_session={value}"
    return ""


def _cookie_format_hint(raw: str) -> str:
    """Short safe hint if parsing failed (no cookie value leaked)."""
    keys = []
    blob = raw.lstrip("\ufeff").strip().splitlines()[0].strip()
    if blob.lower().startswith("cookie:"):
        blob = blob.split(":", 1)[1].strip()
    for part in blob.split(";"):
        p = part.strip()
        if "=" in p:
            keys.append(p.split("=", 1)[0].strip()[:32])
    if keys:
        return "saw cookie name(s): " + ", ".join(keys)
    preview = blob[:60].replace("\n", " ")
    if len(blob) > 60:
        preview += "..."
    return f"first line starts with: {preview!r}"


def atomic_write_json(path: str, obj: dict) -> None:
    dname = os.path.dirname(os.path.abspath(path))
    if dname:
        os.makedirs(dname, exist_ok=True)
    tmp = path + ".tmp"
    payload = json.dumps(obj, indent=2, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


def load_cumulative_minted(stats_path: str) -> int:
    if not stats_path or not os.path.isfile(stats_path):
        return 0
    try:
        with open(stats_path, encoding="utf-8") as f:
            o = json.load(f)
        return int(o.get("cumulative_minted", 0))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return 0


class MinerStats:
    """Thread-safe counters + periodic JSON snapshot for rpow2_miner_status.py."""

    def __init__(self, path: str, cumulative_initial: int):
        self.path = path
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.miner_pid = os.getpid()
        self.email: str = ""
        self.balance = None
        self.minted_api = None
        self.session_minted = 0
        self.cumulative_minted = cumulative_initial
        self.total_hash_attempts = 0
        self.total_solve_seconds = 0.0
        self.session_started_at = time.time()
        self.failures = 0
        self.phase = "starting"
        self.phase_detail = ""
        self.last_mh_s = None
        self.last_difficulty_bits = None
        self.last_solve_ms = None
        self.last_attempts = None
        self.recent_mh: list[float] = []

    def set_account(self, me: dict):
        with self.lock:
            self.email = me.get("email", "")
            self.balance = me.get("balance")
            self.minted_api = me.get("minted")

    def set_phase(self, phase: str, detail: str = ""):
        with self.lock:
            self.phase = phase
            self.phase_detail = detail

    def note_failure(self):
        with self.lock:
            self.failures += 1

    def note_solve_complete(
        self,
        attempts: int,
        solve_seconds: float,
        bits: int,
        solve_ms: float,
    ):
        """GPU 本题结束（尚未提交 /mint）：更新哈希与瞬时/滑动算力。"""
        with self.lock:
            self.total_hash_attempts += attempts
            self.total_solve_seconds += solve_seconds
            mh = attempts / solve_seconds / 1e6 if solve_seconds > 0 else None
            self.last_mh_s = mh
            self.last_difficulty_bits = bits
            self.last_solve_ms = solve_ms
            self.last_attempts = attempts
            if mh is not None:
                self.recent_mh.append(float(mh))
                if len(self.recent_mh) > 10:
                    self.recent_mh.pop(0)

    def note_mint_success(self):
        """链上确认入库成功：增加挖出统计。"""
        with self.lock:
            self.session_minted += 1
            self.cumulative_minted += 1

    def snapshot(self, running: bool = True) -> dict:
        with self.lock:
            wall = max(0.0, time.time() - self.session_started_at)
            total_h = self.total_hash_attempts
            ts = self.total_solve_seconds
            avg_c = (total_h / ts / 1e6) if ts > 0 else None
            avg_w = (total_h / wall / 1e6) if wall > 0 else None
            recent = None
            if self.recent_mh:
                recent = sum(self.recent_mh) / len(self.recent_mh)
            return {
                "schema_version": 1,
                "miner_pid": self.miner_pid,
                "running": running,
                "updated_at": time.time(),
                "updated_at_iso": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "email": self.email,
                "balance": self.balance,
                "minted_api": self.minted_api,
                "session_minted": self.session_minted,
                "cumulative_minted": self.cumulative_minted,
                "total_hash_attempts": total_h,
                "total_solve_seconds": ts,
                "session_wall_seconds": wall,
                "last_mh_s": self.last_mh_s,
                "recent_avg_mh_s": recent,
                "avg_mh_s_compute": avg_c,
                "avg_mh_s_wall": avg_w,
                "last_difficulty_bits": self.last_difficulty_bits,
                "last_solve_ms": self.last_solve_ms,
                "last_attempts": self.last_attempts,
                "failures": self.failures,
                "phase": self.phase,
                "phase_detail": self.phase_detail,
            }

    def flush(self, running: bool = True):
        atomic_write_json(self.path, self.snapshot(running=running))

    def start_heartbeat(self):
        def loop():
            while not self._stop.wait(1.0):
                self.flush(running=True)

        threading.Thread(target=loop, daemon=True).start()

    def stop_heartbeat(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Taichi / Vulkan kernel.
#
# rpow2 PoW spec (matches the on-site browser worker):
#   preimage = nonce_prefix_bytes || little-endian uint64 nonce
#   accept iff trailing_zero_bits(SHA-256(preimage)) >= difficulty_bits
#
# `nonce_prefix` is 16 bytes (32 hex chars) issued by POST /challenge.
# Total preimage is 24 bytes; the SHA-256 padded message fits in one block.
# --------------------------------------------------------------------------
ti.init(arch=ti.vulkan, log_level=ti.WARN)

_K_TABLE = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
_H0_TABLE = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

K = ti.field(dtype=ti.u32, shape=64)
H0 = ti.field(dtype=ti.u32, shape=8)
for _i, _v in enumerate(_K_TABLE):
    K[_i] = _v
for _i, _v in enumerate(_H0_TABLE):
    H0[_i] = _v


@ti.func
def _rotr(x: ti.u32, n: ti.i32) -> ti.u32:
    return (x >> n) | (x << (32 - n))


@ti.kernel
def mine_kernel(
    n_threads: ti.i32,
    iters: ti.i32,
    base_nonce: ti.u32,
    p0: ti.u32, p1: ti.u32, p2: ti.u32, p3: ti.u32,
    bit_mask: ti.u32,
    result: ti.types.ndarray(dtype=ti.u32, ndim=1),  # [found, nonce, _pad]
):
    for gid in range(n_threads):
        local_base = base_nonce + ti.u32(gid) * ti.u32(iters)
        for k in range(iters):
            nonce = local_base + ti.u32(k)
            # SHA-256 reads each 4-byte word big-endian; nonce bytes are LE,
            # so W[4] is the byte-swap of nonce_lo.
            w4 = ((nonce & ti.u32(0xff)) << 24) \
                | (((nonce >> 8) & ti.u32(0xff)) << 16) \
                | (((nonce >> 16) & ti.u32(0xff)) << 8) \
                | ((nonce >> 24) & ti.u32(0xff))

            W = ti.Vector.zero(ti.u32, 64)
            W[0] = p0
            W[1] = p1
            W[2] = p2
            W[3] = p3
            W[4] = w4
            W[5] = ti.u32(0)
            W[6] = ti.u32(0x80000000)  # SHA-256 padding marker
            W[15] = ti.u32(192)        # bit length: 24 bytes = 192 bits

            for t in range(16, 64):
                s0 = _rotr(W[t - 15], 7) ^ _rotr(W[t - 15], 18) ^ (W[t - 15] >> 3)
                s1 = _rotr(W[t - 2], 17) ^ _rotr(W[t - 2], 19) ^ (W[t - 2] >> 10)
                W[t] = W[t - 16] + s0 + W[t - 7] + s1

            a = H0[0]; b = H0[1]; c = H0[2]; d = H0[3]
            e = H0[4]; f = H0[5]; g = H0[6]; h = H0[7]

            for t in range(64):
                S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
                ch = (e & f) ^ ((~e) & g)
                temp1 = h + S1 + ch + K[t] + W[t]
                S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
                mj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = S0 + mj
                h = g; g = f; f = e
                e = d + temp1
                d = c; c = b; b = a
                a = temp1 + temp2

            h7 = H0[7] + h
            # For target_bits in [1..32], "trailing zero bits >= target" iff
            # (h7 & ((1<<bits)-1)) == 0, because h7 is the last 4 digest bytes
            # and the rpow2 trailing-zero count starts at byte[31].
            if (h7 & bit_mask) == ti.u32(0):
                prev = ti.atomic_or(result[0], ti.u32(1))
                if prev == ti.u32(0):
                    result[1] = nonce


def _hex_prefix_to_uint32_be(prefix_hex: str):
    pb = bytes.fromhex(prefix_hex)
    if len(pb) != 16:
        raise ValueError(f"expected 16-byte prefix, got {len(pb)}")
    return tuple(int.from_bytes(pb[i:i + 4], "big") for i in range(0, 16, 4))


def _trailing_zero_bits(d: bytes) -> int:
    z = 0
    for byte in reversed(d):
        if byte == 0:
            z += 8
            continue
        c = 0
        while not (byte & (1 << c)):
            c += 1
        return z + c
    return z


def verify(prefix_hex: str, nonce: int, target_bits: int) -> bool:
    """Sanity-check that a (prefix, nonce) really meets the target."""
    msg = bytes.fromhex(prefix_hex) + nonce.to_bytes(8, "little")
    return _trailing_zero_bits(hashlib.sha256(msg).digest()) >= target_bits


def solve(prefix_hex, target_bits, n_threads, iters, attempt_cap):
    """Return (winning_nonce, total_attempts). Raises RuntimeError on cap."""
    if not (1 <= target_bits <= 31):
        raise ValueError(f"target_bits must be 1..31, got {target_bits}")
    p0, p1, p2, p3 = _hex_prefix_to_uint32_be(prefix_hex)
    bit_mask = np.uint32((1 << target_bits) - 1)
    base = np.uint32(0)
    total = 0
    while total < attempt_cap:
        result_buf = np.zeros(3, dtype=np.uint32)
        mine_kernel(
            n_threads, iters, int(base),
            np.uint32(p0), np.uint32(p1), np.uint32(p2), np.uint32(p3),
            bit_mask, result_buf,
        )
        ti.sync()
        total += n_threads * iters
        if int(result_buf[0]) == 1:
            nonce = int(result_buf[1])
            if not verify(prefix_hex, nonce, target_bits):
                raise RuntimeError(
                    f"kernel returned nonce={nonce} that does not verify on CPU; "
                    "this should never happen — please file a bug"
                )
            return nonce, total
        base = (base + np.uint32(n_threads * iters)) & np.uint32(0xFFFFFFFF)
    raise RuntimeError(f"attempt_cap reached after {total} hashes")


# --------------------------------------------------------------------------
# Live mining loop.
# --------------------------------------------------------------------------
def main():
    global USER_AGENT
    p = argparse.ArgumentParser(
        description="GPU-accelerated rpow2 miner (Vulkan/Taichi).",
    )
    p.add_argument(
        "--cookie",
        default=os.environ.get("RPOW_COOKIE"),
        help="rpow2 session cookie. Format: 'rpow_session=<value>'. "
             "Defaults to $RPOW_COOKIE.",
    )
    p.add_argument(
        "--cookie-file",
        metavar="PATH",
        help="read cookie from file (single line, e.g. rpow_session=...). "
             "Overrides --cookie / $RPOW_COOKIE when set.",
    )
    p.add_argument(
        "--user-agent",
        metavar="STR",
        help="override HTTP User-Agent (env RPOW_USER_AGENT also applies at startup; "
             "this flag wins if set)",
    )
    p.add_argument(
        "--stats-file",
        default=os.environ.get("RPOW_STATS_FILE", "rpow2_miner_stats.json"),
        metavar="PATH",
        help="write live statistics JSON for rpow2_miner_status.py (default: ./rpow2_miner_stats.json; "
             "disabled with --no-stats)",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="do not write statistics JSON",
    )
    p.add_argument(
        "--pid-file",
        metavar="PATH",
        default=os.environ.get("RPOW_PID_FILE"),
        help="write miner PID after GPU warm-up (for systemd / scripts). env: $RPOW_PID_FILE",
    )
    p.add_argument("--rounds", type=int, default=0,
                   help="stop after N successful mints (0 = run forever)")
    p.add_argument("--threads", type=int, default=1 << 20,
                   help="GPU threads per kernel launch (default: 1048576)")
    p.add_argument("--iters", type=int, default=64,
                   help="nonces per thread per launch (default: 64)")
    p.add_argument("--attempt-cap", type=int, default=1 << 36,
                   help="abort a single PoW after this many attempts (safety)")
    p.add_argument("--quiet", action="store_true",
                   help="only print summary, no per-mint lines")
    args = p.parse_args()

    if args.user_agent:
        USER_AGENT = args.user_agent

    cookie_raw = (args.cookie or "").strip()
    if args.cookie_file:
        try:
            with open(args.cookie_file, encoding="utf-8") as f:
                cookie_raw = f.read()
        except OSError as e:
            sys.exit(f"cannot read --cookie-file: {e}")

    args.cookie = normalize_rpow_cookie(cookie_raw)

    if not args.cookie:
        if not cookie_raw.strip():
            sys.exit(
                "no cookie supplied. set $RPOW_COOKIE or pass --cookie / --cookie-file. "
                "tip: in your browser, DevTools → Network → click any /api request "
                "→ copy the 'cookie' header (must include rpow_session=...)."
            )
        sys.exit(
            "could not find rpow_session= in cookie text. "
            "paste the full 'cookie' request header from DevTools (multiple "
            "cookies separated with '; ' are OK), or a single line "
            f"rpow_session=....\n({_cookie_format_hint(cookie_raw)})"
        )
    stats_path = None if args.no_stats else args.stats_file
    pid_file = args.pid_file.strip() if args.pid_file else None

    # Confirm auth before warming up the GPU.
    try:
        status, me = http("GET", "/me", args.cookie)
    except ApiError as e:
        sys.exit(f"auth check failed: {e}")
    if status != 200 or not me or "email" not in me:
        sys.exit(f"unexpected /me response: {me}")
    print(f"signed in: {me['email']}  balance={me['balance']}  minted={me['minted']}",
          file=sys.stderr, flush=True)

    cum0 = load_cumulative_minted(stats_path) if stats_path else 0
    stats = MinerStats(stats_path, cum0) if stats_path else None
    if stats:
        stats.set_account(me)
        stats.set_phase("warming_gpu", "SPIR-V kernel compile / warm-up")
        stats.flush()
        stats.start_heartbeat()

    print("compiling SPIR-V kernel (first launch only)...", file=sys.stderr, flush=True)
    warm = np.zeros(3, dtype=np.uint32)
    mine_kernel(
        args.threads, args.iters, np.uint32(0),
        np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
        np.uint32(0xFFFFFFFF), warm,
    )
    ti.sync()
    print("kernel ready.\n", file=sys.stderr, flush=True)

    if pid_file:
        try:
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except OSError as e:
            sys.exit(f"cannot write --pid-file {pid_file}: {e}")

    if stats:
        stats.set_phase("idle", "waiting for challenge")
        stats.flush()

    minted = 0
    failures = 0
    started_at = time.time()

    def stop_summary(*_):
        if stats:
            stats.stop_heartbeat()
            stats.set_phase("stopped", "miner exit")
            stats.flush(running=False)
        if pid_file:
            try:
                if os.path.isfile(pid_file):
                    os.remove(pid_file)
            except OSError:
                pass
        elapsed = time.time() - started_at
        print(file=sys.stderr)
        print("---- summary ----", file=sys.stderr)
        print(f"minted:    {minted}", file=sys.stderr)
        print(f"failures:  {failures}", file=sys.stderr)
        print(f"elapsed:   {elapsed:.1f}s", file=sys.stderr)
        if stats and stats.total_solve_seconds > 0:
            h = stats.total_hash_attempts
            ts = stats.total_solve_seconds
            print(
                f"hash work: {h:,} hashes in {ts:.2f}s compute  "
                f"avg {h / ts / 1e6:.3f} MH/s (compute-only)",
                file=sys.stderr,
            )
        if stats and elapsed > 0 and stats.total_hash_attempts > 0:
            print(
                f"wall avg:  {stats.total_hash_attempts / elapsed / 1e6:.3f} MH/s "
                f"(includes API / idle)",
                file=sys.stderr,
            )
        if elapsed > 0:
            print(f"avg rate:  {minted/elapsed:.2f} tokens/sec  "
                  f"(~{minted/elapsed*3600:.0f}/hour)", file=sys.stderr)
        if stats_path:
            print(f"stats file: {stats_path}", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT,  stop_summary)
    signal.signal(signal.SIGTERM, stop_summary)

    while True:
        if args.rounds and minted >= args.rounds:
            break

        if stats:
            stats.set_phase("challenge", "POST /challenge")

        try:
            _, ch = http("POST", "/challenge", args.cookie)
        except ApiError as e:
            failures += 1
            if stats:
                stats.note_failure()
            print(f"[!] /challenge failed: {e}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            continue

        cid    = ch["challenge_id"]
        prefix = ch["nonce_prefix"]
        bits   = ch["difficulty_bits"]

        if stats:
            stats.set_phase(
                "solving",
                f"GPU PoW challenge_id={cid[:16]}… diff={bits} bits",
            )

        t0 = time.time()
        try:
            nonce, attempts = solve(
                prefix, bits, args.threads, args.iters, args.attempt_cap,
            )
        except RuntimeError as e:
            failures += 1
            if stats:
                stats.note_failure()
            print(f"[!] solve failed for challenge {cid}: {e}",
                  file=sys.stderr, flush=True)
            continue
        solve_sec = time.time() - t0
        solve_ms = solve_sec * 1000.0
        if stats:
            stats.note_solve_complete(attempts, solve_sec, bits, solve_ms)

        if stats:
            stats.set_phase("minting", f"POST /mint challenge_id={cid[:16]}…")

        try:
            _, m = http(
                "POST", "/mint", args.cookie,
                {"challenge_id": cid, "solution_nonce": str(nonce)},
            )
        except ApiError as e:
            failures += 1
            if stats:
                stats.note_failure()
            print(f"[!] /mint failed (challenge {cid}): {e}",
                  file=sys.stderr, flush=True)
            continue

        minted += 1
        if stats:
            stats.note_mint_success()
            try:
                _, me2 = http("GET", "/me", args.cookie)
                if me2:
                    stats.set_account(me2)
            except ApiError:
                pass
            stats.set_phase("idle", "mint ok, next challenge")
            stats.flush()

        token_id = (m or {}).get("token", {}).get("id", "?")
        inst_mh = attempts / solve_sec / 1e6 if solve_sec > 0 else 0.0
        if not args.quiet:
            print(
                f"minted #{minted:<5d}  bits={bits}  solve={solve_ms:>5.0f}ms  "
                f"MH/s={inst_mh:>8.3f}  attempts={attempts:>10,}  token={token_id}",
                flush=True,
            )

    stop_summary()


if __name__ == "__main__":
    main()
