"""Benchmark for kmd2048.py (pure Python; timing with time.perf_counter()).

Usage::

    python benchmark.py              # full run (about 30 s)
    python benchmark.py --quick      # fewer trials and smaller inputs
    python benchmark.py --markdown   # print the results as a Markdown table
    python benchmark.py --file PATH  # additionally time kmd2048_file(PATH) once

Method
------
Each benchmark is a function that performs ``ops`` operations.  After one
warm-up call each, the benchmarks are timed in interleaved trial rounds
(round 1 runs every benchmark once, then round 2, and so on), so a slow
period on a shared machine affects all benchmarks alike instead of every
trial of a single benchmark.  For each benchmark the best (minimum) and the
median time per operation are reported: the minimum is the least-disturbed
estimate of the code's own cost, the median shows a typical trial.
Throughput is reported only where data is actually processed.  Every number
printed is measured during the run; nothing is extrapolated or estimated.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import kmd2048
from kmd2048 import (
    KECCAK_ROUND_CONSTANTS,
    SHA512_INITIAL_HASH,
    keccak_f1600,
    kmd2048_file,
    sha512,
    shake256,
)

KIB = 1024
MIB = 1024 * 1024


@dataclass
class Benchmark:
    name: str
    operation: str                      # what one timed operation is
    run: Callable[[int], object]        # run(n) performs n operations
    ops: int                            # operations per timed trial
    bytes_per_op: Optional[int] = None  # data processed per operation
    times: List[float] = field(default_factory=list)   # seconds per op, per trial

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def median(self) -> float:
        ordered = sorted(self.times)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    def throughput(self, seconds: float) -> Optional[float]:
        """MiB per second at ``seconds`` per operation, if data is processed."""
        if not self.bytes_per_op:
            return None
        return self.bytes_per_op / seconds / MIB

    def time_once(self) -> None:
        start = time.perf_counter()
        self.run(self.ops)
        self.times.append((time.perf_counter() - start) / self.ops)


def run_interleaved(benchmarks: List[Benchmark], rounds: int) -> None:
    for bench in benchmarks:
        bench.run(1)                     # warm-up
    for _ in range(rounds):
        for bench in benchmarks:
            bench.time_once()


def pattern(n: int) -> bytes:
    """Deterministic, non-trivial input data (the content does not affect timing)."""
    block = bytes((i * 167 + 13) & 0xFF for i in range(4096))
    return (block * (n // len(block) + 1))[:n]


def load_reference():
    """Import the plain reference implementations from test_kmd2048.py, if present.

    They are straightforward, unoptimised transcriptions of FIPS 202 and
    FIPS 180-4 and serve as the "before optimisation" baseline.
    """
    try:
        import test_kmd2048
    except ImportError:
        return None
    return test_kmd2048


def make_scratch_dir() -> Path:
    for base in (os.environ.get("TMPDIR"), os.environ.get("TEMP"), os.environ.get("TMP"),
                 "/tmp", str(Path(__file__).resolve().parent)):
        if base and os.path.isdir(base) and os.access(base, os.W_OK):
            path = Path(base) / f"kmd2048-bench-{os.getpid()}-{time.perf_counter_ns()}"
            path.mkdir(mode=0o700)
            return path
    raise RuntimeError("no writable directory for the benchmark file")


def size_label(n: int) -> str:
    return f"{n // MIB} MiB" if n >= MIB else f"{n // KIB} KiB"


def build_benchmarks(quick: bool, bench_file: Path) -> List[Benchmark]:
    n_perm = 100 if quick else 500
    bulk = 16 * KIB if quick else 64 * KIB
    big = 256 * KIB if quick else MIB
    msg_bulk, msg_69, msg_1k, msg_big = pattern(bulk), pattern(69), pattern(KIB), pattern(big)
    bench_file.write_bytes(msg_big)
    reference = load_reference()
    out = []

    # -- Keccak-f[1600] ------------------------------------------------------
    state0 = keccak_f1600([0] * 25)

    def keccak_public(n):
        s = state0
        for _ in range(n):
            s = keccak_f1600(s)

    def keccak_core(n, rounds=kmd2048._keccak_rounds, rc=KECCAK_ROUND_CONSTANTS):
        s = state0
        for _ in range(n):
            s = rounds(s, rc)

    out.append(Benchmark("Keccak-f[1600] keccak_f1600() (validating)", "permutation",
                         keccak_public, n_perm, 200))
    out.append(Benchmark("Keccak-f[1600] internal core", "permutation",
                         keccak_core, n_perm, 200))
    if reference is not None:
        def keccak_textbook(n, permute=reference.ref_keccak_f1600):
            s = state0
            for _ in range(n):
                s = permute(s)

        out.append(Benchmark("  baseline: textbook FIPS 202 transcription", "permutation",
                             keccak_textbook, max(1, n_perm // 5), 200))

    # -- SHA-512 -------------------------------------------------------------
    words = tuple(range(16))

    def sha_compress(n, compress=kmd2048._sha512_compress, iv=SHA512_INITIAL_HASH):
        h = iv
        for _ in range(n):
            h = compress(h, words)

    out.append(Benchmark("SHA-512 compression function", "128-byte block",
                         sha_compress, n_perm, 128))
    if reference is not None:
        block = b"".join(w.to_bytes(8, "big") for w in words)

        def sha_textbook(n, compress=reference.ref_sha512_compress):
            h = list(SHA512_INITIAL_HASH)
            for _ in range(n):
                h = compress(h, block)

        out.append(Benchmark("  baseline: textbook FIPS 180-4 transcription",
                             "128-byte block", sha_textbook, max(1, n_perm // 5), 128))
    out.append(Benchmark(f"SHA-512 sha512(), {size_label(bulk)} message", "message",
                         lambda n: [sha512(msg_bulk) for _ in range(n)], 1, bulk))
    out.append(Benchmark("SHA-512 sha512(), 69-byte message", "message",
                         lambda n: [sha512(msg_69) for _ in range(n)], n_perm, 69))

    # -- SHAKE256 ------------------------------------------------------------
    out.append(Benchmark(f"SHAKE256 absorb {size_label(bulk)}, 64-byte output", "call",
                         lambda n: [shake256(msg_bulk, 64) for _ in range(n)], 1, bulk))
    out.append(Benchmark(f"SHAKE256 squeeze {size_label(bulk)} from empty input", "call",
                         lambda n: [shake256(b"", bulk) for _ in range(n)], 1, bulk))
    out.append(Benchmark("SHAKE256 69-byte input, 64-byte output", "call",
                         lambda n: [shake256(msg_69, 64) for _ in range(n)], n_perm, 69))

    # -- KMD-2048 ------------------------------------------------------------
    k1 = shake256(b"KMD-K", 64)
    out.append(Benchmark("KMD-2048 chain H_1 .. H_1024 (fast path)", "chain",
                         lambda n: [kmd2048._kmd2048_chain(k1) for _ in range(n)], 1))
    out.append(Benchmark("KMD-2048 kmd2048(b'')", "digest",
                         lambda n: [kmd2048.kmd2048(b"") for _ in range(n)], 1))
    out.append(Benchmark("KMD-2048 kmd2048(), 1 KiB message", "digest",
                         lambda n: [kmd2048.kmd2048(msg_1k) for _ in range(n)], 1, KIB))
    out.append(Benchmark(f"KMD-2048 kmd2048(), {size_label(big)} message", "digest",
                         lambda n: [kmd2048.kmd2048(msg_big) for _ in range(n)], 1, big))
    out.append(Benchmark(f"KMD-2048 kmd2048_file(), {size_label(big)} file", "digest",
                         lambda n: [kmd2048_file(bench_file) for _ in range(n)], 1, big))
    return out


def describe_environment(rounds: int, quick: bool) -> List[str]:
    lines = [f"Python    : {sys.implementation.name} {sys.version.split()[0]} "
             f"({sys.version.splitlines()[0]})"]
    if hasattr(os, "uname"):
        u = os.uname()
        lines.append(f"Platform  : {u.sysname} {u.release} {u.machine}")
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    lines.append(f"CPU       : {cpu or 'unknown'} ({os.cpu_count()} logical CPUs; "
                 "the benchmark uses one)")
    lines.append(f"kmd2048.py: version {kmd2048.__version__}, {kmd2048.ITERATIONS} iterations, "
                 f"{kmd2048.DIGEST_SIZE}-byte digest")
    lines.append(f"Mode      : {'quick' if quick else 'full'}, {rounds} interleaved trial "
                 "rounds, timer time.perf_counter()")
    return lines


def format_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} us"


def format_rate(bench: Benchmark, seconds: float) -> str:
    rate = bench.throughput(seconds)
    return "-" if rate is None else f"{rate:.3f} MiB/s"


def print_text(benchmarks: List[Benchmark]) -> None:
    width = max(len(b.name) for b in benchmarks)
    print(f"{'benchmark'.ljust(width)}  {'best/op':>11}  {'median/op':>11}  "
          f"{'best rate':>13}  ops/s (best)")
    print("-" * (width + 58))
    for b in benchmarks:
        print(f"{b.name.ljust(width)}  {format_time(b.best):>11}  {format_time(b.median):>11}  "
              f"{format_rate(b, b.best):>13}  {1 / b.best:,.1f} {b.operation}/s")


def print_markdown(benchmarks: List[Benchmark]) -> None:
    print("| Benchmark | Best per op | Median per op | Throughput (best) | Ops/s (best) |")
    print("|---|---:|---:|---:|---:|")
    for b in benchmarks:
        print(f"| {b.name.strip()} | {format_time(b.best)} | {format_time(b.median)} | "
              f"{format_rate(b, b.best)} | {1 / b.best:,.1f} {b.operation}/s |")


def main(argv: List[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    quick = "--quick" in argv
    markdown = "--markdown" in argv
    extra_file = None
    if "--file" in argv:
        index = argv.index("--file")
        if index + 1 >= len(argv):
            print("--file needs a path", file=sys.stderr)
            return 2
        extra_file = argv[index + 1]
    rounds = 3 if quick else 7
    for line in describe_environment(rounds, quick):
        print(line)
    print()

    started = time.perf_counter()
    scratch = make_scratch_dir()
    try:
        benchmarks = build_benchmarks(quick, scratch / "bench.bin")
        run_interleaved(benchmarks, rounds)
    finally:
        for child in scratch.iterdir():
            child.unlink()
        scratch.rmdir()
    if extra_file:
        size = os.path.getsize(extra_file)
        extra = Benchmark(f"KMD-2048 kmd2048_file({extra_file})", "digest",
                          lambda n: [kmd2048_file(extra_file) for _ in range(n)], 1,
                          size or None)
        extra.time_once()                # a single timed run (files may be large)
        benchmarks.append(extra)

    if markdown:
        print_markdown(benchmarks)
    else:
        print_text(benchmarks)

    print()
    chain = next(b for b in benchmarks if b.operation == "chain")
    print(f"One KMD-2048 iteration (chain time / {kmd2048.ITERATIONS}, best): "
          f"{format_time(chain.best / kmd2048.ITERATIONS)}")
    by_name = {b.name.strip(): b for b in benchmarks}
    for optimized, baseline in (
            ("Keccak-f[1600] internal core", "baseline: textbook FIPS 202 transcription"),
            ("SHA-512 compression function", "baseline: textbook FIPS 180-4 transcription")):
        if baseline in by_name:
            ratio = by_name[baseline].best / by_name[optimized].best
            print(f"{optimized}: {ratio:.1f}x faster than the "
                  f"{baseline[len('baseline: '):]} (best vs best)")
    print(f"Total benchmark wall time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
