"""Fair side-by-side benchmark for Part A kernels.

This script runs your `matmul_best` and your colleague's `matmul_best`
with exactly the same matrix size, warmup count, repeat count, and timing
metric on the same device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pyopencl as cl


@dataclass
class Tune:
    n: int = 4096
    tile_size: int = 16
    tsm: int = 128
    tsn: int = 128
    tsk: int = 16
    wptm: int = 8
    wptn: int = 8
    width: int = 4
    wpt: int = 4

    @property
    def rtsm(self) -> int:
        return self.tsm // self.wptm

    @property
    def rtsn(self) -> int:
        return self.tsn // self.wptn

    @property
    def tsk4(self) -> int:
        return self.tsk // 4


def kernel_prefix(t: Tune) -> str:
    return f"""
#define TILE_SIZE {t.tile_size}
#define TSM {t.tsm}
#define TSN {t.tsn}
#define TSK {t.tsk}
#define TSK4 {t.tsk4}
#define WPTM {t.wptm}
#define WPTN {t.wptn}
#define RTSM {t.rtsm}
#define RTSN {t.rtsn}
#define WIDTH {t.width}
#define WPT {t.wpt}
"""


def load_program(ctx: cl.Context, kernel_path: Path, tune: Tune) -> cl.Program:
    # Some exported files may carry UTF-8 BOM; strip it for OpenCL compiler.
    source = kernel_path.read_text(encoding="utf-8").lstrip("\ufeff")
    return cl.Program(ctx, kernel_prefix(tune) + source).build()


def bench_kernel(
    queue: cl.CommandQueue,
    kernel,
    global_size: Tuple[int, int],
    local_size: Tuple[int, int],
    args: Tuple,
    warmup: int,
    repeats: int,
) -> Tuple[float, float]:
    for _ in range(warmup):
        kernel(queue, global_size, local_size, *args).wait()

    times = []
    for _ in range(repeats):
        event = kernel(queue, global_size, local_size, *args)
        event.wait()
        times.append((event.profile.end - event.profile.start) * 1e-9)

    t = min(times)
    return t, gflops(int(args[-1]), t)


def gflops(n: int, seconds: float) -> float:
    return 2.0 * (n**3) / (seconds * 1e9)


def correctness_check(
    program: cl.Program,
    queue: cl.CommandQueue,
    tune: Tune,
    check_n: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    a = rng.random((check_n, check_n), dtype=np.float32)
    b = rng.random((check_n, check_n), dtype=np.float32)
    c_ref = np.empty_like(a)
    c_best = np.empty_like(a)

    mf = cl.mem_flags
    a_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(queue.context, mf.WRITE_ONLY, c_ref.nbytes)
    std = (a_buf, b_buf, c_buf, np.int32(check_n))

    program.matmul_coalesced(queue, (check_n, check_n), (tune.tile_size, tune.tile_size), *std).wait()
    cl.enqueue_copy(queue, c_ref, c_buf).wait()

    best_global = (check_n * tune.rtsm // tune.tsm, check_n * tune.rtsn // tune.tsn)
    best_local = (tune.rtsm, tune.rtsn)
    program.matmul_best(queue, best_global, best_local, *std).wait()
    cl.enqueue_copy(queue, c_best, c_buf).wait()

    return float(np.max(np.abs(c_best - c_ref)))


def run_one(
    label: str,
    kernel_path: Path,
    ctx: cl.Context,
    queue: cl.CommandQueue,
    tune: Tune,
    repeats: int,
    warmup: int,
    seed: int,
    check_n: int,
) -> Tuple[float, float, float]:
    program = load_program(ctx, kernel_path, tune)

    rng = np.random.default_rng(seed)
    a = rng.random((tune.n, tune.n), dtype=np.float32)
    b = rng.random((tune.n, tune.n), dtype=np.float32)
    c = np.empty_like(a)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
    std = (a_buf, b_buf, c_buf, np.int32(tune.n))

    naive_global = (tune.n, tune.n)
    naive_local = (tune.tile_size, tune.tile_size)
    best_global = (tune.n * tune.rtsm // tune.tsm, tune.n * tune.rtsn // tune.tsn)
    best_local = (tune.rtsm, tune.rtsn)

    t_coal, gf_coal = bench_kernel(queue, program.matmul_coalesced, naive_global, naive_local, std, warmup, repeats)
    t_best, gf_best = bench_kernel(queue, program.matmul_best, best_global, best_local, std, warmup, repeats)

    max_abs_diff = correctness_check(program, queue, tune, check_n, seed + 7)
    print(f"{label:12s} | coalesced: {gf_coal:8.2f} GFLOPS | best: {gf_best:8.2f} GFLOPS | max_abs_diff(best-vs-coalesced@N={check_n}): {max_abs_diff}")
    return gflops(tune.n, t_coal), gflops(tune.n, t_best), max_abs_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair benchmark: your kernel vs colleague kernel")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--check-n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tune = Tune(n=args.n)

    project_root = Path(__file__).resolve().parents[2]
    your_kernel = project_root / "src" / "kernels" / "matmul_kernels.cl"
    colleague_kernel = project_root / "experiments" / "colleague" / "branch_partieA_snapshot" / "matmul_kernels_colleague.cl"

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print(f"Device: {device.name}")
    print(f"Policy: N={args.n}, warmup={args.warmup}, repeats={args.repeats}, metric=min(event_time)")
    print("Running fair comparison...")

    _, your_best, your_diff = run_one(
        "yours",
        your_kernel,
        ctx,
        queue,
        tune,
        args.repeats,
        args.warmup,
        args.seed,
        args.check_n,
    )
    _, coll_best, coll_diff = run_one(
        "colleague",
        colleague_kernel,
        ctx,
        queue,
        tune,
        args.repeats,
        args.warmup,
        args.seed,
        args.check_n,
    )

    winner = "yours" if your_best >= coll_best else "colleague"
    print("---")
    print(f"Winner by fair policy: {winner}")
    print(f"Best GFLOPS delta (colleague - yours): {coll_best - your_best:.2f}")
    print(f"Correctness checks: yours={your_diff}, colleague={coll_diff}")


if __name__ == "__main__":
    main()
