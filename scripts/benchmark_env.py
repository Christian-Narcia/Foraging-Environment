"""Throughput of VectorForagingEnv under random actions.

Agent frames/s = n_envs x N x steps / time: one 64x64 frame per robot per env step, including
auto-resets (an L2 reset with new occluder sizes compiles a model). If throughput is low, try
2 envs per worker (--envs-per-worker 2), 48x48 images (--img-size 48) or --no-shadows.

Usage, from the repo root:
    python scripts/benchmark_env.py [--n-envs 16 32 64] [--robots 2 4] [--steps 2000] [--level L2]
                                    [--workers W | --envs-per-worker K] [--img-size 64] [--no-shadows]
                                    [--no-render] [--out runs/benchmark.json]

Prints a markdown table and writes every number, with the machine, to --out.
"GPU memory" is the rise in GPU 0's used memory (nvidia-smi) once the workers' GL contexts
exist, so other jobs on the GPU skew it.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from foraging_env import N_ACTIONS, ForagingEnv, VectorForagingEnv   # foraging_env first: it picks the GL backend
from foraging_env.vector_env import available_cpus

import mujoco  # noqa: E402

WARMUP = 50   # untimed steps after reset: GL contexts, first model compiles, page faults


def machine() -> dict:
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = []
    return dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                cpus=os.cpu_count(), cpus_available=available_cpus(), gpu=gpu, mujoco=mujoco.__version__, numpy=np.__version__,
                mujoco_gl=os.environ.get("MUJOCO_GL", "(default)"))


def gpu_mem_mib() -> int | None:
    """Memory in use on GPU 0 (all processes), from nvidia-smi; None without it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def cfg_for(args, n_robots: int) -> dict:
    return {"level": args.level, "n_robots": n_robots, "img_size": args.img_size,
            "shadows": not args.no_shadows, "render": not args.no_render}


def bench_single(args, n_robots: int) -> dict:
    """One ForagingEnv in this process: the per-env cost without any IPC."""
    rng = np.random.default_rng(0)
    with ForagingEnv(cfg_for(args, n_robots), seed=0) as env:
        env.reset()
        for a in rng.integers(N_ACTIONS, size=(WARMUP, n_robots)):
            if env.step(a)[2]:
                env.reset()
        actions = rng.integers(N_ACTIONS, size=(args.steps, n_robots))
        resets, t0 = 0, time.perf_counter()
        for a in actions:
            if env.step(a)[2]:
                env.reset()
                resets += 1
        secs = time.perf_counter() - t0
    return dict(kind="single", n_envs=1, robots=n_robots, workers=0, steps=args.steps, secs=secs,
                resets=resets, frames_per_s=n_robots * args.steps / secs, ms_per_step=1e3 * secs / args.steps)


def bench_vector(args, n_envs: int, n_robots: int) -> dict:
    workers = (math.ceil(n_envs / args.envs_per_worker) if args.envs_per_worker else args.workers)
    rng = np.random.default_rng(0)
    mem0, t0 = gpu_mem_mib(), time.perf_counter()
    with VectorForagingEnv(n_envs, cfg_for(args, n_robots), workers, seed=0) as venv:
        t_start = time.perf_counter() - t0
        t0 = time.perf_counter()
        venv.reset()
        t_reset = time.perf_counter() - t0
        for a in rng.integers(N_ACTIONS, size=(WARMUP, n_envs, n_robots)):
            venv.step(a)
        mem1 = gpu_mem_mib()
        actions = rng.integers(N_ACTIONS, size=(args.steps, n_envs, n_robots))
        resets, t0 = 0, time.perf_counter()
        for a in actions:
            resets += int(venv.step(a)[2].sum())
        secs = time.perf_counter() - t0
        n_workers = venv.n_workers
    return dict(kind="vector", n_envs=n_envs, robots=n_robots, workers=n_workers, steps=args.steps, secs=secs,
                resets=resets, frames_per_s=n_envs * n_robots * args.steps / secs,
                ms_per_step=1e3 * secs / args.steps, startup_s=t_start, first_reset_s=t_reset,
                gpu_mem_mib=None if mem0 is None or mem1 is None else mem1 - mem0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-envs", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--robots", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--level", default="L2")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--workers", type=int, default=None, help="worker processes (default: one per available CPU)")
    group.add_argument("--envs-per-worker", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-single", action="store_true", help="skip the one-env-in-process baseline")
    ap.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    args = ap.parse_args()

    info = machine()
    print(f"{info['host']}: {info['cpus']} CPUs ({info['cpus_available']} available), GPU {', '.join(info['gpu']) or 'unknown'}, MuJoCo "
          f"{info['mujoco']}, MUJOCO_GL={info['mujoco_gl']}")
    print(f"level {args.level}, {args.img_size}x{args.img_size}, shadows {not args.no_shadows}, render "
          f"{not args.no_render}, random actions, {args.steps} timed steps after {WARMUP} warm-up steps\n")
    rows = [] if args.no_single else [bench_single(args, n) for n in args.robots]
    rows += [bench_vector(args, e, n) for n in args.robots for e in args.n_envs]

    print("| envs | robots | workers | agent frames/s | env steps/s | ms per step | resets | startup | GPU memory |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        envs = "1 (in-process)" if r["kind"] == "single" else r["n_envs"]
        startup = f"{r['startup_s']:.1f} s" if r["kind"] == "vector" else "-"
        mem = f"{r['gpu_mem_mib']} MiB" if r.get("gpu_mem_mib") is not None else "-"
        print(f"| {envs} | {r['robots']} | {r['workers'] or '-'} | {r['frames_per_s']:,.0f} | "
              f"{r['frames_per_s'] / r['robots']:,.0f} | {r['ms_per_step']:.2f} | {r['resets']} | {startup} | {mem} |")
    best = max((r for r in rows if r["kind"] == "vector"), key=lambda r: r["frames_per_s"])
    print(f"\nbest: {best['frames_per_s']:,.0f} agent frames/s ({best['n_envs']} envs x {best['robots']} robots, "
          f"{best['workers']} workers)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(machine=info, args={k: str(v) for k, v in vars(args).items()},
                                        warmup=WARMUP, rows=rows), indent=1))
    print(f"wrote {args.out}")
