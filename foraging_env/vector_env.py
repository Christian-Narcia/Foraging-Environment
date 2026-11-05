"""VectorForagingEnv: n_envs ForagingEnvs in worker processes, stepped as one batch.

    from foraging_env import VectorForagingEnv
    if __name__ == "__main__":          # the usual guard for code that starts processes
        with VectorForagingEnv(16, {"level": "L0"}, seed=0) as venv:
            obs, info = venv.reset()
            obs, reward, done, info = venv.step(actions)   # actions int[16, N]

Each worker process owns one or more envs and steps them in turn. Observations, rewards, labels
and privileged state come back through one shared-memory block; only the event lists go
through the pipe. Env k is ForagingEnv(cfg, venv.env_seeds[k]), so it can be reproduced in a
single process.

Auto-reset: an env whose episode ends is reset within the same step (Gymnasium's same-step
autoreset). The step returns the new episode's first observation; what the step reached
before the reset is in info["final_obs"] and info["final_info"].

Teammate mixing: in a fraction of episodes, one robot is driven by a ScriptedTeammate
inside the worker, since scripted agents read env state that never leaves the worker.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import sys
import traceback
import types
from typing import Any, Sequence

import numpy as np

from foraging_env.constants import R_MAX
from foraging_env.env import LABEL_DTYPES, ForagingEnv, priv_state_dim, resolve_cfg
from foraging_env.scripted_agents import TEAMMATE_BEHAVIOURS, ScriptedTeammate

_ALIGN = 64   # bytes; every shared array starts on a cache line


def available_cpus() -> int:
    """CPUs this process may run on: its affinity mask on Linux (a Slurm allocation, taskset),
    else os.cpu_count(), which counts every CPU of the machine."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@contextlib.contextmanager
def _main_hidden():
    """Hide the parent's __main__ while workers start. spawn and forkserver children import it
    before running _worker, which needs nothing from it; a training script imports torch, which
    commits ~1.5 GB per worker on Windows (64 workers ran the machine out of commit)."""
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        sys.modules["__main__"] = main


def _state_spec(n: int, img_size: int | None, priv_dim: int) -> dict[str, tuple[tuple, type]]:
    """Per-env shape and dtype of an observation plus the state part of an info."""
    spec = {} if img_size is None else {"rgb": ((n, img_size, img_size, 3), np.uint8)}
    spec.update(prev_action=((n,), np.int64), carrying=((n,), np.bool_), priv_state=((priv_dim,), np.float32),
                t=((), np.int64), layout_seed=((), np.int64))
    spec.update({f"labels.{f}": ((n, R_MAX), dt) for f, dt in LABEL_DTYPES.items()})
    return spec


def _buffer_spec(n: int, img_size: int | None, priv_dim: int) -> dict[str, tuple[tuple, type]]:
    """Every shared array, per env: the step's input and outputs, the returned state ("now.")
    and, for envs whose episode ended, the state before the reset ("final.")."""
    state = _state_spec(n, img_size, priv_dim)
    return {"actions": ((n,), np.int64), "reward": ((n,), np.float32), "done": ((), np.bool_),
            "truncated": ((), np.bool_), "scripted": ((n,), np.bool_), "delivered": ((), np.float32),
            **{f"now.{k}": v for k, v in state.items()}, **{f"final.{k}": v for k, v in state.items()}}


class _Shared:
    """Named arrays [n_envs, ...] in one block of shared memory. Created in the parent (raw=None)
    and rebuilt in each worker from the same spec and the inherited raw block."""

    def __init__(self, spec: dict, n_envs: int, ctx=None, raw=None):
        offsets, total = {}, 0
        for name, (shape, dtype) in spec.items():
            offsets[name] = total
            total += -(-n_envs * int(np.prod(shape)) * np.dtype(dtype).itemsize // _ALIGN) * _ALIGN
        self.raw = ctx.RawArray("B", max(total, 1)) if raw is None else raw
        buf = np.frombuffer(self.raw, np.uint8)
        self.arrays = {}
        for name, (shape, dtype) in spec.items():
            nbytes = n_envs * int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.arrays[name] = buf[offsets[name]:offsets[name] + nbytes].view(dtype).reshape(n_envs, *shape)


def _write_state(a: dict, prefix: str, k: int, obs: dict, info: dict) -> None:
    if obs["rgb"] is not None:
        a[prefix + "rgb"][k] = obs["rgb"]
    a[prefix + "prev_action"][k] = obs["prev_action"]
    a[prefix + "carrying"][k] = obs["carrying"]
    a[prefix + "priv_state"][k] = info["priv_state"]
    a[prefix + "t"][k] = info["t"]
    a[prefix + "layout_seed"][k] = info["layout_seed"]
    for i, lab in enumerate(info["labels"]):
        for f in LABEL_DTYPES:
            a[f"{prefix}labels.{f}"][k, i] = lab[f]


class _Slot:
    """One env inside a worker, with its teammate mixing and its row k of the shared arrays."""

    def __init__(self, k: int, env: ForagingEnv, seed: int, arrays: dict, teammate_frac: float,
                 teammate_behaviours: tuple[str, ...]):
        self.k, self.env, self.a = k, env, arrays
        self.frac, self.behaviours = teammate_frac, teammate_behaviours
        # Spawn key 0 is the env's own layout-seed stream (ForagingEnv.__init__).
        self.mix_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        self.teammate: ScriptedTeammate | None = None
        self.info = None

    def reset(self, layout_seed: int | None) -> None:
        self._start(*self.env.reset(layout_seed))

    def _start(self, obs: dict, info: dict) -> None:
        """Begin an episode: maybe hand one robot to a scripted teammate, publish the state."""
        a, k = self.a, self.k
        self.teammate = None
        a["scripted"][k] = False
        if self.frac > 0 and self.mix_rng.random() < self.frac:
            robot = int(self.mix_rng.integers(self.env.n))
            behaviour = self.behaviours[int(self.mix_rng.integers(len(self.behaviours)))]
            self.teammate = ScriptedTeammate(robot, behaviour, seed=int(self.mix_rng.integers(2**31)))
            self.teammate.reset(self.env)
            a["scripted"][k, robot] = True
        self.info = info
        _write_state(a, "now.", k, obs, info)

    def step(self) -> list[tuple]:
        a, k, env = self.a, self.k, self.env
        actions = a["actions"][k].copy()
        if self.teammate is not None:
            actions[self.teammate.robot] = self.teammate.act(env, self.info)
        obs, reward, done, info = env.step(actions)
        a["reward"][k], a["done"][k], a["truncated"][k] = reward, done, info["truncated"]
        if not done:
            self.info = info
            _write_state(a, "now.", k, obs, info)
            return info["events"]
        _write_state(a, "final.", k, obs, info)
        a["delivered"][k] = env.deposited.sum() / max(env.exists.sum(), 1)
        self._start(*env.reset())
        return info["events"]


def _worker(conn, env_ids: list[int], seeds: list[int], cfg: dict, raw, spec: dict, n_envs: int,
            teammate_frac: float, teammate_behaviours: tuple[str, ...]) -> None:
    """Commands from the parent: ("reset", [layout seed or None per env]), ("step", None),
    ("close", None). Replies ("ok", payload) or, on an exception, ("error", traceback)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C goes to the parent, which closes the workers
    slots: list[_Slot] = []
    try:
        arrays = _Shared(spec, n_envs, raw=raw).arrays
        slots = [_Slot(k, ForagingEnv(cfg, s), s, arrays, teammate_frac, teammate_behaviours)
                 for k, s in zip(env_ids, seeds)]
        conn.send(("ok", None))
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                conn.send(("ok", [slot.step() for slot in slots]))
            elif cmd == "reset":
                for slot, layout_seed in zip(slots, arg):
                    slot.reset(layout_seed)
                conn.send(("ok", None))
            elif cmd == "close":
                break
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except OSError:   # the parent is gone
            pass
    finally:
        for slot in slots:
            slot.env.close()
        conn.close()


class VectorForagingEnv:
    """n_envs ForagingEnvs with the same cfg, stepped as a batch in n_workers processes.

    n_envs, cfg: number of envs and their cfg (as for ForagingEnv; validated here).
    n_workers: processes, default min(n_envs, available_cpus()); envs are split between them
        evenly. Rendering, not the CPU, limits throughput, so about one worker per physical core
        is enough (on a 16-core machine, 16 workers matched 32).
    seed: env k is ForagingEnv(cfg, env_seeds[k]), where env_seeds is drawn from
        SeedSequence(seed), so vector envs with different seeds share no env streams.
    teammate_frac: probability that an episode hands one robot, chosen uniformly, to a
        ScriptedTeammate with a behaviour drawn uniformly from teammate_behaviours (a collector
        targets a random available cube). Its action inputs are ignored; see info["scripted"].
    start_method: multiprocessing start method; default "forkserver" on Linux, "spawn"
        elsewhere. Workers create their GL contexts on their first reset. With "fork", the
        parent must not have rendered (EGL state does not survive fork).
    copy: return copies (default). With copy=False, the returned arrays are views of shared
        memory that the next step or reset overwrites.

    reset(layout_seeds=None) -> (obs, info); layout_seeds is None or one seed (or None) per
        env; None draws from TRAIN_SEEDS, as ForagingEnv.reset() does.
    step(actions int[n_envs, N]) -> (obs, reward float32[n_envs, N], done bool[n_envs], info),
        or step_async(actions) then step_wait().

    obs: the ForagingEnv obs with a leading n_envs axis: rgb uint8[n_envs, N, H, W, 3] (None if
        cfg["render"] is False), prev_action int64[n_envs, N], carrying bool[n_envs, N].
    info:
        priv_state   float32[n_envs, D]
        labels       dict of field -> [n_envs, N, R_MAX] (field by field, unlike ForagingEnv's
                     list of per-robot dicts)
        t, layout_seed   int64[n_envs]
        scripted     bool[n_envs, N]: robot driven by a ScriptedTeammate this episode; its next
                     action input is ignored and its observations are not the learner's
        events       list of n_envs lists of (kind, robot, resource, t) from this step
        truncated    bool[n_envs]: this step hit the time limit (done is True too)
        final_obs, final_info (step only): rows where done hold the observation and the
            priv_state, labels, t (the episode length) and layout_seed the step reached
            before the reset, and final_info["delivered"], float32[n_envs], the fraction of the
            episode's cubes deposited. Other rows are zeros (copy=True) or stale (copy=False).
    reward, done, truncated and events describe the step; for an env that was reset, the rest
    of info and obs describe the new episode's first state. A centralized critic bootstraps
    from final_info["priv_state"] where truncated, and not at all where done and not truncated.
    """

    def __init__(self, n_envs: int, cfg: dict | None, n_workers: int | None = None, seed: int = 0, *,
                 teammate_frac: float = 0.0, teammate_behaviours: Sequence[str] = ("collector", "wanderer"),
                 start_method: str | None = None, copy: bool = True):
        self._closed, self._waiting = True, False
        self.cfg = resolve_cfg(cfg)
        self.n_envs, self.n = int(n_envs), int(self.cfg["n_robots"])
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        teammate_behaviours = tuple(teammate_behaviours)
        if not 0 <= teammate_frac <= 1:
            raise ValueError(f"teammate_frac must be in [0, 1], got {teammate_frac}")
        if teammate_frac > 0 and self.n < 2:
            raise ValueError("teammate mixing needs n_robots >= 2")
        if teammate_frac > 0 and (not teammate_behaviours
                                  or any(b not in TEAMMATE_BEHAVIOURS for b in teammate_behaviours)):
            raise ValueError(f"teammate_behaviours must be from {TEAMMATE_BEHAVIOURS}, got {teammate_behaviours}")
        self.priv_dim = priv_state_dim(self.n)
        self.env_seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(self.n_envs)]
        self._render, self._copy = bool(self.cfg["render"]), copy

        n_workers = min(int(n_workers or available_cpus()), self.n_envs)
        ctx = mp.get_context(start_method or ("forkserver" if sys.platform.startswith("linux") else "spawn"))
        spec = _buffer_spec(self.n, int(self.cfg["img_size"]) if self._render else None, self.priv_dim)
        self._shared = _Shared(spec, self.n_envs, ctx=ctx)
        self._a = self._shared.arrays
        self._env_ids = [[int(k) for k in ids] for ids in np.array_split(np.arange(self.n_envs), n_workers)]
        self._conns, self._procs = [], []
        self._closed = False
        with _main_hidden():
            for w, ids in enumerate(self._env_ids):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, name=f"VectorForagingEnv-{w}", daemon=True,
                                   args=(child, ids, [self.env_seeds[k] for k in ids], dict(cfg or {}),
                                         self._shared.raw, spec, self.n_envs, float(teammate_frac),
                                         teammate_behaviours))
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)
        for w in range(n_workers):
            self._recv(w)

    @property
    def n_workers(self) -> int:
        return len(self._procs)

    def reset(self, layout_seeds: Sequence[int | None] | None = None) -> tuple[dict, dict]:
        """Start a new episode in every env (see the class docstring)."""
        self._check_idle()
        seeds = [None] * self.n_envs if layout_seeds is None else [None if s is None else int(s) for s in layout_seeds]
        if len(seeds) != self.n_envs:
            raise ValueError(f"need {self.n_envs} layout seeds, got {len(seeds)}")
        for w, ids in enumerate(self._env_ids):
            self._send(w, ("reset", [seeds[k] for k in ids]))
        for w in range(self.n_workers):
            self._recv(w)
        obs, info = self._state("now.")
        info.update(scripted=self._get("scripted"), events=[[] for _ in range(self.n_envs)],
                    truncated=np.zeros(self.n_envs, bool))
        return obs, info

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        """Send int[n_envs, N] actions to the workers and return without waiting."""
        self._check_idle()
        a = np.asarray(actions)
        if a.shape != (self.n_envs, self.n):
            raise ValueError(f"actions must have shape {(self.n_envs, self.n)}, got {a.shape}")
        self._a["actions"][:] = a
        for w in range(self.n_workers):
            self._send(w, ("step", None))
        self._waiting = True

    def step_wait(self) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """Wait for the step sent by step_async and return its results."""
        if not self._waiting:
            raise RuntimeError("step_wait() called without step_async()")
        events: list[list[tuple]] = [[] for _ in range(self.n_envs)]
        for w, ids in enumerate(self._env_ids):
            for k, ev in zip(ids, self._recv(w)):
                events[k] = ev
        self._waiting = False
        done = self._a["done"].copy()
        obs, info = self._state("now.")
        final_obs, final_info = self._state("final.", rows=done)
        final_info["delivered"] = self._get("delivered", rows=done)
        info.update(scripted=self._get("scripted"), events=events, truncated=self._get("truncated"),
                    final_obs=final_obs, final_info=final_info)
        return obs, self._get("reward"), done, info

    def close(self) -> None:
        """Stop the workers (each closes its envs). Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        for conn in self._conns:
            conn.close()

    def __enter__(self) -> VectorForagingEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- internals

    def _check_idle(self) -> None:
        if self._closed:
            raise RuntimeError("the vector env is closed")
        if self._waiting:
            raise RuntimeError("a step is pending: call step_wait() first")

    def _died(self, w: int) -> RuntimeError:
        self.close()
        return RuntimeError(f"worker {w} exited unexpectedly (exit code {self._procs[w].exitcode})")

    def _send(self, w: int, msg: tuple) -> None:
        try:
            self._conns[w].send(msg)
        except OSError:   # broken pipe: the worker is gone
            raise self._died(w) from None

    def _recv(self, w: int) -> Any:
        conn, proc = self._conns[w], self._procs[w]
        try:
            # A worker that dies while starting, before it takes over its end of the pipe, leaves
            # that end open in the parent on Windows, so recv() would wait forever: check liveness.
            while not conn.poll(1.0):
                if not proc.is_alive():
                    raise self._died(w)
            tag, payload = conn.recv()
        except (EOFError, OSError):
            raise self._died(w) from None
        if tag == "error":
            self.close()
            raise RuntimeError(f"worker {w} failed:\n{payload}")
        return payload

    def _get(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        """A shared array: a copy (zeros outside `rows` if given), or the live view if copy=False."""
        x = self._a[name]
        if not self._copy:
            return x
        if rows is None:
            return x.copy()
        out = np.zeros_like(x)
        out[rows] = x[rows]
        return out

    def _state(self, prefix: str, rows: np.ndarray | None = None) -> tuple[dict, dict]:
        get = lambda name: self._get(prefix + name, rows)
        obs = dict(rgb=get("rgb") if self._render else None, prev_action=get("prev_action"),
                   carrying=get("carrying"))
        info = dict(priv_state=get("priv_state"), labels={f: get(f"labels.{f}") for f in LABEL_DTYPES},
                    t=get("t"), layout_seed=get("layout_seed"))
        return obs, info
