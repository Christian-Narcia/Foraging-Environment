"""VectorForagingEnv beyond test_env.py's test 13: other levels and robot
counts, teammate mixing, copy semantics, errors and cleanup. Each test starts worker processes."""
import gc
import multiprocessing as mp
import os
import sys
import threading
from collections import Counter
from multiprocessing import spawn

import numpy as np
import pytest
from helpers import compare_with_reference, reference_run

from foraging_env import HELDOUT_SEEDS, VectorForagingEnv, vector_env


@pytest.fixture(autouse=True)
def no_leftover_workers():
    yield
    gc.collect()
    assert not mp.active_children()


def test_env_seeds():
    seeds = []
    for s in range(3):
        with VectorForagingEnv(64, {"render": False}, n_workers=1, seed=s) as venv:
            seeds.append(venv.env_seeds)
    with VectorForagingEnv(4, {"render": False}, n_workers=1, seed=0) as venv:
        assert venv.env_seeds == seeds[0][:4]                   # env k's seed does not depend on n_envs
    a, b, c = (set(x) for x in seeds)
    assert len(a) == 64 and not (a & b or a & c or b & c)       # runs with different seeds share no env stream


@pytest.mark.parametrize("name,cfg,n_envs,workers,steps,layout_seeds,frac", [
    ("L2 truncated, held-out", {"level": "L2", "max_steps": 120}, 4, 2, 300, list(HELDOUT_SEEDS[:4]), 0.0),
    ("L2, 4 robots", {"level": "L2", "n_robots": 4, "render": False, "max_steps": 150}, 3, 3, 330,
     [HELDOUT_SEEDS[0], None, HELDOUT_SEEDS[2]], 0.0),
    ("teammate in every episode", {"level": "L2", "max_steps": 150}, 4, 2, 450, None, 1.0),
    ("teammate in half", {"level": "L1", "render": False, "max_steps": 100}, 6, 3, 600, None, 0.5),
])
def test_matches_single_envs(name, cfg, n_envs, workers, steps, layout_seeds, frac):
    with VectorForagingEnv(n_envs, cfg, n_workers=workers, seed=1, teammate_frac=frac) as venv:
        refs = [reference_run(cfg, venv.env_seeds[k], steps, None if layout_seeds is None else layout_seeds[k],
                              teammate_frac=frac, policy_seed=k) for k in range(n_envs)]
        bad, stats = compare_with_reference(venv, refs, layout_seeds)
    assert not bad, bad
    assert stats["episodes"] >= n_envs and stats["events"] > 0


def test_teammate_mixing_statistics():
    n_envs = 8
    with VectorForagingEnv(n_envs, {"level": "L0", "render": False, "max_steps": 15}, n_workers=4, seed=3,
                           teammate_frac=0.3) as venv:
        venv.reset()
        robots = Counter()
        for _ in range(1500):
            obs, reward, done, info = venv.step(np.zeros((n_envs, 2), int))
            for k in np.flatnonzero(done):
                s = np.flatnonzero(info["scripted"][k])     # the new episode's assignment
                assert len(s) <= 1
                robots[int(s[0]) if len(s) else None] += 1
    episodes = sum(robots.values())
    assert episodes == 800
    assert abs(1 - robots[None] / episodes - 0.3) < 0.04
    assert abs(robots[0] - robots[1]) < 0.2 * (robots[0] + robots[1])


def test_copy_semantics():
    with VectorForagingEnv(2, {"level": "L0"}, n_workers=1, copy=False) as venv:
        obs, info = venv.reset()
        before = obs["rgb"].copy()
        obs2, *_ = venv.step(np.full((2, 2), 3))
        assert np.shares_memory(obs["rgb"], obs2["rgb"]) and not np.array_equal(obs["rgb"], before)
    with VectorForagingEnv(2, {"level": "L0", "render": False}, n_workers=2) as venv:
        obs, info = venv.reset()
        obs2, reward, done, info2 = venv.step(np.zeros((2, 2), int))
        assert obs["rgb"] is None
        shared = venv._shared.arrays
        assert not np.shares_memory(obs2["prev_action"], shared["now.prev_action"])
        assert not np.shares_memory(info2["priv_state"], shared["now.priv_state"])
        assert not np.shares_memory(reward, shared["reward"])
        assert not info2["final_info"]["priv_state"].any()          # nothing ended: final rows are zero


def test_usage_errors():
    with VectorForagingEnv(2, {"level": "L0", "render": False}, n_workers=2) as venv:
        venv.reset()
        with pytest.raises(ValueError):
            venv.step(np.zeros((2, 3), int))
        venv.step(np.zeros((2, 2), int))                            # still usable
        with pytest.raises(RuntimeError):
            venv.step_wait()
        venv.step_async(np.zeros((2, 2), int))
        with pytest.raises(RuntimeError):
            venv.step_async(np.zeros((2, 2), int))
        venv.step_wait()
        with pytest.raises(ValueError):
            venv.reset([1, 2, 3])
    for kwargs in (dict(cfg={"nope": 1}), dict(cfg={"n_robots": 1}, teammate_frac=0.3),
                   dict(cfg={}, teammate_frac=0.3, teammate_behaviours=("dancer",)), dict(cfg={}, teammate_frac=1.5)):
        cfg = kwargs.pop("cfg")
        with pytest.raises(ValueError):
            VectorForagingEnv(2, cfg, **kwargs)


def test_worker_exception_is_raised_and_closes():
    venv = VectorForagingEnv(3, {"level": "L0", "render": False}, n_workers=3)
    with pytest.raises(RuntimeError, match="expected non-negative integer"):
        venv.reset([0, -1, 0])                                      # np.random.default_rng(-1) fails in a worker
    assert not any(p.is_alive() for p in venv._procs)
    with pytest.raises(RuntimeError, match="closed"):
        venv.step(np.zeros((3, 2), int))
    venv.close()                                                    # closing twice is fine


def test_dead_worker_is_reported():
    venv = VectorForagingEnv(2, {"level": "L0", "render": False}, n_workers=2)
    venv.reset()
    venv._procs[1].terminate()
    venv._procs[1].join()
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        venv.step(np.zeros((2, 2), int))
    venv.close()


class _DieWhileStarting:
    """A worker target whose unpickling ends the child process: it dies while starting, before it
    takes over its end of the pipe (as when a worker can't load a DLL)."""

    def __reduce__(self):
        return os._exit, (3,)


def test_worker_that_dies_while_starting_is_reported(monkeypatch):
    """On Windows, such a worker leaves no EOF on the parent's pipe; the parent must still raise,
    not wait forever (it hung a training run whose workers ran out of memory)."""
    monkeypatch.setattr(vector_env, "_worker", _DieWhileStarting())
    out = {}

    def build():
        try:
            VectorForagingEnv(2, {"level": "L0", "render": False}, n_workers=2)
        except Exception as e:
            out["error"] = e

    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "VectorForagingEnv hung on a worker that died while starting"
    assert isinstance(out.get("error"), RuntimeError) and "exited unexpectedly" in str(out["error"])


def test_workers_do_not_import_the_parents_main():
    """While workers start, the parent's __main__ is hidden, so spawn/forkserver children skip
    importing it (a training script would load torch into every worker)."""
    main = sys.modules["__main__"]
    assert {"init_main_from_path", "init_main_from_name"} & spawn.get_preparation_data("w").keys()
    with vector_env._main_hidden():
        assert not {"init_main_from_path", "init_main_from_name"} & spawn.get_preparation_data("w").keys()
    assert sys.modules["__main__"] is main


def test_dropped_without_close_stops_its_workers():
    venv = VectorForagingEnv(2, {"level": "L0", "render": False}, n_workers=2)
    venv.reset()
    del venv   # the autouse fixture checks that no worker is left
