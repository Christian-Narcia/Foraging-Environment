"""Multi-robot visual foraging in MuJoCo.

    from foraging_env import ForagingEnv
    with ForagingEnv({"level": "L2", "n_robots": 2}, seed=0) as env:
        obs, info = env.reset(layout_seed=10_000)
        obs, reward, done, info = env.step([1, 3])   # robot 0 forward, robot 1 turn left

Modules: constants (fixed world spec, action ids), scenarios (Layout, the layout sampler,
curriculum levels, seed split, probes), mjcf_builder (Layout -> MJCF), rendering (offscreen
cameras), env (ForagingEnv and its cfg), scripted_agents (heuristic forager and scripted
teammates), vector_env (many envs in worker processes), metrics (per-episode task metrics).
"""
import os
import sys

# Headless EGL rendering by default on Linux. This must run before mujoco is imported;
# MuJoCo rejects MUJOCO_GL=egl on other platforms, which keep its default (GLFW).
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"

from foraging_env.constants import (ACTION_NAMES, BACKWARD, DEPOSIT, FORWARD, N_ACTIONS, NOOP,  # noqa: E402
                                    PICK, TURN_LEFT, TURN_RIGHT)
from foraging_env.env import (DEFAULT_CFG, LABEL_DTYPES, ForagingEnv, priv_state_dim,  # noqa: E402
                              priv_state_ego_index, resolve_cfg)
from foraging_env.metrics import EpisodeStats, summarize_episodes  # noqa: E402
from foraging_env.scenarios import (HELDOUT_SEEDS, LEVELS, S2_CONDITIONS, TRAIN_SEEDS, Layout,  # noqa: E402
                                    Occluder, Probe, s1_persistence, s2_ambiguity, s3_full, sample_layout)
from foraging_env.scripted_agents import TEAMMATE_BEHAVIOURS, HeuristicForager, ScriptedTeammate  # noqa: E402
from foraging_env.vector_env import VectorForagingEnv  # noqa: E402

__all__ = ["ForagingEnv", "VectorForagingEnv", "DEFAULT_CFG", "resolve_cfg", "LABEL_DTYPES", "priv_state_dim",
           "priv_state_ego_index", "LEVELS", "TRAIN_SEEDS", "HELDOUT_SEEDS", "Layout", "Occluder", "sample_layout",
           "Probe", "s1_persistence", "s2_ambiguity", "s3_full", "S2_CONDITIONS", "HeuristicForager",
           "ScriptedTeammate", "TEAMMATE_BEHAVIOURS", "EpisodeStats", "summarize_episodes", "N_ACTIONS",
           "ACTION_NAMES", "NOOP", "FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "PICK", "DEPOSIT",
           "__version__"]
