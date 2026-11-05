"""Watch random agents live in MuJoCo's interactive viewer.

Every robot takes a uniformly random action each step. The loop runs in real time, at the env's
decision rate of 10 steps per second, for STEPS steps or until you close the window.

    python examples/live_viewer.py          # on macOS: mjpython examples/live_viewer.py

Needs a display. In the viewer, drag to orbit, right-drag to pan and scroll to zoom. Press ] or [
to cycle through the cameras (the top-down view, then each robot's camera) and Esc to return to
the free camera.
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")   # an on-screen window (foraging_env would pick headless EGL on Linux)

import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from foraging_env import N_ACTIONS, ForagingEnv  # noqa: E402
from foraging_env.constants import TIMESTEP  # noqa: E402

LEVEL = "L2"      # curriculum level: "L0", "L1" or "L2"
N_ROBOTS = 2      # 1 to 4
STEPS = 400       # 40 s at 10 steps per second

rng = np.random.default_rng(0)
# render=False: the viewer draws the scene itself, so the env doesn't need to render the robot cameras.
with ForagingEnv({"level": LEVEL, "n_robots": N_ROBOTS, "max_steps": STEPS, "render": False}, seed=0) as env:
    env.reset()
    step_time = env.cfg["substeps"] * TIMESTEP              # seconds per env step (0.1)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        with viewer.lock():
            viewer.cam.distance = 1.4 * env.layout.arena_size  # the whole arena in view
            viewer.cam.elevation = -60
        for _ in range(STEPS):
            start = time.perf_counter()
            actions = rng.integers(N_ACTIONS, size=env.n)      # one random action per robot
            obs, reward, done, info = env.step(actions)
            viewer.sync()                                      # show the new state
            if done or not viewer.is_running():
                break
            time.sleep(max(0.0, step_time - (time.perf_counter() - start)))
    print(f"{env.t} steps, delivered {int(env.deposited.sum())}/{int(env.exists.sum())} cubes")
