"""Random agents in the multi-robot foraging environment: a minimal end-to-end example.

Every robot picks one of the 7 discrete actions uniformly at random at every step. The script
prints the action, observation and info structure once, then one line per episode. Optionally it
writes a GIF of the first episode (top-down view next to every robot's camera image), or runs
many envs in worker processes with VectorForagingEnv.

    python examples/random_agents.py                                       # 3 episodes, level L0, 2 robots
    python examples/random_agents.py --level L2 --robots 4 --episodes 5
    python examples/random_agents.py --level L2 --gif runs/random_L2.gif   # needs Pillow
    python examples/random_agents.py --n-envs 8                            # VectorForagingEnv, 8 envs
    python examples/random_agents.py --no-render                           # no OpenGL; obs["rgb"] is None

Headless Linux: rendering uses EGL by default (NVIDIA driver, or Mesa's libEGL). Without a GPU,
run with MUJOCO_GL=osmesa (apt install libosmesa6), or pass --no-render.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Import the package before mujoco: on Linux it selects the headless EGL backend.
from foraging_env import ACTION_NAMES, HELDOUT_SEEDS, LEVELS, N_ACTIONS, ForagingEnv, VectorForagingEnv
from foraging_env.constants import TIMESTEP

import mujoco  # noqa: E402  (after foraging_env, see above)

GL_HINT = """
If the error above mentions EGL, GLX, OSMesa or OpenGL, rendering found no OpenGL backend
(MUJOCO_GL={backend}). On a headless Linux machine with an NVIDIA GPU, install the driver and
libegl1; without a GPU, install libosmesa6 and set MUJOCO_GL=osmesa; or run with --no-render."""


def shape(x) -> str:
    return "None" if x is None else f"{x.dtype}{list(np.shape(x))}"


def describe(env: ForagingEnv, obs: dict, info: dict) -> None:
    """Print the spaces once: what a policy sees, what it controls, what only training code reads."""
    cfg, lay = env.cfg, env.layout
    print(f"level {cfg['level']}: {cfg['arena_size']:g} x {cfg['arena_size']:g} m arena, {env.n} robots, "
          f"{int(env.exists.sum())} cubes, {len(lay.distractor_pos)} distractor spheres, {len(lay.occluders)} "
          f"occluders; at most {cfg['max_steps']} steps at {1 / (cfg['substeps'] * TIMESTEP):g} Hz")
    print(f"action space, per robot: Discrete({N_ACTIONS}) = "
          + ", ".join(f"{i} {name}" for i, name in enumerate(ACTION_NAMES)))
    print("observation (policy input), one row per robot:")
    for key, value in obs.items():
        print(f"  obs[{key!r}]".ljust(34) + shape(value))
    print("info (privileged: critic input, training targets, metrics; not policy input):")
    print("  info['priv_state']".ljust(34) + shape(info["priv_state"]))
    for key, value in info["labels"][0].items():
        print(f"  info['labels'][i][{key!r}]".ljust(34) + shape(value) + "   (one entry per cube slot)")
    print("  info['events']".ljust(34) + "list of (kind, robot, cube, t), kind 'pick' or 'deposit'")
    print()


class GifRecorder:
    """Frames of one episode: the top-down camera next to each robot's camera image (upscaled 2x)."""

    def __init__(self, env: ForagingEnv, path: Path, stride: int = 2):
        from PIL import Image, ImageDraw   # optional dependency, only needed for --gif
        self._image, self._draw = Image, ImageDraw
        self.env, self.path, self.stride = env, path, stride
        self.top = mujoco.Renderer(env.model, 256, 256)   # bound to this episode's compiled model
        self.frames = []

    def add(self, obs: dict, force: bool = False) -> None:
        env = self.env
        if env.t % self.stride and not force:
            return
        self.top.update_scene(env.data, camera="top")
        cols = -(-env.n // 2)
        panel = np.zeros((256, 128 * cols, 3), np.uint8)
        for i, img in enumerate(obs["rgb"].repeat(2, axis=1).repeat(2, axis=2)):   # [N, 128, 128, 3]
            panel[(i % 2) * 128:(i % 2 + 1) * 128, (i // 2) * 128:(i // 2 + 1) * 128] = img
        frame = self._image.fromarray(np.concatenate([self.top.render(), panel], axis=1))
        self._draw.Draw(frame).text((4, 2), f"t={env.t}  delivered {int(env.deposited.sum())}/{int(env.exists.sum())}",
                                    fill=(255, 255, 255))
        self.frames.append(frame)

    def save(self) -> None:
        self.top.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=100 * self.stride, loop=0)
        print(f"wrote {self.path} ({len(self.frames)} frames)")


def run_single(args, cfg: dict) -> list[dict]:
    """Episodes of one ForagingEnv on the first held-out layouts."""
    rng = np.random.default_rng(args.seed)                    # the random policy
    results = []
    with ForagingEnv(cfg, seed=args.seed) as env:
        for k in range(args.episodes):
            layout_seed = HELDOUT_SEEDS[k]                    # fixed layouts: every episode is reproducible
            obs, info = env.reset(layout_seed)
            if k == 0:
                describe(env, obs, info)
            gif = GifRecorder(env, args.gif) if args.gif and k == 0 else None
            team_return, picks, done, t0 = 0.0, 0, False, time.perf_counter()
            while not done:
                if gif:
                    gif.add(obs)
                actions = rng.integers(N_ACTIONS, size=env.n)          # int[N]: one action per robot
                obs, reward, done, info = env.step(actions)             # reward: float32[N]
                team_return += float(reward.mean())
                picks += sum(kind == "pick" for kind, *_ in info["events"])
            if gif:
                gif.add(obs, force=True)
                gif.save()
            n_cubes, n_delivered = int(env.exists.sum()), int(env.deposited.sum())
            results.append(dict(layout_seed=layout_seed, steps=env.t, truncated=bool(info["truncated"]),
                                delivered=n_delivered / n_cubes, picks=picks, team_return=team_return))
            end = "time limit" if info["truncated"] else "all cubes delivered"
            print(f"episode {k} | layout {layout_seed} | {env.t} steps ({end}) | delivered {n_delivered}/{n_cubes} "
                  f"| picks {picks} | team return {team_return:+.3f} | {env.t / (time.perf_counter() - t0):,.0f} steps/s")
    return results


def run_vector(args, cfg: dict) -> list[dict]:
    """VectorForagingEnv: envs in worker processes, stepped as one batch, reset automatically."""
    rng = np.random.default_rng(args.seed)
    results = []
    with VectorForagingEnv(args.n_envs, cfg, seed=args.seed) as venv:
        obs, info = venv.reset(list(HELDOUT_SEEDS[:venv.n_envs]))     # first episodes on held-out layouts
        print(f"{venv.n_envs} envs in {venv.n_workers} worker processes; every array gains a leading env axis:")
        for key, value in obs.items():
            print(f"  obs[{key!r}]".ljust(34) + shape(value))
        print("  info['priv_state']".ljust(34) + shape(info["priv_state"]))
        print("  info['labels'][field]".ljust(34) + shape(info["labels"]["visible"]) + "   (stacked field by field)")
        print()
        steps, t0 = 0, time.perf_counter()
        while len(results) < args.episodes:
            actions = rng.integers(N_ACTIONS, size=(venv.n_envs, venv.n))
            obs, reward, done, info = venv.step(actions)    # envs that finish start a new episode in this step
            steps += 1
            final = info["final_info"]                      # the finished episodes' last state, in rows where done
            for k in np.flatnonzero(done):
                results.append(dict(env=int(k), layout_seed=int(final["layout_seed"][k]), steps=int(final["t"][k]),
                                    truncated=bool(info["truncated"][k]), delivered=float(final["delivered"][k])))
                print(f"env {k} | layout {results[-1]['layout_seed']} | {results[-1]['steps']} steps | "
                      f"delivered {results[-1]['delivered']:.0%}")
        secs = time.perf_counter() - t0
        print(f"{steps} vector steps: {steps * venv.n_envs * venv.n / secs:,.0f} agent frames/s "
              f"({venv.n_envs} envs x {venv.n} robots)")
    return results


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="L0", choices=sorted(LEVELS), help="curriculum level (default: L0)")
    ap.add_argument("--robots", type=int, default=2, help="team size, 1-4 (default: 2)")
    ap.add_argument("--episodes", type=int, default=3, help="episodes to run (default: 3)")
    ap.add_argument("--seed", type=int, default=0, help="env seed and random-policy seed (default: 0)")
    ap.add_argument("--max-steps", type=int, default=None, help="episode time limit (default: 400)")
    ap.add_argument("--no-render", action="store_true", help="skip the cameras: no OpenGL needed, obs['rgb'] is None")
    ap.add_argument("--gif", type=Path, default=None, help="write the first episode as a GIF (needs Pillow)")
    ap.add_argument("--n-envs", type=int, default=0, help="run VectorForagingEnv with this many envs instead")
    args = ap.parse_args(argv)
    if args.gif and (args.no_render or args.n_envs):
        ap.error("--gif needs rendering and a single env")
    if not 1 <= args.episodes <= len(HELDOUT_SEEDS):
        ap.error(f"--episodes must be in 1..{len(HELDOUT_SEEDS)}")

    cfg = {"level": args.level, "n_robots": args.robots, "render": not args.no_render}
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    try:
        results = run_vector(args, cfg) if args.n_envs else run_single(args, cfg)
    except Exception:
        if not args.no_render:
            print(GL_HINT.format(backend=os.environ.get("MUJOCO_GL", "default")), file=sys.stderr)
        raise
    print(f"\nrandom agents delivered {np.mean([r['delivered'] for r in results]):.1%} of the cubes "
          f"on average over {len(results)} episodes")
    return results


if __name__ == "__main__":   # the guard is required: VectorForagingEnv starts worker processes
    main()
