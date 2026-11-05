"""Video of the heuristic forager on held-out layouts: the top-down debug camera next to each
robot's egocentric 64x64 view, one frame per env step (10 Hz, real time).

Usage, from the repo root:
    python scripts/render_video.py [out] [--level L2] [--robots 2] [--seconds 30] [--seed 10000] [--noise 0.1]

`out` defaults to runs/heuristic_L2.mp4. An .mp4 needs imageio with imageio-ffmpeg; a .gif needs
only Pillow. Layout seed `seed` with agent seed `seed` is episode `seed - 10000` of
scripts/eval_baselines.py. If an episode ends before the video does, the next held-out layout follows.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from foraging_env import HELDOUT_SEEDS, ForagingEnv, HeuristicForager   # foraging_env first: it picks the GL backend

from preview_env import FONT, Viz, ego_panel, top_panel   # noqa: E402  (same folder)

FPS = 10          # one frame per decision step
TOP, EGO = 512, 256
BAND = 48         # two text lines above and below; 2 robots give 768 x 608, a multiple of 16 for H.264
LEGEND = ("top: heading, lines to cubes in view, planned path, squares = remembered cubes",
          "right: each robot's 64x64 camera image; rings = cubes labelled visible")


def frames(level: str, n_robots: int, n_frames: int, seed: int, noise: float, log: list):
    """Yield n_frames composed frames (PIL images), running episodes on seeds seed, seed + 1, ..."""
    env = ForagingEnv({"level": level, "n_robots": n_robots}, seed=0)
    cols = 1 if n_robots <= 2 else 2
    produced, layout_seed = 0, seed
    while produced < n_frames:
        if layout_seed not in HELDOUT_SEEDS:
            raise ValueError(f"layout seed {layout_seed} is not held out")
        obs, info = env.reset(layout_seed)
        agents = [HeuristicForager(i, layout_seed, noise) for i in range(n_robots)]
        for agent in agents:
            agent.reset(env)
        viz = Viz(env.layout, TOP, TOP)
        n_res, done = int(env.exists.sum()), False
        while True:
            viz.sync(env)
            canvas = Image.new("RGB", (TOP + cols * EGO, BAND + TOP + BAND), (18, 18, 18))
            canvas.paste(top_panel(viz, env, info, agents), (0, BAND))
            for i in range(n_robots):
                canvas.paste(ego_panel(env, obs, info, i, EGO), (TOP + (i // 2) * EGO, BAND + (i % 2) * EGO))
            dr = ImageDraw.Draw(canvas)
            dr.text((8, 5), f"heuristic forager (scripted, privileged detections) | {level}, held-out layout "
                            f"{layout_seed} | {n_robots} robots", fill=(235, 235, 235), font=FONT)
            dr.text((8, 26), f"{env.t / FPS:4.1f} s (step {env.t}/{env.cfg['max_steps']}) | delivered "
                             f"{int(env.deposited.sum())}/{n_res}", fill=(235, 235, 235), font=FONT)
            for k, line in enumerate(LEGEND):
                dr.text((8, BAND + TOP + 5 + 21 * k), line, fill=(170, 170, 170), font=FONT)
            yield canvas
            produced += 1
            if done or produced == n_frames:   # the log then matches the last frame
                break
            obs, _, done, info = env.step([agent.act(env, info) for agent in agents])
        log.append((layout_seed, int(env.deposited.sum()), n_res, env.t, done))
        viz.close()
        layout_seed += 1
    env.close()


def write(path: Path, images, fps: int) -> None:
    if path.suffix == ".gif":
        images = list(images)
        images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
        return
    import imageio.v2 as imageio   # needs imageio-ffmpeg for .mp4
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for img in images:
            writer.append_data(np.asarray(img))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", type=Path, default=Path("runs/heuristic_L2.mp4"))
    ap.add_argument("--level", default="L2")
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=HELDOUT_SEEDS[0], help="first held-out layout seed")
    ap.add_argument("--noise", type=float, default=0.1)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log: list = []
    n = int(round(args.seconds * FPS))
    write(args.out, frames(args.level, args.robots, n, args.seed, args.noise, log), FPS)
    for layout_seed, d, n_res, t, done in log:
        print(f"layout {layout_seed}: delivered {d}/{n_res} by step {t}" + (" (episode over)" if done else ""))
    print(f"{args.out}: {n} frames, {n / FPS:.0f} s at {FPS} fps, {args.out.stat().st_size / 1e6:.1f} MB")
