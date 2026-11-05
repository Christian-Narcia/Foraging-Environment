"""Preview renders of the foraging env: stills of the arena and the robot, GIFs of episodes.

For eyeballing the env only. The README's episodes are driven by a simple privileged-state
heuristic (`Forager` below, not HeuristicForager) through the env's read-only state accessors;
the `heuristic` group runs the scripted agents of foraging_env/scripted_agents.py instead
(HeuristicForager and scripted teammates). Real-time videos come from scripts/render_video.py.
Needs Pillow.

Usage, from the repo root (omit the groups to render all README figures):
    python scripts/preview_env.py docs/media [stills] [episodes] [clusters] [probes]
    python scripts/preview_env.py runs/preview heuristic      # HeuristicForager and scripted teammates
    python scripts/preview_env.py runs/labels_check labels    # 20 frames to check the visible labels by eye
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foraging_env import mjcf_builder as mb   # imported before mujoco so foraging_env picks the GL backend
from foraging_env import scenarios as sc
from foraging_env.constants import CUBE_HALF, NEST_RADIUS, PALETTE, POLE_RADIUS, ROBOT_RADIUS, SPHERE_RADIUS, WALL_RGB
from foraging_env.env import (BACKWARD, DEPOSIT, FORWARD, NOOP, PICK, TURN_LEFT, TURN_RIGHT,
                              ForagingEnv, resolve_cfg)
from foraging_env.scripted_agents import HeuristicForager, ScriptedTeammate

import mujoco  # noqa: E402


def _font(name, size):
    try:
        return ImageFont.truetype(name, size)   # searches the system font directories
    except OSError:
        return ImageFont.load_default(size)


FONT = _font("DejaVuSans.ttf", 14)
FONT_B = _font("DejaVuSans-Bold.ttf", 15)
MARK = [(255, 140, 0), (0, 200, 255), (255, 60, 200), (120, 230, 60)]   # per-robot overlay colours


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


class Viz:
    """High-resolution render model with the same structure as the env's; state is copied over."""

    def __init__(self, layout, h, w):
        self.m = mujoco.MjModel.from_xml_string(mb.build_mjcf(layout, shadow_size=4096, offscreen=(2048, 1536)))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.h, self.w = h, w
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = mb.SCENE_GROUPS

    def sync(self, env):
        src = env.data
        self.d.qpos[:] = src.qpos
        self.d.mocap_pos[:] = src.mocap_pos
        self.d.mocap_quat[:] = src.mocap_quat
        mujoco.mj_forward(self.m, self.d)

    def render(self, cam):
        self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
        return self.r.render()

    def project(self, cam_name, pts):
        """World points [K, 3] -> pixel (u, v) for a fixed camera of this renderer."""
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        f = (self.h / 2) / np.tan(np.deg2rad(self.m.cam_fovy[cid]) / 2)
        loc = (np.atleast_2d(pts) - self.d.cam_xpos[cid]) @ self.d.cam_xmat[cid].reshape(3, 3)
        return np.column_stack([self.w / 2 + f * loc[:, 0] / -loc[:, 2], self.h / 2 - f * loc[:, 1] / -loc[:, 2]])

    def close(self):
        self.r.close()


class Forager:
    """Go to the nearest unclaimed cube, pick, drive to the nest, deposit. Detours around the first
    obstacle on the straight line; backs up and turns when it stops making progress."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.target = [-1] * n
        self.queue = [[] for _ in range(n)]
        self.prev_xy = [None] * n
        self.prev_a = [NOOP] * n
        self.stalled = [0] * n

    def act(self, env):
        lay = env.layout
        pose = env.robot_poses()
        res = env.resource_positions()[:, :2]
        avail = env.available()
        half = lay.arena_size / 2
        acts = np.full(self.n, NOOP)
        for i in range(self.n):
            if self.target[i] >= 0 and not avail[self.target[i]]:
                self.target[i] = -1
        for i in range(self.n):
            x, y, yaw = pose[i]
            if self.prev_xy[i] is not None and self.prev_a[i] == FORWARD:
                moved = np.hypot(x - self.prev_xy[i][0], y - self.prev_xy[i][1])
                self.stalled[i] = self.stalled[i] + 1 if moved < 0.015 else 0
            self.prev_xy[i] = (x, y)
            if self.stalled[i] >= 3 and not self.queue[i]:
                turn = TURN_LEFT if self.rng.random() < 0.5 else TURN_RIGHT
                self.queue[i] = [BACKWARD] * 4 + [turn] * int(self.rng.integers(3, 8)) + [FORWARD] * 5
                self.stalled[i] = 0
            if self.queue[i]:
                acts[i] = self.queue[i].pop(0)
                continue
            carrying = env.carrying[i] >= 0
            if carrying:
                tgt, exclude = lay.nest_pos, -1
                if np.hypot(*(tgt - (x, y))) < NEST_RADIUS - 0.3:
                    acts[i] = DEPOSIT
                    continue
            else:
                if self.target[i] < 0:
                    claimed = {self.target[j] for j in range(self.n) if j != i}
                    cand = [r for r in np.flatnonzero(avail) if r not in claimed] or list(np.flatnonzero(avail))
                    if not cand:
                        continue
                    self.target[i] = int(min(cand, key=lambda r: np.hypot(*(res[r] - (x, y)))))
                exclude = self.target[i]
                tgt = res[exclude]
                ang = wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - yaw)
                if np.hypot(*(tgt - (x, y))) <= 0.33 and abs(ang) <= np.deg2rad(25):
                    acts[i] = PICK
                    continue
            obstacles = ([(*o.pos, o.footprint) for o in lay.occluders]
                         + [(*p, SPHERE_RADIUS) for p in lay.distractor_pos]
                         + [(*res[k], 0.11) for k in np.flatnonzero(avail) if k != exclude]
                         + [(*pose[j, :2], ROBOT_RADIUS) for j in range(self.n) if j != i]
                         + ([] if carrying else [(*lay.nest_pos, POLE_RADIUS)]))
            wx, wy = self._detour(np.array([x, y]), np.asarray(tgt), obstacles, half)
            err = wrap(np.arctan2(wy - y, wx - x) - yaw)
            acts[i] = FORWARD if abs(err) < np.deg2rad(11) else (TURN_LEFT if err > 0 else TURN_RIGHT)
        self.prev_a = list(acts)
        return acts

    @staticmethod
    def _detour(p, t, obstacles, half):
        seg = t - p
        length = np.linalg.norm(seg)
        if length < 1e-6:
            return t
        u = seg / length
        best = None
        for cx, cy, r in obstacles:
            c = np.array([cx, cy])
            reach = r + ROBOT_RADIUS + 0.1
            s = float(np.dot(c - p, u))
            if s <= 0 or s >= length:
                continue
            perp = (c - p) - s * u
            dperp = np.linalg.norm(perp)
            if dperp < reach and (best is None or s < best[0]):
                best = (s, c, reach, perp, dperp)
        if best is None:
            return t
        _, c, reach, perp, dperp = best
        side = -perp / dperp if dperp > 1e-6 else np.array([-u[1], u[0]])
        return np.clip(c + side * (reach + 0.15), -half + 0.35, half - 0.35)


def ego_panel(env, obs, info, i, size):
    """Robot i's 64x64 frame upscaled, with rings on resources labelled visible."""
    img = Image.fromarray(obs["rgb"][i]).resize((size, size), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}")
    cp, R = env.data.cam_xpos[cid], env.data.cam_xmat[cid].reshape(3, 3)
    res3 = env.resource_positions()
    s = size / 64
    for r in np.flatnonzero(info["labels"][i]["visible"]):
        loc = R.T @ (res3[r] - cp)
        u, v = 32 + 32 * loc[0] / -loc[2], 32 - 32 * loc[1] / -loc[2]
        rad = max(5, CUBE_HALF / -loc[2] * 32 * s * 1.6)
        dr.ellipse([u * s - rad, v * s - rad, u * s + rad, v * s + rad], outline=(255, 255, 255), width=2)
    dr.rectangle([0, 0, size - 1, size - 1], outline=MARK[i], width=3)
    status = "carrying" if obs["carrying"][i] else ""
    n_vis = int(info["labels"][i]["visible"].sum())
    dr.text((6, 4), f"robot {i} cam", fill=MARK[i], font=FONT_B, stroke_width=2, stroke_fill=(0, 0, 0))
    dr.text((6, size - 20), f"visible {n_vis} {status}", fill=(255, 255, 255), font=FONT,
            stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def top_panel(viz, env, info, agents=()):
    """Top view; lines link each robot to the resources it labels visible. For scripted agents,
    also each one's planned path and, for HeuristicForager, squares where it remembers cubes."""
    img = Image.fromarray(viz.render("top"))
    dr = ImageDraw.Draw(img)
    pose = env.robot_poses()
    res3 = env.resource_positions()
    for agent in agents:
        c = MARK[agent.robot]
        if len(agent.nav.path) > 1:
            pts = viz.project("top", np.column_stack([agent.nav.path, np.full(len(agent.nav.path), 0.02)]))
            dr.line([tuple(p) for p in pts], fill=c, width=1)
        for xy in getattr(agent, "known", {}).values():
            (u, v), = viz.project("top", [(*xy, CUBE_HALF)])
            dr.rectangle([u - 7, v - 7, u + 7, v + 7], outline=c, width=2)
    for i, (x, y, yaw) in enumerate(pose):
        (u, v), (hu, hv) = viz.project("top", [(x, y, 0.45), (x + 0.45 * np.cos(yaw), y + 0.45 * np.sin(yaw), 0.45)])
        for r in np.flatnonzero(info["labels"][i]["visible"]):
            (ru, rv), = viz.project("top", res3[r])
            dr.line([u, v, ru, rv], fill=MARK[i], width=1)
        dr.line([u, v, hu, hv], fill=MARK[i], width=3)
        dr.text((u + 8, v - 22), str(i), fill=MARK[i], font=FONT_B, stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def still_clusters(out):
    """Top views of clustered layouts next to a uniform one."""
    cases = [("L0", 0, 3), ("L0", 1, 3), ("L1", 2, 5), ("L2", 0, 7), ("L2", 2, 7), ("L2", 3, 8)]
    tiles = []
    for level, clusters, seed in cases:
        env = ForagingEnv({"level": level, "resource_clusters": clusters}, 0)
        obs, info = env.reset(layout_seed=seed)
        v = Viz(env.layout, 400, 400)
        v.sync(env)
        tile = Image.fromarray(v.render("top"))
        v.close()
        lay = env.layout
        label = f"{level}, " + ("uniform" if clusters == 0 else f"{clusters} cluster{'s' * (clusters > 1)}")
        ImageDraw.Draw(tile).text((8, 6), f"{label}  ({len(lay.resource_pos)} cubes)", fill=(255, 255, 255),
                                  font=FONT_B, stroke_width=2, stroke_fill=(0, 0, 0))
        tiles.append(tile)
        env.close()
    img = Image.new("RGB", (1200, 800))
    for k, tile in enumerate(tiles):
        img.paste(tile, ((k % 3) * 400, (k // 3) * 400))
    img.save(out / "clusters.png")
    print("clusters.png")


def run_episode(out, name, level, n_robots, env_seed, layout_seed, stride=2, top=384, ego=192, clusters=0,
                agents=None, label=""):
    """GIF of one episode, driven by `Forager` or, if given, by scripted `agents` (one per robot)."""
    cfg = {"level": level, "n_robots": n_robots, "resource_clusters": clusters}
    env = ForagingEnv(cfg, env_seed)
    obs, info = env.reset(layout_seed=layout_seed)
    viz = Viz(env.layout, top, top)
    if agents is None:
        pol = Forager(n_robots, np.random.default_rng(env_seed))
        act = lambda: pol.act(env)
    else:
        for agent in agents:
            agent.reset(env)
        act = lambda: [agent.act(env, info) for agent in agents]
    n_res = int(env.exists.sum())
    cols = 1 if n_robots <= 2 else 2
    header = 28
    frames, ret, t, done = [], 0.0, 0, False
    while True:
        if t % stride == 0 or done:
            viz.sync(env)
            canvas = Image.new("RGB", (top + cols * ego, top + header), (18, 18, 18))
            canvas.paste(top_panel(viz, env, info, agents or ()), (0, header))
            for i in range(n_robots):
                canvas.paste(ego_panel(env, obs, info, i, ego), (top + (i // 2) * ego if cols == 2 else top,
                                                                 header + (i % 2) * ego))
            tag = f"{level}" + (f" {clusters} clusters" if clusters else "") + (f" | {label}" if label else "")
            ImageDraw.Draw(canvas).text(
                (8, 5), f"{tag} | {n_robots} robots | step {t:3d}/400 | delivered "
                        f"{int(env.deposited.sum())}/{n_res} | return {ret:+.2f}",
                fill=(235, 235, 235), font=FONT)
            frames.append(canvas)
        if done:
            break
        obs, rew, done, info = env.step(act())
        ret += float(rew.mean())
        t += 1
    frames += [frames[-1]] * 15   # hold the last frame
    path = out / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(100 * stride / 2), loop=0)
    delivered = int(env.deposited.sum())
    viz.close()
    env.close()
    print(f"{path.name}: {level}, N={n_robots}, layout_seed={layout_seed}: delivered {delivered}/{n_res} "
          f"in {t} steps, {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


def still_environment(out):
    env = ForagingEnv({"level": "L2"}, 0)
    obs, info = env.reset(layout_seed=4)
    pol = Forager(2, np.random.default_rng(0))
    for _ in range(45):
        obs, _, _, info = env.step(pol.act(env))
    v3 = Viz(env.layout, 800, 1200)
    v3.sync(env)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.0)
    cam.distance, cam.elevation, cam.azimuth = 10.5, -42, 120
    a = Image.fromarray(v3.render(cam))
    v3.close()
    vt = Viz(env.layout, 800, 800)
    vt.sync(env)
    b = top_panel(vt, env, info)
    vt.close()
    img = Image.new("RGB", (2000, 800))
    img.paste(a, (0, 0))
    img.paste(b, (1200, 0))
    dr = ImageDraw.Draw(img)
    lay = env.layout
    dr.text((12, 10), f"L2 layout (seed 4): 8x8 m, {len(lay.resource_pos)} cubes, {len(lay.distractor_pos)} "
                      f"distractor spheres, {len(lay.occluders)} occluders, random nest", fill=(240, 240, 240), font=FONT_B)
    dr.text((1212, 10), "top camera; lines = resources each robot labels visible", fill=(240, 240, 240), font=FONT_B)
    img.save(out / "environment.png")
    env.close()
    print("environment.png")


def still_agent(out):
    lay = sc.Layout(
        arena_size=5.0, nest_pos=np.array([1.0, 1.0]),
        robot_poses=np.array([(-1.25, -1.2, 0.0), (0.3, -1.6, np.pi / 2)]),
        resource_pos=np.array([(-0.95, -1.2), (0.3, -0.35), (-1.6, 0.6), (-0.3, 1.5)]),
        resource_rgb=PALETTE[[0, 2, 3, 1]],
        distractor_pos=np.array([(0.85, -0.55)]), distractor_rgb=PALETTE[[3]],
        occluders=[sc.Occluder("box", (0.3, 0.2, 0.35), (-0.6, 0.35), 0.4)],
        light_dir=np.array([-0.3, 0.2, -1.0]), light_diffuse=np.array([0.6, 0.6, 0.6]), wall_rgb=WALL_RGB.copy())
    env = ForagingEnv({"arena_size": 5.0}, 0)
    env.reset(layout=lay)
    env.step([PICK, NOOP])
    for a in [TURN_LEFT] * 3 + [NOOP]:
        obs, _, _, info = env.step([a, NOOP])
    v = Viz(env.layout, 640, 960)
    v.sync(env)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    look, eye = np.array([-0.5, -1.4, 0.2]), np.array([1.05, 0.05, 1.25])
    fwd = look - eye
    cam.lookat[:] = look
    cam.distance = float(np.linalg.norm(fwd))
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))))
    close = Image.fromarray(v.render(cam))
    v.close()
    img = Image.new("RGB", (960 + 320, 640), (18, 18, 18))
    img.paste(close, (0, 0))
    img.paste(ego_panel(env, obs, info, 0, 320), (960, 0))
    img.paste(ego_panel(env, obs, info, 1, 320), (960, 320))
    dr = ImageDraw.Draw(img)
    dr.text((12, 10), "robot 0 (white) carrying a cube on its mount; robot 1 (dark) facing a cube 1.25 m ahead",
            fill=(240, 240, 240), font=FONT_B)
    dr.text((12, 610), "base r=0.2 m h=0.15 m, mast + carry mount at 0.42 m, camera at (0.15, 0, 0.35) m, "
                       "10 deg down, 90 deg FOV, 64x64 RGB", fill=(220, 220, 220), font=FONT)
    img.save(out / "agent.png")
    env.close()
    print("agent.png")


def _text(dr, xy, s, fill=(255, 255, 255), bold=False):
    dr.text(xy, s, fill=fill, font=FONT_B if bold else FONT, stroke_width=2, stroke_fill=(0, 0, 0))


def _probe_frames(cfg, probe, seed, notes, size):
    """Run the probe's scripted steps; robot 0's frame at each step in `notes` (step -> caption),
    with a ring on the target while it is labelled visible."""
    env = ForagingEnv(cfg, 0)
    obs, info = env.reset(seed, layout=probe.layout)
    tiles = {}
    for step in range(0, max(notes) + 1):
        if step:
            probe.before_step(env, step)
            obs, _, _, info = env.step(probe.actions(step))
        if step not in notes:
            continue
        img = Image.fromarray(obs["rgb"][0]).resize((size, size), Image.NEAREST)
        dr = ImageDraw.Draw(img)
        if info["labels"][0]["visible"][probe.target]:
            cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam0")
            loc = env.data.cam_xmat[cid].reshape(3, 3).T @ (env.resource_positions()[probe.target] - env.data.cam_xpos[cid])
            u, v = (32 + 32 * loc[0] / -loc[2]) * size / 64, (32 - 32 * loc[1] / -loc[2]) * size / 64
            dr.ellipse([u - 9, v - 9, u + 9, v + 9], outline=(255, 255, 255), width=2)
        dr.rectangle([0, 0, size - 1, size - 1], outline=MARK[0], width=2)
        _text(dr, (size - 52, 4), f"t={step}")
        _text(dr, (6, size - 20), notes[step])
        tiles[step] = img
    env.close()
    return [tiles[s] for s in sorted(tiles)]


def _probe_setup(probe, size, title):
    """Top view of a probe layout at reset: robot 0's heading, the target, the scripted paths."""
    lay = probe.layout
    env = ForagingEnv({"level": "L2", "n_robots": lay.n_robots}, 0)
    env.reset(layout=lay)
    viz = Viz(lay, size, size)
    viz.sync(env)
    img = Image.fromarray(viz.render("top"))
    viz.close()
    env.close()
    dr = ImageDraw.Draw(img)
    px = lambda p, z=0.45: tuple(viz.project("top", [(p[0], p[1], z)])[0])
    x0, y0, yaw0 = lay.robot_poses[0]
    target = lay.resource_pos[probe.target]
    dr.line([px((x0, y0)), px((x0 + 0.6 * np.cos(yaw0), y0 + 0.6 * np.sin(yaw0)))], fill=MARK[0], width=4)
    (tu, tv) = px(target, CUBE_HALF)
    dr.ellipse([tu - 10, tv - 10, tu + 10, tv + 10], outline=(255, 255, 255), width=3)
    _text(dr, (tu + 10, tv - 6), "target")
    _text(dr, px((x0, y0)), "0", fill=MARK[0], bold=True)
    if probe.name == "S1":
        back = (x0 - 2.04 * np.cos(yaw0), y0 - 2.04 * np.sin(yaw0))
        dr.line([px((x0, y0)), px(back)], fill=MARK[0], width=2)
        _text(dr, px(back), "release", fill=MARK[0])
    else:
        dr.line([px(lay.robot_poses[1][:2]), px(target)], fill=MARK[1], width=2)
        _text(dr, px(lay.robot_poses[1][:2]), "1", fill=MARK[1], bold=True)
        for k, name in enumerate("AB"):
            _text(dr, px(lay.occluders[k].pos, 0.9), name, fill=(255, 230, 0), bold=True)
        _, pos, yaw = next(iter(probe.occluder_moves.values()))   # pass the occluded_present probe
        o = lay.occluders[1]
        c, s = np.cos(yaw), np.sin(yaw)
        corners = [px((pos[0] + a * o.size[0] * c - b * o.size[1] * s, pos[1] + a * o.size[0] * s + b * o.size[1] * c), 0.7)
                   for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        dr.line(corners, fill=(255, 230, 0), width=2)
    _text(dr, (8, 6), title, bold=True)
    return img


def still_probes(out, seed=10_000):
    """Probe setups (top views) and robot 0's frames at key steps of S1 and three S2 conditions."""
    cfg = {"level": "L2"}
    full = resolve_cfg(cfg)
    big, small = 360, 180
    s1 = sc.s1_persistence(full, seed)
    s2 = {c: sc.s2_ambiguity(full, seed, c) for c in ("collect_visible", "collect_hidden", "occluded_present")}
    rows = [
        ("S1", _probe_frames(cfg, s1, seed, {0: "sees the target", 25: "turned ~180 deg",
                                             40: "driving away", 59: "released"}, small)),
        ("S2 collect_visible", _probe_frames(cfg, s2["collect_visible"], seed, {
            18: "target ahead", 27: "teammate arrives", 30: "teammate picks", 40: "released"}, small)),
        ("S2 collect_hidden", _probe_frames(cfg, s2["collect_hidden"], seed, {
            18: "target ahead", 27: "looking away", 30: "(pick unseen)", 40: "spot is empty"}, small)),
        ("S2 occluded_present", _probe_frames(cfg, s2["occluded_present"], seed, {
            24: "target ahead", 25: "B moved in", 30: "target hidden", 40: "released"}, small)),
    ]
    header = 30
    img = Image.new("RGB", (big + 4 * small, header + 4 * small), (18, 18, 18))
    img.paste(_probe_setup(s1, big, f"S1 setup (seed {seed})"), (0, header))
    img.paste(_probe_setup(s2["occluded_present"], big, f"S2 setup (seed {seed})"), (0, header + big))
    dr = ImageDraw.Draw(img)
    for r, (title, tiles) in enumerate(rows):
        for k, tile in enumerate(tiles):
            img.paste(tile, (big + k * small, header + r * small))
        _text(dr, (big + 6, header + r * small + 22), title, fill=MARK[0], bold=True)
    _text(dr, (8, 6), "Probes on level L2. Left: setups (0 = robot 0, 1 = teammate, A/B = probe occluders, "
                      "outline = where B moves at t=25). Right: robot 0's 64x64 view.", fill=(235, 235, 235))
    img.save(out / "probes.png")
    print("probes.png")


def heuristic_episodes(out):
    """HeuristicForager teams on held-out layouts, and one with a scripted collector and wanderer.
    Top view: thin lines are planned paths, squares the cubes each forager remembers."""
    seed = sc.HELDOUT_SEEDS[0]
    run_episode(out, "heuristic_L0", "L0", 2, 0, seed, agents=[HeuristicForager(i, seed) for i in range(2)],
                label="heuristic")
    run_episode(out, "heuristic_L2", "L2", 2, 0, seed + 1, agents=[HeuristicForager(i, seed) for i in range(2)],
                label="heuristic")
    run_episode(out, "heuristic_L2_4robots", "L2", 4, 0, seed + 2,
                agents=[HeuristicForager(i, seed) for i in range(4)], label="heuristic")
    run_episode(out, "teammates_L2", "L2", 3, 0, seed + 3, label="0 heuristic, 1 collector, 2 wanderer",
                agents=[HeuristicForager(0, seed), ScriptedTeammate(1, "collector", seed=seed),
                        ScriptedTeammate(2, "wanderer", seed=seed)])


def label_check(out, n=20, size=384):
    """Label verification: n camera frames from held-out L2 episodes (heuristic foragers driving),
    each cube whose centre falls inside the frame marked at its projected centre: a green ring if
    labelled visible, a red cross if not. Check by eye that they match the pixels."""
    rng = np.random.default_rng(0)
    for k in range(n):
        seed, i = sc.HELDOUT_SEEDS[20 + k], k % 2
        env = ForagingEnv({"level": "L2"}, 0)
        obs, info = env.reset(layout_seed=seed)
        agents = [HeuristicForager(j, seed) for j in range(2)]
        for agent in agents:
            agent.reset(env)
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}")
        warmup, done = int(rng.integers(10, 200)), False

        def in_frame():
            cp, R = env.data.cam_xpos[cid], env.data.cam_xmat[cid].reshape(3, 3)
            loc = (env.resource_positions() - cp) @ R   # camera frame: x right, y up, looking along -z
            ok = env.exists & (loc[:, 2] < 0) & (np.abs(loc[:, :2]).max(1) <= -loc[:, 2])
            return loc, ok

        while not done and (env.t < warmup or not in_frame()[1].any()) and env.t < 400:
            obs, _, done, info = env.step([agent.act(env, info) for agent in agents])
        loc, ok = in_frame()
        lab = info["labels"][i]
        ego = Image.fromarray(obs["rgb"][i]).resize((size, size), Image.NEAREST)
        dr = ImageDraw.Draw(ego)
        s = size / 64
        lines = []
        for r in np.flatnonzero(env.exists):
            dist = float(np.linalg.norm(env.resource_positions()[r] - env.data.cam_xpos[cid]))
            where = "in frame" if ok[r] else "outside the frame"
            status = ("VISIBLE" if lab["visible"][r] else "not visible" + (", beyond 6 m" if ok[r] and dist >= 6
                      else ", centre hidden" if ok[r] else "")) + ("" if lab["available"][r] else " (carried/deposited)")
            lines.append(f"cube {r}: {where}, {dist:4.1f} m from camera, bearing {np.degrees(lab['bearing'][r]):+5.0f} deg: {status}")
            if not ok[r]:
                continue
            u, v = (32 + 32 * loc[r, 0] / -loc[r, 2]) * s, (32 - 32 * loc[r, 1] / -loc[r, 2]) * s
            if lab["visible"][r]:
                dr.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=3)
            else:
                dr.line([u - 9, v - 9, u + 9, v + 9], fill=(255, 40, 40), width=3)
                dr.line([u - 9, v + 9, u + 9, v - 9], fill=(255, 40, 40), width=3)
            _text(dr, (u + 12, v - 8), str(r))
        dr.rectangle([0, 0, size - 1, size - 1], outline=MARK[i], width=3)
        viz = Viz(env.layout, size, size)
        viz.sync(env)
        top = top_panel(viz, env, info)
        viz.close()
        env.close()
        img = Image.new("RGB", (2 * size, size + 52 + 18 * len(lines)), (18, 18, 18))
        img.paste(ego, (0, 40))
        img.paste(top, (size, 40))
        d = ImageDraw.Draw(img)
        _text(d, (8, 4), f"label check {k + 1}/{n}: held-out L2 layout {seed}, step {env.t}, robot {i}'s 64x64 camera",
              bold=True)
        _text(d, (8, 22), "green ring = labelled visible, red cross = in frame but labelled not visible", fill=(200, 200, 200))
        for m, line in enumerate(lines):
            _text(d, (8, size + 44 + 18 * m), line, fill=(0, 255, 0) if "VISIBLE" in line else (220, 220, 220))
        img.save(out / f"label_check_{k:02d}.png")
    print(f"{n} label check images in {out}")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2:] or ["stills", "episodes", "clusters", "probes"]
    if "labels" in which:
        label_check(out)
    if "heuristic" in which:
        heuristic_episodes(out)
    if "stills" in which:
        still_environment(out)
        still_agent(out)
    if "episodes" in which:
        run_episode(out, "episode_L0", "L0", 2, env_seed=1, layout_seed=11)
        run_episode(out, "episode_L1", "L1", 2, env_seed=2, layout_seed=22)
        run_episode(out, "episode_L2", "L2", 2, env_seed=3, layout_seed=33)
        run_episode(out, "episode_L2_4robots", "L2", 4, env_seed=4, layout_seed=44)
    if "clusters" in which:
        still_clusters(out)
        run_episode(out, "episode_L2_clustered", "L2", 2, env_seed=5, layout_seed=7, clusters=2)
    if "probes" in which:
        still_probes(out)
