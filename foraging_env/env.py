"""ForagingEnv: reset/step, pick/deposit logic, rendering, labels.

N robots share a walled square arena with cube resources, sphere distractors, occluders and a
nest. Each robot sees a 64x64 RGB image from its own camera; the team is rewarded for carrying
cubes to the nest. README.md describes the task, the spaces and every cfg key.

Interface:
    reset(layout_seed) -> (obs, info)
    step(actions int[N]) -> (obs, reward float32[N], done, info)
    obs  = dict(rgb=uint8[N,H,W,3] (None if cfg["render"] is False), prev_action=int64[N], carrying=bool[N])
    info = dict(priv_state=float32[D], labels=[dict per agent], events=[(kind, i, r, t)],
                t, layout_seed, truncated)
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from foraging_env import mjcf_builder as mb
from foraging_env import scenarios
from foraging_env.constants import (ACTION_NAMES, BACKWARD, CAM_FOVY, CARRY_Z, CUBE_HALF, D_MAX,  # noqa: F401
                                    DEPOSIT, FORWARD, MAX_ROBOTS, N_ACTIONS, NEST_RADIUS, NOOP, PARK_Z, PICK,
                                    PILE_SLOTS, PILE_Z, R_MAX, SPHERE_RADIUS, TURN_LEFT, TURN_RIGHT)
from foraging_env.rendering import CameraRenderer, RenderTarget

Obs = dict[str, Any]
Info = dict[str, Any]

DEFAULT_CFG: dict[str, Any] = dict(
    level=None,               # None, or a curriculum preset from scenarios.LEVELS ("L0", "L1", "L2")
    # layout, read by scenarios.sample_layout. Counts are an int or an inclusive [lo, hi].
    n_robots=2,               # 1..MAX_ROBOTS (4)
    arena_size=5.0,           # side of the square arena (m)
    n_resources=4,            # cubes, <= R_MAX (8)
    n_distractors=0,          # spheres, <= D_MAX (4)
    n_occluders=0,            # boxes and cylinders, 0.5-0.9 m tall
    nest="center",            # "center" or "random"
    resource_clusters=0,      # 0 = uniform; else a count or [lo, hi] of clusters
    cluster_radius=1.0,       # every resource lies within this of its cluster centre (m)
    near_occluder_frac=0.0,   # fraction of resources placed within near_occluder_dist of an occluder
    near_occluder_dist=1.0,   # m, from the resource centre to the occluder's footprint
    light_dir=(-0.3, 0.2, -1.0),
    light_diffuse=(0.6, 0.6, 0.6),
    light_azimuth_jitter_deg=0.0,   # rotate light_dir about the vertical by U(-j, j) per episode
    light_intensity_jitter=0.0,     # scale light_diffuse by U(1 - j, 1 + j) per episode
    # dynamics: 10 Hz decisions over 100 Hz physics
    substeps=10,              # physics steps per env step
    v_forward=0.6,            # m/s
    v_backward=0.6,           # m/s
    turn_rate_deg=90.0,       # deg/s
    pick_range=0.35,          # max robot-centre to cube-centre distance for PICK (m)
    pick_half_angle_deg=30.0,  # max |bearing| of the cube for PICK
    # episode and reward
    max_steps=400,
    deposit_reward=1.0,       # to every robot, per deposit
    pick_reward=0.2,          # to the picking robot
    step_penalty=0.001,       # from every robot, every step
    dense_shaping=False,      # add -shaping_coef * (change in distance to nearest available cube)
    shaping_coef=0.01,
    # rendering and labels
    render=True,              # False: obs["rgb"] is None and no GL context is created
    img_size=64,              # square image side (px)
    shadows=True,
    fov_range=6.0,            # max camera-to-cube distance for visible=True (m)
    model_cache_size=8,       # compiled models kept per env, keyed by Layout.structural_key()
)

# Per-robot label fields (ForagingEnv._labels), each an [R_MAX] array of this dtype.
LABEL_DTYPES: dict[str, type] = dict(exists=np.bool_, available=np.bool_, carried_by=np.int8, visible=np.bool_,
                                     ever_seen=np.bool_, bearing=np.float32, distance=np.float32)

_IDENTITY_QUAT = np.array([1.0, 0, 0, 0])


def priv_state_dim(n_robots: int) -> int:
    """Length of info["priv_state"]: 5 per robot, 4 + n_robots per resource slot, 2 for the nest."""
    return 5 * n_robots + R_MAX * (4 + n_robots) + 2


def priv_state_ego_index(n_robots: int) -> np.ndarray:
    """int64[N, D]: priv_state[..., idx[i]] is the privileged state ordered for robot i: the
    robot blocks as (i, then the others in increasing order), and each resource's carried_by
    one-hot in the same order; resources and the nest keep their places. It equals the
    priv_state of the same scene with the robots renumbered that way. It gives a policy shared
    by all robots, which has no agent id, the privileged state from its own robot's side."""
    n, dim = n_robots, priv_state_dim(n_robots)
    idx = []
    for i in range(n):
        order = [i] + [j for j in range(n) if j != i]
        robots = [5 * j + k for j in order for k in range(5)]
        res = [5 * n + r * (4 + n) + k for r in range(R_MAX) for k in (0, 1, 2, 3, *(4 + j for j in order))]
        idx.append(robots + res + [dim - 2, dim - 1])
    return np.array(idx, np.int64)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def resolve_cfg(cfg: dict | None) -> dict:
    """The full cfg: DEFAULT_CFG, then the preset of cfg["level"] if given, then cfg's own keys
    (so explicit keys win over the level's). Raises ValueError on unknown keys or bad values."""
    cfg = dict(cfg or {})
    level = cfg.get("level")
    if level is not None and level not in scenarios.LEVELS:
        raise ValueError(f"unknown level {level!r}; levels are {sorted(scenarios.LEVELS)}")
    full = {**DEFAULT_CFG, **(scenarios.LEVELS[level] if level is not None else {}), **cfg}
    _check_cfg(full)
    return full


def _check_cfg(cfg: dict) -> None:
    unknown = cfg.keys() - DEFAULT_CFG.keys()
    if unknown:
        raise ValueError(f"unknown cfg keys {sorted(unknown)}; valid keys are those of DEFAULT_CFG")
    hi = lambda v: v if np.isscalar(v) else max(v)
    if not 1 <= cfg["n_robots"] <= MAX_ROBOTS:
        raise ValueError(f"n_robots must be in 1..{MAX_ROBOTS}, got {cfg['n_robots']}")
    if hi(cfg["n_resources"]) > R_MAX or hi(cfg["n_distractors"]) > D_MAX:
        raise ValueError(f"at most {R_MAX} resources and {D_MAX} distractors")
    if cfg["nest"] not in ("center", "random"):
        raise ValueError(f'nest must be "center" or "random", got {cfg["nest"]!r}')
    if not 0 <= cfg["near_occluder_frac"] <= 1:
        raise ValueError(f"near_occluder_frac must be in [0, 1], got {cfg['near_occluder_frac']}")


@dataclass
class _Sim:
    """A compiled model with its data, index tables and render target."""
    model: mujoco.MjModel
    data: mujoco.MjData
    qadr: np.ndarray        # [N, 3] qpos address of (x, y, yaw)
    act: np.ndarray         # [N, 3] actuator ids
    robot_body: np.ndarray  # [N]
    cam: np.ndarray         # [N]
    res_mocap: np.ndarray   # [R_MAX]
    res_geom: np.ndarray    # [R_MAX]
    dis_mocap: np.ndarray   # [D_MAX]
    dis_geom: np.ndarray    # [D_MAX]
    occ_mocap: np.ndarray   # [K]
    nest_mocap: int
    view: RenderTarget | None


class ForagingEnv:
    """Multi-robot foraging from egocentric RGB.

    cfg: overrides for DEFAULT_CFG, optionally with a curriculum "level" (see resolve_cfg);
        unknown keys raise ValueError. The resolved cfg is env.cfg.
    seed: seeds two independent streams: one draws layout seeds from scenarios.TRAIN_SEEDS when
        reset() gets none, the other orders the robots when pick/deposit actions are resolved.
        The layout sequence therefore does not depend on what happens inside episodes.

    obs (per robot i):
        rgb[i]          uint8[H, W, 3] from camera cam{i}; obs["rgb"] is None if render=False
        prev_action[i]  int64, the action robot i took last step (NOOP after reset)
        carrying[i]     bool
    reward, float32[N]: -step_penalty each step, +pick_reward to the robot that picks,
        +deposit_reward to every robot per deposit; with dense_shaping, robots not carrying also
        get -shaping_coef * (change in distance to the nearest available cube).
    done: every resource deposited, or t == max_steps (then info["truncated"] is True).
    info:
        priv_state  float32[5N + R_MAX*(4+N) + 2], privileged state for a critic; see _priv_state
        labels      list of N dicts of [R_MAX] arrays, per-robot ground truth; see _labels
        events      ("pick" | "deposit", robot, resource, t) tuples from this step
        t, layout_seed, truncated

    State readable between steps: layout, exists, carried_by, deposited, carrying (resource
    index or -1 per robot), ever_seen, t, robot_poses(), resource_positions(), available(),
    and the live MuJoCo model and data.

    Call close() (or use the env as a context manager) to free the GL context.
    """

    def __init__(self, cfg: dict | None, seed: int):
        self.cfg = resolve_cfg(cfg)
        self.n = int(self.cfg["n_robots"])
        self.rng = np.random.default_rng(seed)    # pick/deposit order
        self._seed_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.priv_dim = priv_state_dim(self.n)
        self._cache: OrderedDict[tuple, _Sim] = OrderedDict()
        self._sim: _Sim | None = None
        self._renderer = (CameraRenderer(int(self.cfg["img_size"]), mb.SCENE_GROUPS)
                          if self.cfg["render"] else None)

    # ------------------------------------------------------------------ API

    def reset(self, layout_seed: int | None = None, *,
              layout: scenarios.Layout | None = None) -> tuple[Obs, Info]:
        """Start an episode on the layout sampled from `layout_seed`, or on an explicit `layout`
        (tests and probes; layout_seed is then only reported). With neither, the layout seed is
        drawn from scenarios.TRAIN_SEEDS; evaluate on scenarios.HELDOUT_SEEDS only."""
        if layout is None:
            if layout_seed is None:
                seeds = scenarios.TRAIN_SEEDS
                layout_seed = int(self._seed_rng.integers(seeds.start, seeds.stop))
            layout = scenarios.sample_layout(self.cfg, np.random.default_rng(layout_seed))
        self._check_layout(layout)
        sim = self._sim = self._get_sim(layout)
        mujoco.mj_resetData(sim.model, sim.data)
        self._apply_layout(layout)

        self.layout, self.layout_seed = layout, layout_seed
        self.exists = np.arange(R_MAX) < len(layout.resource_pos)
        self.carried_by = np.full(R_MAX, -1, np.int8)
        self.deposited = np.zeros(R_MAX, bool)
        self.carrying = np.full(self.n, -1, np.int64)   # resource index or -1
        self.ever_seen = np.zeros((self.n, R_MAX), bool)
        self.prev_action = np.zeros(self.n, np.int64)
        self.t = 0
        mujoco.mj_forward(sim.model, sim.data)
        return self._obs(), self._info([], truncated=False)

    def step(self, actions: np.ndarray) -> tuple[Obs, np.ndarray, bool, Info]:
        """Apply one action per robot for one decision step (cfg["substeps"] physics steps)."""
        cfg, sim = self.cfg, self._sim
        a = np.asarray(actions, np.int64).reshape(self.n)

        pose0 = self.robot_poses()
        self._set_velocity_targets(a, pose0)
        for _ in range(cfg["substeps"]):
            mujoco.mj_step(sim.model, sim.data)
        self.t += 1

        pose = self.robot_poses()
        avail0, res_xy0, carrying0 = self.available(), self.resource_positions()[:, :2], self.carrying.copy()
        reward = np.full(self.n, -cfg["step_penalty"], np.float32)
        events = self._pick_and_deposit(a, pose, reward)
        self._move_carried(pose)
        if cfg["dense_shaping"] and avail0.any():
            self._add_shaping(reward, pose0, pose, res_xy0[avail0], carrying0 < 0)

        mujoco.mj_forward(sim.model, sim.data)
        self.prev_action = a.copy()
        all_deposited = bool(self.deposited[self.exists].all())
        timeout = self.t >= cfg["max_steps"]
        done = all_deposited or timeout
        return self._obs(), reward, done, self._info(events, truncated=timeout and not all_deposited)

    def set_occluder_pose(self, k: int, pos: tuple[float, float], yaw: float) -> None:
        """Move occluder k between steps (probes). The next step's physics, observation and
        labels see it at the new pose; env.layout is replaced by a copy with the new pose."""
        sim = self._sim
        sim.data.mocap_pos[sim.occ_mocap[k]] = (*pos, 0.0)
        sim.data.mocap_quat[sim.occ_mocap[k]] = _yaw_quat(yaw)
        occluders = list(self.layout.occluders)
        occluders[k] = replace(occluders[k], pos=(float(pos[0]), float(pos[1])), yaw=float(yaw))
        self.layout = replace(self.layout, occluders=occluders)
        mujoco.mj_forward(sim.model, sim.data)

    def close(self) -> None:
        """Free the GL context and the cached models. The env can be reset again after."""
        if self._renderer is not None:
            self._renderer.close()
        self._cache.clear()
        self._sim = None

    def __enter__(self) -> ForagingEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------- read-only state

    @property
    def model(self) -> mujoco.MjModel:
        return self._sim.model

    @property
    def data(self) -> mujoco.MjData:
        return self._sim.data

    def robot_poses(self) -> np.ndarray:
        """[N, 3] robot (x, y, yaw)."""
        return self._sim.data.qpos[self._sim.qadr].copy()

    def resource_positions(self) -> np.ndarray:
        """[R_MAX, 3] cube centres; unused slots are parked below the floor."""
        return self._sim.data.mocap_pos[self._sim.res_mocap].copy()

    def available(self) -> np.ndarray:
        """bool[R_MAX]: exists, not carried and not deposited."""
        return self.exists & (self.carried_by < 0) & ~self.deposited

    # ------------------------------------------------------- model and layout

    def _check_layout(self, layout: scenarios.Layout) -> None:
        if layout.n_robots != self.n:
            raise ValueError(f"layout has {layout.n_robots} robots but cfg n_robots={self.n}")
        if len(layout.resource_pos) > R_MAX or len(layout.distractor_pos) > D_MAX:
            raise ValueError(f"a layout holds at most {R_MAX} resources and {D_MAX} distractors")

    def _get_sim(self, layout: scenarios.Layout) -> _Sim:
        """The cached model for this layout's structure, compiling it on a miss (LRU)."""
        key = layout.structural_key()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        sim = self._make_sim(layout)
        self._cache[key] = sim
        if len(self._cache) > self.cfg["model_cache_size"]:
            _, old = self._cache.popitem(last=False)
            if old.view is not None:
                self._renderer.detach(old.view)
        return sim

    def _make_sim(self, layout: scenarios.Layout) -> _Sim:
        size = int(self.cfg["img_size"])
        # The renderer draws the N cameras side by side: the buffer must be at least N*size wide.
        xml = mb.build_mjcf(layout, shadows=self.cfg["shadows"], offscreen=(max(256, self.n * size), max(256, size)))
        m = mujoco.MjModel.from_xml_string(xml)
        O = mujoco.mjtObj

        def ids(kind, names):
            return np.array([mujoco.mj_name2id(m, kind, s) for s in names], dtype=np.int64)

        rng_n, rng_r, rng_d = range(self.n), range(R_MAX), range(D_MAX)
        jnt = ids(O.mjOBJ_JOINT, [f"r{i}_{a}" for i in rng_n for a in ("x", "y", "yaw")])
        body_mocap = lambda names: m.body_mocapid[ids(O.mjOBJ_BODY, names)]
        return _Sim(
            model=m, data=mujoco.MjData(m),
            qadr=m.jnt_qposadr[jnt].reshape(self.n, 3),
            act=ids(O.mjOBJ_ACTUATOR, [f"r{i}_{a}" for i in rng_n for a in ("x", "y", "yaw")]).reshape(self.n, 3),
            robot_body=ids(O.mjOBJ_BODY, [f"robot{i}" for i in rng_n]),
            cam=ids(O.mjOBJ_CAMERA, [f"cam{i}" for i in rng_n]),
            res_mocap=body_mocap([f"res{r}" for r in rng_r]),
            res_geom=ids(O.mjOBJ_GEOM, [f"res{r}" for r in rng_r]),
            dis_mocap=body_mocap([f"dis{k}" for k in rng_d]),
            dis_geom=ids(O.mjOBJ_GEOM, [f"dis{k}" for k in rng_d]),
            occ_mocap=body_mocap([f"occ{k}" for k in range(len(layout.occluders))]),
            nest_mocap=int(body_mocap(["nest"])[0]),
            view=self._renderer.attach(m) if self._renderer is not None else None,
        )

    def _apply_layout(self, layout: scenarios.Layout) -> None:
        """Set every per-layout quantity that is not baked into the compiled model."""
        sim = self._sim
        m, d = sim.model, sim.data
        d.qpos[sim.qadr] = layout.robot_poses
        d.mocap_pos[sim.nest_mocap] = (*layout.nest_pos, 0.0)
        for items, rgb, mocap, geom, z in (
            (layout.resource_pos, layout.resource_rgb, sim.res_mocap, sim.res_geom, CUBE_HALF),
            (layout.distractor_pos, layout.distractor_rgb, sim.dis_mocap, sim.dis_geom, SPHERE_RADIUS),
        ):
            m.geom_contype[geom] = mb.CONTYPE_OBSTACLE
            d.mocap_pos[mocap] = (0.0, 0.0, PARK_Z)
            k = len(items)
            d.mocap_pos[mocap[:k], :2] = items
            d.mocap_pos[mocap[:k], 2] = z
            m.geom_rgba[geom[:k], :3] = rgb
        for k, o in enumerate(layout.occluders):
            d.mocap_pos[sim.occ_mocap[k]] = (*o.pos, 0.0)
            d.mocap_quat[sim.occ_mocap[k]] = _yaw_quat(o.yaw)
        m.light_dir[0] = layout.light_dir / np.linalg.norm(layout.light_dir)
        m.light_diffuse[0] = layout.light_diffuse

    # ------------------------------------------------------------- dynamics

    def _set_velocity_targets(self, a: np.ndarray, pose: np.ndarray) -> None:
        """Planar velocity targets: drive along the current heading, or turn in place."""
        cfg, sim, d = self.cfg, self._sim, self._sim.data
        v = np.select([a == FORWARD, a == BACKWARD], [cfg["v_forward"], -cfg["v_backward"]], 0.0)
        w = np.deg2rad(cfg["turn_rate_deg"])
        d.ctrl[sim.act[:, 0]] = v * np.cos(pose[:, 2])
        d.ctrl[sim.act[:, 1]] = v * np.sin(pose[:, 2])
        d.ctrl[sim.act[:, 2]] = np.select([a == TURN_LEFT, a == TURN_RIGHT], [w, -w], 0.0)

    def _pick_and_deposit(self, a: np.ndarray, pose: np.ndarray, reward: np.ndarray) -> list[tuple]:
        """Resolve PICK and DEPOSIT after physics, in a random robot order so that simultaneous
        picks of the same resource don't always favour robot 0. Adds the pick and deposit
        rewards to `reward` in place; returns this step's events."""
        cfg, sim = self.cfg, self._sim
        m, d = sim.model, sim.data
        events: list[tuple] = []
        n_dep = 0
        for i in self.rng.permutation(self.n):
            if a[i] == PICK and self.carrying[i] < 0:
                r = self._pick_target(pose[i])
                if r >= 0:
                    self.carrying[i], self.carried_by[r] = r, i
                    m.geom_contype[sim.res_geom[r]] = 0   # carried and deposited cubes don't collide
                    reward[i] += cfg["pick_reward"]
                    events.append(("pick", int(i), r, self.t))
            elif (a[i] == DEPOSIT and self.carrying[i] >= 0
                  and np.linalg.norm(pose[i, :2] - self.layout.nest_pos) <= NEST_RADIUS):
                r = int(self.carrying[i])
                slot = PILE_SLOTS[self.deposited.sum()]
                d.mocap_pos[sim.res_mocap[r]] = (*(self.layout.nest_pos + slot), PILE_Z)
                d.mocap_quat[sim.res_mocap[r]] = _IDENTITY_QUAT
                self.deposited[r], self.carried_by[r], self.carrying[i] = True, -1, -1
                n_dep += 1
                events.append(("deposit", int(i), r, self.t))
        reward += cfg["deposit_reward"] * n_dep
        return events

    def _pick_target(self, pose: np.ndarray) -> int:
        """Nearest available resource within pick range and half-angle of the heading, or -1."""
        rel = self._sim.data.mocap_pos[self._sim.res_mocap, :2] - pose[:2]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.angle(np.exp(1j * (np.arctan2(rel[:, 1], rel[:, 0]) - pose[2])))
        ok = (self.available() & (dist <= self.cfg["pick_range"])
              & (np.abs(ang) <= np.deg2rad(self.cfg["pick_half_angle_deg"])))
        return int(np.argmin(np.where(ok, dist, np.inf))) if ok.any() else -1

    def _move_carried(self, pose: np.ndarray) -> None:
        """Carried cubes ride on their robot's mount."""
        d = self._sim.data
        for i in np.flatnonzero(self.carrying >= 0):
            mid = self._sim.res_mocap[self.carrying[i]]
            d.mocap_pos[mid] = (pose[i, 0], pose[i, 1], CARRY_Z)
            d.mocap_quat[mid] = _yaw_quat(pose[i, 2])

    def _add_shaping(self, reward: np.ndarray, pose0: np.ndarray, pose: np.ndarray,
                     res_xy: np.ndarray, free: np.ndarray) -> None:
        """Dense shaping, in place, for robots that were not carrying (`free`): minus
        shaping_coef times the change in distance to the nearest of `res_xy` (resources
        available at the start of the step, at fixed positions so only the robot's motion counts)."""
        for i in np.flatnonzero(free):
            before = np.linalg.norm(res_xy - pose0[i, :2], axis=1).min()
            after = np.linalg.norm(res_xy - pose[i, :2], axis=1).min()
            reward[i] -= self.cfg["shaping_coef"] * (after - before)

    # ------------------------------------------------- observations and info

    def _obs(self) -> Obs:
        rgb = None
        if self._renderer is not None:
            size = int(self.cfg["img_size"])
            rgb = np.empty((self.n, size, size, 3), np.uint8)
            self._renderer.render(self._sim.view, self._sim.model, self._sim.data, self._sim.cam, rgb)
        return dict(rgb=rgb, prev_action=self.prev_action.copy(), carrying=self.carrying >= 0)

    def _visible(self, i: int, res: np.ndarray) -> np.ndarray:
        """bool[R_MAX] for robot i: the resource centre is inside camera i's frustum and closer
        than fov_range, and a group-0 ray from camera i to that centre (excluding robot i's
        body) hits it first. `res` holds the resource centres, [R_MAX, 3]."""
        sim = self._sim
        m, d = sim.model, sim.data
        tan_half = np.tan(np.deg2rad(CAM_FOVY) / 2)
        cam_pos = d.cam_xpos[sim.cam[i]]
        cam_mat = d.cam_xmat[sim.cam[i]].reshape(3, 3)
        geomid = np.zeros(1, np.int32)
        visible = np.zeros(R_MAX, bool)
        for r in np.flatnonzero(self.exists):
            vec = res[r] - cam_pos
            loc = cam_mat.T @ vec       # camera frame: x right, y up, looking along -z
            depth = -loc[2]
            if (depth <= 0 or np.linalg.norm(vec) >= self.cfg["fov_range"]
                    or abs(loc[0]) > tan_half * depth or abs(loc[1]) > tan_half * depth):
                continue
            mujoco.mj_ray(m, d, cam_pos, vec, mb.SCENE_GROUPS, 1, int(sim.robot_body[i]), geomid)
            visible[r] = geomid[0] == sim.res_geom[r]
        return visible

    def _labels(self) -> list[dict]:
        """Per-robot labels, each a dict of [R_MAX] arrays with the dtypes in
        LABEL_DTYPES. Updates ever_seen.

        exists      bool     resource slot is used this episode
        available   bool     exists, not carried, not deposited
        carried_by  int8     robot id, or -1
        visible     bool     see _visible
        ever_seen   bool     visible to this robot at any step of the episode so far
        bearing     float32  radians in the robot frame (0 ahead, positive to the left); 0 if not exists
        distance    float32  planar metres from the robot centre; 0 if not exists
        exists, available and carried_by are global, the same for every robot.
        """
        res = self._sim.data.mocap_pos[self._sim.res_mocap]
        avail = self.available()
        labels = []
        for i, (x, y, yaw) in enumerate(self.robot_poses()):
            dx, dy = res[:, 0] - x, res[:, 1] - y
            lx = np.cos(yaw) * dx + np.sin(yaw) * dy
            ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
            visible = self._visible(i, res)
            self.ever_seen[i] |= visible
            labels.append(dict(
                exists=self.exists.copy(),
                available=avail.copy(),
                carried_by=self.carried_by.copy(),
                visible=visible,
                ever_seen=self.ever_seen[i].copy(),
                bearing=np.where(self.exists, np.arctan2(ly, lx), 0.0).astype(np.float32),
                distance=np.where(self.exists, np.hypot(lx, ly), 0.0).astype(np.float32),
            ))
        return labels

    def _priv_state(self) -> np.ndarray:
        """Robots (x, y, cos, sin, carrying) x N; resources (x, y, available, deposited,
        carried_by one-hot[N]) x R_MAX with zero rows for unused slots; nest (x, y)."""
        pose = self.robot_poses()
        robots = np.column_stack([pose[:, :2], np.cos(pose[:, 2]), np.sin(pose[:, 2]), self.carrying >= 0])
        res = np.zeros((R_MAX, 4 + self.n))
        e = self.exists
        res[e, :2] = self._sim.data.mocap_pos[self._sim.res_mocap[e], :2]
        res[e, 2] = self.available()[e]
        res[e, 3] = self.deposited[e]
        carried = np.flatnonzero(self.carried_by >= 0)
        res[carried, 4 + self.carried_by[carried]] = 1.0
        return np.concatenate([robots.ravel(), res.ravel(), self.layout.nest_pos]).astype(np.float32)

    def _info(self, events: list[tuple], truncated: bool) -> Info:
        return dict(priv_state=self._priv_state(), labels=self._labels(), events=events,
                    t=self.t, layout_seed=self.layout_seed, truncated=truncated)
