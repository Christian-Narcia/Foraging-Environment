"""Layouts, the layout sampler, curriculum levels, the seed split and the probe scenarios.

    Layout, Occluder                  everything that varies between episodes
    sample_layout                     cfg-driven random layouts
    LEVELS                            curriculum presets L0, L1, L2 (cfg overrides; ForagingEnv
                                      resolves cfg["level"])
    TRAIN_SEEDS, HELDOUT_SEEDS        layout seed split; evaluate on held-out seeds only
    Probe, s1_persistence,            scripted evaluation episodes: a layout plus open-loop
    s2_ambiguity, s3_full             actions and events for the first steps
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foraging_env.constants import (BACKWARD, CUBE_HALF, FORWARD, NEST_RADIUS, NOOP, PALETTE, PICK,
                                    ROBOT_RADIUS, SPHERE_RADIUS, TURN_LEFT, TURN_RIGHT, WALL_RGB)

# Curriculum levels: overrides on top of DEFAULT_CFG, selected with cfg["level"]. Keys given
# explicitly in a cfg win over the level's, so e.g. {"level": "L2", "n_robots": 4} works.
LEVELS: dict[str, dict] = {
    "L0": dict(arena_size=5.0, n_resources=4, n_distractors=0, n_occluders=0, nest="center",
               near_occluder_frac=0.0, light_azimuth_jitter_deg=0.0, light_intensity_jitter=0.0,
               dense_shaping=True),
    "L1": dict(arena_size=6.0, n_resources=6, n_distractors=2, n_occluders=3, nest="center",
               near_occluder_frac=0.0, light_azimuth_jitter_deg=0.0, light_intensity_jitter=0.0,
               dense_shaping=False),
    "L2": dict(arena_size=8.0, n_resources=[6, 8], n_distractors=[2, 4], n_occluders=[4, 8], nest="random",
               near_occluder_frac=0.5, light_azimuth_jitter_deg=180.0, light_intensity_jitter=0.3,
               dense_shaping=False),
}

# Layout seeds. Never evaluate on TRAIN_SEEDS; ForagingEnv.reset() without a seed draws from them.
TRAIN_SEEDS = range(0, 10_000)
HELDOUT_SEEDS = range(10_000, 10_200)


@dataclass(frozen=True)
class Occluder:
    """A static box or cylinder standing on the floor."""
    kind: str                 # "box" | "cylinder"
    size: tuple[float, ...]   # box: (hx, hy, hz); cylinder: (r, hz). Half-sizes, MuJoCo convention.
    pos: tuple[float, float]  # x, y of the base centre
    yaw: float

    @property
    def footprint(self) -> float:
        """Radius of the circle enclosing the occluder's floor footprint."""
        return float(np.hypot(*self.size[:2])) if self.kind == "box" else float(self.size[0])

    def distance(self, x, y):
        """Planar distance from (x, y) to the occluder's footprint (0 inside it). x and y may be
        arrays of the same shape."""
        dx, dy = np.subtract(x, self.pos[0]), np.subtract(y, self.pos[1])
        if self.kind == "cylinder":
            return np.maximum(np.hypot(dx, dy) - self.size[0], 0.0)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return np.hypot(np.maximum(np.abs(lx) - self.size[0], 0.0), np.maximum(np.abs(ly) - self.size[1], 0.0))


@dataclass
class Layout:
    """Everything that varies between episodes. World frame: origin at the arena centre,
    z up; yaw in radians, counter-clockwise from +x. Build one by hand to pass to
    `ForagingEnv.reset(layout=...)`, or draw one with `sample_layout`."""
    arena_size: float               # side length of the square arena centred at the origin (m)
    nest_pos: np.ndarray            # [2]
    robot_poses: np.ndarray         # [N, 3] x, y, yaw
    resource_pos: np.ndarray        # [n_res, 2], n_res <= R_MAX
    resource_rgb: np.ndarray        # [n_res, 3]
    distractor_pos: np.ndarray      # [n_dis, 2], n_dis <= D_MAX
    distractor_rgb: np.ndarray      # [n_dis, 3]
    occluders: list[Occluder]
    light_dir: np.ndarray           # [3]
    light_diffuse: np.ndarray       # [3]
    wall_rgb: np.ndarray            # [4, 2, 3]

    @property
    def n_robots(self) -> int:
        return len(self.robot_poses)

    def structural_key(self) -> tuple:
        """Everything baked into the compiled model. Layouts with equal keys share a model;
        poses, colours and lighting are applied at runtime."""
        return (
            self.n_robots,
            round(float(self.arena_size), 4),
            tuple((o.kind, tuple(round(float(s), 4) for s in o.size)) for o in self.occluders),
            tuple(np.round(self.wall_rgb, 4).ravel().tolist()),
        )


# ------------------------------------------------------------------ sampler

def _as_count(v, rng: np.random.Generator) -> int:
    """cfg counts are an int or an inclusive [lo, hi] range."""
    return int(v) if np.isscalar(v) else int(rng.integers(v[0], v[1] + 1))


class _Crowded(RuntimeError):
    """No free spot found for an item; sample_layout redraws the whole layout."""


@dataclass
class _Fixed:
    """Items placed by hand (probes); the sampler fills in the rest of the layout around them."""
    nest: np.ndarray
    robots: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    resources: list[tuple[float, float]] = field(default_factory=list)   # come first, so index 0..
    occluders: list[Occluder] = field(default_factory=list)              # come first, so index 0..
    keepout: list[tuple[float, float, float]] = field(default_factory=list)   # (x, y, r): no sampled item

    def discs(self) -> list[tuple[float, float, float]]:
        return (list(self.keepout) + [(*o.pos, o.footprint) for o in self.occluders]
                + [(x, y, ROBOT_RADIUS) for x, y, _ in self.robots.values()]
                + [(x, y, CUBE_HALF) for x, y in self.resources])


def _place(rng, half, margin, clear, around=None, tries=2000):
    """(x, y) at least `margin` from the walls with `clear(x, y)` True, uniform over the arena
    or, if `around` = (cx, cy, radius) is given, uniform over that disc."""
    for _ in range(tries):
        if around is None:
            x, y = rng.uniform(-half + margin, half - margin, size=2)
        else:
            cx, cy, rad = around
            rho, th = rad * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            x, y = cx + rho * np.cos(th), cy + rho * np.sin(th)
            if max(abs(x), abs(y)) > half - margin:
                continue
        if clear(x, y):
            return float(x), float(y)
    raise _Crowded("layout sampling failed: arena too crowded")


def _sample_nest(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    if cfg["nest"] == "center":
        return np.zeros(2)
    lim = cfg["arena_size"] / 2 - NEST_RADIUS - 0.3
    return rng.uniform(-lim, lim, size=2)


def _sample_light(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    light_dir = np.asarray(cfg["light_dir"], dtype=float)
    light_diffuse = np.asarray(cfg["light_diffuse"], dtype=float)
    azimuth = float(cfg.get("light_azimuth_jitter_deg", 0.0))
    if azimuth > 0:
        a = np.deg2rad(rng.uniform(-azimuth, azimuth))
        c, s = np.cos(a), np.sin(a)
        light_dir = np.array([c * light_dir[0] - s * light_dir[1], s * light_dir[0] + c * light_dir[1],
                              light_dir[2]])
    intensity = float(cfg.get("light_intensity_jitter", 0.0))
    if intensity > 0:
        light_diffuse = light_diffuse * rng.uniform(1 - intensity, 1 + intensity)
    return light_dir, light_diffuse


def sample_layout(cfg: dict, rng: np.random.Generator, max_attempts: int = 100) -> Layout:
    """Sample a layout from cfg keys: n_robots, arena_size, n_resources, n_distractors,
    n_occluders, nest ("center" | "random"), light_dir, light_diffuse, and optionally
    resource_clusters (0 = uniform, else a count or [lo, hi]) with cluster_radius,
    near_occluder_frac with near_occluder_dist, light_azimuth_jitter_deg and
    light_intensity_jitter (see DEFAULT_CFG in foraging_env.py; levels are resolved there).

    A layout that gets stuck placing an item is redrawn from the same rng, so seeds that
    never get stuck give the same layout as without the retry."""
    for _ in range(max_attempts):
        try:
            return _sample_layout(cfg, rng)
        except _Crowded:
            pass
    raise RuntimeError(f"layout sampling failed {max_attempts} times: arena too crowded")


def _sample_layout(cfg: dict, rng: np.random.Generator, fixed: _Fixed | None = None) -> Layout:
    """Placement order: nest, occluders, robots, resources near occluders, cluster centres,
    the other resources, distractors; then colours and lighting. Optional features draw from
    the rng only when enabled, so layouts without them are unchanged by their addition."""
    half = cfg["arena_size"] / 2
    n_res = _as_count(cfg["n_resources"], rng)
    n_dis = _as_count(cfg["n_distractors"], rng)
    n_occ = _as_count(cfg["n_occluders"], rng)
    n_clu = _as_count(cfg.get("resource_clusters", 0), rng)
    cluster_radius = float(cfg.get("cluster_radius", 1.0))
    nest = fixed.nest if fixed is not None else _sample_nest(cfg, rng)

    discs: list[tuple[float, float, float]] = [] if fixed is None else fixed.discs()

    def far(x, y, r, gap):
        return all(np.hypot(x - px, y - py) >= r + pr + gap for px, py, pr in discs)

    def off_nest(x, y, r, gap):
        return np.hypot(x - nest[0], y - nest[1]) >= NEST_RADIUS + r + gap

    occluders = [] if fixed is None else list(fixed.occluders)
    for _ in range(max(n_occ - len(occluders), 0)):
        h = rng.uniform(0.5, 0.9)
        if rng.random() < 0.5:
            size = (rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.5), h / 2)
            kind = "box"
        else:
            size = (rng.uniform(0.15, 0.35), h / 2)
            kind = "cylinder"
        o = Occluder(kind, size, (0.0, 0.0), 0.0)
        r = o.footprint
        # Gap of 0.6 m between occluders and walls leaves room for a robot to pass.
        x, y = _place(rng, half, r + 0.6, lambda x, y: far(x, y, r, 0.6) and off_nest(x, y, r, 0.2))
        occluders.append(Occluder(kind, size, (x, y), float(rng.uniform(-np.pi, np.pi))))
        discs.append((x, y, r))

    robots = []
    for i in range(cfg["n_robots"]):
        if fixed is not None and i in fixed.robots:
            robots.append(fixed.robots[i])
            continue
        x, y = _place(rng, half, ROBOT_RADIUS + 0.2, lambda x, y: far(x, y, ROBOT_RADIUS, 0.2))
        robots.append((x, y, float(rng.uniform(-np.pi, np.pi))))
        discs.append((x, y, ROBOT_RADIUS))

    def place_item(radius, around=None, extra=lambda x, y: True, tries=2000):
        # 0.45 m clearance so a robot can get within pick range from at least one side.
        x, y = _place(rng, half, radius + 0.45,
                      lambda x, y: far(x, y, radius, 0.45) and off_nest(x, y, radius, 0.2) and extra(x, y),
                      around=around, tries=tries)
        discs.append((x, y, radius))
        return x, y

    def place_items(n, radius, centres=()):
        # A full cluster disc is cheaper to redraw than to keep probing, hence fewer tries.
        pos = [place_item(radius, (*centres[k % len(centres)], cluster_radius) if centres else None,
                          tries=300 if centres else 2000) for k in range(n)]
        return np.array(pos, dtype=float).reshape(n, 2)

    # Resources near occluders, dealt to the occluders in turn: centre within near_occluder_dist
    # of the occluder's footprint (and still 0.45 m clear of it).
    fixed_res = [] if fixed is None else list(fixed.resources)
    n_free = max(n_res - len(fixed_res), 0)
    n_near = int(np.floor(cfg.get("near_occluder_frac", 0.0) * n_free + 0.5)) if occluders else 0
    near_dist = float(cfg.get("near_occluder_dist", 1.0))
    near = []
    for k in range(n_near):
        o = occluders[k % len(occluders)]
        near.append(place_item(CUBE_HALF, (*o.pos, o.footprint + near_dist),
                               lambda x, y, o=o: o.distance(x, y) <= near_dist, tries=300))

    # Cluster centres are spots a resource could occupy, 2 * cluster_radius apart so clusters
    # don't merge; the remaining resources are dealt to the clusters in turn.
    centres: list[tuple[float, float]] = []
    for _ in range(min(n_clu, n_free - n_near)):
        centres.append(_place(rng, half, CUBE_HALF + 0.45,
                              lambda x, y: far(x, y, CUBE_HALF, 0.45) and off_nest(x, y, CUBE_HALF, 0.2)
                              and all(np.hypot(x - cx, y - cy) >= 2 * cluster_radius for cx, cy in centres)))

    resource_pos = np.concatenate([np.reshape(fixed_res + near, (-1, 2)),
                                   place_items(n_free - n_near, CUBE_HALF, centres)])
    distractor_pos = place_items(n_dis, SPHERE_RADIUS)
    resource_rgb = PALETTE[rng.integers(len(PALETTE), size=len(resource_pos))]
    distractor_rgb = PALETTE[rng.integers(len(PALETTE), size=n_dis)]
    light_dir, light_diffuse = _sample_light(cfg, rng)

    return Layout(
        arena_size=float(cfg["arena_size"]),
        nest_pos=nest,
        robot_poses=np.array(robots, dtype=float),
        resource_pos=resource_pos,
        resource_rgb=resource_rgb,
        distractor_pos=distractor_pos,
        distractor_rgb=distractor_rgb,
        occluders=occluders,
        light_dir=light_dir,
        light_diffuse=light_diffuse,
        wall_rgb=WALL_RGB.copy(),
    )


# ------------------------------------------------------------------- probes

S2_CONDITIONS = ("collect_visible", "collect_hidden", "no_collect_visible", "no_collect_hidden",
                 "occluded_present")


@dataclass
class Probe:
    """One scripted evaluation episode: a layout, open-loop actions for the first steps and
    scheduled events. Steps count from 1 (the first env.step after reset). Run it with

        obs, info = env.reset(seed, layout=probe.layout)
        for step in range(1, horizon + 1):
            probe.before_step(env, step)
            obs, reward, done, info = env.step(probe.actions(step, policy_actions))

    where policy_actions (int[N]) is only read for policy robots after release_step.
    """
    name: str                          # "S1", "S2/<condition>" or "S3"
    layout: Layout
    target: int                        # resource index the probe is about; -1 for S3
    release_step: int                  # policy robots follow the policy from step release_step + 1
    policy_robots: tuple[int, ...]     # the other robots hold still once their script ends
    scripts: dict[int, list[int]]      # robot -> actions for steps 1, 2, ...
    occluder_moves: dict[int, tuple[int, tuple[float, float], float]]   # step -> (occluder, (x, y), yaw)

    def actions(self, step: int, policy_actions=None) -> np.ndarray:
        """Joint action for env step `step`."""
        a = np.full(self.layout.n_robots, NOOP, np.int64)
        for i in range(len(a)):
            script = self.scripts.get(i, ())
            if step <= len(script):
                a[i] = script[step - 1]
            elif i in self.policy_robots and step > self.release_step:
                if policy_actions is None:
                    raise ValueError(f"step {step}: robot {i} is released and needs a policy action")
                a[i] = policy_actions[i]
        return a

    def before_step(self, env, step: int) -> None:
        """Apply the events scheduled for env step `step`; they show in that step's observation."""
        if step in self.occluder_moves:
            k, pos, yaw = self.occluder_moves[step]
            env.set_occluder_pose(k, pos, yaw)


class _Frame:
    """Rigid transform from a probe's own frame (robot 0 at the origin facing +x) to the world."""

    def __init__(self, x: float, y: float, yaw: float):
        self.x, self.y, self.yaw = x, y, yaw
        self.c, self.s = np.cos(yaw), np.sin(yaw)

    def point(self, p) -> tuple[float, float]:
        return (float(self.x + self.c * p[0] - self.s * p[1]), float(self.y + self.s * p[0] + self.c * p[1]))

    def pose(self, p) -> tuple[float, float, float]:
        return (*self.point(p), float(np.angle(np.exp(1j * (p[2] + self.yaw)))))

    def occluder(self, o: Occluder) -> Occluder:
        return Occluder(o.kind, o.size, self.point(o.pos), float(np.angle(np.exp(1j * (o.yaw + self.yaw)))))


def _segment(a, b, r: float, spacing: float = 0.25) -> list[tuple[float, float, float]]:
    """Keep-out discs of radius r covering the segment a-b."""
    n = max(int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / spacing)), 1)
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, r) for k in range(n + 1)]


# Keep-out radius around a probe target: with _probe_layout's 0.2 m wall gap, the target keeps the
# same 0.45 m wall clearance as sampled cubes.
_TARGET_KEEPOUT = 0.45 + CUBE_HALF - 0.2


def _probe_layout(cfg: dict, rng: np.random.Generator, robots: dict, resources: list,
                  occluders: list[Occluder], keepout: list, max_attempts: int = 100) -> tuple[Layout, _Frame]:
    """Place a probe's hand-built items (given in the probe frame) at a random pose in the arena,
    every disc at least 0.2 m from the walls and the nest disc, then sample the rest of the
    layout around them."""
    half = cfg["arena_size"] / 2
    for _ in range(max_attempts):
        nest = _sample_nest(cfg, rng)
        for _ in range(2000):
            frame = _Frame(*rng.uniform(-half, half, size=2), rng.uniform(-np.pi, np.pi))
            fixed = _Fixed(nest, {i: frame.pose(p) for i, p in robots.items()},
                           [frame.point(p) for p in resources], [frame.occluder(o) for o in occluders],
                           [(*frame.point(d), d[2]) for d in keepout])
            if all(max(abs(x), abs(y)) <= half - r - 0.2
                   and np.hypot(x - nest[0], y - nest[1]) >= NEST_RADIUS + r + 0.2
                   for x, y, r in fixed.discs()):
                break
        else:
            continue
        try:
            return _sample_layout(cfg, rng, fixed), frame
        except _Crowded:
            pass
    raise RuntimeError("probe layout sampling failed: the arena is too small or crowded for the probe")


# S1: robot 0 looks at a target 2 m ahead, turns ~180 deg and drives ~2 m away, then must return
# to it from memory. At 0.6 m/s and 90 deg/s the scripted prefix takes 59 steps: look 5 steps,
# turn 20 steps (178 deg), drive 34 steps (2.04 m).
S1_TARGET_DIST = 2.0
S1_SCRIPT = [NOOP] * 5 + [TURN_LEFT] * 20 + [FORWARD] * 34


def s1_persistence(cfg: dict, seed: int) -> Probe:
    """S1: the target (resource 0) is the only resource. Robot 0 follows S1_SCRIPT and is then
    released about 4 m from the target, facing away; teammates hold still throughout.
    Needs an arena of at least ~8 x 8 m (level L2)."""
    rng = np.random.default_rng(seed)
    layout, _ = _probe_layout(
        {**cfg, "n_resources": 1}, rng,
        robots={0: (0.0, 0.0, 0.0)},
        resources=[(S1_TARGET_DIST, 0.0)],
        occluders=[],
        keepout=([(0.0, 0.0, 0.45), (S1_TARGET_DIST, 0.0, _TARGET_KEEPOUT)]
                 + _segment((-2.3, 0.0), (S1_TARGET_DIST - 0.3, 0.0), 0.4)))
    return Probe("S1", layout, target=0, release_step=len(S1_SCRIPT), policy_robots=(0,),
                 scripts={0: list(S1_SCRIPT)}, occluder_moves={})


# S2, in the probe frame (robot 0 at the origin facing +x):
#   target T at (2.5, 0). Teammate (robot 1) starts 2.04 m from T on robot 0's right, facing T:
#   29 forward steps leave it 0.30 m from T at step 29, it picks (or not) at step 30, then
#   reverses 10 steps back out of view. Occluder A (0.7 m tall) hides its path from robot 0
#   until its last ~7 steps of approach. Occluder B stands left of the sight line and, in
#   occluded_present only, is moved between robot 0 and T before step 25.
#   Hidden conditions: robot 0 turns left 52 deg over steps 19-24, holds, and turns back over
#   steps 35-40, so T is out of view during steps 23-35
#   and robot 0 faces it again, within ~2 deg, at release.
S2_TARGET_DIST = 2.5
S2_TEAMMATE_DIST = 2.04
S2_RELEASE = 40
S2_OCCLUDER_MOVE_STEP = 25
# A covers bearings -49.5..-9.8 deg from robot 0's camera, 0.26 m clear of the sight line to T
# (robot 0 can drive straight to T) and 0.4 m clear of the teammate's path.
_S2_A = Occluder("box", (0.5, 0.15, 0.35), (1.475, -0.75), np.deg2rad(53.0))
_S2_B = Occluder("box", (0.3, 0.15, 0.35), (1.25, 0.9), np.pi / 2)
_S2_B_MOVED = ((1.25, 0.0), np.pi / 2)
_S2_LOOK = [NOOP] * S2_RELEASE
_S2_LOOK_AWAY = [NOOP] * 18 + [TURN_LEFT] * 6 + [NOOP] * 10 + [TURN_RIGHT] * 6


def s2_ambiguity(cfg: dict, seed: int, condition: str) -> Probe:
    """S2: robot 0 faces the target (resource 0) 2.5 m ahead while a scripted teammate (robot 1)
    reaches it at step 30. condition (S2_CONDITIONS) sets whether the teammate picks it, whether
    robot 0 looks away during the pick, or (occluded_present) whether occluder B (index 1) is
    moved in front of it. All conditions share one layout per seed. Robot 0 is released at step
    40; the teammate then holds still. Needs n_robots >= 2 and an ~8 x 8 m arena (level L2)."""
    if condition not in S2_CONDITIONS:
        raise ValueError(f"unknown S2 condition {condition!r}; conditions are {S2_CONDITIONS}")
    if cfg["n_robots"] < 2:
        raise ValueError("S2 needs n_robots >= 2 (robot 1 is the scripted teammate)")
    rng = np.random.default_rng(seed)
    t = (S2_TARGET_DIST, 0.0)
    layout, frame = _probe_layout(
        cfg, rng,
        robots={0: (0.0, 0.0, 0.0), 1: (t[0], -S2_TEAMMATE_DIST, np.pi / 2)},
        resources=[t],
        occluders=[_S2_A, _S2_B],
        keepout=([(0.0, 0.0, 0.45), (*t, _TARGET_KEEPOUT)]
                 + _segment((0.3, 0.0), (t[0] - 0.3, 0.0), 0.4)                       # sight line to T
                 + _segment((0.3, -0.05), (t[0] - 0.2, -0.55), 0.3)                   # ... to the teammate's last approach
                 + _segment((t[0], -S2_TEAMMATE_DIST - 0.1), (t[0], -0.3), 0.45)))    # teammate's path
    collect = condition.startswith("collect")
    teammate = [FORWARD] * 29 + [PICK if collect else NOOP] + [BACKWARD] * 10
    moves = {}
    if condition == "occluded_present":
        pos, yaw = _S2_B_MOVED
        moves[S2_OCCLUDER_MOVE_STEP] = (1, frame.point(pos), float(np.angle(np.exp(1j * (yaw + frame.yaw)))))
    return Probe(f"S2/{condition}", layout, target=0, release_step=S2_RELEASE, policy_robots=(0,),
                 scripts={0: list(_S2_LOOK_AWAY if condition.endswith("hidden") else _S2_LOOK), 1: teammate},
                 occluder_moves=moves)


def s3_full(cfg: dict, seed: int) -> Probe:
    """S3: an ordinary episode on `cfg` (level L2, with 2 or 4 robots); every robot is controlled by the policy.
    The layout is the one ForagingEnv.reset(seed) would sample."""
    layout = sample_layout(cfg, np.random.default_rng(seed))
    return Probe("S3", layout, target=-1, release_step=0, policy_robots=tuple(range(layout.n_robots)),
                 scripts={}, occluder_moves={})
