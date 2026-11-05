"""Scripted agents: the heuristic forager (a reference baseline) and scripted teammates.

Neither learns or reads pixels. Each controls one robot and acts on privileged information
(labels or env state), which scripted agents may use and a learned policy should not:

    HeuristicForager   the non-learning reference. It detects exactly the cubes its own
                       camera sees (labels), knows its own pose and the static map, and
                       remembers what it has seen. Sweeps until it knows of a cube, fetches it,
                       carries it to the nest, repeats; 10% random actions.
    ScriptedTeammate   "collector", "wanderer" or "idle", acting on the full env state. For
                       mixed teams (VectorForagingEnv's teammate_frac) and probes.

Reset every agent after env.reset(), then call act() with the info of the latest step:

    agents = [HeuristicForager(i, seed=0) for i in range(env.n)]
    obs, info = env.reset(seed)
    for agent in agents:
        agent.reset(env)
    while not done:
        obs, reward, done, info = env.step([agent.act(env, info) for agent in agents])

Agents read only the env's public state (robot_poses(), carrying, prev_action, layout, ...).
Navigation is shared: shortest paths on a 0.1 m grid over walls, occluders, distractor spheres,
the nest pole and the floor cubes the agent knows of, followed with straight-line lookahead,
and a back-up-and-turn when the robot stops moving.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from foraging_env.constants import (BACKWARD, CAM_FOVY, CAM_POS, DEPOSIT, FORWARD, N_ACTIONS, NEST_RADIUS,
                                    NOOP, PICK, POLE_RADIUS, ROBOT_RADIUS, SPHERE_RADIUS, TURN_LEFT, TURN_RIGHT)

# Planner grid and path following.
CELL = 0.1                       # m
CLEARANCE = ROBOT_RADIUS + 0.02  # paths keep the robot centre this far from obstacle surfaces
SOFT_CLEARANCE = 0.4             # cells closer than this to an obstacle cost up to 3x, so paths keep a margin
CUBE_DISC = 0.09                 # floor cubes as discs (half-side 0.075, half-diagonal 0.106)
LOOKAHEAD = 10                   # aim at the farthest of the next 10 waypoints (~1 m) in straight sight
TURN_TOL = np.deg2rad(12)        # drive forward while the heading error is below this, else turn
APPROACH = 0.6                   # within this of a cube, with a clear line, drive straight at it
APPROACH_TOL = np.deg2rad(8)
NEST_GOAL = NEST_RADIUS - 0.25   # paths to the nest end this deep in the disc; DEPOSIT at NEST_RADIUS - 0.05
STALL_STEPS = 3                  # FORWARD steps moving < 1.5 cm each before backing up
# The heuristic forager's sweep: floor cells where a cube could lie, marked seen once in view (ray-cast on the map).
COVER_CELL = 0.25
TURN_COST = 0.38                 # m of path per radian of turning (9 deg vs 0.06 m per step)
SWITCH_MARGIN = 0.5              # m: a newly seen cube replaces the target only if this much cheaper
# A remembered cube that is not detected although its spot is in plain view (within MISS_RANGE,
# MISS_ANGLE of the camera axis, sight line MISS_CLEAR from anything on the map) is forgotten
# after MISSES_TO_FORGET such steps. The margins keep cubes half-hidden behind an edge.
MISS_RANGE = 3.0
MISS_ANGLE = np.deg2rad(CAM_FOVY / 2 - 10)
MISS_CLEAR = 0.15
MISSES_TO_FORGET = 3
FOV_HALF = np.deg2rad(CAM_FOVY / 2)
RAY_ANGLES = np.linspace(-FOV_HALF, FOV_HALF, 46)   # 2 deg apart
RAY_STEP = 0.05

TEAMMATE_BEHAVIOURS = ("collector", "wanderer", "idle")


def _wrap(a):
    return np.angle(np.exp(1j * a))


def _rel(pose: np.ndarray, xy) -> tuple[float, float]:
    """Planar distance and bearing (robot frame, positive to the left) from pose to xy."""
    dx, dy = xy[0] - pose[0], xy[1] - pose[1]
    return float(np.hypot(dx, dy)), float(_wrap(np.arctan2(dy, dx) - pose[2]))


def _steer(pose: np.ndarray, xy, tol: float, reverse: bool = False) -> int:
    """FORWARD if xy is within tol of the heading, else a turn toward it. With reverse, the
    same for the robot's back and BACKWARD."""
    _, b = _rel(pose, xy)
    if reverse:
        b = _wrap(b - np.pi)
    if abs(b) <= tol:
        return BACKWARD if reverse else FORWARD
    return TURN_LEFT if b > 0 else TURN_RIGHT


class _Grid:
    """Square cells over the arena; arrays are indexed [row (y), col (x)]."""

    def __init__(self, half: float, cell: float):
        self.half = half
        self.n = max(int(round(2 * half / cell)), 1)
        self.cell = 2 * half / self.n
        c = -half + (np.arange(self.n) + 0.5) * self.cell
        self.x, self.y = np.meshgrid(c, c)

    def index(self, x, y):
        """(row, col) of the cells holding points (x, y), clipped to the grid."""
        col = np.clip(np.floor((np.asarray(x) + self.half) / self.cell).astype(int), 0, self.n - 1)
        row = np.clip(np.floor((np.asarray(y) + self.half) / self.cell).astype(int), 0, self.n - 1)
        return row, col


def _clearance(layout, x, y, walls: bool = True) -> np.ndarray:
    """Distance from points (x, y) to the nearest wall, occluder, distractor sphere or the nest
    pole: everything a robot bumps into apart from cubes and robots."""
    d = layout.arena_size / 2 - np.maximum(np.abs(x), np.abs(y)) if walls else np.full(np.shape(x), np.inf)
    for o in layout.occluders:
        d = np.minimum(d, o.distance(x, y))
    for px, py in layout.distractor_pos:
        d = np.minimum(d, np.hypot(x - px, y - py) - SPHERE_RADIUS)
    return np.minimum(d, np.hypot(x - layout.nest_pos[0], y - layout.nest_pos[1]) - POLE_RADIUS)


_NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _distance_field(cost: np.ndarray, start: tuple[int, int], cell: float) -> np.ndarray:
    """Shortest path lengths from `start` to every cell of an 8-connected grid, where an edge
    costs its length times the mean `cost` of its two cells (inf = blocked). Bellman-Ford with
    whole-grid shifts, repeated until nothing improves."""
    n = cost.shape[0]
    dist = np.full(cost.shape, np.inf)
    dist[start] = 0.0
    edges = []
    for dr, dc in _NEIGHBOURS:
        dst = (slice(max(dr, 0), n + min(dr, 0)), slice(max(dc, 0), n + min(dc, 0)))
        src = (slice(max(-dr, 0), n + min(-dr, 0)), slice(max(-dc, 0), n + min(-dc, 0)))
        edges.append((dst, src, np.hypot(dr, dc) * cell * (cost[dst] + cost[src]) / 2))
    changed = True
    while changed:
        changed = False
        for dst, src, w in edges:
            cand = dist[src] + w
            view = dist[dst]
            better = cand < view
            if better.any():
                view[better] = cand[better]
                changed = True
    return dist


def _backtrack(dist: np.ndarray, cost: np.ndarray, goal: tuple[int, int], cell: float) -> list:
    """Cells of the shortest path from the start of `dist` (where it is 0) to `goal`."""
    n = dist.shape[0]
    r, c = goal
    path = [(r, c)]
    while dist[r, c] > 0:
        best, prev = np.inf, None
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                v = dist[rr, cc] + np.hypot(dr, dc) * cell * (cost[rr, cc] + cost[r, c]) / 2
                if v < best:
                    best, prev = v, (rr, cc)
        r, c = prev
        path.append(prev)
    return path[::-1]


class _Navigator:
    """Shortest paths for one robot over the static map plus the floor cubes it is given, and
    a controller that follows them."""

    def __init__(self):
        self.layout = None

    def set_map(self, layout) -> bool:
        """Rebuild the static map when `layout` is a new object (a new episode, or an occluder
        moved by env.set_occluder_pose). Returns whether it did."""
        if layout is self.layout:
            return False
        self.layout = layout
        self.grid = g = _Grid(layout.arena_size / 2, CELL)
        self.static = _clearance(layout, g.x, g.y)
        self.nest_goal = np.hypot(g.x - layout.nest_pos[0], g.y - layout.nest_pos[1]) <= NEST_GOAL
        self.cubes = None
        self.set_cubes([])
        self.path = np.zeros((0, 2))
        return True

    def set_cubes(self, xy) -> None:
        """The floor cubes to plan around, [k, 2]."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cubes is not None and np.array_equal(xy, self.cubes):
            return
        self.cubes = xy
        g = self.grid
        clear = self.static
        for cx, cy in xy:
            clear = np.minimum(clear, np.hypot(g.x - cx, g.y - cy) - CUBE_DISC)
        self.clear = clear
        soft = np.clip((SOFT_CLEARANCE - clear) / (SOFT_CLEARANCE - CLEARANCE), 0, 1)
        self.cost = np.where(clear < CLEARANCE, np.inf, 1 + 2 * soft)
        self._field_start = None

    def field(self, xy) -> np.ndarray:
        """Path lengths from the cell of xy to every cell; cached while the robot stays in that
        cell and the map is unchanged."""
        start = tuple(int(v) for v in self.grid.index(*xy))
        if start != self._field_start:
            cost = self.cost.copy()
            # A robot pressed against an obstacle starts in a blocked cell: let it back out.
            near = np.hypot(self.grid.x - xy[0], self.grid.y - xy[1]) <= 0.25
            cost[near & np.isinf(cost) & (self.clear > ROBOT_RADIUS - 0.05)] = 3.0
            self._field, self._field_cost = _distance_field(cost, start, self.grid.cell), cost
            self._field_start = start
        return self._field

    def plan(self, xy, goal: np.ndarray) -> bool:
        """Path from xy to the nearest cell of the bool grid `goal`; False if none is reachable."""
        f = self.field(xy)
        g = np.where(goal, f, np.inf)
        k = np.unravel_index(int(np.argmin(g)), g.shape)
        if not np.isfinite(g[k]):
            self.path = np.zeros((0, 2))
            return False
        cells = tuple(np.array(_backtrack(f, self._field_cost, k, self.grid.cell)).T)
        self.path_cells = cells
        self.path_open = np.isfinite(self.cost[cells])   # not counting the cells a robot backs out of
        self.path = np.column_stack([self.grid.x[cells], self.grid.y[cells]])
        return True

    def follow(self, pose: np.ndarray, reverse: bool = False) -> int | None:
        """FORWARD (BACKWARD with reverse) or a turn along the path. None once the path is
        done, if the robot has been pushed off it, or if a cube added since planning blocks it."""
        if not len(self.path):
            return None
        d = np.hypot(*(self.path[:4 * LOOKAHEAD] - pose[:2]).T)
        k = int(np.argmin(d))
        ahead = slice(k, k + 2 * LOOKAHEAD)
        rows, cols = self.path_cells
        if (d[k] > 0.4 or (np.isinf(self.cost[rows[ahead], cols[ahead]]) & self.path_open[ahead]).any()
                or (k == len(self.path) - 1 and d[k] < 0.15)):
            self.path = np.zeros((0, 2))
            return None
        self.path, self.path_open = self.path[k:], self.path_open[k:]
        self.path_cells = (rows[k:], cols[k:])
        aim = min(1, len(self.path) - 1)
        for m in range(2, min(len(self.path), LOOKAHEAD + 1)):
            if not self.line_clear(pose[:2], self.path[m]):
                break
            aim = m
        return _steer(pose, self.path[aim], TURN_TOL, reverse)

    def line_clear(self, p, q, stop: float = 0.0) -> bool:
        """Whether the segment from p to `stop` short of q stays CLEARANCE - 0.02 from obstacles."""
        v = np.subtract(q, p)
        length = float(np.hypot(*v))
        s = np.arange(RAY_STEP, length - stop + 1e-9, RAY_STEP)
        if not len(s):
            return True
        pts = np.asarray(p) + (s / length)[:, None] * v
        return bool((self.clear[self.grid.index(pts[:, 0], pts[:, 1])] >= CLEARANCE - 0.02).all())


class _Agent:
    """Controls one robot. Call reset(env) after every env.reset(), then once per step either
    act(env, info) or, while something else drives the robot, observe(env, info)."""

    def __init__(self, robot: int, seed: int, noise: float):
        self.robot, self.noise = int(robot), float(noise)
        self.rng = np.random.default_rng([seed, robot])   # noise and random choices, across episodes
        self.nav = _Navigator()

    def reset(self, env) -> None:
        self.nav = _Navigator()
        self.goal = None              # key of the goal the current path leads to
        self._reverse = False         # follow the current path backwards
        self.recovery: list[int] = []
        self._recovering = False
        self._stalled, self._last_xy = 0, None
        self.pick_dist = env.cfg["pick_range"] - 0.01
        self.pick_angle = np.deg2rad(env.cfg["pick_half_angle_deg"] - 5)
        self._sync_map(env)

    def _sync_map(self, env) -> None:
        if self.nav.set_map(env.layout):
            self.goal = None
            self._on_map(env.layout)

    def observe(self, env, info) -> None:
        """Update the agent's memory from the latest step without choosing an action."""
        self._sync_map(env)
        self._update(env, info, env.robot_poses()[self.robot])

    def act(self, env, info=None) -> int:
        """This robot's action for the next env.step(); `info` is what the last step returned."""
        self.observe(env, info)
        pose = env.robot_poses()[self.robot]
        self._check_stall(env, pose)
        self._recovering = bool(self.recovery)
        a = self.recovery.pop(0) if self.recovery else self._decide(env, pose)
        if self.noise and self.rng.random() < self.noise:
            a = int(self.rng.integers(N_ACTIONS))
        return int(a)

    # hooks
    def _on_map(self, layout) -> None:
        pass

    def _update(self, env, info, pose: np.ndarray) -> None:
        pass

    def _decide(self, env, pose: np.ndarray) -> int:
        raise NotImplementedError

    # shared behaviour
    def _check_stall(self, env, pose: np.ndarray) -> None:
        """After STALL_STEPS driving steps without moving: back off the other way, turn a random
        amount, replan."""
        xy = pose[:2].copy()
        prev = env.prev_action[self.robot]
        if (self._last_xy is not None and not self._recovering and prev in (FORWARD, BACKWARD)
                and np.hypot(*(xy - self._last_xy)) < 0.015):
            self._stalled += 1
        else:
            self._stalled = 0
        self._last_xy = xy
        if self._stalled >= STALL_STEPS:
            turn = TURN_LEFT if self.rng.random() < 0.5 else TURN_RIGHT
            back = BACKWARD if prev == FORWARD else FORWARD
            self.recovery = [back] * 3 + [turn] * int(self.rng.integers(2, 6))
            self._stalled, self.goal = 0, None

    def _go(self, pose: np.ndarray, key, goal: Callable[[], np.ndarray], may_reverse: bool = False) -> int | None:
        """A step along a path to the cells of goal() (a bool grid). Plans when `key` changes,
        replans once when the path is done, lost or blocked; None if that fails too. With
        may_reverse, a path that starts behind the robot is driven backwards."""
        if key != self.goal:
            self.goal = key
            self._plan(pose, goal(), may_reverse)
        a = self.nav.follow(pose, self._reverse)
        if a is None and self._plan(pose, goal(), may_reverse):
            a = self.nav.follow(pose, self._reverse)
        return a

    def _plan(self, pose: np.ndarray, goal: np.ndarray, may_reverse: bool) -> bool:
        found = self.nav.plan(pose[:2], goal)
        path = self.nav.path
        self._reverse = bool(found and may_reverse
                             and abs(_rel(pose, path[min(LOOKAHEAD, len(path) - 1)])[1]) > np.pi / 2)
        return found

    def _ring(self, xy) -> np.ndarray:
        """Grid cells from which the cube at xy is a short straight drive away."""
        d = np.hypot(self.nav.grid.x - xy[0], self.nav.grid.y - xy[1])
        return (d >= 0.3) & (d <= 0.5)

    def _fetch(self, pose: np.ndarray, xy, key) -> int:
        """A step toward picking the floor cube at xy: PICK once in range, a straight drive
        when close with a clear line, else along a path to the ring around it."""
        d, b = _rel(pose, xy)
        if d <= self.pick_dist and abs(b) <= self.pick_angle:
            return PICK
        if d <= APPROACH and self.nav.line_clear(pose[:2], xy, stop=0.3):
            return _steer(pose, xy, APPROACH_TOL)
        a = self._go(pose, key, lambda: self._ring(xy))
        return a if a is not None else _steer(pose, xy, APPROACH_TOL)

    def _deliver(self, env, pose: np.ndarray) -> int:
        """A step toward the nest, DEPOSIT once inside the disc. Depositing works facing any
        way, so a nest behind the robot is reached in reverse instead of after a U-turn."""
        nest = env.layout.nest_pos
        if np.hypot(*(pose[:2] - nest)) <= NEST_RADIUS - 0.05:
            return DEPOSIT
        a = self._go(pose, "nest", lambda: self.nav.nest_goal, may_reverse=True)
        return a if a is not None else _steer(pose, nest, TURN_TOL)


class HeuristicForager(_Agent):
    """A non-learning reference forager that acts on privileged detections, not pixels (report
    it as such next to vision-based policies). Robot `robot`'s forager uses only:
      - its own pose, carry state and previous action;
      - the static map from env.layout: arena, occluders, distractor spheres, nest;
      - detections: for each cube labelled visible to this robot, its bearing and distance and
        whether it lies on the floor (labels "available").
    It never sees cubes its camera has not seen, teammates' poses or other robots' labels, and
    does not communicate.

    It remembers where it saw floor cubes, and forgets one when it sees it carried or
    deposited, when a pick at its spot fails, or after its spot is in plain view (by the static
    map) for MISSES_TO_FORGET steps without it. Knowing of no cube, it sweeps: it heads for the
    unseen floor cell that is cheapest to reach (path length plus turning) until one comes
    into view. Otherwise it fetches the remembered cube with the lowest path length to it plus
    distance on to the nest, carries it to the nest and deposits it. With probability `noise`
    each action is replaced by a uniformly random one.
    """

    def __init__(self, robot: int, seed: int = 0, noise: float = 0.1):
        super().__init__(robot, seed, noise)

    def reset(self, env) -> None:
        self.known: dict[int, np.ndarray] = {}   # resource slot -> floor position where it was seen
        self._misses: dict[int, int] = {}
        self.target, self._reselect = -1, True
        self._picking = -1                       # cube a PICK was aimed at, to check the outcome
        self._frontier = None
        self._seen = None
        self._range = env.cfg["fov_range"] - 0.5
        self._ray_s = np.arange(RAY_STEP, self._range + 1e-9, RAY_STEP)
        super().reset(env)

    def _on_map(self, layout) -> None:
        half = layout.arena_size / 2
        self._sight = g = _Grid(half, RAY_STEP)
        self._opaque = np.zeros(g.x.shape, bool)   # occluders are taller than the camera
        for o in layout.occluders:
            self._opaque |= o.distance(g.x, g.y) <= 0
        self._sight_clear = _clearance(layout, g.x, g.y, walls=False)
        self._cover = c = _Grid(half, COVER_CELL)
        nest = layout.nest_pos
        self._floor = ((_clearance(layout, c.x, c.y) >= 0.3)
                       & (np.hypot(c.x - nest[0], c.y - nest[1]) >= NEST_RADIUS + 0.1))
        if self._seen is None:
            self._seen = np.zeros(c.x.shape, bool)
        self._frontier = None

    def _sight_lines(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera position, and along each of RAY_ANGLES the distance to the first occluder or
        wall (at most the sweep range)."""
        yaw = pose[2]
        cam = pose[:2] + CAM_POS[0] * np.array([np.cos(yaw), np.sin(yaw)])
        ang = yaw + RAY_ANGLES
        px = cam[0] + np.cos(ang)[:, None] * self._ray_s
        py = cam[1] + np.sin(ang)[:, None] * self._ray_s
        half = self._sight.half
        blocked = (np.abs(px) >= half) | (np.abs(py) >= half) | self._opaque[self._sight.index(px, py)]
        return cam, np.where(blocked.any(1), self._ray_s[np.argmax(blocked, 1)], self._ray_s[-1])

    @staticmethod
    def _in_view(cam, yaw, hit, x, y, near: float, far: float, half_angle: float) -> np.ndarray:
        """Whether points (x, y) are within half_angle of the camera axis, between near and far,
        and in front of the first occluder along their ray."""
        dx, dy = x - cam[0], y - cam[1]
        d = np.hypot(dx, dy)
        a = _wrap(np.arctan2(dy, dx) - yaw)
        k = np.clip(np.rint((a + FOV_HALF) / (RAY_ANGLES[1] - RAY_ANGLES[0])).astype(int), 0, len(RAY_ANGLES) - 1)
        return (np.abs(a) <= half_angle) & (d >= near) & (d <= np.minimum(far, hit[k] - 0.1))

    def _in_plain_view(self, cam: np.ndarray, yaw: float, r: int) -> bool:
        """Whether remembered cube r would surely be detected from here if it were there: close,
        well inside the field of view, and in sight with a margin around everything on the map
        and the other remembered cubes."""
        xy = self.known[r]
        d, b = _rel(np.array([cam[0], cam[1], yaw]), xy)
        if not (0.5 <= d <= MISS_RANGE and abs(b) <= MISS_ANGLE):
            return False
        s = np.arange(RAY_STEP, d - MISS_CLEAR, RAY_STEP)
        pts = cam + (s / d)[:, None] * (xy - cam)
        if (self._sight_clear[self._sight.index(pts[:, 0], pts[:, 1])] < MISS_CLEAR).any():
            return False
        others = np.array([p for k, p in self.known.items() if k != r]).reshape(-1, 2)
        gap = np.hypot(pts[:, None, 0] - others[:, 0], pts[:, None, 1] - others[:, 1])
        return not (gap < CUBE_DISC + MISS_CLEAR).any()

    def _forget(self, r: int) -> None:
        if self.known.pop(r, None) is not None:
            self._misses.pop(r, None)
            if r == self.target:
                self.target, self._reselect = -1, True

    def _update(self, env, info, pose: np.ndarray) -> None:
        i = self.robot
        lab = info["labels"][i]
        if self._picking >= 0 and env.prev_action[i] == PICK and env.carrying[i] < 0:
            self._forget(self._picking)          # nothing to pick where it was remembered
        self._picking = -1
        if env.carrying[i] >= 0:
            self._forget(int(env.carrying[i]))
        cam, hit = self._sight_lines(pose)
        self._seen |= self._in_view(cam, pose[2], hit, self._cover.x, self._cover.y, 0.35, self._range, FOV_HALF)
        for r in range(len(lab["visible"])):
            if lab["visible"][r]:
                if not lab["available"][r]:
                    self._forget(r)              # seen on a mount or on the nest
                    continue
                if r not in self.known:
                    b = pose[2] + lab["bearing"][r]
                    self.known[r] = pose[:2] + lab["distance"][r] * np.array([np.cos(b), np.sin(b)])
                    self._reselect = True
                self._misses[r] = 0
            elif r in self.known and self._in_plain_view(cam, pose[2], r):
                self._misses[r] = self._misses.get(r, 0) + 1
                if self._misses[r] >= MISSES_TO_FORGET:
                    self._forget(r)
        self.nav.set_cubes(list(self.known.values()))

    def _decide(self, env, pose: np.ndarray) -> int:
        if env.carrying[self.robot] >= 0:
            return self._deliver(env, pose)
        r = self._select(pose)
        if r < 0:
            return self._sweep(pose)
        a = self._fetch(pose, self.known[r], ("cube", r))
        if a == PICK:
            self._picking = r
        return a

    def _select(self, pose: np.ndarray) -> int:
        """The remembered cube to fetch, by least path length to it plus straight-line distance
        on to the nest disc. Chosen again when a cube is first seen (switching only if it is
        SWITCH_MARGIN cheaper) or when the target is lost."""
        if not self._reselect:
            return self.target
        self._reselect = False
        f = self.nav.field(pose[:2])
        nest = self.nav.layout.nest_pos
        cost = {r: np.min(f[self._ring(xy)]) + max(np.hypot(*(xy - nest)) - NEST_RADIUS, 0.0)
                for r, xy in self.known.items()}
        best = min(cost, key=cost.get, default=-1)
        if best < 0 or not np.isfinite(cost[best]):
            self.target = -1
        elif self.target not in cost or cost[best] < cost[self.target] - SWITCH_MARGIN:
            self.target = best
        return self.target

    def _sweep(self, pose: np.ndarray) -> int:
        """A step toward the unseen floor cell that is cheapest to reach, path length plus
        TURN_COST per radian of turning; once every cell is seen, the sweep starts over."""
        unseen = self._floor & ~self._seen
        if self._frontier is not None and not unseen[self._frontier]:
            self._frontier = None
        if self._frontier is None:
            if not unseen.any():
                self._seen[:] = False
                unseen = self._floor
            rows, cols = np.nonzero(unseen)
            x, y = self._cover.x[rows, cols], self._cover.y[rows, cols]
            turn = np.abs(_wrap(np.arctan2(y - pose[1], x - pose[0]) - pose[2]))
            cost = self.nav.field(pose[:2])[self.nav.grid.index(x, y)] + TURN_COST * turn
            k = int(np.argmin(cost))
            if not np.isfinite(cost[k]):
                return TURN_LEFT
            self._frontier = (rows[k], cols[k])
        x, y = self._cover.x[self._frontier], self._cover.y[self._frontier]
        g = self.nav.grid
        a = self._go(pose, ("sweep", self._frontier), lambda: np.hypot(g.x - x, g.y - y) <= COVER_CELL / 2)
        if a is None:   # arrived (or stuck) without seeing it from the cell's centre
            self._seen[self._frontier] = True
            self._frontier = None
            return TURN_LEFT
        return a


class ScriptedTeammate(_Agent):
    """A scripted robot for training mixes and probes, acting on the full env state.

    behaviour:
      "collector"  fetch resource `target` (a random available one at reset if None), carry it
                   to the nest and deposit it, then idle. If another robot takes the target
                   first, it idles. If it is already carrying a cube, it delivers that one.
      "wanderer"   drive between random free spots; never picks or deposits.
      "idle"       NOOP.
    noise: probability of replacing each action by a uniformly random one.
    """

    def __init__(self, robot: int, behaviour: str, target: int | None = None, seed: int = 0,
                 noise: float = 0.0):
        if behaviour not in TEAMMATE_BEHAVIOURS:
            raise ValueError(f"unknown behaviour {behaviour!r}; behaviours are {TEAMMATE_BEHAVIOURS}")
        super().__init__(robot, seed, noise)
        self.behaviour, self._target_arg = behaviour, target

    def reset(self, env) -> None:
        super().reset(env)
        self.target = self._target_arg
        if self.behaviour == "collector" and self.target is None:
            avail = np.flatnonzero(env.available())
            self.target = int(self.rng.choice(avail)) if len(avail) else -1
        self.done = False
        self._carrying = env.carrying[self.robot] >= 0
        self._spot, self._spot_steps = None, 0

    def _decide(self, env, pose: np.ndarray) -> int:
        if self.behaviour == "idle" or self.done:
            return NOOP
        floor = env.available()
        if self.behaviour == "wanderer":
            self.nav.set_cubes(env.resource_positions()[floor, :2])
            return self._wander(pose)
        carrying = env.carrying[self.robot] >= 0
        if self._carrying and not carrying:      # delivered
            self.done = True
            return NOOP
        self._carrying = carrying
        if carrying:
            return self._deliver(env, pose)
        if self.target < 0 or not floor[self.target]:
            self.done = True
            return NOOP
        floor[self.target] = False
        self.nav.set_cubes(env.resource_positions()[floor, :2])
        return self._fetch(pose, env.resource_positions()[self.target, :2], ("cube", self.target))

    def _wander(self, pose: np.ndarray) -> int:
        g = self.nav.grid
        if self._spot is None:
            free = np.flatnonzero(self.nav.clear.ravel() >= SOFT_CLEARANCE)
            k = int(self.rng.choice(free))
            self._spot, self._spot_steps = (g.x.flat[k], g.y.flat[k]), 0
        self._spot_steps += 1
        x, y = self._spot
        a = self._go(pose, ("wander", self._spot), lambda: np.hypot(g.x - x, g.y - y) <= CELL)
        if a is None or self._spot_steps > 200:   # arrived, unreachable or taking too long
            self._spot = None
            return NOOP
        return a
