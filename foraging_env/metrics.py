"""Per-episode task metrics, defined once here for training logs and evaluation alike.

EpisodeStats follows a VectorForagingEnv step by step and returns one record per finished
episode:
    delivered        fraction of the episode's cubes deposited (final_info["delivered"])
    length           steps (final_info["t"])
    picks            pick events, all robots
    wasted_revisits  approaches by learner robots to where a cube was before it was picked,
                     when the robot had seen the cube there and not since seen it gone (below)
    scripted         a ScriptedTeammate drove one robot in this episode

Wasted revisit. Robot i makes one when it enters the
WASTED_REVISIT_RADIUS (1 m) disc around the spot where cube r stood when it was picked, and
    - r was picked earlier in the episode, by another robot,
    - robot i had seen r (labels ever_seen) and has not seen it as unavailable since: no step
      with r visible to i while not available (carried, or on the nest pile); both judged on
      the state the step started from, what the robot knew when it moved in,
    - robot i is not carrying a cube (a carrying robot cannot be going there to pick), and
    - robot i is a learner, not a scripted teammate.
Being inside the disc when r is picked is not an approach; leaving and re-entering is a new one.
Positions come from info["priv_state"] (cubes never move while available).

Every info field read here is privileged (priv_state, labels): metrics may read them, a
learned policy should not.

To follow a single ForagingEnv, feed it as a vector env of one: priv_state [None], labels
stacked per field, events [events], and its final state as final_info (tests/test_metrics.py).
"""
from __future__ import annotations

import numpy as np

from foraging_env.constants import R_MAX

WASTED_REVISIT_RADIUS = 1.0   # m

EPISODE_FIELDS = ("delivered", "length", "picks", "wasted_revisits", "scripted")


def split_priv_state(priv: np.ndarray, n_robots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(robot_xy [..., N, 2], carrying bool[..., N], resource_xy [..., R_MAX, 2]) from
    priv_state [..., D], in the layout of ForagingEnv._priv_state."""
    n = n_robots
    robots = priv[..., :5 * n].reshape(*priv.shape[:-1], n, 5)
    res = priv[..., 5 * n:5 * n + R_MAX * (4 + n)].reshape(*priv.shape[:-1], R_MAX, 4 + n)
    return robots[..., :2], robots[..., 4] > 0.5, res[..., :2]


class EpisodeStats:
    """Per-episode task metrics for a VectorForagingEnv's outputs (module docstring).

    reset(info) after venv.reset(); step(done, info) after every venv.step(), which returns the
    records of the episodes that ended in that step (dicts with EPISODE_FIELDS)."""

    def __init__(self, n_envs: int, n_robots: int):
        self.E, self.n = n_envs, n_robots
        self._start(np.ones(n_envs, bool), None)

    def reset(self, info: dict) -> None:
        self._start(np.ones(self.E, bool), info)
        self._before(info)

    def step(self, done: np.ndarray, info: dict) -> list[dict]:
        done = np.asarray(done, bool)
        fin = info["final_info"]
        post = lambda x, f: np.where(done.reshape(-1, *(1,) * (x.ndim - 1)), f, x)   # the state the step reached
        priv = post(info["priv_state"], fin["priv_state"])
        labels = {f: post(info["labels"][f], fin["labels"][f]) for f in ("available", "visible")}
        robot_xy, carrying, _ = split_priv_state(priv, self.n)
        _, _, res_xy_before = split_priv_state(self._priv, self.n)

        picked = np.zeros((self.E, R_MAX), bool)
        for e, events in enumerate(info["events"]):
            for kind, i, r, _ in events:
                if kind == "pick":
                    self.picks[e] += 1
                    self.gone_xy[e, r] = res_xy_before[e, r]
                    self.knows[e, i, r] = True
                    picked[e, r] = True

        with np.errstate(invalid="ignore"):          # NaN for cubes not picked: never inside
            inside = np.linalg.norm(robot_xy[:, :, None] - self.gone_xy[:, None], axis=-1) <= WASTED_REVISIT_RADIUS
        self.inside |= inside & picked[:, None]      # already there when it was picked: not an approach
        approach = (inside & ~self.inside & self._seen & ~self.knows
                    & ~carrying[:, :, None] & ~self.scripted[:, :, None])
        self.revisits += approach.sum((1, 2))
        self.inside = inside
        self.knows |= labels["visible"] & ~labels["available"]

        episodes = [dict(delivered=float(fin["delivered"][e]), length=int(fin["t"][e]), picks=int(self.picks[e]),
                         wasted_revisits=int(self.revisits[e]), scripted=bool(self.scripted[e].any()))
                    for e in np.flatnonzero(done)]
        self._start(done, info)
        self._before(info)
        return episodes

    def _before(self, info: dict) -> None:
        """Keep what the next step starts from: priv_state and the robots' ever_seen."""
        self._priv = np.array(info["priv_state"], np.float32)
        self._seen = np.array(info["labels"]["ever_seen"], bool)

    def _start(self, rows: np.ndarray, info: dict | None) -> None:
        """Begin new episodes in `rows` (all state for the first call), from `info`."""
        E, N = self.E, self.n
        if info is None:
            self.gone_xy = np.full((E, R_MAX, 2), np.nan)
            self.knows = np.zeros((E, N, R_MAX), bool)
            self.inside = np.zeros((E, N, R_MAX), bool)
            self.picks = np.zeros(E, np.int64)
            self.revisits = np.zeros(E, np.int64)
            self.scripted = np.zeros((E, N), bool)
            self._priv = np.zeros((E, 5 * N + R_MAX * (4 + N) + 2), np.float32)   # priv_state before the step
            self._seen = np.zeros((E, N, R_MAX), bool)                             # ever_seen before the step
            return
        self.gone_xy[rows] = np.nan
        for x in (self.knows, self.inside, self.picks, self.revisits):
            x[rows] = 0
        self.scripted[rows] = info["scripted"][rows]


def summarize_episodes(episodes: list[dict]) -> dict[str, float]:
    """Means of the EPISODE_FIELDS over `episodes` as episode/<field>, the delivery fraction
    of episodes without a scripted teammate (episode/delivered_solo), and their count. Empty
    if there are none."""
    if not episodes:
        return {}
    out = {f"episode/{f}": float(np.mean([ep[f] for ep in episodes])) for f in EPISODE_FIELDS}
    solo = [ep["delivered"] for ep in episodes if not ep["scripted"]]
    if solo:
        out["episode/delivered_solo"] = float(np.mean(solo))
    out["episode/count"] = len(episodes)
    return out
