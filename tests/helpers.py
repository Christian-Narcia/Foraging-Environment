"""Shared helpers for the tests: explicit layouts, and single-env reference runs for comparing
VectorForagingEnv step for step."""
from collections import Counter

import numpy as np

from foraging_env import N_ACTIONS, ForagingEnv, HeuristicForager, Layout, ScriptedTeammate
from foraging_env.constants import PALETTE, WALL_RGB
from foraging_env.env import LABEL_DTYPES


def make_layout(robots, resources=(), occluders=(), nest=(2.0, 2.0), arena=5.0, distractors=(), rgb=None) -> Layout:
    """A hand-built Layout. robots: (x, y, yaw) each; resources and distractors: (x, y) each;
    rgb: one colour per resource (PALETTE in turn by default)."""
    res = np.array(resources, float).reshape(-1, 2)
    dis = np.array(distractors, float).reshape(-1, 2)
    return Layout(
        arena_size=float(arena), nest_pos=np.array(nest, float), robot_poses=np.array(robots, float).reshape(-1, 3),
        resource_pos=res, resource_rgb=PALETTE[np.arange(len(res)) % len(PALETTE)] if rgb is None else np.asarray(rgb, float),
        distractor_pos=dis, distractor_rgb=PALETTE[np.arange(len(dis)) % len(PALETTE)], occluders=list(occluders),
        light_dir=np.array([-0.3, 0.2, -1.0]), light_diffuse=np.array([0.6, 0.6, 0.6]), wall_rgb=WALL_RGB.copy())


def env_on(layout: Layout, render: bool = False, **cfg) -> tuple[ForagingEnv, dict, dict]:
    """A ForagingEnv reset on `layout` (close it when done)."""
    env = ForagingEnv({"n_robots": layout.n_robots, "arena_size": layout.arena_size, "render": render, **cfg}, seed=0)
    obs, info = env.reset(layout=layout)
    return env, obs, info


def reference_run(cfg, env_seed, steps, layout_seed=None, teammate_frac=0.0,
                  behaviours=("collector", "wanderer"), policy_seed=0) -> list[dict]:
    """One ForagingEnv(cfg, env_seed) driven by heuristic foragers, reset by hand when an episode ends, with
    teammate mixing done the way VectorForagingEnv documents it. Returns per-step records of the
    action inputs and everything the vector env should return for this env."""
    env = ForagingEnv(cfg, env_seed)
    mix = np.random.default_rng(np.random.SeedSequence(env_seed, spawn_key=(1,)))
    rng = np.random.default_rng(policy_seed)

    def start():
        tm = None
        if teammate_frac > 0 and mix.random() < teammate_frac:
            robot = int(mix.integers(env.n))
            behaviour = behaviours[int(mix.integers(len(behaviours)))]
            tm = ScriptedTeammate(robot, behaviour, seed=int(mix.integers(2**31)))
            tm.reset(env)
        agents = [HeuristicForager(i, int(rng.integers(2**31))) for i in range(env.n)]
        for agent in agents:
            agent.reset(env)
        return tm, agents

    obs, info = env.reset(layout_seed)
    tm, agents = start()
    recs = [dict(obs=obs, info=info, scripted=None if tm is None else tm.robot)]
    for _ in range(steps):
        inp = np.array([agent.act(env, info) for agent in agents])
        executed = inp.copy()
        if tm is not None:
            inp[tm.robot] = rng.integers(N_ACTIONS)    # the vector env must ignore it
            executed[tm.robot] = tm.act(env, info)
        obs, reward, done, info = env.step(executed)
        rec = dict(inp=inp, reward=reward, done=done, truncated=info["truncated"], events=info["events"])
        if done:
            rec.update(final_obs=obs, final_info=info, delivered=np.float32(env.deposited.sum() / max(env.exists.sum(), 1)))
            obs, info = env.reset()
            tm, agents = start()
        rec.update(obs=obs, info=info, scripted=None if tm is None else tm.robot)
        recs.append(rec)
    env.close()
    return recs


def _state_mismatches(out: Counter, tag, obs, info, vobs, vinfo, k, rgb_tol) -> None:
    if obs["rgb"] is None:
        out[tag + "rgb is not None"] += vobs["rgb"] is not None
    else:
        out[tag + "rgb"] += int(np.abs(obs["rgb"].astype(int) - vobs["rgb"][k]).max() > rgb_tol)
    for key in ("prev_action", "carrying"):
        out[tag + key] += not np.array_equal(obs[key], vobs[key][k])
    out[tag + "priv_state"] += not np.array_equal(info["priv_state"], vinfo["priv_state"][k])
    for key in ("t", "layout_seed"):
        out[tag + key] += info[key] != vinfo[key][k]
    for i, lab in enumerate(info["labels"]):
        for f in LABEL_DTYPES:
            v = vinfo["labels"][f][k, i]
            out[tag + "labels." + f] += not (np.array_equal(lab[f], v) and lab[f].dtype == v.dtype)


def compare_with_reference(venv, refs: list[list[dict]], layout_seeds=None, rgb_tol: int = 1) -> tuple[dict, Counter]:
    """Step `venv` with the references' action inputs; return the nonzero mismatch counts per
    field and some event statistics."""
    n_envs, n = venv.n_envs, venv.n
    bad, stats = Counter(), Counter()
    scripted = lambda s: np.arange(n) == (-1 if s is None else s)
    vobs, vinfo = venv.reset(layout_seeds)
    for k in range(n_envs):
        _state_mismatches(bad, "reset.", refs[k][0]["obs"], refs[k][0]["info"], vobs, vinfo, k, rgb_tol)
        bad["scripted"] += not np.array_equal(vinfo["scripted"][k], scripted(refs[k][0]["scripted"]))
    for t in range(1, len(refs[0])):
        vobs, vrew, vdone, vinfo = venv.step(np.stack([refs[k][t]["inp"] for k in range(n_envs)]))
        for k in range(n_envs):
            r = refs[k][t]
            bad["reward"] += not np.array_equal(r["reward"], vrew[k])
            bad["done"] += r["done"] != vdone[k]
            bad["truncated"] += r["truncated"] != vinfo["truncated"][k]
            bad["events"] += r["events"] != vinfo["events"][k]
            bad["scripted"] += not np.array_equal(vinfo["scripted"][k], scripted(r["scripted"]))
            stats["events"] += len(r["events"])
            if r["done"]:
                stats["episodes"] += 1
                stats["truncated"] += bool(r["truncated"])
                _state_mismatches(bad, "final.", r["final_obs"], r["final_info"], vinfo["final_obs"],
                                  vinfo["final_info"], k, rgb_tol)
                bad["final.delivered"] += r["delivered"] != vinfo["final_info"]["delivered"][k]
            else:
                bad["final row not zero"] += bool(vinfo["final_info"]["priv_state"][k].any())
            _state_mismatches(bad, "", r["obs"], r["info"], vobs, vinfo, k, rgb_tol)
    return {key: v for key, v in bad.items() if v}, stats
