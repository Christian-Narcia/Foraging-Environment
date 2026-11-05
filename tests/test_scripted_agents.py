"""HeuristicForager and ScriptedTeammate. No rendering needed."""
import copy

import numpy as np
import pytest

from foraging_env import (HELDOUT_SEEDS, NOOP, ForagingEnv, HeuristicForager, ScriptedTeammate, resolve_cfg,
                          s1_persistence)


def run(level, agents, layout_seed, steps=400, n_robots=None, info_filter=None):
    """Run the agents for up to `steps`; returns the env state at the end, the actions and events."""
    n = n_robots or len(agents)
    env = ForagingEnv({"level": level, "n_robots": n, "render": False}, seed=0)
    obs, info = env.reset(layout_seed)
    for agent in agents:
        agent.reset(env)
    actions, events, done = [], [], False
    while not done and env.t < steps:
        a = [agent.act(env, info if info_filter is None else info_filter(info, agent.robot)) for agent in agents]
        obs, reward, done, info = env.step(a)
        actions.append(a)
        events += info["events"]
    result = dict(deposited=int(env.deposited.sum()), n=int(env.exists.sum()), t=env.t, done=done)
    env.close()
    return result, np.array(actions), events


# -------------------------------------------------------------------- heuristic forager

def test_heuristic_delivers_on_l0():
    for seed in HELDOUT_SEEDS[:3]:
        r, _, _ = run("L0", [HeuristicForager(i, seed) for i in range(2)], seed)
        assert r["done"] and r["deposited"] == r["n"]


def test_heuristic_is_deterministic():
    seed = HELDOUT_SEEDS[4]
    a = run("L2", [HeuristicForager(i, seed) for i in range(2)], seed, steps=200)[1]
    b = run("L2", [HeuristicForager(i, seed) for i in range(2)], seed, steps=200)[1]
    assert np.array_equal(a, b)


def _scramble(rng):
    """Replace everything the heuristic may not read with random values. It may use, for its own robot
    only, the labels of cubes labelled visible: visible, bearing, distance, available."""
    def f(info, robot):
        info = copy.deepcopy(info)
        info["priv_state"] = rng.normal(size=info["priv_state"].shape).astype(np.float32)
        info["events"] = [("pick", 1, 3, 0)]
        for j, lab in enumerate(info["labels"]):
            keep = lab["visible"].copy() if j == robot else np.zeros_like(lab["visible"])
            for key, v in lab.items():
                if key == "visible" and j == robot:
                    continue
                noise = (rng.random(v.shape) < 0.5 if v.dtype == bool else
                         rng.integers(-1, 2, v.shape).astype(v.dtype) if v.dtype == np.int8 else
                         rng.uniform(-5, 5, v.shape).astype(v.dtype))
                lab[key] = np.where(keep & (key in ("bearing", "distance", "available")), v, noise)
        return info
    return f


@pytest.mark.parametrize("level,seed", [("L0", HELDOUT_SEEDS[10]), ("L0", HELDOUT_SEEDS[11]),
                                        ("L2", HELDOUT_SEEDS[12]), ("L2", HELDOUT_SEEDS[13])])
def test_heuristic_uses_only_its_own_detections(level, seed):
    """The heuristic's actions must not change when every label field it may not read is randomised."""
    honest = run(level, [HeuristicForager(i, seed) for i in range(2)], seed, steps=250)
    scrambled = run(level, [HeuristicForager(i, seed) for i in range(2)], seed, steps=250,
                    info_filter=_scramble(np.random.default_rng(seed)))
    assert np.array_equal(honest[1], scrambled[1])
    assert honest[2]                                    # it did pick and deposit along the way


def test_heuristic_remembers_through_a_probe_prefix():
    """observe() during a probe's scripted prefix keeps the heuristic's memory."""
    cfg = resolve_cfg({"level": "L2"})
    probe = s1_persistence(cfg, HELDOUT_SEEDS[0])
    with ForagingEnv({**cfg, "render": False}, seed=0) as env:
        obs, info = env.reset(layout=probe.layout)
        forager = HeuristicForager(0, seed=0, noise=0.0)
        forager.reset(env)
        for step in range(1, probe.release_step + 1):
            forager.observe(env, info)
            obs, reward, done, info = env.step(probe.actions(step))
        assert not info["labels"][0]["visible"][0]           # the target is behind robot 0 now
        assert 0 in forager.known                             # ... but the forager remembers it
        events = []
        for _ in range(200):
            obs, reward, done, info = env.step([forager.act(env, info), NOOP])
            events += info["events"]
        assert ("pick", 0, 0) in [e[:3] for e in events]     # and goes back for it


# --------------------------------------------------------------- scripted teammates

@pytest.mark.parametrize("seed", HELDOUT_SEEDS[:3])
def test_collector_delivers_its_target_then_idles(seed):
    agents = [ScriptedTeammate(0, "idle"), ScriptedTeammate(1, "collector", target=2, seed=seed)]
    r, actions, events = run("L0", agents, seed)
    assert [e[:3] for e in events] == [("pick", 1, 2), ("deposit", 1, 2)]
    after = actions[events[-1][3]:]                      # steps after the deposit
    assert (after == NOOP).all() and (actions[:, 0] == NOOP).all()


def test_collector_idles_when_its_target_is_taken():
    seed = HELDOUT_SEEDS[5]
    agents = [ScriptedTeammate(0, "collector", target=1, seed=seed), ScriptedTeammate(1, "collector", target=1, seed=seed)]
    r, actions, events = run("L0", agents, seed, steps=300)
    picks = [e for e in events if e[0] == "pick"]
    assert len(picks) == 1 and picks[0][2] == 1
    loser = 1 - picks[0][1]
    assert (actions[picks[0][3]:, loser] == NOOP).all()


def test_collector_without_target_picks_a_random_available_cube():
    targets = []
    for _ in range(2):
        with ForagingEnv({"level": "L2", "render": False}, seed=0) as env:
            env.reset(HELDOUT_SEEDS[6])
            agent = ScriptedTeammate(1, "collector", seed=3)
            agent.reset(env)
            assert env.available()[agent.target]
            targets.append(agent.target)
    assert targets[0] == targets[1]


def test_wanderer_moves_but_never_picks_or_deposits():
    seed = HELDOUT_SEEDS[8]
    agents = [ScriptedTeammate(0, "idle"), ScriptedTeammate(1, "wanderer", seed=seed)]
    with ForagingEnv({"level": "L2", "render": False}, seed=0) as env:
        obs, info = env.reset(seed)
        for agent in agents:
            agent.reset(env)
        path, events = 0.0, []
        for _ in range(300):
            before = env.robot_poses()[1, :2]
            obs, reward, done, info = env.step([agent.act(env, info) for agent in agents])
            path += float(np.hypot(*(env.robot_poses()[1, :2] - before)))
            events += info["events"]
    assert not events and path > 8.0


def test_unknown_behaviour_raises():
    with pytest.raises(ValueError):
        ScriptedTeammate(0, "dancer")
