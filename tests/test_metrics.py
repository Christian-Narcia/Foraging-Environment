"""Task metrics (foraging_env/metrics.py): wasted revisits on hand-built episodes, and
EpisodeStats following real episodes of the heuristic forager."""
import numpy as np

from foraging_env import HELDOUT_SEEDS, ForagingEnv, HeuristicForager
from foraging_env.constants import R_MAX
from foraging_env.env import LABEL_DTYPES, priv_state_dim
from foraging_env.metrics import EpisodeStats, split_priv_state, summarize_episodes

N = 2
D = priv_state_dim(N)
CUBE = (2.0, 0.0)

# A hand-built episode of 8 states with one cube at CUBE. State 0: robot 0 sees the cube. Step 1:
# robot 1 picks it and carries it from then on. Robot 0 enters the 1 m disc around CUBE at
# steps 2 (revisit 1) and 5 (revisit 2), stays at step 3, leaves at 4 and 6, and at step 6 sees
# the cube on robot 1, so entering again at step 7 is not a wasted revisit.
R0 = [(0, 0), (0, 0), (1.5, 0), (1.5, 0), (0, 0), (1.2, 0), (0, 0), (1.5, 0)]
R1 = [(2.3, 0), (2.3, 0), (2.3, 0), (3.5, 0), (3.5, 0), (2.0, 0), (3.0, 0), (3.0, 0)]
R0_INSIDE = [(0, 0), (1.5, 0), (1.5, 0), (1.5, 0), (0, 0), (1.2, 0), (0, 0), (1.5, 0)]   # there at the pick


def episode(r0=R0, seen0=True, carry0=False, sees_pick=False, sees_entry=False, picker_returns=False):
    """The states of the hand-built episode, with variations for robot 0: it never saw the
    cube before the pick (seen0=False), carries a cube of its own throughout, sees the cube
    carried right after the pick, or sees it carried at step 2, as it enters the disc
    (sees_entry). picker_returns: robot 1 has seen the cube, is empty-handed from step 3 on (as
    if it had delivered) and enters the disc at step 5; it took the cube, so that is no wasted
    revisit."""
    frames, ever = [], False
    for k, (p0, p1) in enumerate(zip(r0, R1)):
        vis0 = (k == 0 and seen0) or (k == 1 and sees_pick) or (k == 2 and sees_entry) or k == 6
        ever |= vis0
        frames.append(dict(robots=(p0, p1), avail=k == 0, vis=(vis0, False), seen=(ever, picker_returns),
                           carry=(carry0, k > 0 and not (picker_returns and k >= 3)),
                           events=[("pick", 1, 0, 1)] if k == 1 else []))
    return frames


def to_info(frames, scripted):
    """One state of each env as a vector env reports it (priv_state, labels, events, scripted).
    The carried cube rides on robot 1."""
    E = len(frames)
    priv = np.zeros((E, D), np.float32)
    labels = {k: np.zeros((E, N, R_MAX), dt) for k, dt in LABEL_DTYPES.items()}
    for e, f in enumerate(frames):
        for i, (x, y) in enumerate(f["robots"]):
            priv[e, 5 * i:5 * i + 5] = x, y, 1.0, 0.0, f["carry"][i]
        priv[e, 5 * N:5 * N + 2] = CUBE if f["avail"] else f["robots"][1]
        labels["exists"][e, :, 0] = True
        labels["available"][e, :, 0] = f["avail"]
        labels["visible"][e, :, 0] = f["vis"]
        labels["ever_seen"][e, :, 0] = f["seen"]
    return dict(priv_state=priv, labels=labels, events=[f["events"] for f in frames], scripted=np.array(scripted))


def run(scenarios, scripted):
    """Feed equal-length state lists, one env each: state 0 to reset(), the others as steps, the
    last of which ends every episode (auto-reset to state 0). Returns the episodes' records."""
    E, K = len(scenarios), len(scenarios[0])
    stats = EpisodeStats(E, N)
    stats.reset(to_info([s[0] for s in scenarios], scripted))
    for k in range(1, K):
        done = k == K - 1
        step = to_info([s[k] for s in scenarios], scripted)
        info = to_info([s[0] for s in scenarios], scripted) if done else step
        info = dict(info, events=step["events"],
                    final_info=dict(step, delivered=np.full(E, 0.25, np.float32), t=np.full(E, k)))
        episodes = stats.step(np.full(E, done), info)
        assert len(episodes) == (E if done else 0)
    return episodes


CASES = {   # name: (episode() arguments, robot 0 scripted, wasted revisits)
    "taken out of sight": ({}, False, 2),
    "never seen": ({"seen0": False}, False, 0),
    "first seen carried, on entry": ({"seen0": False, "sees_entry": True}, False, 0),   # not seen before moving in
    "seen carried away": ({"sees_pick": True}, False, 0),
    "carrying": ({"carry0": True}, False, 0),
    "scripted robot": ({}, True, 0),
    "inside at the pick": ({"r0": R0_INSIDE}, False, 1),
    "picker comes back": ({"picker_returns": True}, False, 2),
}


def test_wasted_revisits():
    """Every case in one batch, and each alone: the counts follow the definition in
    foraging_env/metrics.py, and envs don't affect each other."""
    names = list(CASES)
    scripted = [(CASES[n][1], False) for n in names]
    episodes = run([episode(**CASES[n][0]) for n in names], scripted)
    for name, s, ep in zip(names, scripted, episodes):
        assert ep["wasted_revisits"] == CASES[name][2], name
        assert ep == dict(delivered=0.25, length=7, picks=1, wasted_revisits=CASES[name][2], scripted=s[0])
        assert run([episode(**CASES[name][0])], [s]) == [ep]
    summary = summarize_episodes(episodes)
    assert summary["episode/wasted_revisits"] == 5 / len(names) and summary["episode/count"] == len(names)
    assert summarize_episodes([]) == {}


def as_vector(info):
    """A ForagingEnv info as a vector env of one reports it."""
    return dict(priv_state=info["priv_state"][None], events=[info["events"]], scripted=np.zeros((1, N), bool),
                labels={f: np.stack([lab[f] for lab in info["labels"]])[None] for f in LABEL_DTYPES})


def test_episode_stats_follow_real_episodes():
    """Fed one ForagingEnv as a vector env of one, EpisodeStats
    reports the length, delivery fraction and pick count the env has; split_priv_state reads the
    robot and cube positions the env has."""
    env = ForagingEnv({"level": "L1", "render": False}, seed=0)
    for seed in HELDOUT_SEEDS[:2]:
        _, info = env.reset(seed)
        agents = [HeuristicForager(i, seed) for i in range(N)]
        for agent in agents:
            agent.reset(env)
        stats = EpisodeStats(1, N)
        stats.reset(as_vector(info))
        picks, done = 0, False
        while not done:
            _, _, done, info = env.step([agent.act(env, info) for agent in agents])
            picks += sum(ev[0] == "pick" for ev in info["events"])
            robot_xy, carrying, res_xy = split_priv_state(info["priv_state"], N)
            assert np.allclose(robot_xy, env.robot_poses()[:, :2]) and np.array_equal(carrying, env.carrying >= 0)
            assert np.allclose(res_xy[env.exists], env.resource_positions()[env.exists, :2])
            v = as_vector(info)
            v["final_info"] = dict(v, delivered=np.array([env.deposited.sum() / env.exists.sum()], np.float32),
                                   t=np.array([info["t"]]))
            episodes = stats.step(np.array([done]), v)
        (ep,) = episodes
        assert ep["length"] == env.t and ep["picks"] == picks > 0 and not ep["scripted"]
        assert np.isclose(ep["delivered"], env.deposited.sum() / env.exists.sum())
        assert ep["wasted_revisits"] >= 0
    env.close()
