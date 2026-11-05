"""Reference baselines on held-out layouts: uniformly random actions and the heuristic forager.

Usage, from the repo root:
    python scripts/eval_baselines.py [--policies random heuristic] [--levels L0 L1 L2] [--robots 2]
                                     [--episodes 100] [--noise 0.1]

Episode k runs on layout seed HELDOUT_SEEDS[k], with policy seed HELDOUT_SEEDS[k] (random actions,
or the heuristic's action noise), so episodes are reproducible one by one. Each (policy, level)
pair runs in its own env with env seed 0, episodes in order. Rendering is off: neither baseline
reads images (the heuristic acts on privileged detections; see README, "Scripted agents").

Reported per (policy, level): the fraction of all cubes delivered, the per-episode mean and
minimum of that fraction, the fraction of episodes with every cube delivered, and the mean
episode length in steps.
"""
import argparse
import time

import numpy as np

from foraging_env import HELDOUT_SEEDS, N_ACTIONS, ForagingEnv, HeuristicForager

POLICIES = ("random", "heuristic")


def make_policy(name: str, env: ForagingEnv, seed: int, noise: float):
    """A function from the last step's info to the joint action, for one episode."""
    if name == "random":
        rng = np.random.default_rng(seed)
        return lambda info: rng.integers(N_ACTIONS, size=env.n)
    agents = [HeuristicForager(i, seed, noise) for i in range(env.n)]
    for agent in agents:
        agent.reset(env)
    return lambda info: [agent.act(env, info) for agent in agents]


def evaluate(policy: str, level: str, episodes: int, robots: int, noise: float) -> dict:
    env = ForagingEnv({"level": level, "n_robots": robots, "render": False}, seed=0)
    delivered = total = 0
    frac, steps = [], []
    t0 = time.perf_counter()
    for seed in HELDOUT_SEEDS[:episodes]:
        obs, info = env.reset(seed)
        act = make_policy(policy, env, seed, noise)
        done = False
        while not done:
            obs, reward, done, info = env.step(act(info))
        n, d = int(env.exists.sum()), int(env.deposited.sum())
        delivered, total = delivered + d, total + n
        frac.append(d / n)
        steps.append(env.t)
    env.close()
    frac = np.array(frac)
    return dict(policy=policy, level=level, delivered=delivered / total, per_episode=float(frac.mean()),
                min=float(frac.min()), all=float(np.mean(frac == 1)), steps=float(np.mean(steps)),
                secs=time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2"])
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.1, help="heuristic forager: probability of a random action")
    args = ap.parse_args()
    if args.episodes > len(HELDOUT_SEEDS):
        ap.error(f"at most {len(HELDOUT_SEEDS)} held-out layouts")
    print(f"{args.robots} robots, held-out layout seeds {HELDOUT_SEEDS.start}..{HELDOUT_SEEDS.start + args.episodes - 1}, "
          f"heuristic noise {args.noise}\n")
    print("| policy | level | delivered | per-episode mean | min | all delivered | mean steps | time |")
    print("|---|---|---|---|---|---|---|---|")
    for policy in args.policies:
        for level in args.levels:
            r = evaluate(policy, level, args.episodes, args.robots, args.noise)
            print(f"| {policy} | {level} | {r['delivered']:.3f} | {r['per_episode']:.3f} | {r['min']:.2f} | "
                  f"{r['all']:.2f} | {r['steps']:.0f} | {r['secs']:.0f} s |", flush=True)
