"""Curriculum levels, the layout sampler and the probes."""
import numpy as np
import pytest

from foraging_env import (HELDOUT_SEEDS, LEVELS, S2_CONDITIONS, ForagingEnv, resolve_cfg, s1_persistence, s2_ambiguity,
                          s3_full, sample_layout)
from foraging_env.constants import CUBE_HALF, NEST_RADIUS

SEEDS = list(HELDOUT_SEEDS[:3])


def count_range(v):
    return (v, v) if np.isscalar(v) else tuple(v)


# -------------------------------------------------------------------------- levels

def test_level_presets_and_precedence():
    cfg = resolve_cfg({"level": "L2", "n_robots": 4, "max_steps": 300})
    assert cfg["arena_size"] == 8.0 and cfg["nest"] == "random" and cfg["near_occluder_frac"] == 0.5
    assert cfg["n_robots"] == 4 and cfg["max_steps"] == 300        # explicit keys win
    assert resolve_cfg({"level": "L0"})["dense_shaping"] and not resolve_cfg({"level": "L1"})["dense_shaping"]
    assert resolve_cfg({"level": "L0", "dense_shaping": False})["dense_shaping"] is False


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_sampled_layouts_follow_the_level(level):
    cfg = resolve_cfg({"level": level})
    half = cfg["arena_size"] / 2
    ranges = {k: count_range(cfg[k]) for k in ("n_resources", "n_distractors", "n_occluders")}
    azimuths = set()
    for seed in HELDOUT_SEEDS[:60]:
        lay = sample_layout(cfg, np.random.default_rng(seed))
        res = lay.resource_pos
        for key, n in (("n_resources", len(res)), ("n_distractors", len(lay.distractor_pos)),
                       ("n_occluders", len(lay.occluders))):
            assert ranges[key][0] <= n <= ranges[key][1]
        # Every cube can be reached: clear of walls, occluders and the nest by the sampler's margins.
        assert (np.abs(res) <= half - CUBE_HALF - 0.45 + 1e-9).all()
        for o in lay.occluders:
            assert (o.distance(res[:, 0], res[:, 1]) >= 0.45 + CUBE_HALF - 1e-9).all()
        assert (np.hypot(*(res - lay.nest_pos).T) >= NEST_RADIUS + CUBE_HALF + 0.2 - 1e-9).all()
        if cfg["near_occluder_frac"] and lay.occluders:
            near = [min(o.distance(x, y) for o in lay.occluders) <= cfg["near_occluder_dist"] for x, y in res]
            assert sum(near) >= round(cfg["near_occluder_frac"] * len(res))
        azimuths.add(round(float(np.arctan2(lay.light_dir[1], lay.light_dir[0])), 3))
    assert (len(azimuths) > 30) == (cfg["light_azimuth_jitter_deg"] > 0)   # lighting jitter on L2 only


def test_sampling_is_deterministic():
    cfg = resolve_cfg({"level": "L2"})
    a, b = (sample_layout(cfg, np.random.default_rng(HELDOUT_SEEDS[7])) for _ in range(2))
    assert np.array_equal(a.resource_pos, b.resource_pos) and np.array_equal(a.robot_poses, b.robot_poses)
    assert a.occluders == b.occluders and np.array_equal(a.light_dir, b.light_dir)


# -------------------------------------------------------------------------- probes

def _run_probe(probe, cfg):
    """Run the scripted steps; robot 0's visibility of the target and the events, per step."""
    with ForagingEnv({**cfg, "render": False}, seed=0) as env:
        obs, info = env.reset(HELDOUT_SEEDS[0], layout=probe.layout)
        start_yaw = env.robot_poses()[0, 2]
        vis, events = [bool(info["labels"][0]["visible"][probe.target])], []
        for step in range(1, probe.release_step + 1):
            probe.before_step(env, step)
            obs, reward, done, info = env.step(probe.actions(step))
            vis.append(bool(info["labels"][0]["visible"][probe.target]))
            events += info["events"]
        lab = info["labels"][0]
        end = dict(ever_seen=bool(lab["ever_seen"][probe.target]), distance=float(lab["distance"][probe.target]),
                   bearing=float(lab["bearing"][probe.target]),
                   turned=float(np.degrees(np.angle(np.exp(1j * (env.robot_poses()[0, 2] - start_yaw))))))
    return np.array(vis), events, end


@pytest.mark.parametrize("seed", SEEDS)
def test_s1_persistence(seed):
    cfg = resolve_cfg({"level": "L2"})
    probe = s1_persistence(cfg, seed)
    vis, events, end = _run_probe(probe, cfg)
    assert probe.release_step == 59 and len(probe.layout.resource_pos) == 1
    assert vis[:6].all()                  # sees the target before turning
    assert not vis[26:].any()             # turned away, then driving off
    assert end["ever_seen"] and 4.0 < end["distance"] < 4.1 and abs(end["bearing"]) > np.radians(179)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("condition", S2_CONDITIONS)
def test_s2_ambiguity(condition, seed):
    cfg = resolve_cfg({"level": "L2"})
    probe = s2_ambiguity(cfg, seed, condition)
    vis, events, end = _run_probe(probe, cfg)
    assert probe.release_step == 40 and vis[:19].all()
    assert (("pick", 1, 0, 30) in events) == condition.startswith("collect")
    if condition.endswith("hidden"):
        assert not vis[23:36].any()                      # robot 0 looks away during the pick
    if condition == "no_collect_hidden":
        assert vis[40]                                   # ... and sees the target again at release
    if condition == "no_collect_visible":
        assert vis.all()
    if condition == "occluded_present":                  # occluder B hides the target
        assert vis[:25].all() and not vis[25:41].any()
    assert abs(end["turned"]) < 3                        # robot 0 faces the target again at release


def test_s2_conditions_share_a_layout():
    cfg = resolve_cfg({"level": "L2"})
    layouts = [s2_ambiguity(cfg, SEEDS[0], c).layout for c in S2_CONDITIONS]
    for lay in layouts[1:]:
        assert np.array_equal(lay.resource_pos, layouts[0].resource_pos) and lay.occluders == layouts[0].occluders


def test_s3_is_the_reset_layout():
    cfg = resolve_cfg({"level": "L2"})
    probe = s3_full(cfg, SEEDS[1])
    with ForagingEnv({"level": "L2", "render": False}, seed=0) as env:
        env.reset(SEEDS[1])
        assert np.array_equal(env.layout.resource_pos, probe.layout.resource_pos)
        assert env.layout.occluders == probe.layout.occluders


def test_probes_need_the_large_arena():
    with pytest.raises(RuntimeError):
        s1_persistence(resolve_cfg({"level": "L0"}), SEEDS[0])


def test_set_occluder_pose_updates_layout_and_labels():
    cfg = resolve_cfg({"level": "L2"})
    probe = s2_ambiguity(cfg, SEEDS[0], "occluded_present")
    k, pos, yaw = probe.occluder_moves[25]
    with ForagingEnv({**cfg, "render": False}, seed=0) as env:
        obs, info = env.reset(layout=probe.layout)
        assert info["labels"][0]["visible"][0]
        env.set_occluder_pose(k, pos, yaw)
        assert env.layout.occluders[k].pos == pytest.approx(pos) and env.layout is not probe.layout
        obs, reward, done, info = env.step([0, 0])
        assert not info["labels"][0]["visible"][0]
