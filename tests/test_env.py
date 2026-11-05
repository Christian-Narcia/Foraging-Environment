"""ForagingEnv tests. The numbered tests (1-13) are the core specification checks: spaces,
determinism, pick, carry, deposit, team reward, episode end, visibility, ever_seen, bearing and
distance, the seed split and the vector env. The others cover dynamics, configuration and
rendering. Rendering tests need a GL backend (EGL or OSMesa on Linux)."""
import numpy as np
import pytest
from helpers import compare_with_reference, env_on, make_layout, reference_run

from foraging_env import (BACKWARD, DEPOSIT, FORWARD, HELDOUT_SEEDS, N_ACTIONS, NOOP, PICK, TRAIN_SEEDS, TURN_LEFT,
                          ForagingEnv, Occluder, VectorForagingEnv)
from foraging_env.constants import CARRY_Z, PILE_SLOTS, PILE_Z, R_MAX, ROBOT_RADIUS
from foraging_env.env import LABEL_DTYPES, priv_state_dim, priv_state_ego_index


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


# ------------------------------------------------------------- 1. shapes and dtypes

@pytest.mark.parametrize("n,img,render", [(1, 64, True), (2, 64, True), (4, 64, True), (2, 48, True), (4, 128, True),
                                          (2, 64, False)])
def test_1_shapes_and_dtypes(n, img, render):
    with ForagingEnv({"level": "L2", "n_robots": n, "img_size": img, "render": render}, seed=0) as env:
        obs, info = env.reset(HELDOUT_SEEDS[0])
        for step in range(3):
            if step:
                obs, reward, done, info = env.step(np.full(n, FORWARD))
                assert reward.shape == (n,) and reward.dtype == np.float32
                assert isinstance(done, bool)
            if render:
                assert obs["rgb"].shape == (n, img, img, 3) and obs["rgb"].dtype == np.uint8
                assert obs["rgb"].any()
            else:
                assert obs["rgb"] is None
            assert obs["prev_action"].shape == (n,) and obs["prev_action"].dtype == np.int64
            assert obs["carrying"].shape == (n,) and obs["carrying"].dtype == bool
            assert info["priv_state"].shape == (priv_state_dim(n),) and info["priv_state"].dtype == np.float32
            assert len(info["labels"]) == n
            for lab in info["labels"]:
                assert set(lab) == set(LABEL_DTYPES)
                for f, dt in LABEL_DTYPES.items():
                    assert lab[f].shape == (R_MAX,) and lab[f].dtype == dt
            assert info["t"] == step and info["layout_seed"] == HELDOUT_SEEDS[0] and info["truncated"] is False
            assert isinstance(info["events"], list)


# ---------------------------------------------------------------- 2. determinism

def _rollout(env_seed, layout_seed, n, steps, render):
    rng = np.random.default_rng(7)
    with ForagingEnv({"level": "L2", "n_robots": n, "render": render}, seed=env_seed) as env:
        out = [env.reset(layout_seed)]
        for a in rng.integers(N_ACTIONS, size=(steps, n)):
            out.append(env.step(a))
    return out


def test_2_determinism_given_seeds():
    a = _rollout(3, HELDOUT_SEEDS[1], 2, 150, render=True)
    b = _rollout(3, HELDOUT_SEEDS[1], 2, 150, render=True)
    differing = 0
    for x, y in zip(a, b):
        ox, oy, ix, iy = x[0], y[0], x[-1], y[-1]
        assert np.array_equal(ox["prev_action"], oy["prev_action"]) and np.array_equal(ox["carrying"], oy["carrying"])
        assert np.array_equal(ix["priv_state"], iy["priv_state"]) and ix["events"] == iy["events"]
        for lx, ly in zip(ix["labels"], iy["labels"]):
            assert all(np.array_equal(lx[f], ly[f]) for f in LABEL_DTYPES)
        if len(x) == 4:
            assert np.array_equal(x[1], y[1]) and x[2] == y[2]
        d = np.abs(ox["rgb"].astype(int) - oy["rgb"])   # two GL contexts: up to 1 level (README)
        assert d.max() <= 1
        differing += int((d > 0).sum())
    assert differing <= 1e-3 * a[0][0]["rgb"].size * len(a)


def test_2_reset_without_seed_is_reproducible():
    seqs = []
    for steps in (0, 37):   # the layout sequence must not depend on what happens in episodes
        with ForagingEnv({"level": "L0", "render": False}, seed=11) as env:
            seq = []
            for _ in range(5):
                seq.append(env.reset()[1]["layout_seed"])
                for _ in range(steps):
                    env.step([NOOP, NOOP])
            seqs.append(seq)
    assert seqs[0] == seqs[1]


def test_rgb_replay_is_bit_exact_within_an_env():
    rng = np.random.default_rng(3)
    actions = rng.integers(N_ACTIONS, size=(120, 4))
    with ForagingEnv({"level": "L2", "n_robots": 4}, seed=0) as env:
        runs = []
        for _ in range(2):
            frames = [env.reset(HELDOUT_SEEDS[5])[0]["rgb"]]
            frames += [env.step(a)[0]["rgb"] for a in actions]
            runs.append(np.array(frames))
    assert np.array_equal(runs[0], runs[1])


# ------------------------------------------------------------------------- dynamics

def test_dynamics_speeds_and_walls():
    env, obs, info = env_on(make_layout([(-1.0, 0.0, 0.0)], [(1.5, 1.5)]))
    with env:
        def moves(action, k):
            out = []
            for _ in range(k):
                p0 = env.robot_poses()[0]
                env.step([action])
                p1 = env.robot_poses()[0]
                out.append((np.hypot(*(p1[:2] - p0[:2])), np.degrees(wrap(p1[2] - p0[2]))))
            return np.array(out)
        fwd = moves(FORWARD, 8)
        assert 0.05 < fwd[0, 0] < 0.06 and np.allclose(fwd[3:, 0], 0.06, atol=1e-3)
        stop = moves(NOOP, 3)
        assert stop[-1, 0] < 1e-3
        turn = moves(TURN_LEFT, 8)
        assert np.allclose(turn[3:, 1], 9.0, atol=0.05) and np.allclose(turn[:, 0], 0, atol=1e-3)
        moves(NOOP, 3)
        back = moves(BACKWARD, 8)
        assert np.allclose(back[3:, 0], 0.06, atol=1e-3)
    # Driving into the +x wall (face at 2.5): the robot stops with its edge at the wall.
    env, obs, info = env_on(make_layout([(1.0, -1.5, 0.0)], [(-1.5, 1.5)]))
    with env:
        for _ in range(60):
            env.step([FORWARD])
        x = env.robot_poses()[0, 0]
    assert 2.5 - ROBOT_RADIUS - 0.01 < x < 2.5 - ROBOT_RADIUS + 0.01


# ----------------------------------------------------- 3. pick range and angle

@pytest.mark.parametrize("dist,angle_deg,picks", [(0.33, 0, True), (0.37, 0, False), (0.32, 25, True),
                                                  (0.32, -25, True), (0.32, 35, False), (0.32, -35, False)])
def test_3_pick_only_within_range_and_angle(dist, angle_deg, picks):
    a = np.radians(angle_deg)
    env, obs, info = env_on(make_layout([(0.0, 0.0, 0.0)], [(dist * np.cos(a), dist * np.sin(a)), (1.8, 1.8)]))
    with env:
        obs, reward, done, info = env.step([PICK])
    assert obs["carrying"][0] == picks
    if picks:
        assert info["events"] == [("pick", 0, 0, 1)]
        assert reward[0] == pytest.approx(0.2 - 0.001)
        assert not info["labels"][0]["available"][0] and info["labels"][0]["carried_by"][0] == 0
    else:
        assert info["events"] == [] and reward[0] == pytest.approx(-0.001)


# ----------------------------------------------- 4. carried item follows the mount

def test_4_carried_item_follows_mount():
    env, obs, info = env_on(make_layout([(0.0, 0.0, 0.3)], [(0.3 * np.cos(0.3), 0.3 * np.sin(0.3)), (1.8, 1.8)]))
    with env:
        obs, *_ = env.step([PICK])
        assert obs["carrying"][0]
        for a in [FORWARD] * 6 + [TURN_LEFT] * 6 + [BACKWARD] * 4 + [NOOP]:
            obs, reward, done, info = env.step([a])
            x, y, yaw = env.robot_poses()[0]
            assert np.array_equal(env.resource_positions()[0], [x, y, CARRY_Z])
            assert info["labels"][0]["carried_by"][0] == 0 and not info["labels"][0]["available"][0]


# ------------------------------- 5-7. deposit, team reward, episode end (one scenario)

def _carry_to_nest(n_cubes):
    """Robot 0 picks cube 0 at x=-1.18, drives into the nest at (1, 0) and deposits; robot 1
    idles outside the nest. Returns the records of the interesting steps."""
    cubes = [(-1.18, 0.0), (-1.5, -1.8)][:n_cubes]
    env, obs, info = env_on(make_layout([(-1.5, 0.0, 0.0), (-1.5, 1.6, 0.0)], cubes, nest=(1.0, 0.0)))
    rec = {}
    with env:
        rec["empty_deposit"] = env.step([DEPOSIT, DEPOSIT])
        rec["pick"] = env.step([PICK, NOOP])
        rec["outside_deposit"] = env.step([DEPOSIT, NOOP])
        for _ in range(25):
            env.step([FORWARD, NOOP])
        rec["pose_before_deposit"] = env.robot_poses()
        rec["deposit"] = env.step([DEPOSIT, NOOP])
        rec["cube_pos"] = env.resource_positions()[0]
        rec["deposited"], rec["t"] = env.deposited.copy(), env.t
    return rec


def test_5_deposit_only_inside_nest_while_carrying():
    rec = _carry_to_nest(1)
    obs, reward, done, info = rec["empty_deposit"]
    assert info["events"] == [] and not obs["carrying"].any()
    obs, reward, done, info = rec["outside_deposit"]
    assert info["events"] == [] and obs["carrying"][0]
    assert np.hypot(*(rec["pose_before_deposit"][0, :2] - (1.0, 0.0))) < 1.2
    obs, reward, done, info = rec["deposit"]
    assert info["events"] == [("deposit", 0, 0, rec["t"])] and not obs["carrying"][0]
    assert rec["deposited"][0]
    assert np.allclose(rec["cube_pos"], [1.0 + PILE_SLOTS[0, 0], PILE_SLOTS[0, 1], PILE_Z])
    lab = info["labels"][1]
    assert not lab["available"][0] and lab["carried_by"][0] == -1


def test_6_team_reward_on_deposit():
    rec = _carry_to_nest(2)
    obs, reward, done, info = rec["deposit"]
    assert reward == pytest.approx([1.0 - 0.001, 1.0 - 0.001])   # every robot, not just the depositor
    obs, reward, done, info = rec["pick"]
    assert reward == pytest.approx([0.2 - 0.001, -0.001])         # the pick bonus is individual


def test_7_episode_ends_when_all_deposited():
    for n_cubes, done_after in ((1, True), (2, False)):
        obs, reward, done, info = _carry_to_nest(n_cubes)["deposit"]
        assert done == done_after and not info["truncated"]


def test_truncation_at_max_steps():
    with ForagingEnv({"level": "L0", "render": False, "max_steps": 5}, seed=0) as env:
        env.reset(HELDOUT_SEEDS[0])
        outs = [env.step([NOOP, NOOP]) for _ in range(5)]
    assert [o[2] for o in outs] == [False] * 4 + [True]
    assert outs[-1][3]["truncated"] and not outs[-2][3]["truncated"]


def test_priv_state_layout():
    env, obs, info = env_on(make_layout([(0.0, 0.0, 0.4), (-1.0, 1.0, -1.0)],
                                        [(0.3 * np.cos(0.4), 0.3 * np.sin(0.4)), (1.5, -1.0)], nest=(1.5, 1.5)))
    with env:
        obs, reward, done, info = env.step([PICK, NOOP])
        p = info["priv_state"]
        pose = env.robot_poses()
    robots = p[:10].reshape(2, 5)
    assert np.allclose(robots[:, :2], pose[:, :2], atol=1e-6)
    assert np.allclose(robots[:, 2], np.cos(pose[:, 2]), atol=1e-6) and np.allclose(robots[:, 3], np.sin(pose[:, 2]), atol=1e-6)
    assert list(robots[:, 4]) == [1.0, 0.0]
    res = p[10:10 + R_MAX * 6].reshape(R_MAX, 6)
    assert list(res[0, 2:]) == [0.0, 0.0, 1.0, 0.0]               # carried by robot 0
    assert np.allclose(res[1, :2], [1.5, -1.0]) and list(res[1, 2:]) == [1.0, 0.0, 0.0, 0.0]
    assert not res[2:].any()
    assert np.allclose(p[-2:], [1.5, 1.5])


def test_priv_state_ego_index_renumbers_robots():
    """priv_state[idx[i]] equals the priv_state of the same scene with robot i first and the
    others after it in order; robot 1 carries cube 1, so the carried_by one-hots move too."""
    poses = [(0.0, 0.0, 0.0), (1.5, 1.0, 0.0), (-1.5, -1.0, 0.0)]
    cubes = [(0.3, 0.0), (1.8, 1.0), (-1.0, 1.5)]

    def priv(order):      # robots renumbered: new robot k is old robot order[k]; old robot 1 picks
        env, obs, info = env_on(make_layout([poses[j] for j in order], cubes, nest=(-1.5, 1.5)))
        with env:
            return env.step([PICK if j == 1 else NOOP for j in order])[3]["priv_state"]

    p, idx = priv([0, 1, 2]), priv_state_ego_index(3)
    assert idx.shape == (3, priv_state_dim(3)) and np.array_equal(idx[0], np.arange(priv_state_dim(3)))
    assert list(p[15:15 + R_MAX * 7].reshape(R_MAX, 7)[1, 4:]) == [0.0, 1.0, 0.0]   # cube 1 carried by robot 1
    for i in range(3):
        assert sorted(idx[i]) == list(range(len(p)))
        assert np.allclose(p[idx[i]], priv([i] + [j for j in range(3) if j != i]), atol=1e-6)


# ---------------------------------------------------------------- 8-9. visibility

def _visible(robots, cubes, occluders=(), arena=8.0):
    env, obs, info = env_on(make_layout(robots, cubes, occluders, nest=(3.0, 3.0), arena=arena))
    env.close()
    return info["labels"][0]["visible"][:len(cubes)]


def test_8_visible_false_behind_occluder():
    wall = Occluder("box", (0.1, 0.4, 0.4), (-2.0, 0.0), 0.0)
    assert not _visible([(-3.0, 0.0, 0.0)], [(-1.0, 0.0)], [wall])[0]


def test_8_visible_false_behind_teammate():
    # Robot 1's base (0.15 m tall) cuts the sight line from robot 0's camera to the cube.
    assert not _visible([(-3.0, 0.0, 0.0), (-1.3, 0.0, 0.0)], [(-1.0, 0.0)])[0]


def test_9_visible_true_when_unobstructed_and_in_fov():
    assert _visible([(-3.0, 0.0, 0.0)], [(-1.0, 0.0)])[0]


@pytest.mark.parametrize("cube,expected", [((-3.6, 0.0), False),     # behind the camera
                                           ((-2.0, 1.8), False),     # 53 deg to the left, outside the 90 deg FOV
                                           ((-1.5, 1.0), True),      # 28 deg to the left
                                           ((2.3, 0.0), True),       # 5.65 m from the camera
                                           ((2.8, 0.0), False)])     # 6.15 m, beyond fov_range
def test_9_visible_needs_fov_and_range(cube, expected):
    assert _visible([(-3.5, 0.0, 0.0)], [cube])[0] == expected


# ---------------------------------------------------------------- 10. ever_seen

def test_10_ever_seen_monotone():
    rng = np.random.default_rng(5)
    with ForagingEnv({"level": "L2", "n_robots": 2, "render": False}, seed=0) as env:
        obs, info = env.reset(HELDOUT_SEEDS[3])
        seen = np.array([lab["visible"] for lab in info["labels"]])
        steps_with_new = 0
        for a in rng.choice([FORWARD, FORWARD, TURN_LEFT, NOOP], size=(300, 2)):
            prev = np.array([lab["ever_seen"] for lab in info["labels"]])
            obs, reward, done, info = env.step(a)
            ever = np.array([lab["ever_seen"] for lab in info["labels"]])
            vis = np.array([lab["visible"] for lab in info["labels"]])
            seen |= vis
            assert (ever >= prev).all() and np.array_equal(ever, seen)
            steps_with_new += bool((ever > prev).any())
    assert steps_with_new > 0


# --------------------------------------------------------- 11. bearing and distance

def test_11_bearing_and_distance_match_analytic():
    robots = [(0.5, -0.3, 0.7), (-1.2, 1.0, -2.5)]
    cubes = np.array([(1.5, 0.4), (-1.0, -1.2), (0.2, 1.8)])
    env, obs, info = env_on(make_layout(robots, cubes))
    env.close()
    for (x, y, yaw), lab in zip(robots, info["labels"]):
        d = cubes - (x, y)
        assert np.allclose(lab["distance"][:3], np.hypot(d[:, 0], d[:, 1]), atol=1e-5)
        assert np.allclose(wrap(lab["bearing"][:3] - (np.arctan2(d[:, 1], d[:, 0]) - yaw)), 0, atol=1e-5)
        assert not lab["distance"][3:].any() and not lab["bearing"][3:].any() and not lab["exists"][3:].any()


# ------------------------------------------------------------------ 12. seed split

def test_12_heldout_seeds_disjoint_from_train():
    assert not set(TRAIN_SEEDS) & set(HELDOUT_SEEDS)
    assert len(HELDOUT_SEEDS) == 200 and min(HELDOUT_SEEDS) > max(TRAIN_SEEDS)
    with ForagingEnv({"level": "L0", "render": False}, seed=2) as env:
        assert all(env.reset()[1]["layout_seed"] in TRAIN_SEEDS for _ in range(50))


# ------------------------------------------------------ 13. vector env matches single env

def test_13_vector_env_matches_single_env_step_for_step():
    cfg = {"level": "L0"}
    with VectorForagingEnv(3, cfg, n_workers=2, seed=0) as venv:
        refs = [reference_run(cfg, venv.env_seeds[k], 250, policy_seed=k) for k in range(venv.n_envs)]
        bad, stats = compare_with_reference(venv, refs)
    assert not bad, bad
    assert stats["episodes"] >= 3 and stats["events"] >= 20   # auto-resets and picks/deposits were exercised


# -------------------------------------------------------------- random-policy sanity

def test_random_policy_delivers_little_on_l2():
    """Reactive sanity: one robot acting uniformly at random delivers < 10% on L2 (100 held-out layouts)."""
    rng = np.random.default_rng(0)
    delivered = total = 0
    with ForagingEnv({"level": "L2", "n_robots": 1, "render": False}, seed=0) as env:
        for seed in HELDOUT_SEEDS[:100]:
            env.reset(seed)
            while not env.step(rng.integers(N_ACTIONS, size=1))[2]:
                pass
            delivered, total = delivered + env.deposited.sum(), total + env.exists.sum()
    assert delivered / total < 0.10


# --------------------------------------------------------------------- configuration

def test_cfg_validation():
    for cfg in ({"nope": 1}, {"n_robots": 5}, {"n_robots": 0}, {"level": "L9"}, {"n_resources": 9},
                {"nest": "corner"}, {"near_occluder_frac": 1.5}):
        with pytest.raises(ValueError):
            ForagingEnv(cfg, seed=0)
    with ForagingEnv({"n_robots": 2, "render": False}, seed=0) as env:
        with pytest.raises(ValueError):
            env.reset(layout=make_layout([(0, 0, 0)], [(1, 1)]))   # 1 robot in a 2-robot env


# ------------------------------------------------------------------------- rendering

def _colour_pixels(img, colour):
    r, g, b = (img[..., c].astype(int) for c in range(3))
    mask = (r > 2 * g) & (r > 2 * b) & (r > 60) if colour == "red" else (b > 2 * r) & (b > 1.4 * g) & (b > 60)
    return np.argwhere(mask)


def test_each_robot_gets_its_own_camera_image():
    # Robot 0 has a red cube 1 m straight ahead; robot 1 a blue cube 1 m ahead, 20 deg to its left.
    # Each sees only its own, so a mix-up of the cameras in the batched readback shows.
    b = np.radians(20)
    red, blue = (0.85, 0.10, 0.10), (0.10, 0.30, 0.90)
    env, obs, info = env_on(make_layout([(-1.5, -1.0, 0.0), (-1.5, 1.0, 0.0)],
                                        [(-0.35, -1.0), (-1.35 + np.cos(b), 1.0 + np.sin(b))],
                                        rgb=[red, blue], nest=(1.5, 0.0)), render=True)
    env.close()
    rgb = obs["rgb"]
    r0, b0, r1, b1 = (_colour_pixels(rgb[i], c) for i, c in ((0, "red"), (0, "blue"), (1, "red"), (1, "blue")))
    assert len(r0) > 20 and len(b1) > 20 and len(b0) == 0 and len(r1) == 0
    assert abs(r0[:, 1].mean() - 31.5) < 3          # straight ahead: centred horizontally
    assert r0[:, 0].mean() > 32                     # below the camera: lower half of the image
    assert b1[:, 1].mean() < 26                     # to the robot's left: left part of the image
