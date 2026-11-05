"""Procedural MJCF generation (arena, robots, resources, nest, occluders).

Collision scheme (contype/conaffinity bits):
  robot    1 / 3  collides with robots and obstacles
  obstacle 2 / 0  walls, occluders, nest pole, resources and distractors on the ground
  visual   0 / 0  floor, nest disc, robot mast/mount/camera housing
The env sets a resource's contype to 0 while it is carried or deposited.

Geom groups: 0 = rendered scene (robot cameras and visibility rays use group 0 only);
3 = reserved for invisible collision helpers.

Resources, distractors, occluders and the nest are mocap bodies, so their poses are set
at runtime; robots are planar (slide x, slide y, hinge z) with qpos = absolute (x, y, yaw).

Names the env looks up: joints and actuators r{i}_x, r{i}_y, r{i}_yaw; bodies robot{i},
res{r}, dis{d}, occ{k}, nest; cameras cam{i} (egocentric) and top (debug, whole arena).
Dimensions and colours come from `constants.py`.
"""
from __future__ import annotations

import numpy as np

from foraging_env.constants import (CAM_FOVY, CAM_POS, CAM_TILT_DEG, CUBE_HALF, D_MAX, MOUNT_HALF,
                                    MOUNT_Z, NEST_HALF_H, NEST_RADIUS, NEST_RGB, PARK_Z, POLE_HEIGHT,
                                    POLE_RADIUS, POLE_RGB, POLE_SEGMENTS, R_MAX, ROBOT_HALF_H, ROBOT_MASS,
                                    ROBOT_RADIUS, ROBOT_RGB, SHADOW_SIZE, SPHERE_RADIUS, TIMESTEP, ZNEAR)
from foraging_env.scenarios import Layout

CONTYPE_OBSTACLE = 2
SCENE_GROUPS = np.array([1, 0, 0, 0, 0, 0], np.uint8)   # geom group mask: rendered scene only

# Velocity actuators. kv = inertia / timestep: the implicitfast integrator then halves the
# velocity error every substep. The force cap (which keeps wall pushes gentle) is kv * 0.3:
# while clamped, the integrator gets no damping from the actuator, so a clamped substep must
# not jump over the unclamped band, or the speed limit-cycles around the target.
ROBOT_YAW_INERTIA = 0.5 * ROBOT_MASS * ROBOT_RADIUS**2
KV_LIN = ROBOT_MASS / TIMESTEP
KV_YAW = ROBOT_YAW_INERTIA / TIMESTEP
FORCE_LIN, FORCE_YAW = 0.3 * KV_LIN, 0.3 * KV_YAW


def _f(*v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _yaw_quat(yaw: float) -> str:
    return _f(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def build_mjcf(layout: Layout, *, shadows: bool = True, shadow_size: int = SHADOW_SIZE,
               offscreen: tuple[int, int] = (256, 256)) -> str:
    """MJCF for `layout`. Mocap bodies are written at the layout's poses and robots get a
    `layout` keyframe, so the XML is viewable on its own; the env re-applies poses itself.

    shadows: directional-light shadows (off is faster and makes RGB bit-exact across GL contexts).
    shadow_size: shadow map side in texels. SHADOW_SIZE suits the 64x64 cameras; renders at
        hundreds of pixels want MuJoCo's default 4096.
    offscreen: (width, height) of the offscreen framebuffer, the largest image renderable.
    """
    half = layout.arena_size / 2
    n = layout.n_robots
    t = np.deg2rad(CAM_TILT_DEG)
    cam_xyaxes = _f(0, -1, 0, np.sin(t), 0, np.cos(t))
    top_h = (half + 0.3) / np.tan(np.deg2rad(30))
    # MuJoCo scales the near plane by stat.extent. The compiled extent (~2x the arena) put the
    # default near plane 0.10-0.16 m out, so robots saw through walls and occluders they were
    # close to. Pin the extent so the near plane is ZNEAR in metres.
    extent = 2 * layout.arena_size

    asset = [
        '<texture name="floor" type="2d" builtin="checker" rgb1=".58 .58 .54" rgb2=".46 .46 .43" '
        'width="256" height="256"/>',
        f'<material name="floor" texture="floor" texrepeat="{_f(layout.arena_size, layout.arena_size)}" '
        'texuniform="true"/>',
        '<texture name="occluder" type="cube" builtin="checker" rgb1=".40 .36 .32" rgb2=".28 .26 .24" '
        'width="64" height="64"/>',
        '<material name="occluder" texture="occluder" texrepeat="2 2"/>',
    ]
    for k, (rgb1, rgb2) in enumerate(layout.wall_rgb):
        asset.append(f'<texture name="wall{k}" type="cube" builtin="gradient" rgb1="{_f(*rgb1)}" '
                     f'rgb2="{_f(*rgb2)}" width="64" height="64"/>')
        asset.append(f'<material name="wall{k}" texture="wall{k}"/>')

    world = [
        f'<light name="sun" type="directional" pos="0 0 6" dir="{_f(*layout.light_dir)}" '
        f'diffuse="{_f(*layout.light_diffuse)}" castshadow="{str(shadows).lower()}"/>',
        f'<geom name="floor" type="plane" size="{_f(half, half, 0.1)}" material="floor"/>',
        f'<camera name="top" pos="{_f(0, 0, top_h)}" fovy="60"/>',
    ]
    wall_h, wall_t = 0.5, 0.05
    for k, ang in enumerate((0, np.pi / 2, np.pi, 3 * np.pi / 2)):
        c = half + wall_t
        world.append(f'<geom name="wall{k}" class="obstacle" type="box" material="wall{k}" '
                     f'pos="{_f(c * np.cos(ang), c * np.sin(ang), wall_h)}" quat="{_yaw_quat(ang)}" '
                     f'size="{_f(wall_t, half + 2 * wall_t, wall_h)}"/>')

    # Nest: flat disc + striped pole; deposited items go to PILE_SLOTS around the pole.
    nest = [f'<body name="nest" mocap="true" pos="{_f(*layout.nest_pos, 0)}">',
            f'  <geom name="nest_disc" type="cylinder" size="{_f(NEST_RADIUS, NEST_HALF_H)}" '
            f'pos="{_f(0, 0, NEST_HALF_H)}" rgba="{_f(*NEST_RGB, 1)}"/>']
    seg = POLE_HEIGHT / POLE_SEGMENTS / 2
    for s in range(POLE_SEGMENTS):
        nest.append(f'  <geom name="pole{s}" class="obstacle" type="cylinder" '
                    f'size="{_f(POLE_RADIUS, seg)}" pos="{_f(0, 0, (2 * s + 1) * seg)}" '
                    f'rgba="{_f(*POLE_RGB[s % 2], 1)}"/>')
    nest.append('</body>')
    world += nest

    for i in range(n):
        world += [
            f'<body name="robot{i}">',
            f'  <joint name="r{i}_x" type="slide" axis="1 0 0"/>',
            f'  <joint name="r{i}_y" type="slide" axis="0 1 0"/>',
            f'  <joint name="r{i}_yaw" type="hinge" axis="0 0 1"/>',
            f'  <geom name="robot{i}" class="robot" type="cylinder" size="{_f(ROBOT_RADIUS, ROBOT_HALF_H)}" '
            f'pos="{_f(0, 0, ROBOT_HALF_H)}" mass="{ROBOT_MASS}" rgba="{_f(*ROBOT_RGB[i], 1)}"/>',
            f'  <geom name="mast{i}" class="visual" type="cylinder" size="0.02" '
            f'fromto="{_f(0, 0, 2 * ROBOT_HALF_H, 0, 0, MOUNT_Z)}" rgba="{_f(*ROBOT_RGB[i], 1)}"/>',
            f'  <geom name="mount{i}" class="visual" type="box" size="{_f(0.06, 0.06, MOUNT_HALF)}" '
            f'pos="{_f(0, 0, MOUNT_Z)}" rgba="{_f(*ROBOT_RGB[i], 1)}"/>',
            f'  <geom name="camhouse{i}" class="visual" type="box" size="0.04 0.05 0.035" '
            f'pos="{_f(CAM_POS[0] - 0.05, 0, CAM_POS[2])}" rgba=".1 .1 .1 1"/>',
            f'  <camera name="cam{i}" pos="{_f(*CAM_POS)}" xyaxes="{cam_xyaxes}" fovy="{CAM_FOVY:g}"/>',
            '</body>',
        ]

    for r in range(R_MAX):
        if r < len(layout.resource_pos):
            pos, rgb = (*layout.resource_pos[r], CUBE_HALF), layout.resource_rgb[r]
        else:
            pos, rgb = (0, 0, PARK_Z), (0.5, 0.5, 0.5)
        world += [f'<body name="res{r}" mocap="true" pos="{_f(*pos)}">',
                  f'  <geom name="res{r}" class="obstacle" type="box" size="{_f(CUBE_HALF, CUBE_HALF, CUBE_HALF)}" '
                  f'rgba="{_f(*rgb, 1)}"/>',
                  '</body>']
    for d in range(D_MAX):
        if d < len(layout.distractor_pos):
            pos, rgb = (*layout.distractor_pos[d], SPHERE_RADIUS), layout.distractor_rgb[d]
        else:
            pos, rgb = (0, 0, PARK_Z), (0.5, 0.5, 0.5)
        world += [f'<body name="dis{d}" mocap="true" pos="{_f(*pos)}">',
                  f'  <geom name="dis{d}" class="obstacle" type="sphere" size="{SPHERE_RADIUS}" '
                  f'rgba="{_f(*rgb, 1)}"/>',
                  '</body>']
    for k, o in enumerate(layout.occluders):
        hz = o.size[-1]
        world += [f'<body name="occ{k}" mocap="true" pos="{_f(*o.pos, 0)}" quat="{_yaw_quat(o.yaw)}">',
                  f'  <geom name="occ{k}" class="obstacle" type="{o.kind}" size="{_f(*o.size)}" '
                  f'pos="{_f(0, 0, hz)}" material="occluder"/>',
                  '</body>']

    actuators = []
    for i in range(n):
        for j, (kv, f) in zip(("x", "y", "yaw"), ((KV_LIN, FORCE_LIN),) * 2 + ((KV_YAW, FORCE_YAW),)):
            actuators.append(f'<velocity name="r{i}_{j}" joint="r{i}_{j}" kv="{kv:g}" '
                             f'forcerange="{_f(-f, f)}"/>')

    key_qpos = _f(*np.asarray(layout.robot_poses).ravel())

    return "\n".join([
        '<mujoco model="multi_robot_foraging">',
        '<compiler angle="radian" autolimits="true"/>',
        f'<option timestep="{TIMESTEP}" integrator="implicitfast"/>',
        f'<statistic extent="{extent:g}" center="0 0 0.5"/>',
        '<visual>',
        f'  <global offwidth="{offscreen[0]}" offheight="{offscreen[1]}"/>',
        f'  <map znear="{ZNEAR / extent:.6g}"/>',
        '  <headlight ambient=".35 .35 .35" diffuse=".25 .25 .25" specular="0 0 0"/>',
        f'  <quality shadowsize="{shadow_size if shadows else 0}"/>',
        '</visual>',
        '<default>',
        '  <geom contype="0" conaffinity="0" group="0"/>',
        '  <default class="robot"><geom contype="1" conaffinity="3"/></default>',
        f'  <default class="obstacle"><geom contype="{CONTYPE_OBSTACLE}" conaffinity="0"/></default>',
        '  <default class="visual"><geom contype="0" conaffinity="0" density="0"/></default>',
        '</default>',
        '<asset>', *("  " + a for a in asset), '</asset>',
        '<worldbody>', *("  " + w for w in world), '</worldbody>',
        '<actuator>', *("  " + a for a in actuators), '</actuator>',
        f'<keyframe><key name="layout" qpos="{key_qpos}"/></keyframe>',
        '</mujoco>',
    ])
