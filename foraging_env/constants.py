"""Fixed world specification shared by the layout sampler, the MJCF builder and the env.

Everything here is the same in every episode. Per-episode quantities live in `Layout`
(`scenarios.py`) and tunables in `DEFAULT_CFG` (`foraging_env.py`). Lengths are in metres;
box and cylinder sizes follow MuJoCo's half-size convention.
"""
from __future__ import annotations

import numpy as np

# Discrete actions, the same for every robot.
NOOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, PICK, DEPOSIT = range(7)
N_ACTIONS = 7
ACTION_NAMES = ("noop", "forward", "backward", "turn_left", "turn_right", "pick", "deposit")

# Padding: the model always has R_MAX resource and D_MAX distractor bodies. Unused ones are
# parked below the floor at PARK_Z and labelled exists=False; labels and priv_state have
# R_MAX rows.
R_MAX = 8
D_MAX = 4
PARK_Z = -1.0

TIMESTEP = 0.01           # physics step (100 Hz); the env takes cfg["substeps"] per decision

# Robot: planar base (slide x, slide y, hinge z) with a mast, a carry mount and a camera.
ROBOT_RADIUS = 0.2
ROBOT_HALF_H = 0.075      # base cylinder r=0.2, h=0.15
ROBOT_MASS = 5.0
MOUNT_Z = 0.42            # carry mount centre height
MOUNT_HALF = 0.02
CAM_POS = (0.15, 0.0, 0.35)   # robot frame: x forward, y left, z up
CAM_TILT_DEG = 10.0       # pitch below horizontal
CAM_FOVY = 90.0           # degrees; images are square, so the horizontal FOV is the same
ZNEAR = 0.01              # camera near plane (m), see mjcf_builder.build_mjcf
SHADOW_SIZE = 1024        # shadow map side (texels); MuJoCo's default 4096 costs 64 MiB of GPU memory per env

# Items. Resources are cubes and distractors spheres; they share PALETTE, so shape is the only cue.
CUBE_HALF = 0.075
SPHERE_RADIUS = 0.09
CARRY_Z = MOUNT_Z + MOUNT_HALF + CUBE_HALF   # carried cube centre height

# Nest: flat disc with a striped pole at its centre. Deposited cubes go to PILE_SLOTS.
NEST_RADIUS = 1.2         # a robot can deposit while its centre is inside this disc
NEST_HALF_H = 0.005
POLE_SEGMENTS = 8
POLE_HEIGHT = 2.0
POLE_RADIUS = 0.05
PILE_RADIUS = 0.45
# Pile slots for deposited items, relative to the nest centre, filled in order.
PILE_SLOTS = np.array([[PILE_RADIUS * np.cos(a), PILE_RADIUS * np.sin(a)]
                       for a in np.arange(R_MAX) * 2 * np.pi / R_MAX])
PILE_Z = 2 * NEST_HALF_H + CUBE_HALF

# Colours, RGB in [0, 1].
PALETTE = np.array([      # resources and distractors
    [0.85, 0.10, 0.10],  # red
    [0.10, 0.65, 0.15],  # green
    [0.10, 0.30, 0.90],  # blue
    [0.95, 0.80, 0.10],  # yellow
])
# rgb1/rgb2 of the gradient texture on each wall, ordered +x, +y, -x, -y.
WALL_RGB = np.array([
    [[0.75, 0.55, 0.55], [0.45, 0.25, 0.25]],
    [[0.55, 0.75, 0.55], [0.25, 0.45, 0.25]],
    [[0.55, 0.60, 0.80], [0.25, 0.30, 0.50]],
    [[0.80, 0.75, 0.50], [0.50, 0.45, 0.20]],
])
ROBOT_RGB = np.array([    # one per robot: white, dark grey, brown, teal
    [0.92, 0.92, 0.92],
    [0.20, 0.20, 0.22],
    [0.60, 0.40, 0.20],
    [0.30, 0.60, 0.62],
])
MAX_ROBOTS = len(ROBOT_RGB)
NEST_RGB = (0.55, 0.25, 0.65)
POLE_RGB = ((0.95, 0.95, 0.95), NEST_RGB)   # alternating pole segments
