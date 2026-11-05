"""Offscreen rendering of fixed cameras (the robots' egocentric views).

A `CameraRenderer` owns one GL context for its lifetime and one `RenderTarget` (MjrContext +
MjvScene) per compiled model, so switching between cached models never recreates the GL
context: creating a GL context costs ~50 ms, a RenderTarget ~5 ms, a 64x64 frame ~0.2 ms.

The cameras of one render() call are drawn side by side into the offscreen buffer and read back
together: each readback stalls until the GPU is done, and with many envs on one GPU these
stalls, not drawing, limit the frame rate.

The GL backend comes from MUJOCO_GL, which MuJoCo reads when it is first imported (`foraging_env`
defaults it to egl on Linux). The GL context is created on first use, so an env with
render=False never needs a working GL setup. Every call makes the context current first, so
several renderers can share a process. EGL contexts do not survive fork: create renderers
inside worker processes.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(eq=False)
class RenderTarget:
    """Render state for one compiled model; create with `CameraRenderer.attach`."""
    mjr: mujoco.MjrContext
    scene: mujoco.MjvScene


class CameraRenderer:
    """Renders square `size` x `size` RGB frames from named cameras, drawing only the geom
    groups set in `geomgroup` (a length-6 uint8 mask)."""

    def __init__(self, size: int, geomgroup: np.ndarray):
        self.size = size
        self._gl = None
        self._targets: list[RenderTarget] = []
        self._strips: dict[int, tuple[mujoco.MjrRect, np.ndarray]] = {}   # n cameras -> readback rect, buffer
        self._vopt = mujoco.MjvOption()
        self._vopt.geomgroup[:] = geomgroup
        self._vopt.sitegroup[:] = 0
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def attach(self, model: mujoco.MjModel) -> RenderTarget:
        """Upload `model`'s meshes and textures. The model's offscreen buffer must be at least
        (n * size) x size to render n cameras per call."""
        if self._gl is None:
            from mujoco.gl_context import GLContext   # imported here so render=False needs no GL
            self._gl = GLContext(self.size, self.size)
        self._gl.make_current()
        mjr = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_50)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, mjr)
        target = RenderTarget(mjr, mujoco.MjvScene(model, maxgeom=256))
        self._targets.append(target)
        return target

    def detach(self, target: RenderTarget) -> None:
        self._gl.make_current()
        target.mjr.free()
        self._targets.remove(target)

    def render(self, target: RenderTarget, model: mujoco.MjModel, data: mujoco.MjData,
               cam_ids: np.ndarray, out: np.ndarray) -> None:
        """Render camera cam_ids[k] into out[k], a uint8 [K, size, size, 3] array, top row first.
        Camera k is drawn at x = k * size of the offscreen buffer; one readback returns all K."""
        self._gl.make_current()
        s, n = self.size, len(cam_ids)
        if n not in self._strips:
            self._strips[n] = (mujoco.MjrRect(0, 0, n * s, s), np.empty((s, n * s, 3), np.uint8))
        rect, strip = self._strips[n]
        for k, cam_id in enumerate(cam_ids):
            self._cam.fixedcamid = cam_id
            mujoco.mjv_updateScene(model, data, self._vopt, None, self._cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, target.scene)
            mujoco.mjr_render(mujoco.MjrRect(k * s, 0, s, s), target.scene, target.mjr)
        mujoco.mjr_readPixels(strip, None, rect, target.mjr)
        out[:] = strip[::-1].reshape(s, n, s, 3).transpose(1, 0, 2, 3)   # GL rows are bottom-up

    def close(self) -> None:
        """Free every RenderTarget and the GL context. The renderer can be used again after."""
        if self._gl is not None:
            self._gl.make_current()
            for target in self._targets:
                target.mjr.free()
            self._gl.free()
            self._gl = None
        self._targets.clear()
