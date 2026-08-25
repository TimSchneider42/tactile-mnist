"""Tests for the process-wide shared EGL context (tactile_mnist.shared_egl).

Requires a GPU/EGL-capable machine; run with ``PYOPENGL_PLATFORM=egl``.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import threading

import numpy as np
import pyrender
import trimesh
import trimesh.creation
from trimesh.visual.material import PBRMaterial

from tactile_mnist.shared_egl import (
    AlphaToCoverageOffscreenRenderer,
    SharedContextOffscreenRenderer,
    _shared,
)


# A pyrender.Mesh can only be bound to one Renderer, so every renderer gets its
# own (deterministically identical) scene.
def _make_scene() -> pyrender.Scene:
    scene = pyrender.Scene(
        ambient_light=np.array([0.3, 0.3, 0.3, 1.0]),
        bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    box.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=(200, 30, 30), metallicFactor=0.2, roughnessFactor=0.8
        )
    )
    scene.add(pyrender.Mesh.from_trimesh(box, smooth=False))
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 0.3
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4), pose=camera_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=camera_pose
    )
    return scene


def test_single_shared_context_and_safe_delete():
    r_small = SharedContextOffscreenRenderer(64, 64)
    r_large = SharedContextOffscreenRenderer(640, 480)
    scene_small = _make_scene()
    scene_large = _make_scene()

    for _ in range(10):
        color_small, depth_small = r_small.render(scene_small)
        color_large, depth_large = r_large.render(scene_large)
        assert color_small.shape == (64, 64, 3)
        assert color_large.shape == (480, 640, 3)
        assert color_small.any(), "small render is all black"
        assert color_large.any(), "large render is all black"
        assert (depth_small > 0).any()
        assert (depth_large > 0).any()

    assert _shared.context_create_count == 1

    # Deleting one renderer must not invalidate the shared context (plain
    # pyrender would eglTerminate the display here and kill every sibling).
    r_small.delete()
    color_large, _ = r_large.render(scene_large)
    assert color_large.any()
    assert _shared.context_create_count == 1


# The plain renderer must never be garbage collected: its delete()/__del__
# calls eglTerminate on the (shared) EGL display, which would invalidate the
# shared context for all tests that run afterwards.
_plain_renderer_keepalive = []


def test_pixel_parity_with_plain_offscreen_renderer():
    plain = AlphaToCoverageOffscreenRenderer(200, 150)
    _plain_renderer_keepalive.append(plain)
    color_plain, depth_plain = plain.render(_make_scene())

    shared = SharedContextOffscreenRenderer(200, 150)
    color_shared, depth_shared = shared.render(_make_scene())

    np.testing.assert_array_equal(color_plain, color_shared)
    np.testing.assert_array_equal(depth_plain, depth_shared)


def test_read_depth_false_skips_depth_and_matches_color():
    renderer = SharedContextOffscreenRenderer(200, 150)
    scene = _make_scene()

    color_full, depth_full = renderer.render(scene)
    color_only = renderer.render(scene, read_depth=False)

    assert isinstance(color_only, np.ndarray)
    np.testing.assert_array_equal(color_only, color_full)

    # The read_depth=False render must not degrade subsequent full renders.
    color_full_2, depth_full_2 = renderer.render(scene)
    np.testing.assert_array_equal(color_full_2, color_full)
    np.testing.assert_array_equal(depth_full_2, depth_full)

    rgba = renderer.render(scene, flags=pyrender.RenderFlags.RGBA, read_depth=False)
    assert rgba.shape == (150, 200, 4)

    try:
        renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY, read_depth=False)
    except ValueError:
        pass
    else:
        raise AssertionError("DEPTH_ONLY + read_depth=False should raise ValueError")


def test_render_from_multiple_threads():
    renderer = SharedContextOffscreenRenderer(64, 64)
    scene = _make_scene()
    errors = []

    def worker():
        try:
            color, _ = renderer.render(scene)
            assert color.any()
        except Exception as e:  # noqa: BLE001 - re-raised via the main thread
            errors.append(e)

    # Sequential renders from distinct threads: without releasing the context
    # after each render (eglMakeCurrent with EGL_NO_CONTEXT), the second
    # thread's eglMakeCurrent fails with EGL_BAD_ACCESS.
    for _ in range(2):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    color, _ = renderer.render(scene)
    assert color.any()
    assert not errors, f"thread render failed: {errors}"
    assert _shared.context_create_count == 1
