"""Process-wide shared EGL context for pyrender offscreen rendering.

Plain :class:`pyrender.OffscreenRenderer` creates one EGL display + context per
instance and its ``delete()`` calls ``eglTerminate`` on the display, which
invalidates every other context in the process. In addition, its
``make_uncurrent()`` is a no-op, so a context stays current on whatever thread
last rendered — making renders from a different thread (e.g. a JAX io_callback
thread) illegal under the EGL spec (``EGL_BAD_ACCESS``).

:class:`SharedContextOffscreenRenderer` is a drop-in replacement that routes all
rendering through a single lazily-created EGL context shared by the whole
process. Each instance still owns its own :class:`pyrender.Renderer` (and thus
its own framebuffers and mesh bindings — a ``pyrender.Mesh`` can only be bound
to one ``Renderer``), so the existing scene/mesh partitioning must be kept.
Renders are serialized by a lock and the context is properly released
(``eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)``) after
each render, so any thread may render. The shared display is never terminated.
"""

from __future__ import annotations

import os
import threading

from pyrender import OffscreenRenderer, RenderFlags
from pyrender.renderer import Renderer


class _SharedEglPlatform:
    """Singleton owning the process-wide EGL display + context.

    The context is created lazily inside the first render call (i.e. on the
    rendering thread). All access must happen while holding :attr:`lock`, which
    also serializes rendering across all :class:`SharedContextOffscreenRenderer`
    instances.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._platform = None
        # Instrumentation for tests: number of EGL contexts ever created.
        self.context_create_count = 0

    def ensure_context_current(self):
        """Create the shared context if necessary and make it current.

        Must be called with :attr:`lock` held.
        """
        if self._platform is None:
            from pyrender.platforms import egl

            device_id = int(os.environ.get("EGL_DEVICE_ID", "0"))
            device = egl.get_device_by_index(device_id)
            # The pbuffer surface size is irrelevant: pyrender renders into its
            # own framebuffer objects, never into the default framebuffer.
            platform = egl.EGLPlatform(1, 1, device=device)
            platform.init_context()  # also makes the context current
            self._platform = platform
            self.context_create_count += 1
        else:
            self._platform.make_current()

    def release(self):
        """Release the context from the current thread.

        ``EGLPlatform.make_uncurrent()`` is a no-op, so we call
        ``eglMakeCurrent`` with ``EGL_NO_CONTEXT`` directly. Without this, the
        context would stay current on the last rendering thread and could never
        legally be made current on another one. Must be called with
        :attr:`lock` held.
        """
        if self._platform is None:
            return
        from OpenGL import EGL

        if not EGL.eglMakeCurrent(
            self._platform._egl_display,
            EGL.EGL_NO_SURFACE,
            EGL.EGL_NO_SURFACE,
            EGL.EGL_NO_CONTEXT,
        ):
            raise RuntimeError("eglMakeCurrent(EGL_NO_CONTEXT) failed")


_shared = _SharedEglPlatform()


class SharedContextOffscreenRenderer:
    """Drop-in replacement for :class:`pyrender.OffscreenRenderer`.

    Owns a :class:`pyrender.Renderer` (framebuffers sized to this viewport,
    created lazily on first render) but no EGL context — all instances render
    through the single shared context. ``delete()`` frees this instance's GL
    resources only and never invalidates other renderers.

    GL resources are freed only by an explicit :meth:`delete` call; there is
    deliberately no ``__del__``, as garbage collection may run on a thread that
    holds the render lock or has no right to make the context current.
    """

    def __init__(self, viewport_width, viewport_height, point_size=1.0):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size
        self._renderer = None

    @property
    def viewport_width(self):
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = int(value)

    @property
    def viewport_height(self):
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = int(value)

    @property
    def point_size(self):
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def render(self, scene, flags=RenderFlags.NONE, seg_node_map=None):
        with _shared.lock:
            _shared.ensure_context_current()
            try:
                if self._renderer is None:
                    self._renderer = Renderer(self.viewport_width, self.viewport_height)
                self._renderer.viewport_width = self.viewport_width
                self._renderer.viewport_height = self.viewport_height
                self._renderer.point_size = self.point_size
                return self._renderer.render(
                    scene, flags | RenderFlags.OFFSCREEN, seg_node_map
                )
            finally:
                _shared.release()

    def delete(self):
        """Free this renderer's GL resources (framebuffers, mesh bindings).

        The shared EGL display/context is left untouched, so other renderers
        keep working.
        """
        with _shared.lock:
            if self._renderer is None:
                return
            _shared.ensure_context_current()
            try:
                self._renderer.delete()
            finally:
                self._renderer = None
                _shared.release()


def make_offscreen_renderer(
    viewport_width: int, viewport_height: int
) -> OffscreenRenderer | SharedContextOffscreenRenderer:
    """Create an offscreen renderer, sharing one EGL context per process.

    Falls back to plain :class:`pyrender.OffscreenRenderer` when
    ``PYOPENGL_PLATFORM`` is not ``egl`` (e.g. pyglet on a desktop) — the
    shared-context machinery is EGL-specific.
    """
    if os.environ.get("PYOPENGL_PLATFORM") == "egl":
        return SharedContextOffscreenRenderer(viewport_width, viewport_height)
    return OffscreenRenderer(viewport_width, viewport_height)
