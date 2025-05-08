import jax
import jax.numpy as jnp

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from .tactile_renderer import Device
from .tactile_renderer_jax import TactileRendererJAX


class DepthRendererJAX(TactileRendererJAX):
    def __init__(self, device: Device | None = None):
        super().__init__(channels=1, device=device)

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return output_size

    def _render_direct(
        self, depth: jax.Array, output_size: tuple[int, int]
    ) -> jax.Array:
        gel_thickness_m = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000
        depth = jnp.clip(depth, 0, gel_thickness_m)
        return depth[..., None] / gel_thickness_m
